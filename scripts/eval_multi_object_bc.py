"""Multi-object pick-and-place with CNN detection + per-object state-BC policies.

Pipeline per round:
  1. CNN scan -> detect objects on pegs
  2. Sort detected objects by distance to basket (closest first)
  3. For each object: run its per-object BC policy (APPROACH -> GRIP -> CARRY -> RELEASE)
  4. go_home() between objects (no env.reset mid-round — deadlocks cameras)
  5. Constrained-random spawn ensures tools land on slots their policies can handle

Proven mechanics integrated from eval_sequential.py:
  - obs normalization (obs_mean / obs_std from checkpoint)
  - subtract_frame_transforms for robot-root-frame obs override
  - closed-loop servo approach within servo_dist
  - safe lift above SAFE_LIFT_Z before lateral moves
  - go_home() with IK-relative EE servo between objects (no raw joint driving)
  - constrained slot assignment per tool
  - tight RELEASE_XY + RELEASE_HOLD + physics landing check

Usage (GUI):
    cd /mnt/extra/IsaacLab && ./isaaclab.sh -p \\
        /mnt/extra/ai_ws/AI_Robot_Manipulation_UR3e/scripts/eval_multi_object_bc.py \\
        --num_rounds 3 --enable_cameras

Headless:
    Add --headless to the command above.

Without CNN (GT physics positions):
    Add --no_cnn --headless
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-object CNN + state-BC orchestrator.")
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Brush-Play-v0")
parser.add_argument("--ckpt_dir", default=str(REPO_ROOT / "checkpoints"),
                    help="Directory containing policy_state_bc_*.pth files.")
parser.add_argument("--seg_ckpt", default=str(REPO_ROOT / "checkpoints/air2_segmentation_v3.pth"))
parser.add_argument("--num_rounds", type=int, default=1)
parser.add_argument("--max_picks", type=int, default=2,
                    help="Max objects to attempt per round (default 2).")
parser.add_argument("--max_steps_per_object", type=int, default=2000)
parser.add_argument("--servo_dist", type=float, default=0.18,
                    help="EE distance threshold to switch from BC to closed-loop servo.")
parser.add_argument("--detect_every", type=int, default=10,
                    help="Re-run CNN every N steps during approach (unused if --no_cnn).")
parser.add_argument("--no_cnn", action="store_true",
                    help="Skip CNN — use GT physics positions for detection + ordering.")
parser.add_argument("--episode_length_s", type=float, default=300.0)
parser.add_argument("--out", default="eval_results/multi_object_bc.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.no_cnn

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import cv2
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from isaaclab_ext.tasks.air2_franka.objects import OBJECT_SPECS, OBJECT_BY_LABEL
from isaaclab_ext.tasks.air2_franka.cnn.model import build_model, build_resnet_model
from isaaclab_ext.tasks.air2_franka.cnn.postprocess import extract_detections

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_ORDER = ["brush", "pliers", "scissors", "screwdriver"]

TOOL_SCENE_KEY = {
    "brush":       "object",
    "pliers":      "tool_pliers",
    "scissors":    "tool_scissors",
    "screwdriver": "tool_screwdriver",
}

BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])
HOME_JOINT_TOL   = 0.01

# Phase-loop constants (proven in eval_sequential.py)
NEAR_THRESH     = 250
GRIP_HOLD       = 50
MIN_CARRY_STEPS = 200
RELEASE_XY      = 0.12
RELEASE_HOLD    = 120
SUCCESS_RADIUS  = 0.25
SUCCESS_Z_MAX   = 1.27

# Safe-transit height — lift EE above pegs before lateral moves
SAFE_LIFT_Z     = 1.95
LIFT_MAX_STEPS  = 120

# CNN thresholds
CNN_CONF_THRESH = 0.35
PEG_Z_MIN       = 1.10

# Constrained-random spawn: each tool lands on slots its policy can handle
SLOT_WORLD = {
    "R0": [-4.272, -5.960, 1.611],
    "R1": [-4.445, -5.960, 1.611],
    "R2": [-4.272, -5.960, 1.326],
    "R3": [-4.445, -5.960, 1.326],
}
TOOL_QUAT = [0.7071, 0.0, 0.0, -0.7071]
VALID_ASSIGNMENTS = [
    {"object": "R2", "tool_pliers": "R0", "tool_screwdriver": "R1", "tool_scissors": "R3"},
    {"object": "R1", "tool_pliers": "R0", "tool_screwdriver": "R2", "tool_scissors": "R3"},
]

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim, output_dim, hidden_dims, activation="elu"):
    act = nn.ELU if activation == "elu" else nn.ReLU
    layers, in_d = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_d, h), act()]
        in_d = h
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


def load_policy(ckpt_path: str, device: str):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _make_mlp(blob["input_dim"], blob["action_dim"],
                      blob["hidden_dims"], blob.get("activation", "elu")).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    obs_mean = obs_std = None
    if blob.get("obs_mean") is not None:
        obs_mean = torch.as_tensor(blob["obs_mean"], dtype=torch.float32, device=device)
        obs_std  = torch.as_tensor(blob["obs_std"],  dtype=torch.float32, device=device)
    obs_aligned = bool(blob.get("obs_aligned", False))
    return model, obs_mean, obs_std, obs_aligned


def load_seg_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(ckpt.get("num_classes", 9))
    if str(ckpt.get("backbone", "")).startswith("resnet"):
        model = build_resnet_model(num_classes=num_classes, pretrained=False).to(device)
    else:
        model = build_model(num_classes=num_classes,
                            base_channels=int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tool placement (no env.reset mid-round)
# ---------------------------------------------------------------------------

def place_tools(env, device, assignment, placed, attempted, settle_steps: int = 30):
    """Write tool poses without env.reset().

    - placed: scene_keys that landed in basket -> move to basket
    - attempted: scene_keys that were attempted but failed -> leave where they are
    - remaining: not yet attempted -> restore to assigned peg slot
    """
    env_ids = torch.tensor([0], device=device)
    quat = torch.tensor(TOOL_QUAT, device=device).unsqueeze(0)
    basket = BASKET_POS_LOCAL.to(device).unsqueeze(0)
    for scene_key, slot in assignment.items():
        if scene_key in placed:
            # Landed successfully — put in basket
            asset = env.unwrapped.scene[scene_key]
            pos = env.unwrapped.scene.env_origins[env_ids] + basket
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
        elif scene_key in attempted:
            # Failed attempt — leave where it fell, don't respawn
            pass
        else:
            # Not yet attempted — restore to peg
            asset = env.unwrapped.scene[scene_key]
            pos = env.unwrapped.scene.env_origins[env_ids] + \
                torch.tensor(SLOT_WORLD[slot], device=device).unsqueeze(0)
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
    neutral = torch.zeros(1, 7, device=device)
    neutral[0, 6] = 1.0
    for _ in range(settle_steps):
        env.step(neutral)


# ---------------------------------------------------------------------------
# go_home — return arm to start pose between objects
# ---------------------------------------------------------------------------

def go_home(env, robot, device: str, max_steps: int = 600):
    """Return to home pose: raw joint drive → action_manager reset → IK flush.

    Uses raw joint-level control to converge joints to default (avoids IK
    redundancy — EE-only servo lands in wrong joint config). Then resets the
    action manager to clear stale IK internal state, and runs an extended
    env.step() flush so the IK controller re-initializes from the now-correct
    joint/EE position.

    Steps:
      1) Lift EE above pegs via env.step() (stays in IK pipeline)
      2) Raw joint drive to default joint positions (tight tolerance)
      3) action_manager.reset() to clear stale IK target + last_action
      4) Extended env.step(neutral) flush to re-sync IK controller
    """
    ee = env.unwrapped.scene["ee_frame"]

    # 1) Open gripper + lift straight up to clear pegs
    for _ in range(LIFT_MAX_STEPS):
        if ee.data.target_pos_w[0, 0, 2].item() > SAFE_LIFT_Z:
            break
        lift = torch.zeros(1, 7, device=device)
        lift[0, 2] = 0.12   # +z position delta (lift)
        lift[0, 6] = 1.0    # gripper open
        env.step(lift)

    # 2) Raw joint drive to near-default position.
    home = robot.data.default_joint_pos[:1].clone()
    home_vel = torch.zeros_like(home)
    converged = False
    for i in range(max_steps):
        robot.set_joint_position_target(home)
        robot.write_data_to_sim()
        env.unwrapped.sim.step()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        if (robot.data.joint_pos - home).abs().max() < HOME_JOINT_TOL:
            converged = True
            break

    # 3) Snap joints to EXACT default position + zero velocity.
    #    The PD controller can't converge perfectly; small residuals (~0.01-0.05 rad)
    #    accumulate across go_home cycles and push later policies OOD.
    #    Writing exact positions eliminates this drift entirely.
    robot.write_joint_state_to_sim(home, home_vel)
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.physics_dt)

    # 4) Reset action manager — clears _prev_action, _action, _raw_actions,
    #    so the next env.step() starts fresh with no stale IK target.
    env.unwrapped.action_manager.reset()

    # 5) Extended neutral flush: env.step(neutral) calls process_actions(zero)
    #    which sets ee_pos_des = current_ee_pos (relative mode), then compute()
    #    returns current joint_pos + 0 delta. This re-syncs the IK controller
    #    to the exact default joint configuration.
    neutral = torch.zeros(1, 7, device=device)
    neutral[0, 6] = 1.0  # gripper open
    for _ in range(100):
        env.step(neutral)

    joint_err = (robot.data.joint_pos - home).abs().max().item()
    print(f"[orch] go_home: converged={converged} joint_err={joint_err:.4f}rad", flush=True)


# ---------------------------------------------------------------------------
# CNN detection + visualization
# ---------------------------------------------------------------------------

# Per-label colors (BGR for OpenCV)
LABEL_COLORS_BGR = {
    "brush":       (80, 175, 76),     # green
    "pliers":      (0, 152, 255),     # orange
    "scissors":    (54, 67, 244),     # red
    "screwdriver": (243, 150, 33),    # blue
    "robot":       (180, 180, 180),
    "basket":      (0, 200, 200),
    "table":       (100, 100, 100),
}


def run_cnn(seg_model, cam, device: str, debug: bool = False):
    """Return (positions, raw_detections, rgb, pred_mask).

    positions: {label: position_world} for tools on pegs.
    raw_detections: full list from extract_detections (for overlay).
    rgb: raw camera image (H, W, 3) uint8.
    pred_mask: argmax segmentation (H, W) uint8.
    """
    rgb   = cam.data.output["rgb"][0].detach().cpu().numpy().astype("uint8")
    depth = cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
    intr  = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
    pos_w = cam.data.pos_w[0].detach().cpu().numpy()
    quat_w = cam.data.quat_w_ros[0].detach().cpu().numpy()
    H, W = rgb.shape[0], rgb.shape[1]
    seg_dev = next(seg_model.parameters()).device
    with torch.no_grad():
        rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(seg_dev) / 255.0
        rgb_224 = torch.nn.functional.interpolate(rgb_t, size=(224, 224),
                                                  mode="bilinear", align_corners=False)
        logits_224 = seg_model(rgb_224)
        logits = torch.nn.functional.interpolate(logits_224, size=(H, W),
                                                 mode="bilinear", align_corners=False)
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_mask = probs.argmax(axis=0).astype(np.uint8)

    dets = extract_detections(logits, depth=depth, intrinsic_matrix=intr,
                               pos_w=pos_w, rot_w_quat=quat_w,
                               min_confidence=CNN_CONF_THRESH)
    positions = {}
    for d in dets:
        if d["class_id"] == 0:
            continue
        pw = d["position_world"]
        if debug:
            reason = ("KEEP" if pw is not None and pw[2] > PEG_Z_MIN
                      else f"DROP(z={pw[2]:.3f})" if pw is not None else "DROP(no pos)")
            print(f"[cnn]   {d['label']:<11} conf={d.get('confidence', float('nan')):.2f} "
                  f"pos={None if pw is None else [round(v,3) for v in pw]} -> {reason}", flush=True)
        if pw is not None and pw[2] > PEG_Z_MIN:
            positions[d["label"]] = pw
    return positions, dets, rgb, pred_mask


def draw_detection_overlay(rgb, pred_mask, dets, positions, basket_w,
                           closest_label=None):
    """Draw bounding boxes + labels + distances on the camera image.

    Returns annotated BGR image for cv2.imshow().
    """
    # RGB -> BGR for OpenCV, make writable copy
    vis = cv2.cvtColor(rgb[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
    H, W = vis.shape[:2]

    # Semi-transparent segmentation overlay (tools only)
    SKIP_LABELS = {"table", "robot", "basket", "environment", "background"}
    overlay = vis.copy()
    for d in dets:
        if d["class_id"] == 0 or d["label"] in SKIP_LABELS:
            continue
        color = LABEL_COLORS_BGR.get(d["label"], (200, 200, 200))
        mask = (pred_mask == d["class_id"])
        overlay[mask] = color
    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    # Compute distances for detected tools
    tool_dists = {}
    for d in dets:
        label = d["label"]
        pw = positions.get(label)
        if pw is not None and basket_w is not None:
            dist = sum((a - b) ** 2 for a, b in zip(pw, basket_w)) ** 0.5
            tool_dists[label] = dist

    # Draw bounding boxes + labels (skip non-tool classes)
    SKIP_LABELS = {"table", "robot", "basket", "environment", "background"}
    for d in dets:
        if d["class_id"] == 0 or d["centroid_px"] is None:
            continue
        label = d["label"]
        if label in SKIP_LABELS:
            continue
        color = LABEL_COLORS_BGR.get(label, (200, 200, 200))
        conf = d.get("confidence", 0.0)

        # Bounding box from segmentation mask
        mask = (pred_mask == d["class_id"]).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = 4
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W - 1, x2 + pad), min(H - 1, y2 + pad)

        thickness = 3 if label == closest_label else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label text
        txt = f"{label} ({conf:.0%})"
        if label in tool_dists:
            txt += f" d={tool_dists[label]:.2f}m"
        if label == closest_label:
            txt += " << NEXT"

        # Draw label background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, 1)
        ty = max(y1 - 6, th + 2)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(vis, txt, (x1 + 2, ty - 2), font, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Title bar (bottom) — show which tool is closest
    if closest_label:
        title = f"CNN Object Detection - Picking: {closest_label.upper()} (closest to basket)"
    else:
        title = "CNN Object Detection - No tool detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 0.55, 1)
    cv2.rectangle(vis, (0, H - th - 12), (tw + 20, H), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, H - 8), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def get_gt_object_positions(env, origins):
    """Fallback: get object positions from physics (--no_cnn mode)."""
    result = {}
    for label, scene_key in TOOL_SCENE_KEY.items():
        try:
            obj = env.unwrapped.scene[scene_key]
            pos_w = obj.data.root_pos_w[0].cpu().tolist()
            pos_local = [pos_w[i] - origins[0, i].item() for i in range(3)]
            if pos_local[2] > PEG_Z_MIN:
                result[label] = pos_w
        except KeyError:
            pass
    return result


# ---------------------------------------------------------------------------
# Per-object rollout (proven phase machine from eval_sequential.py)
# ---------------------------------------------------------------------------

def run_single_object(env, policy, obs_mean, obs_std, obs_aligned,
                      target_scene_key: str, target_label: str,
                      device, max_steps: int) -> dict:
    """Run phase-based rollout for one object. Returns result dict."""
    robot    = env.unwrapped.scene["robot"]
    tool_obj = env.unwrapped.scene[target_scene_key]
    ee       = env.unwrapped.scene["ee_frame"]
    basket_dev = BASKET_POS_LOCAL.to(device)
    env_origins = env.unwrapped.scene.env_origins

    phase         = torch.zeros(1, dtype=torch.long, device=device)
    prev_phase    = torch.full((1,), -1, dtype=torch.long, device=device)
    near_counter  = torch.zeros(1, dtype=torch.long, device=device)
    grip_steps    = torch.zeros(1, dtype=torch.long, device=device)
    carry_steps   = torch.zeros(1, dtype=torch.long, device=device)
    release_steps = torch.zeros(1, dtype=torch.long, device=device)
    released      = torch.zeros(1, dtype=torch.bool, device=device)

    _PHASE_NAMES = {0: "APPROACH", 1: "GRIP", 2: "CARRY", 3: "RELEASE"}
    landed = torch.zeros(1, dtype=torch.bool, device=device)
    dist_gt = torch.zeros(1, device=device)

    for step in range(max_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            # Track TRUE object pose (CNN is too coarse for grasp precision)
            obj_pos_w = tool_obj.data.root_pos_w.clone()

            # Object in robot-root frame (for obs override + servo)
            obj_root, _ = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w)

            # Object in local frame (for distance calcs)
            obj_local = obj_pos_w - env_origins

            # EE positions
            ee_pos  = ee.data.target_pos_w[:, 0, :] - env_origins
            ee_obj_dist = torch.linalg.norm(ee_pos - obj_local, dim=-1)

            # Build obs
            obs_dict = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)
            obs_policy = obs_policy.clone()

            # Override object position dims with robot-root-frame position
            obs_policy[:, 18:21] = obj_root

            if obs_aligned:
                id_quat = torch.zeros(1, 4, device=device)
                id_quat[:, 0] = 1.0
                obs_policy[:, 21:24] = obj_root
                obs_policy[:, 24:28] = id_quat

            phase_bit = (phase >= 2).float().unsqueeze(-1)
            obs_input = torch.cat([obs_policy, phase_bit], dim=-1)

            # One-time obs diagnostic at step 0
            if step == 0:
                print(f"  [{target_label}] joint_pos[0:7]={[round(v,3) for v in obs_input[0,:7].cpu().tolist()]}"
                      f"  obj_pos[18:21]={[round(v,3) for v in obs_input[0,18:21].cpu().tolist()]}"
                      f"  last_action[28:35]={[round(v,3) for v in obs_input[0,28:35].cpu().tolist()]}"
                      f"  ee_pos[35:38]={[round(v,3) for v in obs_input[0,35:38].cpu().tolist()]}", flush=True)

            if obs_mean is not None:
                obs_input = (obs_input - obs_mean) / obs_std

            action = policy(obs_input)

            # Phase 0 — APPROACH
            near_counter = torch.where(
                (phase == 0) & (ee_obj_dist < 0.08), near_counter + 1, near_counter)
            phase = torch.where(
                (phase == 0) & (near_counter >= NEAR_THRESH),
                torch.ones_like(phase), phase)

            # Phase 1 — GRIP
            grip_steps = torch.where(phase == 1, grip_steps + 1, grip_steps)
            phase = torch.where(
                (phase == 1) & (grip_steps >= GRIP_HOLD),
                torch.full_like(phase, 2), phase)

            # Phase 2 — CARRY
            carry_steps = torch.where(phase == 2, carry_steps + 1, carry_steps)
            xy_dist = torch.linalg.norm(obj_local[:, :2] - basket_dev[:2], dim=-1)
            phase = torch.where(
                (phase == 2) & (carry_steps >= MIN_CARRY_STEPS) & (xy_dist < RELEASE_XY),
                torch.full_like(phase, 3), phase)

            released      = released | (phase == 3)
            release_steps = torch.where(phase == 3, release_steps + 1, release_steps)

            # Arm action
            arm_action = action[:, :-1].clamp(-0.15, 0.15)

            # SERVO: closed-loop approach using object position
            if args_cli.servo_dist > 0:
                ee_root, _ = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w,
                    ee.data.target_pos_w[:, 0, :])
                servo_cmd = ((obj_root - ee_root) * 2.0).clamp(-0.12, 0.12)
                servo_active = (phase == 0) & (ee_obj_dist < args_cli.servo_dist)
                arm_action[:, 0:3] = torch.where(
                    servo_active.unsqueeze(-1).expand(-1, 3),
                    servo_cmd, arm_action[:, 0:3])

            # Hold arm still during GRIP and RELEASE
            hold_arm = (phase == 1) | (phase == 3)
            arm_action = torch.where(
                hold_arm.unsqueeze(-1).expand_as(arm_action),
                torch.zeros_like(arm_action), arm_action)

            grip_val = torch.where(phase == 0, torch.ones(1, device=device),
                       torch.where(phase == 3, torch.ones(1, device=device),
                                   -torch.ones(1, device=device)))
            action_out = torch.cat([arm_action, grip_val.unsqueeze(-1)], dim=-1)

            # Phase transition logging
            if phase[0] != prev_phase[0]:
                print(f"  [{target_label}] step={step:4d}  "
                      f"{_PHASE_NAMES.get(prev_phase[0].item(),'')} -> "
                      f"{_PHASE_NAMES.get(phase[0].item(),'')}  "
                      f"ee_dist={ee_obj_dist[0].item():.3f}m  "
                      f"xy_basket={xy_dist[0].item():.3f}m", flush=True)
            # Periodic CARRY diagnostics (every 200 steps)
            if phase[0].item() == 2 and step % 200 == 0:
                arm_mag = arm_action[0].norm().item()
                ee_w = ee.data.target_pos_w[0, 0].cpu().tolist()
                obj_w = obj_pos_w[0].cpu().tolist()
                print(f"  [{target_label}] CARRY step={step:4d}  "
                      f"xy_basket={xy_dist[0].item():.3f}m  "
                      f"arm_mag={arm_mag:.4f}  "
                      f"arm_action={[round(v, 3) for v in arm_action[0].cpu().tolist()]}  "
                      f"ee={[round(v,3) for v in ee_w]}  "
                      f"obj={[round(v,3) for v in obj_w]}", flush=True)
            prev_phase = phase.clone()

            env.step(action_out)

            # Success check (GT physics)
            dist_gt = torch.linalg.norm(obj_local - basket_dev, dim=1)
            landed = released & (dist_gt < SUCCESS_RADIUS) & (obj_local[:, 2] <= SUCCESS_Z_MAX)

            done = (phase == 3) & (release_steps >= RELEASE_HOLD)
            if done[0]:
                break

    return {
        "object": target_label,
        "scene_key": target_scene_key,
        "steps": step + 1,
        "landed": bool(landed[0].item()),
        "released": bool(released[0].item()),
        "final_dist": float(dist_gt[0].item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env_cfg.episode_length_s = args_cli.episode_length_s

    # Disable env terminations so tools can physically settle
    for name in list(vars(env_cfg.terminations).keys()):
        if name.startswith("_") or name in ("time_out", "time_outs"):
            continue
        if getattr(env_cfg.terminations, name) is not None:
            setattr(env_cfg.terminations, name, None)

    # Add main_camera for CNN (matches segmentation_env_cfg.py)
    if not args_cli.no_cnn:
        env_cfg.scene.main_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/main_camera",
            update_period=0.0,
            height=360, width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-4.8, -5.2, 2.2),
                rot=(0.1598, -0.3477, 0.8395, -0.3857),
                convention="ros",
            ),
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    env_origins = env.unwrapped.scene.env_origins
    basket_dev = BASKET_POS_LOCAL.to(device)
    robot = env.unwrapped.scene["robot"]



    # ---- discover per-object checkpoints -----------------------------------
    ckpt_dir = Path(args_cli.ckpt_dir)
    policies = {}
    for spec in OBJECT_SPECS:
        candidates = [
            f"policy_state_bc_mimic_{spec.label}_v2.pth",
            f"policy_state_bc_mimic_v2.pth" if spec.label == "brush" else None,
            f"policy_state_bc_{spec.label}.pth",
            f"policy_state_bc_{spec.label}_mimic.pth",
        ]
        for pattern in candidates:
            if pattern is None:
                continue
            p = ckpt_dir / pattern
            if p.exists():
                model, obs_mean, obs_std, obs_aligned = load_policy(str(p), device)
                policies[spec.label] = {
                    "scene_key": spec.scene_key,
                    "model": model,
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "obs_aligned": obs_aligned,
                }
                print(f"[orch] loaded {spec.label}: {p.name}", flush=True)
                break

    if not policies:
        print("[FATAL] No per-object checkpoints found in", ckpt_dir, flush=True)
        env.close()
        return

    print(f"[orch] found {len(policies)} policies: {list(policies.keys())}", flush=True)

    # ---- CNN segmentation model -------------------------------------------
    if args_cli.no_cnn:
        seg_model = cam = None
        print("[orch] --no_cnn: using GT physics positions.", flush=True)
    else:
        seg_model = load_seg_model(args_cli.seg_ckpt, "cuda:0")
        cam = env.unwrapped.scene["main_camera"]
        print(f"[orch] seg model loaded from {args_cli.seg_ckpt}", flush=True)

    # ---- run rounds --------------------------------------------------------
    all_rounds = []

    # First reset is safe (before cameras are fully active)
    first_round = True

    for round_idx in range(args_cli.num_rounds):
        if not simulation_app.is_running():
            break

        # Constrained-random spawn — only env.reset() on the first round.
        # Mid-run env.reset() with cameras active deadlocks the render pipeline.
        assignment = random.choice(VALID_ASSIGNMENTS)
        placed = set()
        if first_round:
            env.reset()
            first_round = False
        else:
            go_home(env, robot, device)
        attempted_keys = set()  # scene_keys of tools attempted (failed = leave on ground)
        place_tools(env, device, assignment, placed, attempted_keys)

        print(f"\n{'#'*60}", flush=True)
        print(f"[orch] ROUND {round_idx+1}/{args_cli.num_rounds}", flush=True)
        print(f"[orch] spawn: " + "  ".join(
            f"{k}={v}" for k, v in assignment.items()), flush=True)

        round_results = []
        attempted = set()
        picks_done = 0

        while picks_done < args_cli.max_picks:
            if not simulation_app.is_running():
                break

            # Detect objects (CNN or GT fallback)
            raw_dets = []
            cnn_rgb = cnn_mask = None
            if args_cli.no_cnn:
                detections = get_gt_object_positions(env, env_origins)
            else:
                detections, raw_dets, cnn_rgb, cnn_mask = run_cnn(
                    seg_model, cam, device, debug=True)

            # Find remaining tools with detections + policies
            remaining = [t for t in TOOL_ORDER if t not in attempted]
            candidates = [t for t in remaining if t in detections and t in policies]

            # Auto-skip detected tools without policies
            for t in remaining:
                if t in detections and t not in policies:
                    print(f"[orch] {t} detected but no policy — skipping", flush=True)
                    attempted.add(t)
                    round_results.append({"object": t, "steps": 0, "landed": False,
                                          "released": False, "final_dist": -1.0,
                                          "skipped": True})

            if not candidates:
                # No more detectable tools — skip undetected ones
                for t in remaining:
                    if t not in attempted:
                        print(f"[orch] {t} not detected — skipping", flush=True)
                        attempted.add(t)
                break

            # Sort candidates by distance to basket (closest first)
            basket_w = (env_origins[0] + basket_dev).tolist()
            def _basket_dist(t):
                p = detections[t]
                return sum((a - b) ** 2 for a, b in zip(p, basket_w)) ** 0.5

            order = sorted(candidates, key=_basket_dist)
            print(f"[orch] pick order (closest to basket): "
                  + ", ".join(f"{t}({_basket_dist(t):.2f}m)" for t in order), flush=True)

            # Save CNN overlay with bounding boxes + distances
            if cnn_rgb is not None and cnn_mask is not None:
                vis = draw_detection_overlay(
                    cnn_rgb, cnn_mask, raw_dets, detections,
                    basket_w, closest_label=order[0])
                vis_dir = out_path.parent / (out_path.stem + "_overlays")
                vis_dir.mkdir(parents=True, exist_ok=True)
                scan_idx = len([f for f in vis_dir.iterdir() if f.suffix == ".png"])
                vis_path = vis_dir / f"round{round_idx+1}_scan{scan_idx:02d}.png"
                cv2.imwrite(str(vis_path), vis)
                print(f"[orch] overlay saved: {vis_path}", flush=True)

            # Pick the closest one
            tool_name = order[0]
            pol = policies[tool_name]
            scene_key = pol["scene_key"]

            # Log CNN vs GT position
            grasp_obj = env.unwrapped.scene[scene_key]
            gt_pos = grasp_obj.data.root_pos_w[0].cpu().tolist()
            det_pos = detections.get(tool_name)
            if det_pos is not None:
                err = sum((a - b) ** 2 for a, b in zip(gt_pos, det_pos)) ** 0.5
                print(f"[orch] {tool_name}: CNN={[round(v,3) for v in det_pos]}  "
                      f"GT={[round(v,3) for v in gt_pos]}  err={err:.3f}m", flush=True)

            print(f"\n[orch] >>> picking {tool_name} <<<", flush=True)
            result = run_single_object(
                env, pol["model"], pol["obs_mean"], pol["obs_std"], pol["obs_aligned"],
                scene_key, tool_name, device, args_cli.max_steps_per_object,
            )
            result["skipped"] = False
            round_results.append(result)
            attempted.add(tool_name)
            attempted_keys.add(scene_key)
            picks_done += 1

            if result["landed"]:
                placed.add(scene_key)

            status = "LANDED" if result["landed"] else ("RELEASED" if result["released"] else "TIMEOUT")
            print(f"[orch] {tool_name}: {status}  dist={result['final_dist']:.3f}m  "
                  f"steps={result['steps']}", flush=True)

            # go_home + restore arrangement before next tool
            if picks_done < args_cli.max_picks:
                go_home(env, robot, device)
                place_tools(env, device, assignment, placed, attempted_keys)
                print(f"[orch] ready for next tool ({len(placed)} in basket)", flush=True)

        attempted_results = [r for r in round_results if not r.get("skipped")]
        n_landed = sum(1 for r in attempted_results if r.get("landed", False))
        print(f"\n[orch] round {round_idx+1} complete: {n_landed}/{len(attempted_results)} landed",
              flush=True)
        all_rounds.append({
            "round": round_idx + 1,
            "pick_order": [r["object"] for r in round_results if not r.get("skipped")],
            "landed": n_landed,
            "total": len([r for r in round_results if not r.get("skipped")]),
            "objects": round_results,
        })

    # ---- summary -----------------------------------------------------------
    total_attempts = sum(r["total"] for r in all_rounds)
    total_landed = sum(r["landed"] for r in all_rounds)
    per_tool = {}
    for tool in TOOL_ORDER:
        att = [r for rd in all_rounds for r in rd["objects"]
               if r["object"] == tool and not r.get("skipped")]
        landed = sum(1 for r in att if r.get("landed"))
        per_tool[tool] = f"{landed}/{len(att)}"

    summary = {
        "num_rounds": len(all_rounds),
        "overall_land_rate": f"{total_landed}/{total_attempts}",
        "per_tool": per_tool,
        "rounds": all_rounds,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"[orch] DONE: {total_landed}/{total_attempts} landed "
          f"across {len(all_rounds)} rounds", flush=True)
    print(f"[orch] per tool: {per_tool}", flush=True)
    print(f"[orch] results -> {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
