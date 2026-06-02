"""Sequential multi-tool eval: run 4 per-tool BC policies back-to-back in one episode.

Pipeline per episode:
  env.reset() → [CNN scan → policy loop → go_home()] × 4 tools → log results

CNN provides object position, replacing GT physics in obs + servo + ee_dist.
GT physics kept as fallback if CNN confidence drops below threshold.
go_home() runs between tools — no env.reset() mid-episode.

Usage:
    DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_sequential.py \
        --brush_ckpt      checkpoints/working/R0_R1_R2_ALL_WORKS_FOR_BRUSH/policy_state_bc_mimic_v2.pth \
        --pliers_ckpt     checkpoints/policy_state_bc_pliers.pth \
        --scissors_ckpt   checkpoints/policy_state_bc_scissors.pth \
        --screwdriver_ckpt checkpoints/policy_state_bc_screwdriver_merged.pth \
        --num_episodes 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--brush_ckpt",
    default="checkpoints/working/R0_R1_R2_ALL_WORKS_FOR_BRUSH/policy_state_bc_mimic_v2.pth")
parser.add_argument("--pliers_ckpt",
    default="checkpoints/working/PLIERS_R0_R2_WORKS/PLIERS_R0_R2_WORKS/policy_state_bc_mimic_pliers_v2.pth")
parser.add_argument("--scissors_ckpt",
    default="checkpoints/policy_state_bc_scissors.pth")
parser.add_argument("--screwdriver_ckpt",
    default="checkpoints/policy_state_bc_screwdriver.pth")
parser.add_argument("--task",             default="Isaac-AIR2-Robotis-Franka-Brush-Play-v0")
parser.add_argument("--num_episodes",     type=int,   default=10)
parser.add_argument("--max_steps_per_tool", type=int, default=2000)
parser.add_argument("--seg_ckpt",         default="checkpoints/air2_segmentation_unet.pth")
parser.add_argument("--detect_every",     type=int,   default=10)
parser.add_argument("--no_cnn",           action="store_true",
                    help="Skip CNN entirely — use GT physics positions. No board_camera loaded.")
parser.add_argument("--servo_dist",       type=float, default=0.18)
parser.add_argument("--episode_length_s", type=float, default=300.0)
parser.add_argument("--out",              default="eval_results/sequential.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.no_cnn

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from isaaclab_ext.tasks.air2_franka.cnn.model import build_model, build_resnet_model
from isaaclab_ext.tasks.air2_franka.cnn.postprocess import extract_detections

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_ORDER = ["brush", "pliers", "scissors", "screwdriver"]

# Scene key for each tool in the Brush play env
TOOL_SCENE_KEY = {
    "brush":       "object",
    "pliers":      "tool_pliers",
    "scissors":    "tool_scissors",
    "screwdriver": "tool_screwdriver",
}

BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])
HOME_JOINTS      = torch.tensor([-1.5708, -0.569, 0.0, -2.26892803, 0.0, 3.037, 0.741, 0.04, 0.04])
HOME_JOINT_TOL   = 0.05

# Phase-loop constants (mirror eval_state_bc.py)
NEAR_THRESH     = 250
GRIP_HOLD       = 50
MIN_CARRY_STEPS = 200
RELEASE_XY      = 0.12
RELEASE_HOLD    = 120
SUCCESS_RADIUS  = 0.25
SUCCESS_Z_MAX   = 1.27

# CNN: minimum confidence to trust a detection; below this falls back to physics
CNN_CONF_THRESH = 0.35
# Peg-area world-frame Z threshold — tool is "on peg" if z > this
PEG_Z_MIN = 1.20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim, output_dim, hidden_dims, activation="elu"):
    import torch.nn as nn
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
    if ckpt.get("backbone") == "resnet":
        model = build_resnet_model(num_classes=num_classes, pretrained=False).to(device)
    else:
        model = build_model(num_classes=num_classes,
                            base_channels=int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def go_home(env, robot, device: str, max_steps: int = 200) -> None:
    home = HOME_JOINTS.to(device).unsqueeze(0)
    for _ in range(max_steps):
        robot.set_joint_position_target(home)
        env.unwrapped.sim.step()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        if (robot.data.joint_pos - home).abs().max() < HOME_JOINT_TOL:
            break


def run_cnn(seg_model, board_cam, device: str) -> dict[str, list[float]]:
    """Return {label: position_world} for tools detected on the pegs."""
    rgb   = board_cam.data.output["rgb"][0].detach().cpu().numpy().astype("uint8")
    depth = board_cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
    intr  = board_cam.data.intrinsic_matrices[0].detach().cpu().numpy()
    pos_w = board_cam.data.pos_w[0].detach().cpu().numpy()
    quat_w = board_cam.data.quat_w_ros[0].detach().cpu().numpy()
    with torch.no_grad():
        rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to("cpu") / 255.0
        logits = seg_model(rgb_t)
    dets = extract_detections(logits, depth=depth, intrinsic_matrix=intr,
                               pos_w=pos_w, rot_w_quat=quat_w,
                               min_confidence=CNN_CONF_THRESH)
    result = {}
    for d in dets:
        if d["class_id"] == 0 or d["position_world"] is None:
            continue
        pw = d["position_world"]
        if pw[2] > PEG_Z_MIN:  # only count detections in peg area
            result[d["label"]] = pw
    return result


# ---------------------------------------------------------------------------
# Per-tool policy loop
# ---------------------------------------------------------------------------

def run_tool(
    tool_name: str,
    env,
    policy,
    obs_mean,
    obs_std,
    obs_aligned: bool,
    seg_model,
    board_cam,
    device: str,
    cnn_pos_world: list[float],   # initial CNN position from pre-loop scan
) -> dict:
    """Run one tool's BC policy until LANDED or timeout. Returns result dict."""
    robot    = env.unwrapped.scene["robot"]
    tool_obj = env.unwrapped.scene[TOOL_SCENE_KEY[tool_name]]
    ee       = env.unwrapped.scene["ee_frame"]
    basket_dev = BASKET_POS_LOCAL.to(device)
    env_origins = env.unwrapped.scene.env_origins

    # Phase state
    phase         = torch.zeros(1, dtype=torch.long, device=device)
    prev_phase    = torch.full((1,), -1, dtype=torch.long, device=device)
    near_counter  = torch.zeros(1, dtype=torch.long, device=device)
    grip_steps    = torch.zeros(1, dtype=torch.long, device=device)
    carry_steps   = torch.zeros(1, dtype=torch.long, device=device)
    release_steps = torch.zeros(1, dtype=torch.long, device=device)
    released      = torch.zeros(1, dtype=torch.bool, device=device)

    _PHASE_NAMES = {0: "APPROACH", 1: "GRIP", 2: "CARRY", 3: "RELEASE"}

    # CNN position cache (world frame tensor, shape (1,3))
    cnn_world = torch.tensor(cnn_pos_world, dtype=torch.float32, device=device).unsqueeze(0)

    for step in range(args_cli.max_steps_per_tool):
        with torch.inference_mode():

            # Refresh CNN position every detect_every steps (skipped in --no_cnn mode)
            if not args_cli.no_cnn and step % args_cli.detect_every == 0 and step > 0:
                fresh = run_cnn(seg_model, board_cam, device)
                if tool_name in fresh:
                    cnn_world = torch.tensor(fresh[tool_name], dtype=torch.float32,
                                             device=device).unsqueeze(0)

            # CNN world → robot-root frame (for obs override + SERVO)
            obj_root, _ = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, cnn_world)

            # CNN local frame (world - env_origins, for dist calcs)
            cnn_local = cnn_world - env_origins

            # EE positions
            ee_pos  = ee.data.target_pos_w[:, 0, :] - env_origins
            ee_obj_dist = torch.linalg.norm(ee_pos - cnn_local, dim=-1)

            # Build obs
            obs_dict   = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)
            obs_policy = obs_policy.clone()

            # Override object position dims with CNN-derived position
            obs_policy[:, 18:21] = obj_root

            # Obs-alignment: dims 21:28 = object pos + identity quat
            if obs_aligned:
                id_quat = torch.zeros(1, 4, device=device)
                id_quat[:, 0] = 1.0
                obs_policy[:, 21:24] = obj_root
                obs_policy[:, 24:28] = id_quat

            phase_bit = (phase >= 2).float().unsqueeze(-1)
            obs_input = torch.cat([obs_policy, phase_bit], dim=-1)
            if obs_mean is not None:
                obs_input = (obs_input - obs_mean) / obs_std

            action = policy(obs_input)

            # Phase transitions
            near_counter = torch.where(
                (phase == 0) & (ee_obj_dist < 0.08), near_counter + 1, near_counter)
            phase = torch.where(
                (phase == 0) & (near_counter >= NEAR_THRESH),
                torch.ones_like(phase), phase)

            grip_steps = torch.where(phase == 1, grip_steps + 1, grip_steps)
            phase = torch.where(
                (phase == 1) & (grip_steps >= GRIP_HOLD),
                torch.full_like(phase, 2), phase)

            carry_steps = torch.where(phase == 2, carry_steps + 1, carry_steps)

            # xy_dist_to_basket uses GT physics for release trigger
            obj_pos_gt = tool_obj.data.root_pos_w
            obj_local_gt = obj_pos_gt - env_origins
            xy_dist = torch.linalg.norm(obj_local_gt[:, :2] - basket_dev[:2], dim=-1)
            phase = torch.where(
                (phase == 2) & (carry_steps >= MIN_CARRY_STEPS) & (xy_dist < RELEASE_XY),
                torch.full_like(phase, 3), phase)

            released      = released | (phase == 3)
            release_steps = torch.where(phase == 3, release_steps + 1, release_steps)

            # Arm + gripper
            arm_action = action[:, :-1].clamp(-0.15, 0.15)

            # SERVO: closed-loop approach using CNN position
            if args_cli.servo_dist > 0:
                ee_root, _ = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w,
                    ee.data.target_pos_w[:, 0, :])
                servo_cmd = ((obj_root - ee_root) * 2.0).clamp(-0.12, 0.12)
                servo_active = (phase == 0) & (ee_obj_dist < args_cli.servo_dist)
                arm_action[:, 0:3] = torch.where(
                    servo_active.unsqueeze(-1).expand(-1, 3),
                    servo_cmd, arm_action[:, 0:3])

            hold_arm = (phase == 1) | (phase == 3)
            arm_action = torch.where(
                hold_arm.unsqueeze(-1).expand_as(arm_action),
                torch.zeros_like(arm_action), arm_action)

            grip_val = torch.where(phase == 0, torch.ones(1, device=device),
                       torch.where(phase == 3, torch.ones(1, device=device),
                                   -torch.ones(1, device=device)))
            action = torch.cat([arm_action, grip_val.unsqueeze(-1)], dim=-1)

            # Phase transition print
            if phase[0] != prev_phase[0]:
                print(f"  [{tool_name}] step={step:4d}  "
                      f"{_PHASE_NAMES.get(prev_phase[0].item(),'')} → "
                      f"{_PHASE_NAMES.get(phase[0].item(),'')}  "
                      f"ee_dist={ee_obj_dist[0].item():.3f}m  "
                      f"xy_basket={xy_dist[0].item():.3f}m", flush=True)
            prev_phase = phase.clone()

            env.step(action)

            # Success check (GT physics)
            dist_gt = torch.linalg.norm(obj_local_gt - basket_dev, dim=1)
            landed = released & (dist_gt < SUCCESS_RADIUS) & (obj_local_gt[:, 2] <= SUCCESS_Z_MAX)

            done = (phase == 3) & (release_steps >= RELEASE_HOLD)
            if done[0]:
                break

    return {
        "tool":    tool_name,
        "steps":   step + 1,
        "landed":  bool(landed[0].item()),
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

    # Disable terminations so tools can settle
    for name in list(vars(env_cfg.terminations).keys()):
        if name.startswith("_") or name in ("time_out", "time_outs"):
            continue
        if getattr(env_cfg.terminations, name) is not None:
            setattr(env_cfg.terminations, name, None)

    if not args_cli.no_cnn:
        env_cfg.scene.board_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/board_camera",
        update_period=0.0,
        height=360, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.0, -3.5, 1.8),
            rot=(-0.2068, 0.2807, 0.7545, -0.5560),
            convention="ros",
        ),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device

    # Load all 4 policies — skip missing checkpoints gracefully
    ckpt_map = {
        "brush":       args_cli.brush_ckpt,
        "pliers":      args_cli.pliers_ckpt,
        "scissors":    args_cli.scissors_ckpt,
        "screwdriver": args_cli.screwdriver_ckpt,
    }
    policies = {}
    for tool, ckpt in ckpt_map.items():
        if not Path(ckpt).exists():
            print(f"[seq] {tool}: checkpoint not found ({ckpt}) — tool will be skipped", flush=True)
            continue
        model, obs_mean, obs_std, obs_aligned = load_policy(ckpt, device)
        policies[tool] = (model, obs_mean, obs_std, obs_aligned)
        print(f"[seq] loaded {tool} policy from {ckpt}", flush=True)

    if args_cli.no_cnn:
        seg_model = board_cam = None
        print("[seq] --no_cnn: using GT physics positions.", flush=True)
    else:
        seg_model = load_seg_model(args_cli.seg_ckpt, "cpu")
        board_cam = env.unwrapped.scene["board_camera"]
        print("[seq] seg model loaded.", flush=True)
    robot     = env.unwrapped.scene["robot"]
    print(f"[seq] Starting {args_cli.num_episodes} episodes.", flush=True)

    basket_dev  = BASKET_POS_LOCAL.to(device)
    all_episodes = []

    for ep_idx in range(args_cli.num_episodes):
        env.reset()
        ep_results = []
        print(f"\n[seq] ===== EPISODE {ep_idx + 1}/{args_cli.num_episodes} =====", flush=True)

        attempted = set()
        while len(attempted) < len(TOOL_ORDER):
            detections = {} if args_cli.no_cnn else run_cnn(seg_model, board_cam, device)
            remaining = [t for t in TOOL_ORDER if t not in attempted]
            on_peg    = [t for t in remaining if t in detections]

            print(f"\n[seq] CNN detected on peg: {on_peg if on_peg else 'none'}", flush=True)
            print(f"[seq] Remaining: {remaining}", flush=True)
            print(f"[seq] Commands: {on_peg} | 'auto' (next in order) | 'skip <tool>' | 'done'",
                  flush=True)

            try:
                user_input = input("[seq] Pick tool > ").strip().lower()
            except EOFError:
                user_input = "auto"

            if user_input == "done":
                break
            elif user_input == "auto":
                # Pick next in priority order that's on peg
                tool_name = next((t for t in remaining if t in detections), None)
                if tool_name is None:
                    print("[seq] No tools detected on peg — ending episode", flush=True)
                    break
            elif user_input.startswith("skip "):
                skip_tool = user_input.split()[1]
                if skip_tool in remaining:
                    attempted.add(skip_tool)
                    ep_results.append({"tool": skip_tool, "steps": 0,
                                       "landed": False, "released": False,
                                       "final_dist": -1.0, "skipped": True})
                    print(f"[seq] skipped {skip_tool}", flush=True)
                continue
            elif user_input in TOOL_ORDER:
                tool_name = user_input
                if tool_name not in policies:
                    print(f"[seq] no policy loaded for {tool_name} — skipping", flush=True)
                    continue
            else:
                print(f"[seq] Unknown command '{user_input}'", flush=True)
                continue

            if tool_name not in detections:
                tool_obj_fb = env.unwrapped.scene[TOOL_SCENE_KEY[tool_name]]
                gt_pos = tool_obj_fb.data.root_pos_w[0].cpu().tolist()
                print(f"[seq] {tool_name} not detected — using GT physics pos", flush=True)
                cnn_pos = gt_pos
            else:
                cnn_pos = detections[tool_name]

            print(f"[seq] running {tool_name} at ({cnn_pos[0]:.3f},{cnn_pos[1]:.3f},{cnn_pos[2]:.3f})",
                  flush=True)

            model, obs_mean, obs_std, obs_aligned = policies[tool_name]
            result = run_tool(
                tool_name, env, model, obs_mean, obs_std, obs_aligned,
                seg_model, board_cam, device, cnn_pos,
            )
            result["skipped"] = False
            ep_results.append(result)
            attempted.add(tool_name)
            print(f"[seq] {tool_name}: LANDED={result['landed']}  "
                  f"dist={result['final_dist']:.3f}m  steps={result['steps']}", flush=True)

            go_home(env, robot, device)
            print(f"[seq] returned home", flush=True)

        landed_n = sum(1 for r in ep_results if r["landed"])
        print(f"[seq] episode {ep_idx+1} done — {landed_n}/{len(TOOL_ORDER)} landed", flush=True)
        all_episodes.append({"ep_idx": ep_idx, "tools": ep_results, "landed_count": landed_n})

    # Summary
    total_tools    = sum(len(e["tools"]) for e in all_episodes)
    total_landed   = sum(e["landed_count"] for e in all_episodes)
    per_tool_rates = {}
    for tool in TOOL_ORDER:
        attempts = [r for e in all_episodes for r in e["tools"] if r["tool"] == tool and not r["skipped"]]
        landed   = sum(1 for r in attempts if r["landed"])
        per_tool_rates[tool] = f"{landed}/{len(attempts)}"

    summary = {
        "num_episodes": args_cli.num_episodes,
        "overall_land_rate": f"{total_landed}/{total_tools}",
        "per_tool": per_tool_rates,
        "episodes": all_episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[seq] DONE — overall {total_landed}/{total_tools} landed", flush=True)
    print(f"[seq] per tool: {per_tool_rates}", flush=True)
    print(f"[seq] results → {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
