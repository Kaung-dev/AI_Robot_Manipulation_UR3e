"""Roll out a trained STATE-only BC policy (from train_state_bc.py) and report metrics.

This is the smaller MLP that mirrors rsl_rl's actor architecture — the one we
use to warm-start PPO ([bc_to_ppo.py](bc_to_ppo.py)). Same metrics + JSON
schema as [eval_bc.py](eval_bc.py) so [plot_training_curves.py](plot_training_curves.py)
can plot a state-BC line on the same axes.

Use in GUI (drop --headless to watch):
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/eval_state_bc.py ^
        --state_bc_ckpt checkpoints/policy_state_bc.pth ^
        --task Isaac-AIR2-Franka-Play-v0 ^
        --enable_cameras ^
        --num_envs 1 --num_episodes 5

Headless eval (for metrics + JSON):
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/eval_state_bc.py ^
        --state_bc_ckpt checkpoints/policy_state_bc.pth ^
        --task Isaac-AIR2-Franka-v0 ^
        --headless --enable_cameras ^
        --num_envs 4 --num_episodes 20 ^
        --out eval_results/state_bc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained STATE-only BC policy.")
parser.add_argument("--state_bc_ckpt", required=True,
                    help="Path to checkpoints/policy_state_bc.pth from train_state_bc.py.")
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Brush-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--reset_delay", type=int, default=100,
                    help="Visual pause steps after release before resetting (~2s at 50Hz).")
parser.add_argument("--episode_length_s", type=float, default=40.0)
parser.add_argument("--servo_dist", type=float, default=0.18,
                    help="When EE is within this dist of the object during APPROACH, servo the EE "
                         "straight onto the object (robot-root frame) instead of trusting BC alone. "
                         "Fixes R2 downward overshoot. Set 0 to disable.")
parser.add_argument("--out", default="eval_results/state_bc_cnn.json")
parser.add_argument("--seg_ckpt", default="checkpoints/air2_segmentation_unet.pth",
                    help="Path to U-Net segmentation checkpoint for object detection.")
parser.add_argument("--detect_every", type=int, default=10,
                    help="Run CNN detection every N steps (default 10 to keep sim smooth).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # board_camera requires camera rendering

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


def _load_seg_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "unet")
    num_classes = int(ckpt.get("num_classes", 9))
    if backbone == "resnet":
        model = build_resnet_model(num_classes=num_classes, pretrained=False).to(device)
    else:
        model = build_model(num_classes=num_classes,
                            base_channels=int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _make_mlp(input_dim, output_dim, hidden_dims, activation="elu"):
    import torch.nn as nn
    act = nn.ELU if activation == "elu" else nn.ReLU
    layers = []
    in_d = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_d, h), act()]
        in_d = h
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


# Same proxy success metric used by eval_bc.py / eval_ppo.py.
BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])
BASKET_REACH_RADIUS = 0.40

# Default joint config: Franka base defaults + env overrides (joint1=-90°, joint4=-2.269)
HOME_JOINTS = torch.tensor([-1.5708, -0.569, 0.0, -2.26892803, 0.0, 3.037, 0.741, 0.04, 0.04])
HOME_JOINT_TOL = 0.05  # rad — stop early if within this of home


def go_home(env, robot, num_envs: int, device: str, max_steps: int = 200) -> None:
    """Drive the robot to its default joint configuration without a full env reset.

    Bypasses the Cartesian-delta action space by writing joint position targets
    directly. The high-stiffness implicit PD actuators snap the arm to home in
    ~50-100 steps. Tool objects are untouched — only the robot moves.
    """
    home = HOME_JOINTS.to(device).unsqueeze(0).expand(num_envs, -1)
    for _ in range(max_steps):
        robot.set_joint_position_target(home)
        env.unwrapped.sim.step()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        if (robot.data.joint_pos - home).abs().max() < HOME_JOINT_TOL:
            break


def load_state_bc_policy(ckpt_path: str, device: str):
    """Reconstruct the MLP from the checkpoint's metadata + state_dict."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in blob:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'state_dict' key")
    input_dim  = blob["input_dim"]
    action_dim = blob["action_dim"]
    hidden_dims = blob["hidden_dims"]
    activation  = blob.get("activation", "elu")
    model = _make_mlp(input_dim, action_dim, hidden_dims, activation).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    obs_mean = obs_std = None
    if blob.get("obs_mean") is not None and blob.get("obs_std") is not None:
        obs_mean = torch.as_tensor(blob["obs_mean"], dtype=torch.float32, device=device)
        obs_std  = torch.as_tensor(blob["obs_std"],  dtype=torch.float32, device=device)
    obs_aligned = bool(blob.get("obs_aligned", False))
    print(f"[eval-state-bc] loaded MLP: input={input_dim} hidden={hidden_dims} "
          f"output={action_dim} act={activation} | normalized={obs_mean is not None} aligned={obs_aligned}")
    return model, obs_mean, obs_std, obs_aligned


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s

    # Add board_camera with RGB + depth — same position as segmentation_env_cfg.py.
    # Camera pose may need tuning if the new scene layout differs from the old env.
    env_cfg.scene.board_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/board_camera",
        update_period=0.0,
        height=360,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.0, -3.5, 1.8),
            rot=(-0.2068, 0.2807, 0.7545, -0.5560),
            convention="ros",
        ),
    )

    # Disable env terminations so the object can physically settle after release
    # before we score the episode. Keep only time_out so episodes can't run forever.
    _disabled = []
    for _tname in list(vars(env_cfg.terminations).keys()):
        if _tname.startswith("_") or _tname in ("time_out", "time_outs"):
            continue
        if getattr(env_cfg.terminations, _tname) is not None:
            setattr(env_cfg.terminations, _tname, None)
            _disabled.append(_tname)
    print(f"[eval-state-bc] disabled env terminations: {_disabled}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    policy, obs_mean, obs_std, obs_aligned = load_state_bc_policy(args_cli.state_bc_ckpt, device=device)

    seg_model = _load_seg_model(args_cli.seg_ckpt, device)
    board_cam = env.unwrapped.scene["board_camera"]
    print(f"[eval-state-bc-cnn] segmentation model loaded from {args_cli.seg_ckpt}", flush=True)
    print(f"[eval-state-bc] task={args_cli.task} envs={num_envs} target_episodes={args_cli.num_episodes} episode_length_s={env_cfg.episode_length_s}",
          flush=True)

    basket_pos_dev = BASKET_POS_LOCAL.to(device)

    ep_step      = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward    = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)
    # Phase state: 0=APPROACH, 1=GRIP, 2=CARRY, 3=RELEASE
    phase         = torch.zeros(num_envs, dtype=torch.long, device=device)
    prev_phase    = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    near_counter  = torch.zeros(num_envs, dtype=torch.long, device=device)
    grip_steps    = torch.zeros(num_envs, dtype=torch.long, device=device)
    carry_steps   = torch.zeros(num_envs, dtype=torch.long, device=device)
    release_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    released      = torch.zeros(num_envs, dtype=torch.bool, device=device)
    NEAR_THRESH      = 250   # steps within 0.08m before closing (5s)
    GRIP_HOLD        = 50    # steps holding gripper closed while gripping (1s)
    MIN_CARRY_STEPS  = 200   # min carry steps before XY check activates (~5s)
    RELEASE_XY       = 0.12  # tight XY radius to trigger release (centred over basket)
    RELEASE_HOLD     = 120   # steps to hold after gripper opens so object settles (~2.4s)
    SUCCESS_RADIUS   = 0.25  # 3D dist from basket centre to count as landed
    SUCCESS_Z_MAX    = 1.27  # object must settle at/below this height (basket z=1.14)
    saved_episodes: list[dict] = []

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            ee    = env.unwrapped.scene["ee_frame"]
            brush = env.unwrapped.scene["object"]

            robot   = env.unwrapped.scene["robot"]
            ee_pos  = ee.data.target_pos_w[:, 0, :] - env.unwrapped.scene.env_origins
            ee_quat = ee.data.target_quat_w[:, 0, :]
            obj_pos = brush.data.root_pos_w
            ee_obj_dist = torch.linalg.norm(ee_pos - (obj_pos - env.unwrapped.scene.env_origins), dim=-1)

            obs_dict   = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)
            # obs-alignment: replace target_object_position (dims 21:28) with the
            # actual object position in robot-root frame + identity quat, matching
            # what the training script wrote into the HDF5 at load time.
            obs_policy = obs_policy.clone()
            if obs_aligned:
                id_quat = torch.zeros(num_envs, 4, device=device)
                id_quat[:, 0] = 1.0
                obs_policy[:, 21:24] = obs_policy[:, 18:21]
                obs_policy[:, 24:28] = id_quat
            phase_bit  = (phase >= 2).float().unsqueeze(-1)
            obs_input  = torch.cat([obs_policy, phase_bit], dim=-1)  # (N,43)
            if obs_mean is not None:
                obs_input = (obs_input - obs_mean) / obs_std

            action = policy(obs_input)

            # Phase 0 — APPROACH: BC arm, gripper open, count time near object
            near_counter = torch.where(
                (phase == 0) & (ee_obj_dist < 0.08),
                near_counter + 1, near_counter)
            phase = torch.where((phase == 0) & (near_counter >= NEAR_THRESH),
                                torch.ones_like(phase), phase)

            # Phase 1 — GRIP: hold arm still, gripper closed, wait GRIP_HOLD steps
            grip_steps = torch.where(phase == 1, grip_steps + 1, grip_steps)
            phase = torch.where((phase == 1) & (grip_steps >= GRIP_HOLD),
                                torch.full_like(phase, 2), phase)

            # Phase 2 — CARRY: BC arm, gripper closed
            carry_steps = torch.where(phase == 2, carry_steps + 1, carry_steps)
            obj_local = obj_pos - env.unwrapped.scene.env_origins
            xy_dist_to_basket = torch.linalg.norm(
                obj_local[:, :2] - basket_pos_dev[:2], dim=-1)
            # Release once centred over basket in XY — no Z gate (arm carries above rim)
            phase = torch.where(
                (phase == 2) & (carry_steps >= MIN_CARRY_STEPS)
                & (xy_dist_to_basket < RELEASE_XY),
                torch.full_like(phase, 3), phase)

            # Phase 3 — RELEASE: gripper open, arm held still, let object settle
            released = released | (phase == 3)
            release_steps = torch.where(phase == 3, release_steps + 1, release_steps)

            arm_action = action[:, :-1].clamp(-0.15, 0.15)

            # SERVO: closed-loop final approach when EE is close to the object.
            # Overrides BC XYZ with a direct delta toward the object's true pose
            # in robot-root frame. Fixes R2 downward overshoot.
            if args_cli.servo_dist > 0:
                robot = env.unwrapped.scene["robot"]
                root_pos_w  = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                ee_root,  _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee.data.target_pos_w[:, 0, :])
                obj_root, _ = subtract_frame_transforms(root_pos_w, root_quat_w, brush.data.root_pos_w)
                servo_cmd = ((obj_root - ee_root) * 2.0).clamp(-0.12, 0.12)
                servo_active = (phase == 0) & (ee_obj_dist < args_cli.servo_dist)
                arm_action[:, 0:3] = torch.where(
                    servo_active.unsqueeze(-1).expand(-1, 3), servo_cmd, arm_action[:, 0:3])

            # Hold arm still during GRIP (1) and RELEASE (3)
            hold_arm = (phase == 1) | (phase == 3)
            arm_action = torch.where(hold_arm.unsqueeze(-1).expand_as(arm_action),
                                     torch.zeros_like(arm_action), arm_action)
            grip_val = torch.where(phase == 0,
                                   torch.ones(num_envs, device=device),
                                   torch.where(phase == 3,
                                               torch.ones(num_envs, device=device),
                                               -torch.ones(num_envs, device=device)))
            action = torch.cat([arm_action, grip_val.unsqueeze(-1)], dim=-1)

            if ep_step[0] == 0:
                _SLOTS = {"R0": [-4.272,-5.960,1.611], "R1": [-4.445,-5.960,1.611],
                          "R2": [-4.272,-5.960,1.326], "R3": [-4.445,-5.960,1.326]}
                bp = obj_pos[0].cpu().tolist()
                slot_name = min(_SLOTS, key=lambda s: sum((a-b)**2 for a,b in zip(_SLOTS[s], bp)))
                print(f"[episode start] brush world=({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f})  slot={slot_name}", flush=True)

            # CNN detection (env 0 only, every detect_every steps)
            if ep_step[0] % args_cli.detect_every == 0:
                rgb   = board_cam.data.output["rgb"][0].detach().cpu().numpy().astype("uint8")
                depth = board_cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
                intr  = board_cam.data.intrinsic_matrices[0].detach().cpu().numpy()
                pos_w_cam  = board_cam.data.pos_w[0].detach().cpu().numpy()
                quat_w_cam = board_cam.data.quat_w_ros[0].detach().cpu().numpy()
                with torch.no_grad():
                    rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                    logits = seg_model(rgb_t)
                dets = extract_detections(logits, depth=depth, intrinsic_matrix=intr,
                                          pos_w=pos_w_cam, rot_w_quat=quat_w_cam)
                det_strs = [f"{d['label']}@({d['position_world'][0]:.2f},{d['position_world'][1]:.2f},{d['position_world'][2]:.2f})"
                            if d["position_world"] else d["label"]
                            for d in dets if d["class_id"] != 0]
                print(f"[cnn step {ep_step[0].item():4d}] {det_strs}", flush=True)

            # phase transition print
            _PHASE_NAMES = {0: "APPROACH", 1: "GRIP", 2: "CARRY", 3: "RELEASE"}
            if phase[0] != prev_phase[0]:
                print(f"[phase->] step={ep_step[0].item():4d}  {_PHASE_NAMES.get(prev_phase[0].item(),'')} → {_PHASE_NAMES.get(phase[0].item(),'')}  "
                      f"ee_dist={ee_obj_dist[0].item():.3f}m  obj_z={obj_pos[0,2].item():.3f}  xy_basket={xy_dist_to_basket[0].item():.3f}m", flush=True)
            prev_phase = phase.clone()

            if ep_step[0] % 50 == 0:
                jvel = robot.data.joint_vel[0].abs().max().item()
                raw_arm_mag = action[0, :-1].abs().max().item()
                grip_state = "OPEN" if grip_val[0].item() > 0 else "CLOSED"
                print(f"[step {ep_step[0].item():4d}]  ph={phase[0].item()}({_PHASE_NAMES[phase[0].item()][:3]})  "
                      f"near={near_counter[0].item():3d}  ee_dist={ee_obj_dist[0].item():.3f}m  "
                      f"obj_z={obj_pos[0,2].item():.3f}  carry={carry_steps[0].item():3d}  "
                      f"xy_bskt={xy_dist_to_basket[0].item():.3f}m  rel={release_steps[0].item():3d}  "
                      f"grip={grip_state}  arm={raw_arm_mag:.3f}  jvel={jvel:.3f}", flush=True)

            _, rew, terminated, truncated, _ = env.step(action)
            ep_step += 1
            ep_reward += rew

            brush_local = brush.data.root_pos_w - env.unwrapped.scene.env_origins
            dist = torch.linalg.norm(brush_local - basket_pos_dev, dim=1)
            ep_min_basket_dist = torch.minimum(ep_min_basket_dist, dist)

            # End episode after settle window, max steps, or env truncation
            done = truncated | (ep_step >= args_cli.max_steps) \
                   | ((phase == 3) & (release_steps >= RELEASE_HOLD))
            # Success = released AND object settled inside basket post-drop
            landed = released & (dist < SUCCESS_RADIUS) & (brush_local[:, 2] <= SUCCESS_Z_MAX)
            if done.any():
                env.reset()
                for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                    if len(saved_episodes) >= args_cli.num_episodes:
                        break
                    success = bool(landed[i].item())
                    saved_episodes.append({
                        "ep_idx": len(saved_episodes),
                        "env_idx": int(i),
                        "steps": int(ep_step[i].item()),
                        "cumulative_reward": float(ep_reward[i].item()),
                        "min_basket_dist": float(ep_min_basket_dist[i].item()),
                        "final_basket_dist": float(dist[i].item()),
                        "released": bool(released[i].item()),
                        "landed_in_basket": success,
                        "reached_basket": success,
                        "terminated": bool(terminated[i].item()),
                        "truncated": bool(truncated[i].item()),
                    })
                    print(f"[eval-state-bc] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
                          f"steps={int(ep_step[i].item())}  reward={float(ep_reward[i].item()):.2f}  "
                          f"released={bool(released[i].item())}  final_dist={float(dist[i].item()):.3f}m  "
                          f"LANDED={success}",
                          flush=True)
                ep_step[done] = 0
                ep_reward[done] = 0.0
                ep_min_basket_dist[done] = float("inf")
                phase[done] = 0
                prev_phase[done] = -1
                near_counter[done] = 0
                grip_steps[done] = 0
                carry_steps[done] = 0
                release_steps[done] = 0
                released[done] = False

    rewards = [e["cumulative_reward"] for e in saved_episodes]
    landed_n = sum(1 for e in saved_episodes if e["reached_basket"])
    n = len(saved_episodes)
    summary = {
        "policy": "state_bc",
        "checkpoint": str(args_cli.state_bc_ckpt),
        "num_episodes": n,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "basket_reach_rate": landed_n / n if n else 0.0,
        "mean_steps": float(np.mean([e["steps"] for e in saved_episodes])) if n else 0.0,
        "episodes": saved_episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-state-bc] LANDED: {summary['basket_reach_rate']*100:.0f}% "
          f"({landed_n}/{n}), mean reward: {summary['mean_reward']:.2f}, results -> {out_path}",
          flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
