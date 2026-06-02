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
parser.add_argument("--task", default="Isaac-AIR2-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--reset_delay", type=int, default=250,
                    help="Steps to hold at release before resetting (lets you see the result). 250=~5s at 50Hz.")
parser.add_argument("--episode_length_s", type=float, default=40.0)
parser.add_argument("--out", default="eval_results/state_bc.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
    # Observation standardization stats (train_state_bc_from_hdf5.py). If absent
    # (older checkpoint), fall back to identity (mean=0, std=1) = no-op.
    if "obs_mean" in blob and "obs_std" in blob:
        obs_mean = torch.as_tensor(blob["obs_mean"], dtype=torch.float32, device=device)
        obs_std  = torch.as_tensor(blob["obs_std"],  dtype=torch.float32, device=device)
        print("[eval-state-bc] applying obs standardization from checkpoint")
    else:
        obs_mean = torch.zeros(input_dim, device=device)
        obs_std  = torch.ones(input_dim, device=device)
        print("[eval-state-bc] WARNING: no obs_mean/obs_std in checkpoint -> using raw obs (retrain to get normalized policy)")
    print(f"[eval-state-bc] loaded MLP: input={input_dim} hidden={hidden_dims} "
          f"output={action_dim} act={activation}")
    return model, obs_mean, obs_std


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    policy, obs_mean, obs_std = load_state_bc_policy(args_cli.state_bc_ckpt, device=device)
    print(f"[eval-state-bc] task={args_cli.task} envs={num_envs} target_episodes={args_cli.num_episodes} episode_length_s={env_cfg.episode_length_s}",
          flush=True)

    basket_pos_dev = BASKET_POS_LOCAL.to(device)

    ep_step      = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward    = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)
    # Strict drop-in success (matches PPO env's target_dropped_in_basket):
    # object released (gripper open) AND inside basket footprint AND below rim.
    ep_dropped_in = torch.zeros(num_envs, dtype=torch.bool, device=device)
    release_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    DROP_XY_RADIUS   = 0.18   # object within this XY of basket center
    DROP_RIM_OFFSET  = 0.15   # object below basket_z + this
    FINGER_OPEN_THR  = 0.03   # finger sum above this = released
    RELEASE_SETTLE   = 150    # steps to keep stepping after release so object falls+settles
    # Phase state: 0=APPROACH, 1=GRIP, 2=CARRY, 3=RELEASE
    phase        = torch.zeros(num_envs, dtype=torch.long, device=device)
    near_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    grip_steps   = torch.zeros(num_envs, dtype=torch.long, device=device)
    carry_steps  = torch.zeros(num_envs, dtype=torch.long, device=device)
    NEAR_THRESH      = 250   # steps within 0.08m before closing (5s)
    GRIP_HOLD        = 50    # steps holding gripper closed while gripping (1s)
    MIN_CARRY_STEPS  = 200   # min carry steps before XY check activates (~5s)
    BASKET_XY_RADIUS = 0.35  # XY radius around basket to trigger release
    saved_episodes: list[dict] = []

    # Map a brush world (x,z) to its right-slot label (R0/R1/R2/R3).
    _SLOT_COORDS = {"R0": (-4.272, 1.611), "R1": (-4.445, 1.611),
                    "R2": (-4.272, 1.326), "R3": (-4.445, 1.326)}
    def _slot_label(x, z):
        return min(_SLOT_COORDS, key=lambda k: (_SLOT_COORDS[k][0] - x) ** 2 + (_SLOT_COORDS[k][1] - z) ** 2)
    ep_start_slot = ["?"] * num_envs   # brush slot captured at each episode's first step

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            ee    = env.unwrapped.scene["ee_frame"]
            brush = env.unwrapped.scene["object"]

            ee_pos  = ee.data.target_pos_w[:, 0, :] - env.unwrapped.scene.env_origins
            ee_quat = ee.data.target_quat_w[:, 0, :]
            obj_pos = brush.data.root_pos_w
            ee_obj_dist = torch.linalg.norm(ee_pos - (obj_pos - env.unwrapped.scene.env_origins), dim=-1)

            # Capture the brush's starting slot at each episode's first step.
            for _i in (ep_step == 0).nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                ep_start_slot[_i] = _slot_label(float(obj_pos[_i, 0].item()), float(obj_pos[_i, 2].item()))

            obs_dict   = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)
            # Patch target_object_position (dims 21-27) with actual brush world pose.
            # generated_commands returns a stale default — overwrite with live brush pos.
            id_quat = torch.zeros(num_envs, 4, device=device)
            id_quat[:, 0] = 1.0
            obs_policy = obs_policy.clone()
            obs_policy[:, 21:24] = obj_pos  # world pos
            obs_policy[:, 24:28] = id_quat
            phase_bit  = (phase >= 2).float().unsqueeze(-1)
            obs_input  = torch.cat([obs_policy, phase_bit], dim=-1)  # (N,43)
            obs_input  = (obs_input - obs_mean) / obs_std            # same z-score as training

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
            # After MIN_CARRY_STEPS, trigger release when object enters basket XY radius
            carry_steps = torch.where(phase == 2, carry_steps + 1, carry_steps)
            obj_local = obj_pos - env.unwrapped.scene.env_origins
            xy_dist_to_basket = torch.linalg.norm(
                obj_local[:, :2] - basket_pos_dev[:2], dim=-1)
            phase = torch.where(
                (phase == 2) & (carry_steps >= MIN_CARRY_STEPS)
                & (xy_dist_to_basket < BASKET_XY_RADIUS) & (obj_local[:, 2] <= 1.4),
                torch.full_like(phase, 3), phase)

            # Phase 3 — RELEASE: BC arm, gripper open

            # STRICT drop-in success (latched): object released AND in basket AND below rim.
            # Mirrors PPO env's target_dropped_in_basket — NOT a gripped fly-over.
            robot = env.unwrapped.scene["robot"]
            finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)
            gripper_open = finger_sum > FINGER_OPEN_THR
            inside_xy = xy_dist_to_basket < DROP_XY_RADIUS
            below_rim = obj_local[:, 2] < (basket_pos_dev[2] + DROP_RIM_OFFSET)
            ep_dropped_in = ep_dropped_in | (gripper_open & inside_xy & below_rim)
            release_steps = torch.where(phase == 3, release_steps + 1, release_steps)

            # Build action based on phase — clip arm to prevent OOD large outputs
            arm_action = action[:, :-1].clamp(-0.15, 0.15)
            # Hold the arm still in GRIP (1) and RELEASE (3) so the object drops cleanly.
            arm_action = torch.where(((phase == 1) | (phase == 3)).unsqueeze(-1).expand_as(arm_action),
                                     torch.zeros_like(arm_action), arm_action)
            grip_val = torch.where(phase == 0,
                                   torch.ones(num_envs, device=device),
                                   torch.where(phase == 3,
                                               torch.ones(num_envs, device=device),
                                               -torch.ones(num_envs, device=device)))
            action = torch.cat([arm_action, grip_val.unsqueeze(-1)], dim=-1)

            if ep_step[0] % 100 == 0:
                robot = env.unwrapped.scene["robot"]
                jvel = robot.data.joint_vel[0].abs().max().item()
                raw_arm_mag = action[0, :-1].abs().max().item()
                print(f"[phase] step={ep_step[0].item():4d}  phase={phase[0].item()}  "
                      f"near={near_counter[0].item()}  ee_dist={ee_obj_dist[0].item():.3f}m  "
                      f"carry={carry_steps[0].item()}  xy_basket={xy_dist_to_basket[0].item():.3f}m  "
                      f"max_jvel={jvel:.3f}  raw_arm={raw_arm_mag:.3f}", flush=True)

            _, rew, terminated, truncated, _ = env.step(action)
            ep_step += 1
            ep_reward += rew

            brush_local = brush.data.root_pos_w - env.unwrapped.scene.env_origins
            dist = torch.linalg.norm(brush_local - basket_pos_dev, dim=1)
            ep_min_basket_dist = torch.minimum(ep_min_basket_dist, dist)

            # End on env termination, timeout, max steps, OR after the post-release
            # settle window (so the object actually falls + lands before we score it).
            done = terminated | truncated | (ep_step >= args_cli.max_steps) | (release_steps >= RELEASE_SETTLE)
            if done.any():
                # Pause before reset so you can see the result (phase 3 = release/success)
                if int(phase[0].item()) == 3 and args_cli.reset_delay > 0:
                    import time; time.sleep(args_cli.reset_delay / 50.0)
                # Force env reset so next episode always starts clean
                if (ep_step >= args_cli.max_steps).any():
                    env.reset()
                for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                    if len(saved_episodes) >= args_cli.num_episodes:
                        break
                    # STRICT: success only if object was released + landed in basket
                    # (not a gripped fly-over). Loose min-dist kept for diagnostics.
                    reached = bool(ep_dropped_in[i].item())
                    flyover = bool(ep_min_basket_dist[i].item() < BASKET_REACH_RADIUS)
                    saved_episodes.append({
                        "ep_idx": len(saved_episodes),
                        "env_idx": int(i),
                        "steps": int(ep_step[i].item()),
                        "cumulative_reward": float(ep_reward[i].item()),
                        "min_basket_dist": float(ep_min_basket_dist[i].item()),
                        "reached_basket": reached,
                        "flyover_only": flyover and not reached,
                        "slot": ep_start_slot[i],
                        "terminated": bool(terminated[i].item()),
                        "truncated": bool(truncated[i].item()),
                    })
                    print(f"[eval-state-bc] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
                          f"slot={ep_start_slot[i]}  "
                          f"steps={int(ep_step[i].item())}  reward={float(ep_reward[i].item()):.2f}  "
                          f"min_dist={float(ep_min_basket_dist[i].item()):.3f}m  "
                          f"dropped_in={reached}  flyover_only={flyover and not reached}",
                          flush=True)
                ep_step[done] = 0
                ep_reward[done] = 0.0
                ep_min_basket_dist[done] = float("inf")
                phase[done] = 0
                near_counter[done] = 0
                grip_steps[done] = 0
                carry_steps[done] = 0
                ep_dropped_in[done] = False
                release_steps[done] = 0

    rewards = [e["cumulative_reward"] for e in saved_episodes]
    reached = sum(1 for e in saved_episodes if e["reached_basket"])
    n = len(saved_episodes)
    summary = {
        "policy": "state_bc",
        "checkpoint": str(args_cli.state_bc_ckpt),
        "num_episodes": n,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "basket_reach_rate": reached / n if n else 0.0,
        "mean_steps": float(np.mean([e["steps"] for e in saved_episodes])) if n else 0.0,
        "episodes": saved_episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-state-bc] basket-reach: {summary['basket_reach_rate']*100:.0f}% "
          f"({reached}/{n}), mean reward: {summary['mean_reward']:.2f}, results -> {out_path}",
          flush=True)
    # Per-slot success breakdown — shows which slots the BC generalises to.
    per_slot: dict[str, list[int]] = {}
    for e in saved_episodes:
        s = e.get("slot", "?")
        per_slot.setdefault(s, [0, 0])
        per_slot[s][1] += 1
        if e["reached_basket"]:
            per_slot[s][0] += 1
    print("[eval-state-bc] per-slot reach:", flush=True)
    for s in sorted(per_slot):
        ok, tot = per_slot[s]
        print(f"    {s}: {ok}/{tot}  ({100*ok/tot:.0f}%)", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
