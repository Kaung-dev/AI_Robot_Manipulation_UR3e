"""Roll out a trained BC policy in the AIR2 segmentation env and report metrics.

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/eval_bc.py ^
        --bc_ckpt checkpoints/policy_bc.pth ^
        --unet_ckpt checkpoints/air2_segmentation_unet.pth ^
        --task Isaac-Lift-AIR2-Robotis-Segmentation-Play-v0 ^
        --enable_cameras --headless ^
        --num_envs 4 --num_episodes 50 ^
        --out eval_results/bc_rollouts.json

Computes per-episode:
  - episode length (steps)
  - whether time_out fired or env terminated
  - cumulative reward
  - whether the policy reached the basket region (proxy success metric since
    we don't actually grasp anything in the simplified demos)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.objects import OBJECT_BY_LABEL, TARGET_LABELS

# Windows workaround: load h5py's bundled HDF5 DLLs before Isaac Sim extensions.
import h5py  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained BC policy.")
parser.add_argument("--bc_ckpt", required=True, help="Path to policy_bc.pth from train_bc.py.")
parser.add_argument("--unet_ckpt", default=None, help="U-Net checkpoint (must match the one used at train time).")
parser.add_argument("--num_classes", type=int, default=9)
parser.add_argument("--task", default="Isaac-Lift-AIR2-Robotis-Segmentation-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=800)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--target_object", choices=TARGET_LABELS, default="brush")
parser.add_argument("--out", default="eval_results/bc_rollouts.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
import isaaclab_ext.tasks.lift_air2_robotis  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.policy import (
    BCPolicy, load_frozen_encoder, JOINT_DIM, ACTION_DIM, COMMAND_DIM,
)


# Basket position in env-local frame (matches collect_air2_demos.py).
BASKET_POS_LOCAL = torch.tensor([-3.560, -5.370, 1.040])
BASKET_REACH_RADIUS = 0.40  # 40 cm — proxy success metric


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    print(f"[eval] launched {args_cli.task} with {num_envs} envs", flush=True)

    # Load policy
    encoder = load_frozen_encoder(args_cli.unet_ckpt, num_classes=args_cli.num_classes)
    policy = BCPolicy(encoder).to(device)
    ckpt = torch.load(args_cli.bc_ckpt, map_location=device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"[eval] loaded BC ckpt from {args_cli.bc_ckpt} (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f})", flush=True)

    last_action = torch.zeros(num_envs, ACTION_DIM, device=device)
    target_spec = OBJECT_BY_LABEL[args_cli.target_object]
    target_one_hot = torch.zeros(num_envs, COMMAND_DIM, device=device)
    target_one_hot[:, target_spec.class_id - 1] = 1.0
    print(f"[eval] target_object={target_spec.label} class_id={target_spec.class_id}", flush=True)
    basket_pos_dev = BASKET_POS_LOCAL.to(device)

    # Per-env episode tracking
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)
    saved_episodes: list[dict] = []

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            wrist = env.unwrapped.scene["wrist_camera"].data.output["rgb"]
            board = env.unwrapped.scene["board_camera"].data.output["rgb"]
            jp = env.unwrapped.scene["robot"].data.joint_pos
            jv = env.unwrapped.scene["robot"].data.joint_vel
            state = torch.cat([jp[:, :JOINT_DIM], jv[:, :JOINT_DIM], last_action], dim=1)

            # Channel-first uint8 for BCPolicy
            wrist_chw = wrist.permute(0, 3, 1, 2).contiguous()
            board_chw = board.permute(0, 3, 1, 2).contiguous()

            # Chunked policy returns (B, chunk, 7); execute the first action.
            # (Could be upgraded to temporal ensembling — average overlapping
            # chunks from the last K steps — see ACT paper.)
            chunk = policy(wrist_chw, board_chw, state, target_one_hot)
            action = chunk[:, 0, :]   # (B, 7)

            # Gripper logit → discrete {-1, +1}
            grip = torch.where(action[:, 6] > 0, 1.0, -1.0)
            action = torch.cat([action[:, :6], grip.unsqueeze(-1)], dim=1)

            obs, rew, term, trunc, info = env.step(action)
            last_action = action

            ep_step += 1
            ep_reward += rew

            # Track distance to basket (proxy success for the visual-demo task)
            ee = env.unwrapped.scene["ee_frame"]
            ee_world = ee.data.target_pos_w[..., 0, :]
            origins = env.unwrapped.scene.env_origins
            ee_local = ee_world - origins
            dist_to_basket = torch.linalg.norm(ee_local - basket_pos_dev, dim=1)
            ep_min_basket_dist = torch.minimum(ep_min_basket_dist, dist_to_basket)

            done = term | trunc | (ep_step >= args_cli.max_steps)
            if done.any():
                for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                    if len(saved_episodes) >= args_cli.num_episodes:
                        break
                    reached_basket = bool(ep_min_basket_dist[i].item() < BASKET_REACH_RADIUS)
                    saved_episodes.append({
                        "ep_idx": len(saved_episodes),
                        "env_idx": int(i),
                        "steps": int(ep_step[i].item()),
                        "cumulative_reward": float(ep_reward[i].item()),
                        "min_basket_dist": float(ep_min_basket_dist[i].item()),
                        "reached_basket": reached_basket,
                        "terminated": bool(term[i].item()),
                        "truncated": bool(trunc[i].item()),
                        "target_object": target_spec.label,
                        "target_class_id": target_spec.class_id,
                    })
                    print(f"[eval] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
                          f"env={i}  steps={ep_step[i].item()}  "
                          f"reward={ep_reward[i].item():.2f}  "
                          f"min_basket_dist={ep_min_basket_dist[i].item():.3f}m  "
                          f"reached_basket={reached_basket}", flush=True)

                # Reset trackers for finished envs
                ep_step[done] = 0
                ep_reward[done] = 0.0
                ep_min_basket_dist[done] = float("inf")
                last_action[done] = 0.0

    # Summary
    n = len(saved_episodes)
    rewards = [e["cumulative_reward"] for e in saved_episodes]
    reached = sum(1 for e in saved_episodes if e["reached_basket"])
    summary = {
        "num_episodes": n,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "basket_reach_rate": reached / n if n else 0.0,
        "mean_steps": float(np.mean([e["steps"] for e in saved_episodes])) if n else 0.0,
        "target_object": target_spec.label,
        "target_class_id": target_spec.class_id,
        "episodes": saved_episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval] SUMMARY")
    print(f"[eval]   episodes:           {n}")
    print(f"[eval]   mean reward:        {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"[eval]   basket reach rate:  {summary['basket_reach_rate']*100:.1f}%  ({reached}/{n})")
    print(f"[eval]   mean episode steps: {summary['mean_steps']:.0f}")
    print(f"[eval] full results -> {out_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
