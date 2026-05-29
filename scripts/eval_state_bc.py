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
parser.add_argument("--max_steps", type=int, default=800)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--out", default="eval_results/state_bc.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

from rsl_rl.networks import MLP

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# Same proxy success metric used by eval_bc.py / eval_ppo.py.
BASKET_POS_LOCAL = torch.tensor([-3.560, -5.370, 1.040])
BASKET_REACH_RADIUS = 0.40


def load_state_bc_policy(ckpt_path: str, device: str) -> MLP:
    """Reconstruct the MLP from the checkpoint's metadata + state_dict."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in blob:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'state_dict' key — "
                           f"is this really a train_state_bc.py output?")
    input_dim = blob["input_dim"]
    action_dim = blob["action_dim"]
    hidden_dims = blob["hidden_dims"]
    activation = blob.get("activation", "elu")
    model = MLP(input_dim, action_dim, hidden_dims, activation).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    print(f"[eval-state-bc] loaded MLP: input={input_dim} hidden={hidden_dims} "
          f"output={action_dim} act={activation}")
    return model


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    policy = load_state_bc_policy(args_cli.state_bc_ckpt, device=device)
    print(f"[eval-state-bc] task={args_cli.task} envs={num_envs} target_episodes={args_cli.num_episodes}",
          flush=True)

    basket_pos_dev = BASKET_POS_LOCAL.to(device)

    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)
    saved_episodes: list[dict] = []

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            obs_dict = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)

            action = policy(obs_policy)

            _, rew, terminated, truncated, _ = env.step(action)
            ep_step += 1
            ep_reward += rew

            ee = env.unwrapped.scene["ee_frame"]
            ee_local = ee.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
            dist = torch.linalg.norm(ee_local - basket_pos_dev, dim=1)
            ep_min_basket_dist = torch.minimum(ep_min_basket_dist, dist)

            done = terminated | truncated | (ep_step >= args_cli.max_steps)
            if done.any():
                for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                    if len(saved_episodes) >= args_cli.num_episodes:
                        break
                    reached = bool(ep_min_basket_dist[i].item() < BASKET_REACH_RADIUS)
                    saved_episodes.append({
                        "ep_idx": len(saved_episodes),
                        "env_idx": int(i),
                        "steps": int(ep_step[i].item()),
                        "cumulative_reward": float(ep_reward[i].item()),
                        "min_basket_dist": float(ep_min_basket_dist[i].item()),
                        "reached_basket": reached,
                        "terminated": bool(terminated[i].item()),
                        "truncated": bool(truncated[i].item()),
                    })
                    print(f"[eval-state-bc] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
                          f"steps={int(ep_step[i].item())}  reward={float(ep_reward[i].item()):.2f}  "
                          f"min_dist={float(ep_min_basket_dist[i].item()):.3f}m  reached={reached}",
                          flush=True)
                ep_step[done] = 0
                ep_reward[done] = 0.0
                ep_min_basket_dist[done] = float("inf")

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

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
