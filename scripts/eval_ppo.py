"""Roll out a trained PPO policy (from bc_to_ppo.py / rsl_rl) and report metrics.

Loads an rsl_rl checkpoint (model_*.pt produced by OnPolicyRunner.save) and
runs it in the AIR2 env. Saves the same JSON schema as eval_bc.py so
plot_training_curves.py can plot a BC-vs-PPO comparison.

Usage:
    isaaclab.bat -p scripts/eval_ppo.py \
        --ppo_ckpt logs/rsl_rl/air2_ppo/<run>/model_final.pt \
        --task Isaac-AIR2-Franka-v0 \
        --headless --num_envs 4 --num_episodes 20 \
        --out eval_results/ppo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
parser.add_argument("--ppo_ckpt", required=True, help="Path to rsl_rl model_*.pt from bc_to_ppo.py.")
parser.add_argument("--task", default="Isaac-AIR2-Franka-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=800)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--out", default="eval_results/ppo.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


# Same proxy success metric as eval_bc.py — distance to basket position.
BASKET_POS_LOCAL = torch.tensor([-3.560, -5.370, 1.040])
BASKET_REACH_RADIUS = 0.40


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Build the same OnPolicyRunner the trainer used, then load the checkpoint.
    log_dir = str(Path(args_cli.ppo_ckpt).parent)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.ppo_ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[eval-ppo] loaded {args_cli.ppo_ckpt}", flush=True)

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    basket_pos_dev = BASKET_POS_LOCAL.to(device)

    obs, _ = env.get_observations() if hasattr(env, "get_observations") else (env.reset(), None)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)
    saved_episodes: list[dict] = []

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            action = policy(obs)
            obs, rew, terminated, truncated, info = env.step(action)
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
                    print(f"[eval-ppo] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
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
        "num_episodes": n,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "basket_reach_rate": reached / n if n else 0.0,
        "mean_steps": float(np.mean([e["steps"] for e in saved_episodes])) if n else 0.0,
        "episodes": saved_episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-ppo] basket-reach: {summary['basket_reach_rate']*100:.0f}% ({reached}/{n}), "
          f"mean reward: {summary['mean_reward']:.2f}, results -> {out_path}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
