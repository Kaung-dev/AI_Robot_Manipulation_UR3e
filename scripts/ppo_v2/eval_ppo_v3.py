"""Roll out a trained PPO policy (bc_to_ppo_v2 + GraspAssistWrapperV2) and report
OBJECT-landed-in-basket success.

Must mirror the TRAINING contract: the policy was trained with GraspAssistWrapperV2
(auto-grip + B2 kinematic attach), so we wrap the env the same way here, else the
gripper never closes and nothing is graspable.

Usage (GUI on the display GPU 0; or --headless on a spare GPU):
    DISPLAY=:0 isaaclab.sh -p scripts/ppo_v2/eval_ppo_v2.py \
        --ppo_ckpt logs/rsl_rl/air2_ppo/<run>/model_final.pt \
        --task Isaac-AIR2-Robotis-Franka-Brush-v0 \
        --num_envs 1 --num_episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local v2 wrapper

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy (grasp-assist v2).")
parser.add_argument("--ppo_ckpt", required=True, help="rsl_rl model_*.pt")
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Brush-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--episode_length_s", type=float, default=40.0)
parser.add_argument("--out", default="eval_results/ppo_v3.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from grasp_assist_wrapper_v3 import GraspAssistWrapperV3

BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])
SUCCESS_RADIUS = 0.25   # 3D object->basket dist that counts as "in the basket"
SUCCESS_Z_MAX  = 1.27   # object must settle at/below this height (basket z=1.14)


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # strip cameras + per-tool ring markers (crash non-brush scenes); keep ee_frame/basket
    for cam in ("wrist_camera", "board_camera"):
        if hasattr(env_cfg.scene, cam) and getattr(env_cfg.scene, cam) is not None:
            setattr(env_cfg.scene, cam, None)
    for mk in ("brush_frame", "pliers_frame", "scissors_frame", "screwdriver_frame", "ee_tcp_marker"):
        if hasattr(env_cfg.scene, mk) and getattr(env_cfg.scene, mk) is not None:
            setattr(env_cfg.scene, mk, None)
    # disable early terminations so the object physically drops + settles before scoring
    disabled = []
    for tn in list(vars(env_cfg.terminations).keys()):
        if tn.startswith("_") or tn in ("time_out", "time_outs"):
            continue
        if getattr(env_cfg.terminations, tn) is not None:
            setattr(env_cfg.terminations, tn, None); disabled.append(tn)
    print(f"[eval-ppo-v3] disabled terminations: {disabled}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = GraspAssistWrapperV3(env, target_key="object")   # SAME contract as training (V3 drop-assist)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = str(Path(args_cli.ppo_ckpt).parent)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.ppo_ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[eval-ppo-v3] loaded {args_cli.ppo_ckpt}  (grasp-assist v2 + B2 attach)", flush=True)

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    basket = BASKET_POS_LOCAL.to(device)

    obs = env.get_observations()   # rsl_rl wrapper returns a TensorDict (no unpack)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward = torch.zeros(num_envs, device=device)
    ep_min = torch.full((num_envs,), float("inf"), device=device)
    saved: list[dict] = []

    while len(saved) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            action = policy(obs)
            obs, rew, dones, info = env.step(action)   # rsl_rl: 4-tuple (obs, reward, dones, extras)
            ep_step += 1
            ep_reward += rew

            obj = env.unwrapped.scene["object"]
            obj_local = obj.data.root_pos_w - env.unwrapped.scene.env_origins
            dist = torch.linalg.norm(obj_local - basket, dim=1)
            ep_min = torch.minimum(ep_min, dist)

            done = dones.bool() | (ep_step >= args_cli.max_steps)
            if done.any():
                landed = (dist < SUCCESS_RADIUS) & (obj_local[:, 2] <= SUCCESS_Z_MAX)
                for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                    if len(saved) >= args_cli.num_episodes:
                        break
                    ok = bool(landed[i].item())
                    saved.append({
                        "ep_idx": len(saved), "env_idx": int(i),
                        "steps": int(ep_step[i].item()),
                        "cumulative_reward": float(ep_reward[i].item()),
                        "min_basket_dist": float(ep_min[i].item()),
                        "final_basket_dist": float(dist[i].item()),
                        "landed_in_basket": ok,
                        "reached_basket": ok,
                    })
                    print(f"[eval-ppo-v3] ep {len(saved)}/{args_cli.num_episodes}  "
                          f"steps={int(ep_step[i].item())}  reward={float(ep_reward[i].item()):.1f}  "
                          f"final_dist={float(dist[i].item()):.3f}m  LANDED={ok}", flush=True)
                ep_step[done] = 0; ep_reward[done] = 0.0; ep_min[done] = float("inf")

    n = len(saved)
    landed = sum(1 for e in saved if e["landed_in_basket"])
    summary = {
        "policy": "ppo_v3", "checkpoint": str(args_cli.ppo_ckpt), "task": args_cli.task,
        "num_episodes": n, "landed_rate": landed / n if n else 0.0,
        "mean_reward": float(np.mean([e["cumulative_reward"] for e in saved])) if n else 0.0,
        "episodes": saved,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-ppo-v3] LANDED: {summary['landed_rate']*100:.0f}% ({landed}/{n}), "
          f"mean reward {summary['mean_reward']:.1f} -> {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
