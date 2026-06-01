"""Roll out a trained PPO checkpoint in IsaacLab GUI.

Loads `model_best.pt` (or any --ppo_ckpt) into the rsl_rl OnPolicyRunner, then
plays deterministically in the matching task env. Same env as training so the
grasp-assist wrapper + reward setup is identical (no surprises).

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\_play_ppo.py ^
        --task Isaac-AIR2-Robotis-Franka-Brush-v0 ^
        --ppo_ckpt logs\\rsl_rl\\air2_ppo\\brush_ppo_grasp_v1\\model_best.pt ^
        --grasp_assist --grasp_assist_target_key object ^
        --num_episodes 5 --num_envs 1 --enable_cameras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ppo_ckpt", required=True, help="Path to model_*.pt (e.g. model_best.pt) from a PPO run dir")
parser.add_argument("--task", required=True, help="Gym task ID, e.g. Isaac-AIR2-Robotis-Franka-Brush-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--grasp_assist", action="store_true",
                    help="Wrap env in GraspAssistWrapper — same as during training")
parser.add_argument("--grasp_assist_target_key", default="object")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# post-sim-init imports
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_ext.tasks.air2_franka.mdp.constants import BASKET_POS_LOCAL


def main() -> None:
    # Build cfg + env. Strip cameras (unless --enable_cameras explicitly used by
    # the GUI launcher) and force play-time speed (no curriculum, low noise).
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    # MUST match training-time PPO config so the checkpoint loads cleanly:
    agent_cfg.policy.noise_std_type = "log"
    agent_cfg.empirical_normalization = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.grasp_assist:
        from grasp_assist_wrapper import GraspAssistWrapper
        env = GraspAssistWrapper(env, target_key=args_cli.grasp_assist_target_key)
        print(f"[play] grasp-assist ENABLED target={args_cli.grasp_assist_target_key} "
              f"NEAR={GraspAssistWrapper.NEAR_THRESH}@{GraspAssistWrapper.NEAR_RADIUS}m "
              f"GRIP_HOLD={GraspAssistWrapper.GRIP_HOLD}", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[play] loading ppo ckpt: {args_cli.ppo_ckpt}", flush=True)
    runner.load(args_cli.ppo_ckpt)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    print(f"[play] policy ready. task={args_cli.task} envs={args_cli.num_envs} episodes={args_cli.num_episodes}", flush=True)

    basket = BASKET_POS_LOCAL.to(agent_cfg.device)
    obs, _ = env.reset()
    ep_step = torch.zeros(args_cli.num_envs, dtype=torch.long, device=agent_cfg.device)
    ep_min_basket = torch.full((args_cli.num_envs,), float("inf"), device=agent_cfg.device)
    saved: list[dict] = []

    target_scene_key = args_cli.grasp_assist_target_key  # the target tool rigid body key

    while len(saved) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            action = policy(obs)
        out = env.step(action)
        obs = out[0]
        terminated = out[2]
        truncated = out[3] if len(out) >= 5 else torch.zeros_like(terminated)
        ep_step += 1

        # Track min basket distance for the target tool
        tgt = env.unwrapped.scene[target_scene_key]
        tgt_local = tgt.data.root_pos_w - env.unwrapped.scene.env_origins
        dist = torch.linalg.norm(tgt_local - basket, dim=1)
        ep_min_basket = torch.minimum(ep_min_basket, dist)

        if ep_step[0].item() % 50 == 0:
            print(f"[play] step={int(ep_step[0])}  "
                  f"min_basket_dist={ep_min_basket[0].item():.3f}m  "
                  f"action_mag={action[0, :-1].abs().max().item():.3f}  "
                  f"grip={action[0, -1].item():+.2f}", flush=True)

        done = terminated | truncated | (ep_step >= args_cli.max_steps)
        if done.any():
            for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                if len(saved) >= args_cli.num_episodes:
                    break
                reached = bool(ep_min_basket[i].item() < 0.30)
                saved.append({
                    "ep": len(saved),
                    "steps": int(ep_step[i]),
                    "min_basket_dist": float(ep_min_basket[i]),
                    "reached_basket": reached,
                    "terminated": bool(terminated[i]),
                    "truncated": bool(truncated[i] if isinstance(truncated, torch.Tensor) else False),
                })
                print(f"[play] ep {len(saved)}/{args_cli.num_episodes}  steps={int(ep_step[i])}  "
                      f"min_basket_dist={float(ep_min_basket[i]):.3f}m  reached={reached}", flush=True)
            ep_step[done] = 0
            ep_min_basket[done] = float("inf")

    n = len(saved)
    reached = sum(1 for e in saved if e["reached_basket"])
    print(f"\n[play] DONE  n={n}  reached={reached}/{n} ({100*reached/max(1,n):.0f}%)", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
