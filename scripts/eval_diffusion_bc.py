"""Roll out a Diffusion-Policy state-BC in IsaacLab GUI.

Receding-horizon execution: at each control tick, sample a chunk of K actions
from the diffusion policy conditioned on the current obs, execute the FIRST
`--exec_horizon` of them, then re-sample. Smaller exec_horizon = more re-planning
= robust to perturbations but slower; larger = faster but more open-loop.

Matches eval_state_bc.py's CLI for drop-in convenience: same --task and --num_envs
semantics, same JSON output schema for plotting.

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\eval_diffusion_bc.py ^
        --diffusion_ckpt checkpoints\\policy_diffusion_bc_brush.pth ^
        --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 ^
        --num_envs 1 --num_episodes 5 --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Diffusion-Policy state-BC.")
parser.add_argument("--diffusion_ckpt", required=True)
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Brush-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--episode_length_s", type=float, default=40.0)
parser.add_argument("--exec_horizon", type=int, default=8,
                    help="How many actions from each sampled chunk to execute before re-sampling.")
parser.add_argument("--num_inference_steps", type=int, default=32,
                    help="Diffusion denoising steps at inference (DDIM). Lower = faster, higher = better quality.")
parser.add_argument("--out", default=None, help="Optional JSON metrics output path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- post-sim-init imports (Isaac/USD/pxr ready after AppLauncher) ----------

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from train_diffusion_bc_from_raw import DiffusionPolicy
from isaaclab_ext.tasks.air2_franka.mdp.constants import BASKET_POS_LOCAL


# ----- load checkpoint --------------------------------------------------------

def load_policy(ckpt_path: str, device: str) -> DiffusionPolicy:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiffusionPolicy(
        state_dim=blob["state_dim"], action_dim=blob["action_dim"],
        chunk_size=blob["chunk_size"], num_timesteps=blob["num_diffusion_steps"],
        state_embed_dim=blob["state_embed_dim"], time_embed_dim=blob["time_embed_dim"],
        hidden_dim=blob["hidden_dim"], num_hidden_layers=blob["num_hidden_layers"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"[diff-eval] loaded {ckpt_path}: state_dim={blob['state_dim']}  "
          f"chunk={blob['chunk_size']}  T={blob['num_diffusion_steps']}", flush=True)
    return model


# ----- main loop --------------------------------------------------------------

BASKET_REACH_RADIUS = 0.30  # legacy euclidean — same as eval_state_bc.py


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    policy = load_policy(args_cli.diffusion_ckpt, device=device)
    print(f"[diff-eval] task={args_cli.task} envs={num_envs} "
          f"target_episodes={args_cli.num_episodes} exec_horizon={args_cli.exec_horizon}", flush=True)

    basket_pos_dev = BASKET_POS_LOCAL.to(device)
    saved_episodes: list[dict] = []
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)

    # Chunk buffer: action_buffer[env, t, :] is the t-th action in env's current chunk.
    # exec_idx[env] is the index into action_buffer to execute next.
    action_buffer = torch.zeros(num_envs, policy.chunk_size, policy.action_dim, device=device)
    exec_idx = torch.full((num_envs,), policy.chunk_size, dtype=torch.long, device=device)  # forces sample on first step

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Sample a fresh chunk for any env whose buffer is exhausted (or in
            # receding-horizon mode every `exec_horizon` steps).
            need_resample = exec_idx >= args_cli.exec_horizon
            if need_resample.any():
                obs_dict = env.unwrapped.observation_manager.compute()
                obs_policy = obs_dict["policy"]
                if isinstance(obs_policy, dict):
                    obs_policy = torch.cat(list(obs_policy.values()), dim=-1)
                env_ids = need_resample.nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() > 0:
                    chunk = policy.sample(obs_policy[env_ids],
                                          num_inference_steps=args_cli.num_inference_steps)
                    action_buffer[env_ids] = chunk
                    exec_idx[env_ids] = 0

            # Build the per-env action for this control tick.
            action = action_buffer[torch.arange(num_envs, device=device), exec_idx, :]
            exec_idx = exec_idx + 1

        # Clamp/binarise like eval_state_bc.py for safety + Franka binary gripper convention.
        arm = action[:, :-1].clamp(-0.5, 0.5)
        grip = torch.sign(action[:, -1:]).clamp(min=-1.0, max=1.0)
        # If sign is 0 (rare), default to open.
        grip = torch.where(grip == 0, torch.ones_like(grip), grip)
        action_to_step = torch.cat([arm, grip], dim=-1)

        _, rew, terminated, truncated, _ = env.step(action_to_step)
        ep_step += 1
        ep_reward += rew
        # Track min brush-to-basket distance for the "reached" metric.
        brush = env.unwrapped.scene["object"]
        brush_local = brush.data.root_pos_w - env.unwrapped.scene.env_origins
        dist = torch.linalg.norm(brush_local - basket_pos_dev, dim=1)
        ep_min_basket_dist = torch.minimum(ep_min_basket_dist, dist)

        if ep_step[0].item() % 50 == 0:
            print(f"[diff-eval] step={ep_step[0].item():4d}  "
                  f"min_dist={ep_min_basket_dist[0].item():.3f}m  "
                  f"raw_arm={action[0, :-1].abs().max().item():.3f}  "
                  f"grip={action[0, -1].item():+.2f}", flush=True)

        done = terminated | truncated
        if done.any():
            for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                if len(saved_episodes) >= args_cli.num_episodes:
                    break
                reached = bool(ep_min_basket_dist[i].item() < BASKET_REACH_RADIUS)
                saved_episodes.append({
                    "ep_idx": len(saved_episodes),
                    "steps": int(ep_step[i].item()),
                    "reward": float(ep_reward[i].item()),
                    "min_basket_dist": float(ep_min_basket_dist[i].item()),
                    "reached_basket": reached,
                    "terminated": bool(terminated[i].item()),
                    "truncated": bool(truncated[i].item()),
                })
                print(f"[diff-eval] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
                      f"steps={int(ep_step[i].item())}  reward={float(ep_reward[i].item()):.2f}  "
                      f"min_dist={float(ep_min_basket_dist[i].item()):.3f}m  reached={reached}",
                      flush=True)
            ep_step[done] = 0
            ep_reward[done] = 0.0
            ep_min_basket_dist[done] = float("inf")
            exec_idx[done] = policy.chunk_size  # force re-sample on the next tick

    n = len(saved_episodes)
    rewards = [e["reward"] for e in saved_episodes]
    reached = sum(1 for e in saved_episodes if e["reached_basket"])
    print(f"\n[diff-eval] DONE  n={n}  mean_reward={sum(rewards)/max(1,n):.2f}  "
          f"reached={reached}/{n} ({100*reached/max(1,n):.0f}%)", flush=True)

    if args_cli.out:
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out).write_text(json.dumps({
            "num_episodes": n,
            "mean_reward": sum(rewards) / max(1, n),
            "reached_count": reached,
            "reached_fraction": reached / max(1, n),
            "episodes": saved_episodes,
        }, indent=2))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
