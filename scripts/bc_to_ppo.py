"""PPO training for the AIR2 task using rsl_rl, with optional BC warm-start.

Two modes:

1. **PPO from scratch** (default):
       python scripts/bc_to_ppo.py --task Isaac-Lift-AIR2-UR3e-RG2-v0 \\
           --num_envs 16 --max_iterations 1000 --headless

   Trains PPO with rsl_rl's standard `OnPolicyRunner` using the AIR2 task's
   `rsl_rl_cfg_entry_point` (defined in agents/rsl_rl_ppo_cfg.py).

2. **PPO warm-started from a BC checkpoint**:
       python scripts/bc_to_ppo.py --task Isaac-Lift-AIR2-UR3e-RG2-v0 \\
           --bc_ckpt checkpoints/policy_bc.pth \\
           --num_envs 16 --max_iterations 1000 --headless

   The BC checkpoint's action head is copied into the rsl_rl actor's last
   layer (if shapes are compatible). The critic stays randomly initialized.
   Note: rsl_rl's actor is a flat MLP and operates on flat observations, so
   only state-compatible portions of the BC policy can be transferred.

Architecture note: this matches the design doc's "module 3 — PPO fine-tune"
section. The CNN is NOT used during PPO (rsl_rl's ActorCritic doesn't accept
image observations directly). The BC vision policy and the PPO state policy
are two separate demonstrations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO trainer for AIR2 task (with optional BC warm-start).")
parser.add_argument("--task", default="Isaac-Lift-AIR2-UR3e-RG2-v0")
parser.add_argument("--bc_ckpt", default=None, help="Optional BC checkpoint (.pth) to warm-start the actor's action head.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", default="logs/rsl_rl/air2_ppo")
parser.add_argument("--experiment_name", default=None, help="Subdir under log_dir; defaults to a timestamp.")
parser.add_argument("--resume", default=None, help="Path to a previous PPO checkpoint to resume from.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import os
import time
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


def warm_start_actor_from_bc(runner: OnPolicyRunner, bc_ckpt_path: str) -> None:
    """Copy what we can from a BC checkpoint into the rsl_rl actor.

    rsl_rl's actor is a flat MLP: input_dim → hidden ... → action_dim.
    Our BCPolicy's action_head is also `nn.Linear(hidden, action_dim)`.
    If the action dims match (they should = 7), copy the action_head weights
    into the actor's last layer. Hidden-layer transfer is best-effort and
    skipped if shapes mismatch.
    """
    print(f"[bc->ppo] loading BC ckpt: {bc_ckpt_path}", flush=True)
    bc_state = torch.load(bc_ckpt_path, map_location="cpu")
    if isinstance(bc_state, dict) and "state_dict" in bc_state:
        bc_state = bc_state["state_dict"]

    actor = runner.alg.policy.actor  # rsl_rl MLP
    last_linear = None
    for m in reversed(list(actor.modules())):
        if isinstance(m, torch.nn.Linear):
            last_linear = m
            break
    if last_linear is None:
        print("[bc->ppo] WARNING: rsl_rl actor has no Linear layer? skipping warm-start")
        return

    bc_head_w = bc_state.get("action_head.weight")
    bc_head_b = bc_state.get("action_head.bias")
    if bc_head_w is None or bc_head_b is None:
        print("[bc->ppo] WARNING: BC checkpoint has no action_head.{weight,bias}, skipping")
        return

    if bc_head_w.shape == last_linear.weight.shape and bc_head_b.shape == last_linear.bias.shape:
        with torch.no_grad():
            last_linear.weight.copy_(bc_head_w)
            last_linear.bias.copy_(bc_head_b)
        print(f"[bc->ppo] action_head copied: {tuple(bc_head_w.shape)} -> rsl_rl actor last layer", flush=True)
    else:
        print(
            f"[bc->ppo] WARNING: shape mismatch — BC head {tuple(bc_head_w.shape)} "
            f"vs rsl_rl actor last {tuple(last_linear.weight.shape)}. "
            f"Skipping warm-start (PPO trains from scratch)."
        )


def main():
    # Env cfg + agent cfg
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # Logging
    log_root = Path(args_cli.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    exp_name = args_cli.experiment_name or time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bc->ppo] logging to {log_dir}", flush=True)

    # Make env + wrap for rsl_rl
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Build runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Optional warm-start
    if args_cli.bc_ckpt:
        warm_start_actor_from_bc(runner, args_cli.bc_ckpt)
    elif args_cli.resume:
        print(f"[bc->ppo] resuming from {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)
    else:
        print("[bc->ppo] training PPO from scratch (no BC ckpt, no resume)", flush=True)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Final checkpoint
    final = log_dir / "model_final.pt"
    runner.save(str(final))
    print(f"[bc->ppo] saved final policy to {final}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
