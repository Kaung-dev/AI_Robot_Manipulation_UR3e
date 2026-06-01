"""PPO training for the AIR2 task using rsl_rl, with optional BC warm-start.

Two modes:

1. **PPO from scratch** (default):
       python scripts/bc_to_ppo.py --task Isaac-AIR2-Franka-v0 \\
           --num_envs 16 --max_iterations 1000 --headless

   Trains PPO with rsl_rl's standard `OnPolicyRunner` using the AIR2 task's
   `rsl_rl_cfg_entry_point` (defined in agents/rsl_rl_ppo_cfg.py).

2. **PPO warm-started from a BC checkpoint**:
       python scripts/bc_to_ppo.py --task Isaac-AIR2-Franka-v0 \\
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
parser.add_argument("--task", default="Isaac-AIR2-Franka-v0")
parser.add_argument("--bc_ckpt", default=None,
                    help="DEPRECATED: vision+chunked BC ckpt. Only the head shape is checked — almost never compatible. "
                         "Use --state_bc_ckpt for a real warm-start.")
parser.add_argument("--state_bc_ckpt", default=None,
                    help="State-only BC checkpoint from train_state_bc.py. Full actor state-dict load.")
parser.add_argument("--warm_start_noise_std", type=float, default=0.3,
                    help="When warm-starting, override init_noise_std (default 1.0 destroys the warm-started behaviour).")
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
import isaaclab_ext.tasks.air2_franka  # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401 — registers per-target Brush/Pliers/Scissors/Screwdriver task IDs
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg


def warm_start_actor_from_state_bc(runner: OnPolicyRunner, ckpt_path: str) -> None:
    """Load the state-BC MLP weights into rsl_rl's actor.mlp sub-module.

    New rsl_rl (>=4.0) wraps the MLP inside MLPModel(.mlp), so BC keys
    like '0.weight' must be remapped to 'mlp.0.weight'.
    """
    print(f"[bc->ppo] loading STATE-BC ckpt: {ckpt_path}", flush=True)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob

    # Remap plain Sequential keys → mlp.* (new rsl_rl MLPModel structure)
    remapped = {f"mlp.{k}": v for k, v in sd.items()}

    actor = runner.alg.actor
    actor_sd = actor.state_dict()

    # Verify shapes for the keys we're loading
    for key, val in remapped.items():
        if key not in actor_sd:
            raise RuntimeError(f"[bc->ppo] remapped key '{key}' not found in actor. "
                               f"Actor keys: {list(actor_sd.keys())}")
        if actor_sd[key].shape != val.shape:
            raise RuntimeError(f"[bc->ppo] shape mismatch '{key}': "
                               f"bc={tuple(val.shape)} actor={tuple(actor_sd[key].shape)}")

    # Non-strict: only loads mlp.* weights, leaves obs_normalizer/distribution alone
    missing, unexpected = actor.load_state_dict(remapped, strict=False)
    print(f"[bc->ppo] loaded {len(remapped)} tensors into actor.mlp  "
          f"(missing={missing}, unexpected={unexpected})", flush=True)


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

    # PPO doesn't use camera obs (rsl_rl is flat-state only). Cameras add
    # ~35s/iter of rendering overhead on T4 and require --enable_cameras flag.
    # Strip them from the scene so PPO runs at ~10s/iter and doesn't need
    # the flag. (Mimic data-gen and BC eval still spawn cameras as needed.)
    for cam_attr in ("wrist_camera", "board_camera"):
        if hasattr(env_cfg.scene, cam_attr) and getattr(env_cfg.scene, cam_attr) is not None:
            setattr(env_cfg.scene, cam_attr, None)
            print(f"[bc->ppo] stripped scene.{cam_attr} (PPO doesn't need cameras)", flush=True)
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # When warm-starting, two changes are critical or PPO destroys the prior:
    #   1) Lower init_noise_std (default 1.0 = random walk overwhelms the
    #      warm-started mean within ~5 grad steps).
    #   2) Disable empirical normalization. The state-BC was trained on raw obs,
    #      so the actor expects raw inputs. Empirical normalization would
    #      re-center observations during PPO and silently invalidate the prior.
    is_warm_start = bool(args_cli.state_bc_ckpt or args_cli.bc_ckpt)
    if is_warm_start:
        agent_cfg.policy.init_noise_std = args_cli.warm_start_noise_std
        agent_cfg.policy.noise_std_type = "log"
        agent_cfg.empirical_normalization = False
        # Adaptive LR + a warm-started actor = KL spike on iter 1 (the prior is
        # far from the freshly-init critic's value estimates), which adaptive
        # then jacks LR up by 1.5x repeatedly until grads explode and log_std
        # goes NaN ("normal expects std >= 0"). Fixed schedule with a smaller
        # LR keeps the warm-started behaviour intact while PPO learns the
        # critic.
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.learning_rate = 1.0e-5
        agent_cfg.algorithm.entropy_coef = 0.01
        agent_cfg.algorithm.max_grad_norm = 0.5
        print(f"[bc->ppo] warm-start mode: init_noise_std={agent_cfg.policy.init_noise_std}, "
              f"noise_std_type=log, schedule=fixed, lr={agent_cfg.algorithm.learning_rate}, "
              f"entropy_coef={agent_cfg.algorithm.entropy_coef}, "
              f"empirical_normalization=False", flush=True)

    # Logging
    log_root = Path(args_cli.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    exp_name = args_cli.experiment_name or time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bc->ppo] logging to {log_dir}", flush=True)

    # Make env + wrap for rsl_rl
    print("[debug] before gym.make", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[debug] after gym.make, before RslRlVecEnvWrapper", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[debug] after RslRlVecEnvWrapper, before OnPolicyRunner", flush=True)

    # Upgrade cfg to new rsl_rl API (adds class_name fields expected by newer rsl_rl)
    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # Build runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    print("[debug] after OnPolicyRunner", flush=True)
    runner.add_git_repo_to_log(__file__)
    print("[debug] after add_git_repo_to_log", flush=True)

    # Optional warm-start
    if args_cli.state_bc_ckpt:
        warm_start_actor_from_state_bc(runner, args_cli.state_bc_ckpt)
    elif args_cli.bc_ckpt:
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
    except BaseException as e:
        import traceback
        print(f"[debug] FATAL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
