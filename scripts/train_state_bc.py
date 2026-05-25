"""Train a STATE-only BC policy whose architecture matches rsl_rl's PPO actor.

This is the missing piece between our vision+chunked BC ([train_bc.py](train_bc.py))
and PPO warm-start ([bc_to_ppo.py](bc_to_ppo.py)). The vision BC cannot be loaded
into the PPO actor directly — vision != state, chunked != single-step. So we
collect a parallel set of (obs, action) pairs where:

  - `obs` is the same flat vector PPO sees (joint_pos_rel + joint_vel_rel +
    object_position + target_object_position + last_action ≈ 39-D for AIR2)
  - `action` is the scripted controller's action (same 7-D as PPO)

We then train an MLP with the EXACT same architecture as rsl_rl's actor
(`MLP(input → 256 → 128 → 64 → 7, elu)`), so [bc_to_ppo.py](bc_to_ppo.py)
can `load_state_dict()` the whole thing — not just the head.

Usage:
    C:/isaac/IsaacLab/isaaclab.bat -p scripts/train_state_bc.py ^
        --task Isaac-Lift-AIR2-UR3e-RG2-Play-v0 ^
        --num_envs 4 --num_episodes 40 ^
        --max_episode_steps 800 --episode_length_s 20 ^
        --enable_cameras --headless ^
        --epochs 200 --lr 3e-4 ^
        --out checkpoints/policy_state_bc.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect + train a state-only BC matching rsl_rl actor.")
parser.add_argument("--task", default="Isaac-Lift-AIR2-UR3e-RG2-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=40)
parser.add_argument("--max_episode_steps", type=int, default=800)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--out", default="checkpoints/policy_state_bc.pth")
parser.add_argument("--single_object", action="store_true",
                    help="Pick ONE random object per episode (faster, cleaner) instead of all 4.")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64],
                    help="Must match rsl_rl_ppo_cfg.policy.actor_hidden_dims.")
parser.add_argument("--activation", default="elu",
                    help="Must match rsl_rl_ppo_cfg.policy.activation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

from rsl_rl.networks import MLP

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.scripted_controller import (
    PickPlaceController, OBJECT_NAMES,
)


def collect_demos(env, controller, num_episodes: int, max_steps: int):
    """Run scripted controller, capture (obs, action) per env step.

    Returns (obs_buf, act_buf) as numpy arrays of shape (N, obs_dim) / (N, 7).
    """
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    def collect_object_positions():
        per_env = []
        origins = env.unwrapped.scene.env_origins
        for env_idx in range(num_envs):
            objs = [
                (env.unwrapped.scene[name].data.root_pos_w[env_idx] - origins[env_idx]).cpu()
                for name in OBJECT_NAMES
            ]
            per_env.append(objs)
        return per_env

    controller.reset(collect_object_positions())

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    action = torch.zeros(env.unwrapped.action_space.shape, device=device)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    saved_episodes = 0
    step_counter = 0
    while saved_episodes < num_episodes and simulation_app.is_running():
        step_counter += 1
        with torch.inference_mode():
            ee = env.unwrapped.scene["ee_frame"]
            ee_local = ee.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
            delta_pos, grip = controller.step(ee_local)
            action.zero_()
            action[:, 0:3] = delta_pos
            action[:, 6] = grip

            # PPO obs is the manager-computed "policy" group, concatenated.
            obs_dict = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]  # (num_envs, obs_dim) — already concat
            if isinstance(obs_policy, dict):
                # if observation_manager returns a dict-of-tensors (non-concat), stitch
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)

            obs_list.append(obs_policy.detach().cpu().numpy().copy())
            act_list.append(action.detach().cpu().numpy().copy())

            obs, rew, term, trunc, info = env.step(action)
            ep_step += 1

            cap_hit = (ep_step >= max_steps)
            episode_finished = controller.done | term | trunc | cap_hit
            if episode_finished.any():
                n_fin = int(episode_finished.sum().item())
                saved_episodes += n_fin
                print(f"[state-bc] step {step_counter}: +{n_fin} episodes -> {saved_episodes}/{num_episodes} "
                      f"(buffer={len(obs_list)} steps)", flush=True)
                ep_step[episode_finished] = 0
                controller.reset(collect_object_positions())

    obs_buf = np.concatenate(obs_list, axis=0)
    act_buf = np.concatenate(act_list, axis=0)
    print(f"[state-bc] collected {len(obs_buf)} (obs, action) pairs, obs_dim={obs_buf.shape[1]}", flush=True)
    return obs_buf, act_buf


def train_mlp(obs_np: np.ndarray, act_np: np.ndarray, hidden_dims, activation,
              epochs: int, batch_size: int, lr: float, val_ratio: float,
              device: str):
    """Train an MLP matching rsl_rl's actor architecture."""
    obs = torch.from_numpy(obs_np).float()
    act = torch.from_numpy(act_np).float()
    n = obs.shape[0]
    n_val = max(1, int(n * val_ratio))
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = TensorDataset(obs[train_idx], act[train_idx])
    val_ds = TensorDataset(obs[val_idx], act[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = obs.shape[1]
    action_dim = act.shape[1]
    model = MLP(input_dim, action_dim, hidden_dims, activation).to(device)
    print(f"[state-bc] MLP: input={input_dim}, hidden={hidden_dims}, output={action_dim}, act={activation}")
    print(f"[state-bc] arch:\n{model}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0; tr_n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss_sum += loss.item() * xb.size(0); tr_n += xb.size(0)
        tr_loss = tr_loss_sum / max(1, tr_n)

        model.eval()
        with torch.no_grad():
            vl_loss_sum = 0.0; vl_n = 0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = nn.functional.smooth_l1_loss(pred, yb)
                vl_loss_sum += loss.item() * xb.size(0); vl_n += xb.size(0)
            vl_loss = vl_loss_sum / max(1, vl_n)
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": vl_loss})
        if ep % 10 == 0 or ep == 1:
            print(f"[state-bc] ep {ep}/{epochs}  train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}", flush=True)
    return model, log, input_dim


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device

    controller = PickPlaceController(num_envs=env.unwrapped.num_envs, device=device,
                                     single_object=args_cli.single_object)
    if args_cli.single_object:
        print("[state-bc] single_object mode: each episode picks ONE random object", flush=True)

    obs_np, act_np = collect_demos(env, controller, args_cli.num_episodes, args_cli.max_episode_steps)
    env.close()

    model, log, input_dim = train_mlp(
        obs_np, act_np,
        hidden_dims=args_cli.hidden_dims, activation=args_cli.activation,
        epochs=args_cli.epochs, batch_size=args_cli.batch_size,
        lr=args_cli.lr, val_ratio=args_cli.val_ratio, device="cuda:0",
    )

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "action_dim": int(act_np.shape[1]),
        "hidden_dims": args_cli.hidden_dims,
        "activation": args_cli.activation,
        "num_pairs": int(obs_np.shape[0]),
    }, out_path)
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps({"epochs": log, "input_dim": input_dim,
                                    "hidden_dims": args_cli.hidden_dims}, indent=2))
    print(f"[state-bc] saved actor checkpoint -> {out_path}")
    print(f"[state-bc] saved training log     -> {log_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
