"""State-only BC from friend's per-target HDF5 — produces a checkpoint that
bc_to_ppo.py --state_bc_ckpt can load via strict load_state_dict() into the
rsl_rl actor.

Obs layout MATCHES Stephen's air2_franka env (LiftEnvCfg.PolicyCfg order),
term-by-term:
    [joint_pos(9), joint_vel(9), object_position(3),
     target_object_position(7), actions(7)] = 35-D

Architecture: rsl_rl.networks.MLP(35 -> 256 -> 128 -> 64 -> 7, elu) —
exactly matching air2_franka/agents/rsl_rl_ppo_cfg.UR3eRG2AIR2LiftPPORunnerCfg.

Loss: smooth-L1 across all 7 dims.

No Isaac Sim boot. Plain torch + h5py + numpy. Trains in ~3-5 min on a T4.

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\train_state_bc_from_hdf5.py `
        --hdf5 _archive\\v2_friend_hdf5\\teleop_air2_robotis.hdf5 `
        --epochs 200 `
        --out checkpoints\\policy_state_bc_brush.pth
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rsl_rl.networks import MLP

JOINT_DIM = 9
OBJECT_POS_DIM = 3
TARGET_POSE_DIM = 7
ACTION_DIM = 7
OBS_DIM = JOINT_DIM * 2 + OBJECT_POS_DIM + TARGET_POSE_DIM + ACTION_DIM   # = 35
HIDDEN_DIMS = [256, 128, 64]
ACTIVATION = "elu"


def load_hdf5(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    """One HDF5 file -> (obs (N, 35), actions (N, 7)) concatenated across all demos."""
    obs_all, act_all = [], []
    n_demos = 0
    with h5py.File(fp, "r") as f:
        demo_names = sorted(
            f["data"].keys(),
            key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else 0,
        )
        for d in demo_names:
            grp = f["data"][d]
            actions = grp["actions"][...].astype(np.float32)              # (T, 7)
            joint_pos = grp["obs/joint_pos"][...].astype(np.float32)      # (T, 9)
            joint_vel = grp["obs/joint_vel"][...].astype(np.float32)      # (T, 9)
            object_pos = grp["obs/object_position"][...].astype(np.float32)        # (T, 3)
            target = grp["obs/target_object_position"][...].astype(np.float32)     # (T, 7)
            T = actions.shape[0]
            last_action = np.zeros_like(actions)
            last_action[1:] = actions[:-1]
            obs = np.concatenate([joint_pos, joint_vel, object_pos, target, last_action], axis=-1)
            if obs.shape[-1] != OBS_DIM:
                raise ValueError(f"{fp.name}/{d}: obs dim {obs.shape[-1]} != {OBS_DIM}")
            obs_all.append(obs)
            act_all.append(actions)
        n_demos = len(demo_names)
    obs_cat = np.concatenate(obs_all, axis=0)
    act_cat = np.concatenate(act_all, axis=0)
    print(f"[hdf5] {fp.name}: {n_demos} demos / {obs_cat.shape[0]} frames")
    return obs_cat, act_cat


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] device={device}")

    obs_np, act_np = load_hdf5(Path(args.hdf5))

    obs_t = torch.from_numpy(obs_np).float()
    act_t = torch.from_numpy(act_np).float()
    n = obs_t.shape[0]
    n_val = max(1, int(n * args.val_frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_dl = DataLoader(TensorDataset(obs_t[train_idx], act_t[train_idx]),
                          batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(TensorDataset(obs_t[val_idx], act_t[val_idx]),
                        batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"[train] split: train={len(train_idx)}  val={len(val_idx)}")

    model = MLP(OBS_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION).to(device)
    print(f"[train] MLP {OBS_DIM} -> {HIDDEN_DIMS} -> {ACTION_DIM}, activation={ACTIVATION}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    log = []
    best_val = math.inf
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_sum, tr_n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            tr_sum += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        tr_loss = tr_sum / max(1, tr_n)

        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl_sum += nn.functional.smooth_l1_loss(model(xb), yb).item() * xb.size(0)
                vl_n += xb.size(0)
        vl_loss = vl_sum / max(1, vl_n)
        sched.step()
        dt = time.time() - t0
        log.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss, "lr": sched.get_last_lr()[0]})
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs}  train={tr_loss:.5f}  val={vl_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.1f}s", flush=True)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim": OBS_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dims": HIDDEN_DIMS,
                "activation": ACTIVATION,
                "input_layout": ["joint_pos(9)", "joint_vel(9)", "object_position(3)",
                                  "target_object_position(7)", "actions(7)"],
                "trained_on": str(Path(args.hdf5).resolve()),
                "num_pairs": int(obs_np.shape[0]),
            }, out_path)

    (out_path.with_suffix(".log.json")).write_text(json.dumps(
        {"epochs": log, "input_dim": OBS_DIM, "hidden_dims": HIDDEN_DIMS,
         "activation": ACTIVATION, "best_val": best_val}, indent=2))
    print(f"[train] best val={best_val:.5f};  checkpoint -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", required=True, help="Single per-target HDF5 (from friend).")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
