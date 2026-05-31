"""Train a state-BC policy from Mimic-generated HDF5 demos.

Obs:  joint_pos(9) + joint_vel(9) + object_position(3) +
      target_object_position(7) + last_action(7)  = 35-D
Action: 7-D (matches PPO actor layout — checkpoint is load_state_dict compatible)

No Isaac Sim required — pure PyTorch.

Usage:
    python scripts/train_state_bc_from_hdf5.py \
        --hdf5 datasets/air2_mimic_generated.hdf5 \
        --epochs 300 --lr 3e-4 \
        --out checkpoints/policy_state_bc_mimic.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", default="datasets/air2_mimic_generated.hdf5")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
parser.add_argument("--out", default="checkpoints/policy_state_bc_mimic.pth")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

OBS_KEYS = ["joint_pos", "joint_vel", "object_position", "target_object_position", "actions"]
# "actions" in the obs group = last_action recorded at that timestep

def load_dataset(hdf5_path: str):
    obs_list, act_list = [], []
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"Loading {len(demos)} demos from {hdf5_path}")
        for demo_key in demos:
            d = f["data"][demo_key]
            obs_parts = [d["obs"][k][:] for k in OBS_KEYS]
            obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)  # (T, 35)
            act = d["actions"][:].astype(np.float32)                     # (T, 7)
            obs_list.append(obs)
            act_list.append(act)

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    print(f"Dataset: {obs_all.shape[0]} steps, obs_dim={obs_all.shape[1]}, act_dim={act_all.shape[1]}")
    return obs_all, act_all


# ---------------------------------------------------------------------------
# Model — matches rsl_rl actor (ELU activations, no BN)
# ---------------------------------------------------------------------------

def make_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_d = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_d, h), nn.ELU()]
        in_d = h
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(obs_np: np.ndarray, act_np: np.ndarray):
    n = obs_np.shape[0]
    n_val = max(1, int(n * args.val_ratio))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    obs_t = torch.from_numpy(obs_np)
    act_t = torch.from_numpy(act_np)
    train_dl = DataLoader(TensorDataset(obs_t[train_idx], act_t[train_idx]),
                          batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl   = DataLoader(TensorDataset(obs_t[val_idx],  act_t[val_idx]),
                          batch_size=args.batch_size, shuffle=False, drop_last=False)

    input_dim  = obs_np.shape[1]
    action_dim = act_np.shape[1]
    model = make_mlp(input_dim, action_dim, args.hidden_dims).to(args.device)
    print(f"MLP: {input_dim} → {args.hidden_dims} → {action_dim}  (ELU)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    log = []

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_idx)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(args.device), yb.to(args.device)
                vl_loss += nn.functional.smooth_l1_loss(model(xb), yb).item() * xb.size(0)
        vl_loss /= max(1, len(val_idx))
        scheduler.step()

        log.append({"epoch": epoch, "train": tr_loss, "val": vl_loss})
        if epoch % 25 == 0 or epoch == 1:
            print(f"  ep {epoch:>3}/{args.epochs}  train={tr_loss:.5f}  val={vl_loss:.5f}")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best val loss: {best_val:.5f}")
    return model, best_state, log, input_dim, action_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs_np, act_np = load_dataset(args.hdf5)
    model, best_state, log, input_dim, action_dim = train(obs_np, act_np)

    torch.save({
        "state_dict": best_state,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "hidden_dims": args.hidden_dims,
        "activation": "elu",
        "obs_keys": OBS_KEYS,
        "num_steps": int(obs_np.shape[0]),
    }, out_path)

    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, indent=2))

    print(f"Saved → {out_path}")
    print(f"Log   → {log_path}")
