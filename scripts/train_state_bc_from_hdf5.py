"""Train a phase-conditioned state-BC policy from Mimic-generated HDF5 demos.

Obs:  joint_pos(9) + joint_vel(9) + object_position(3) +
      target_object_position(7) + last_action(7) + eef_pos(3) + eef_quat(4) + phase(1) = 43-D
Action: 7-D (matches PPO actor layout — checkpoint is load_state_dict compatible)

Phase label is derived per-step from obs/datagen_info/subtask_term_signals/grasp_brush:
  0 = approach (before grasp_brush fires), 1 = carry (after grasp_brush fires)

No Isaac Sim required — pure PyTorch.

Usage:
    python scripts/train_state_bc_from_hdf5.py \
        --hdf5 datasets/air2_mimic_generated_v2.hdf5 \
        --epochs 300 --lr 3e-4 \
        --out checkpoints/policy_state_bc_mimic_v2.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn


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
parser.add_argument("--object", default="brush",
                    choices=["brush", "pliers", "scissors", "screwdriver"],
                    help="Target object — determines grasp signal name in annotated HDF5.")
parser.add_argument("--out", default="checkpoints/policy_state_bc_mimic.pth")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

OBS_KEYS = ["joint_pos", "joint_vel", "object_position", "target_object_position", "actions", "eef_pos", "eef_quat"]
# "actions" in the obs group = last_action recorded at that timestep

DWELL_ARM_THRESH = 0.01   # BC-DW1: arm steps with max|action[:6]| below this are "dwell"
DWELL_LOSS_WEIGHT = 0.1   # BC-DW1: loss weight for dwell steps (vs 1.0 for active steps)


def load_dataset(hdf5_path: str, grasp_signal: str = "grasp_brush"):
    obs_list, act_list, weight_list = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"Loading {len(demos)} demos from {hdf5_path} (grasp_signal={grasp_signal})")
        for demo_key in demos:
            d = f["data"][demo_key]
            obs_parts = [d["obs"][k][:] for k in OBS_KEYS]
            obs_42 = np.concatenate(obs_parts, axis=-1).astype(np.float32)  # (T, 42)

            # Phase label: 0=approach, 1=carry
            # Prefer datagen_info grasp signal (annotated HDF5); fall back to
            # gripper action column (generated HDF5, which omits datagen_info).
            T = obs_42.shape[0]
            try:
                signals = d["obs"]["datagen_info"]["subtask_term_signals"][grasp_signal][:].flatten()
                transitions = np.where(np.diff(signals.astype(np.int32)) > 0)[0]
                boundary = int(transitions[0]) + 1 if len(transitions) > 0 else T
            except (KeyError, ValueError):
                grip = d["actions"][:, -1]
                trans = np.where(np.diff((grip < 0).astype(np.int32)) > 0)[0]
                boundary = int(trans[0]) + 1 if len(trans) > 0 else T
            phase = np.zeros((T, 1), dtype=np.float32)
            phase[boundary:] = 1.0

            obs = np.concatenate([obs_42, phase], axis=-1)  # (T, 43)
            act = d["actions"][:].astype(np.float32)        # (T, 7)

            # BC-DW1: down-weight dwell steps so approach-direction signal dominates
            arm_mag = np.abs(act[:, :6]).max(axis=1)        # (T,)
            w = np.where(arm_mag < DWELL_ARM_THRESH, DWELL_LOSS_WEIGHT, 1.0).astype(np.float32)

            obs_list.append(obs)
            act_list.append(act)
            weight_list.append(w)

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    weight_all = np.concatenate(weight_list, axis=0)
    dwell_frac = (weight_all < 1.0).mean() * 100
    print(f"Dataset: {obs_all.shape[0]} steps, obs_dim={obs_all.shape[1]}, act_dim={act_all.shape[1]}")
    print(f"BC-DW1: dwell steps={dwell_frac:.1f}% (weight={DWELL_LOSS_WEIGHT}), active={100-dwell_frac:.1f}% (weight=1.0)")
    return obs_all, act_all, weight_all


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

def train(obs_np: np.ndarray, act_np: np.ndarray, weight_np: np.ndarray):
    n = obs_np.shape[0]
    n_val = max(1, int(n * args.val_ratio))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Pin entire dataset on GPU — ~55MB total, avoids 641 CPU→GPU transfers per epoch
    dev = args.device
    obs_t    = torch.from_numpy(obs_np).to(dev)
    act_t    = torch.from_numpy(act_np).to(dev)
    weight_t = torch.from_numpy(weight_np).to(dev)
    obs_train,  act_train,  w_train = obs_t[train_idx],  act_t[train_idx],  weight_t[train_idx]
    obs_val,    act_val              = obs_t[val_idx],    act_t[val_idx]
    n_train = len(train_idx)
    print(f"Device: {dev}  |  train={n_train}  val={len(val_idx)}")

    input_dim  = obs_np.shape[1]
    action_dim = act_np.shape[1]
    model = make_mlp(input_dim, action_dim, args.hidden_dims).to(dev)
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
        # Shuffle on GPU
        perm_t = torch.randperm(n_train, device=dev)
        obs_shuf, act_shuf, w_shuf = obs_train[perm_t], act_train[perm_t], w_train[perm_t]
        tr_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            xb = obs_shuf[i:i + args.batch_size]
            yb = act_shuf[i:i + args.batch_size]
            wb = w_shuf[i:i + args.batch_size]          # BC-DW1: per-sample weights
            pred = model(xb)
            # BC-DW1: weight each sample's loss; val uses unweighted loss for a clean metric
            loss_raw = nn.functional.smooth_l1_loss(pred, yb, reduction="none")  # (B, 7)
            loss = (loss_raw.mean(dim=-1) * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= n_train

        model.eval()
        with torch.no_grad():
            pred_val = model(obs_val)
            vl_loss = nn.functional.smooth_l1_loss(model(obs_val), act_val).item()
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
    import time
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[diag] device arg: {args.device}")
    import torch as _t
    print(f"[diag] CUDA available: {_t.cuda.is_available()}")
    if _t.cuda.is_available():
        print(f"[diag] GPU: {_t.cuda.get_device_name(0)}")

    t0 = time.time()
    grasp_signal = f"grasp_{args.object}"
    obs_np, act_np, weight_np = load_dataset(args.hdf5, grasp_signal=grasp_signal)
    print(f"[diag] data load: {time.time()-t0:.1f}s")

    t1 = time.time()
    model, best_state, log, input_dim, action_dim = train(obs_np, act_np, weight_np)
    print(f"[diag] total train time: {time.time()-t1:.1f}s")

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
