"""Train a state-only BC policy from raw manual-demo HDF5s (no Mimic needed).

Path B from the 2026-06-01 session: Mimic's annotate step was failing to replay
the demos cleanly, so we train BC directly on the 40 raw demos per tool. The
resulting BC is the same 35-D MLP that Stephen's `policy_state_bc_brush.pth`
used pre-Mimic, so all downstream eval/PPO scripts still work.

Obs (35-D): joint_pos(9) + joint_vel(9) + object_position(3) +
            eef_pos(3) + eef_quat(4) + last_action(7) = 35
Action: 7-D (6 IK pose delta + 1 binary gripper) — same layout as Mimic-trained.

No Isaac Sim required — pure PyTorch.

Usage (one tool):
    python scripts/train_state_bc_from_raw.py \
        --hdf5 datasets/air2_manual_demos_brush/air2_mimic_source.hdf5 \
        --out checkpoints/policy_state_bc_brush.pth

Usage (all 4 tools in sequence on one machine, no GPU coordination needed):
    for tool in brush pliers scissors screwdriver; do
        python scripts/train_state_bc_from_raw.py \
            --hdf5 datasets/air2_manual_demos_${tool}/air2_mimic_source.hdf5 \
            --out checkpoints/policy_state_bc_${tool}.pth
    done

With --device cuda:N you can fan 4 tools across 4 GPUs in parallel.
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
parser.add_argument("--hdf5", required=True, help="Path to raw demos HDF5 (with /data/demo_N/obs/* and /data/demo_N/actions)")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
parser.add_argument("--out", required=True, help="Where to save the .pth (also writes .log.json next to it)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading: raw FreshDatas HDF5 -> 35-D obs + 7-D action arrays
# ---------------------------------------------------------------------------

# Obs order MUST match what the env emits at runtime, otherwise the BC's
# first layer reads the wrong feature per dim and eval outputs garbage.
# After our env disables target_object_position and appends eef_pos/eef_quat,
# the env emits in this order:
#   joint_pos(9) + joint_vel(9) + object_position(3) + actions=last_action(7)
#   + eef_pos(3) + eef_quat(4) = 35-D
HDF5_OBS_HEAD = ["joint_pos", "joint_vel", "object_position"]   # 21 dims
# last_action (7) is derived from demo['actions'] shifted by 1
HDF5_OBS_TAIL = ["eef_pos", "eef_quat"]                          # 7 dims
OBS_KEYS_FROM_HDF5 = HDF5_OBS_HEAD + ["<last_action>"] + HDF5_OBS_TAIL  # for ckpt metadata


def load_raw_dataset(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    obs_list, act_list = [], []
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        print(f"[bc-raw] {hdf5_path}: {len(demos)} demos")
        for demo_key in demos:
            d = f["data"][demo_key]
            try:
                head_parts = [d["obs"][k][:] for k in HDF5_OBS_HEAD]
                tail_parts = [d["obs"][k][:] for k in HDF5_OBS_TAIL]
            except KeyError as e:
                print(f"[bc-raw] {demo_key}: missing obs key {e}, skipping")
                continue
            T = head_parts[0].shape[0]
            # Derive last_action: shift actions[t] -> last_action[t+1]; t=0 zeros.
            act_arr = d["actions"][:].astype(np.float32)  # (T, 7)
            if act_arr.shape[0] != T:
                print(f"[bc-raw] {demo_key}: actions len {act_arr.shape[0]} != obs len {T}, skipping")
                continue
            last_action = np.zeros_like(act_arr)
            last_action[1:] = act_arr[:-1]
            # Concat in env emission order: head + last_action + tail.
            obs = np.concatenate(head_parts + [last_action] + tail_parts, axis=-1).astype(np.float32)
            obs_list.append(obs)
            act_list.append(act_arr)

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    n_demos = len(obs_list)
    print(f"[bc-raw] loaded {n_demos} demos -> {obs_all.shape[0]} steps, "
          f"obs_dim={obs_all.shape[1]}, act_dim={act_all.shape[1]}")
    return obs_all, act_all, n_demos


# ---------------------------------------------------------------------------
# Model — same MLP shape as Stephen's 35-D BC (ELU activations, no BN)
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

def train(obs_np: np.ndarray, act_np: np.ndarray) -> tuple[dict, list[dict], int, int]:
    n = obs_np.shape[0]
    n_val = max(1, int(n * args.val_ratio))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    dev = args.device
    obs_t = torch.from_numpy(obs_np).to(dev)
    act_t = torch.from_numpy(act_np).to(dev)
    obs_train, act_train = obs_t[train_idx], act_t[train_idx]
    obs_val, act_val = obs_t[val_idx], act_t[val_idx]
    n_train = len(train_idx)
    print(f"[bc-raw] device={dev}  train_steps={n_train}  val_steps={len(val_idx)}")

    input_dim = obs_np.shape[1]
    action_dim = act_np.shape[1]
    model = make_mlp(input_dim, action_dim, args.hidden_dims).to(dev)
    print(f"[bc-raw] MLP: {input_dim} -> {args.hidden_dims} -> {action_dim}  (ELU)")
    print(f"[bc-raw] params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    log = []
    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm_t = torch.randperm(n_train, device=dev)
        obs_shuf, act_shuf = obs_train[perm_t], act_train[perm_t]
        tr_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            xb = obs_shuf[i:i + args.batch_size]
            yb = act_shuf[i:i + args.batch_size]
            pred = model(xb)
            # Split loss like Stephen's recipe: smooth-L1 on pose (cols 0-5),
            # BCE-on-sign for gripper (col 6) so it doesn't collapse to a mean.
            pose_loss = nn.functional.smooth_l1_loss(pred[:, :6], yb[:, :6])
            grip_target = (yb[:, 6:7] > 0).float()  # +1 open -> 1, -1 close -> 0
            grip_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 6:7], grip_target)
            loss = pose_loss + grip_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= n_train

        model.eval()
        with torch.no_grad():
            pred_val = model(obs_val)
            pv_pose = nn.functional.smooth_l1_loss(pred_val[:, :6], act_val[:, :6]).item()
            pv_grip = nn.functional.binary_cross_entropy_with_logits(
                pred_val[:, 6:7], (act_val[:, 6:7] > 0).float()).item()
            vl_loss = pv_pose + pv_grip
        scheduler.step()

        log.append({"epoch": epoch, "train": tr_loss, "val": vl_loss,
                    "val_pose": pv_pose, "val_grip": pv_grip})
        if epoch % 25 == 0 or epoch == 1:
            print(f"  ep {epoch:>3}/{args.epochs}  train={tr_loss:.5f}  "
                  f"val={vl_loss:.5f}  (pose={pv_pose:.5f} grip={pv_grip:.5f})")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[bc-raw] best val loss: {best_val:.5f}")
    return best_state, log, input_dim, action_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs_np, act_np, n_demos = load_raw_dataset(args.hdf5)
    best_state, log, input_dim, action_dim = train(obs_np, act_np)

    torch.save({
        "state_dict": best_state,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "hidden_dims": args.hidden_dims,
        "activation": "elu",
        "obs_keys": OBS_KEYS_FROM_HDF5 + ["last_action"],
        "obs_dim_breakdown": "9 joint_pos + 9 joint_vel + 3 object_position + 3 eef_pos + 4 eef_quat + 7 last_action = 35",
        "num_steps": int(obs_np.shape[0]),
        "num_demos": n_demos,
        "source_hdf5": str(args.hdf5),
    }, out_path)

    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, indent=2))

    print(f"[bc-raw] saved ckpt   -> {out_path}")
    print(f"[bc-raw] saved log    -> {log_path}")
