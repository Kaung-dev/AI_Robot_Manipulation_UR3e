"""Offline BC smoke test — load policy, run on dummy/random inputs, print actions.

Bypasses Isaac Sim entirely. Confirms whether the trained checkpoint
produces meaningful actions or is outputting zeros/garbage. Runs in seconds.

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\_smoke_bc_offline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from isaaclab_ext.tasks.air2_franka.policy import (
    BCPolicy, load_frozen_resnet_encoder,
    JOINT_DIM, ACTION_DIM, COMMAND_DIM,
)

CKPT = REPO_ROOT / "checkpoints" / "policy_bc.pth"
SAMPLE_EPISODE = REPO_ROOT / "datasets" / "air2_manual_demos" / "ep_000"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[smoke] device={device}")

# 1. Load policy with same backbone train_bc.py used (resnet18 default).
print(f"[smoke] loading checkpoint {CKPT}")
ckpt = torch.load(CKPT, map_location=device)
print(f"[smoke] checkpoint keys: {list(ckpt.keys())}")
print(f"[smoke] saved at epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', 'n/a')}")

encoder = load_frozen_resnet_encoder(None, num_classes=9)
policy = BCPolicy(encoder).to(device)
policy.load_state_dict(ckpt["state_dict"])
policy.eval()
print(f"[smoke] BC policy loaded, params={sum(p.numel() for p in policy.parameters()):,}")

# 2. Run on REAL data from the first training episode so we know the shape is right.
print(f"\n[smoke] loading real obs from {SAMPLE_EPISODE.name}")
states = np.load(SAMPLE_EPISODE / "states.npz")
wrist_img = np.array(Image.open(SAMPLE_EPISODE / "wrist_rgb" / "t_0000.png").convert("RGB"))
board_img = np.array(Image.open(SAMPLE_EPISODE / "board_rgb" / "t_0000.png").convert("RGB"))
print(f"[smoke] wrist {wrist_img.shape} dtype={wrist_img.dtype}  board {board_img.shape} dtype={board_img.dtype}")
print(f"[smoke] joint_pos[0] {states['joint_pos'][0]}")
print(f"[smoke] target_one_hot[0] {states['target_one_hot'][0]}")

# Channel-first uint8 batched
wrist_t = torch.from_numpy(wrist_img).permute(2, 0, 1).unsqueeze(0).to(device).contiguous()
board_t = torch.from_numpy(board_img).permute(2, 0, 1).unsqueeze(0).to(device).contiguous()
jp = torch.from_numpy(states["joint_pos"][0]).float().unsqueeze(0).to(device)
jv = torch.from_numpy(states["joint_vel"][0]).float().unsqueeze(0).to(device)
last_action = torch.zeros(1, ACTION_DIM, device=device)
state = torch.cat([jp[:, :JOINT_DIM], jv[:, :JOINT_DIM], last_action], dim=1)
tgt = torch.from_numpy(states["target_one_hot"][0]).float().unsqueeze(0).to(device)
print(f"[smoke] state shape {state.shape}  target_one_hot shape {tgt.shape}")

# 3. Forward pass
print("\n[smoke] forward...")
with torch.inference_mode():
    chunk = policy(wrist_t, board_t, state, tgt)
print(f"[smoke] chunk shape {chunk.shape}  (B, chunk, 7)")
action0 = chunk[0, 0].cpu().numpy()
print(f"[smoke] first-step action = {action0}")
print(f"[smoke]   pose_delta (xyz + axis-angle) = {action0[:6]}")
print(f"[smoke]   gripper_logit                  = {action0[6]:.4f}  -> {'OPEN' if action0[6] > 0 else 'CLOSE'}")

# 4. Compare to the recorded action
recorded_action = states["action"][0]
print(f"[smoke] recorded action @ t=0     = {recorded_action}")
print(f"[smoke] action L2 error           = {np.linalg.norm(action0[:6] - recorded_action[:6]):.4f}")

# 5. Run a few more frames and print pose-delta magnitudes — if they're all ~0
# the policy collapsed to no-op (training data was mostly zero deltas).
print("\n[smoke] action magnitudes for frames 0..9:")
for t in range(10):
    if t >= len(states["joint_pos"]):
        break
    wrist_t = torch.from_numpy(np.array(Image.open(SAMPLE_EPISODE / "wrist_rgb" / f"t_{t:04d}.png").convert("RGB"))).permute(2,0,1).unsqueeze(0).to(device).contiguous()
    board_t = torch.from_numpy(np.array(Image.open(SAMPLE_EPISODE / "board_rgb" / f"t_{t:04d}.png").convert("RGB"))).permute(2,0,1).unsqueeze(0).to(device).contiguous()
    jp = torch.from_numpy(states["joint_pos"][t]).float().unsqueeze(0).to(device)
    jv = torch.from_numpy(states["joint_vel"][t]).float().unsqueeze(0).to(device)
    last = torch.from_numpy(states["action"][max(0, t - 1)]).float().unsqueeze(0).to(device) if t > 0 else torch.zeros(1, ACTION_DIM, device=device)
    state = torch.cat([jp[:, :JOINT_DIM], jv[:, :JOINT_DIM], last], dim=1)
    tgt = torch.from_numpy(states["target_one_hot"][t]).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        chunk = policy(wrist_t, board_t, state, tgt)
    a = chunk[0, 0].cpu().numpy()
    rec = states["action"][t]
    print(f"  t={t:2d}  pred_pose_mag={np.linalg.norm(a[:6]):.4f}  rec_pose_mag={np.linalg.norm(rec[:6]):.4f}  pred_grip={a[6]:+.2f}  rec_grip={rec[6]:+.0f}")

print("\n[smoke] done")
