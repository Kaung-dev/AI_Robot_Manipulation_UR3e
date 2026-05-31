"""Custom PyTorch behaviour-cloning trainer for the AIR2 pick-place task.

Reads PNG+npz episodes produced by scripts/collect_air2_manual_demos.py or
scripts/collect_air2_demos.py and trains
a BCPolicy (frozen U-Net encoder + small MLP) to predict the recorded action
from (wrist_rgb, board_rgb, joint_pos, joint_vel, last_action, target_object).

Loss: MSE on the 6 IK-Rel pose-delta dims + BCE on the gripper bit.
Optim: Adam, lr 1e-4 by default, cosine schedule.

Usage:
    python scripts/train_bc.py \
        --demos datasets/air2_demos_test \
        --unet_ckpt checkpoints/air2_segmentation_unet.pth \
        --epochs 50 --batch_size 32 \
        --out checkpoints/policy_bc.pth

The U-Net encoder is *frozen* — only the fusion + action head get gradients.
This is the standard "perception module is supervised separately, control
module is imitation-learned" decomposition from CNN_PERCEPTION_AND_POLICY.md.

Does not import isaacsim — runs in a plain conda env with torch + pillow + numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab_ext.tasks.air2_franka.policy import (
    BCPolicy, FrozenUNetEncoder, load_frozen_encoder,
    FrozenResNetEncoder, load_frozen_resnet_encoder,
    ACTION_DIM, STATE_DIM, JOINT_DIM, CHUNK_SIZE, COMMAND_DIM,
    BASKET_POS_LOCAL,
)


# ----- dataset --------------------------------------------------------------


class AIR2DemoDataset(Dataset):
    """Per-frame samples with action *chunks* (ACT-style).

    Each sample = (wrist_rgb_chw, board_rgb_chw, state_vec, action_chunk, mask).
      state_vec    = concat[joint_pos(9), joint_vel(9), last_action(7),
                            centroid(2), basket_pos(3)] = (30,)
      action_chunk = next CHUNK_SIZE actions starting at this step       = (CHUNK_SIZE, 7)
      mask         = 1.0 for valid actions, 0.0 for padded                = (CHUNK_SIZE,)

    centroid is (cx, cy) of the target object in board_camera, normalised [0,1].
    (-1, -1) is used when the object is not detected. Run precompute_centroids.py
    before training to generate ep_XXX/centroids.npy files.

    For frames within `chunk_size` of the episode end, the remaining
    positions are padded with the last action and `mask=0` so the loss
    ignores them.
    """

    WRIST_H = WRIST_W = 224
    BOARD_H, BOARD_W = 360, 640

    def __init__(self, demos_root: Path, frame_stride: int = 1, chunk_size: int = CHUNK_SIZE,
                 success_only: bool = True):
        self.demos_root = Path(demos_root)
        self.frame_stride = frame_stride
        self.chunk_size = chunk_size
        self.success_only = success_only
        self.index: list[tuple[Path, int]] = []   # (ep_dir, frame_idx_within_ep)
        self.actions: dict[Path, np.ndarray] = {}
        self.joint_pos: dict[Path, np.ndarray] = {}
        self.joint_vel: dict[Path, np.ndarray] = {}
        self.target_one_hot: dict[Path, np.ndarray] = {}
        self.target_valid: dict[Path, np.ndarray] = {}
        self.centroids: dict[Path, np.ndarray] = {}

        episodes = sorted(p for p in self.demos_root.iterdir() if p.is_dir() and p.name.startswith("ep_"))
        if not episodes:
            raise FileNotFoundError(f"No ep_* directories under {demos_root}")
        kept, skipped_failed = 0, 0
        for ep in episodes:
            states_path = ep / "states.npz"
            if not states_path.exists():
                print(f"[dataset] skip {ep.name}: no states.npz")
                continue
            # Optionally filter to success=True only — failed demos contain
            # stuck/flailing frames that pollute the imitation signal.
            if self.success_only:
                meta_path = ep / "meta.json"
                if meta_path.exists():
                    import json as _json
                    try:
                        meta = _json.loads(meta_path.read_text())
                        if not meta.get("success", False):
                            skipped_failed += 1
                            continue
                    except Exception:
                        pass  # if meta unreadable, fall through and keep the episode
            data = np.load(states_path)
            n = len(data["action"])
            self.actions[ep] = data["action"].astype(np.float32)
            self.joint_pos[ep] = data["joint_pos"].astype(np.float32)
            self.joint_vel[ep] = data["joint_vel"].astype(np.float32)
            if "target_one_hot" not in data:
                raise KeyError(
                    f"{states_path} does not contain target_one_hot. "
                    "Record manual object-annotated demos with collect_air2_manual_demos.py."
                )
            target = data["target_one_hot"].astype(np.float32)
            if target.shape != (n, COMMAND_DIM):
                raise ValueError(f"{states_path} target_one_hot has shape {target.shape}, expected {(n, COMMAND_DIM)}")
            self.target_one_hot[ep] = target
            self.target_valid[ep] = data["target_valid"].astype(bool) if "target_valid" in data else target.sum(axis=1) > 0.0
            centroids_path = ep / "centroids.npy"
            if centroids_path.exists():
                self.centroids[ep] = np.load(str(centroids_path)).astype(np.float32)
            else:
                self.centroids[ep] = np.full((n, 2), -1.0, dtype=np.float32)
            # Skip very first frame (last_action is meaningless before any action).
            # We DO include frames near the end — padding handles the short tail.
            for i in range(1, n, self.frame_stride):
                if not bool(self.target_valid[ep][i]):
                    continue
                wrist = ep / "wrist_rgb" / f"t_{i:04d}.png"
                board = ep / "board_rgb" / f"t_{i:04d}.png"
                if wrist.exists() and board.exists():
                    self.index.append((ep, i))
            kept += 1
        print(f"[dataset] kept {kept}/{len(episodes)} episodes "
              f"(skipped {skipped_failed} as success=False), "
              f"{len(self.index)} usable frames, chunk_size={self.chunk_size}")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load_rgb(path: Path, target_h: int, target_w: int) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()  # writable copy
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W) uint8

    def __getitem__(self, idx: int):
        ep, t = self.index[idx]
        wrist = self._load_rgb(ep / "wrist_rgb" / f"t_{t:04d}.png", self.WRIST_H, self.WRIST_W)
        board = self._load_rgb(ep / "board_rgb" / f"t_{t:04d}.png", self.BOARD_H, self.BOARD_W)
        jp = torch.from_numpy(self.joint_pos[ep][t].copy())          # (9,)
        jv = torch.from_numpy(self.joint_vel[ep][t].copy())          # (9,)
        last_action = torch.from_numpy(self.actions[ep][t - 1].copy())   # (7,)
        centroid = torch.from_numpy(self.centroids[ep][t].copy())    # (2,) — (-1,-1) if not detected
        basket = BASKET_POS_LOCAL.clone()                             # (3,)
        state = torch.cat([jp, jv, last_action, centroid, basket], dim=0)  # (30,)
        target_one_hot = torch.from_numpy(self.target_one_hot[ep][t].copy())

        # Action chunk + mask
        actions_arr = self.actions[ep]
        n = len(actions_arr)
        end = min(t + self.chunk_size, n)
        valid = end - t
        chunk = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
        chunk[:valid] = actions_arr[t:end]
        # Pad the rest with the last valid action so the network still sees
        # a sensible target (mask=0 means loss ignores it anyway).
        if valid < self.chunk_size:
            chunk[valid:] = actions_arr[end - 1]
        mask = np.zeros(self.chunk_size, dtype=np.float32)
        mask[:valid] = 1.0
        return wrist, board, state, target_one_hot, torch.from_numpy(chunk), torch.from_numpy(mask)


# ----- losses ---------------------------------------------------------------


def bc_loss(
    pred: torch.Tensor,          # (B, chunk, 7)
    target: torch.Tensor,        # (B, chunk, 7)
    mask: torch.Tensor,          # (B, chunk) — 1.0 valid, 0.0 padded
) -> tuple[torch.Tensor, dict[str, float]]:
    """Smoothed-L1 on the 6 pose-delta dims + BCE on the gripper bit, masked over the chunk.

    Smoothed-L1 (a.k.a. Huber loss) is borrowed from ACT — more robust than
    plain MSE to occasional outlier actions in human demos.
    """
    pose_pred, grip_pred = pred[..., :6], pred[..., 6]
    pose_tgt, grip_tgt = target[..., :6], target[..., 6]

    # Pose: per-element smooth-L1, then mask + mean across valid (B, chunk) positions.
    pose_per = nn.functional.smooth_l1_loss(pose_pred, pose_tgt, reduction="none").mean(dim=-1)  # (B, chunk)
    pose_loss = (pose_per * mask).sum() / mask.sum().clamp(min=1.0)

    # Gripper: per-step BCE, masked.
    grip_tgt_01 = (grip_tgt + 1.0) * 0.5
    grip_per = nn.functional.binary_cross_entropy_with_logits(
        grip_pred, grip_tgt_01, reduction="none"
    )  # (B, chunk)
    grip_loss = (grip_per * mask).sum() / mask.sum().clamp(min=1.0)

    loss = pose_loss + 1.0 * grip_loss
    return loss, {
        "pose_l1": float(pose_loss.item()),
        "grip_bce": float(grip_loss.item()),
        "total": float(loss.item()),
    }


# ----- main -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="BC trainer for AIR2 pick-place.")
    parser.add_argument("--demos", required=True, help="Root dir containing ep_*/ subdirs.")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "unet"],
                        help="Visual encoder backbone. Must match the segmentation checkpoint.")
    parser.add_argument("--unet_ckpt", default=None, help="Path to segmentation .pth checkpoint.")
    parser.add_argument("--out", default="checkpoints/policy_bc.pth", help="Output BC checkpoint.")
    parser.add_argument("--num_classes", type=int, default=9, help="Segmentation model output classes.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1, help="Take every Nth frame; 1 = all.")
    parser.add_argument("--include_failed", action="store_true",
                        help="Train on success=False demos too. Default: only success=True.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log.json")
    log: list[dict] = []

    print(f"[bc] device={args.device}")
    dataset = AIR2DemoDataset(Path(args.demos), frame_stride=args.frame_stride,
                              success_only=not args.include_failed)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"[bc] train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    if args.backbone == "resnet18":
        encoder = load_frozen_resnet_encoder(args.unet_ckpt, num_classes=args.num_classes)
    else:
        encoder = load_frozen_encoder(args.unet_ckpt, num_classes=args.num_classes)
    policy = BCPolicy(encoder).to(args.device)
    # Only fusion + action_head have grads (encoder is frozen by load_frozen_encoder).
    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[bc] trainable params: {n_trainable:,} / {n_total:,} total")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = math.inf

    for epoch in range(args.epochs):
        # --- train ---
        policy.train()
        t0 = time.time()
        epoch_metrics = {"pose_l1": 0.0, "grip_bce": 0.0, "total": 0.0, "n": 0}
        for wrist, board, state, target_one_hot, action, mask in train_loader:
            wrist = wrist.to(args.device, non_blocking=True)
            board = board.to(args.device, non_blocking=True)
            state = state.to(args.device, non_blocking=True)
            target_one_hot = target_one_hot.to(args.device, non_blocking=True)
            action = action.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)

            pred = policy(wrist, board, state, target_one_hot)  # (B, chunk, 7)
            loss, m = bc_loss(pred, action, mask)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()

            b = wrist.size(0)
            for k, v in m.items():
                epoch_metrics[k] += v * b
            epoch_metrics["n"] += b

        for k in ("pose_l1", "grip_bce", "total"):
            epoch_metrics[k] /= max(1, epoch_metrics["n"])

        # --- val ---
        policy.eval()
        val_metrics = {"pose_l1": 0.0, "grip_bce": 0.0, "total": 0.0, "n": 0}
        with torch.no_grad():
            for wrist, board, state, target_one_hot, action, mask in val_loader:
                wrist = wrist.to(args.device, non_blocking=True)
                board = board.to(args.device, non_blocking=True)
                state = state.to(args.device, non_blocking=True)
                target_one_hot = target_one_hot.to(args.device, non_blocking=True)
                action = action.to(args.device, non_blocking=True)
                mask = mask.to(args.device, non_blocking=True)
                pred = policy(wrist, board, state, target_one_hot)
                _, m = bc_loss(pred, action, mask)
                b = wrist.size(0)
                for k, v in m.items():
                    val_metrics[k] += v * b
                val_metrics["n"] += b
        for k in ("pose_l1", "grip_bce", "total"):
            val_metrics[k] /= max(1, val_metrics["n"])

        sched.step()
        dt = time.time() - t0
        print(f"[bc] ep {epoch+1:03d}/{args.epochs}  "
              f"train pose={epoch_metrics['pose_l1']:.4f} grip={epoch_metrics['grip_bce']:.4f}  "
              f"val pose={val_metrics['pose_l1']:.4f} grip={val_metrics['grip_bce']:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({dt:.1f}s)", flush=True)

        log.append({"epoch": epoch + 1, "train": epoch_metrics, "val": val_metrics, "lr": sched.get_last_lr()[0]})
        log_path.write_text(json.dumps(log, indent=2))

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({
                "state_dict": policy.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "val_loss": val_metrics["total"],
                "trainable_only": False,
            }, out_path)
            print(f"[bc]   saved best to {out_path}")

    print(f"[bc] done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
