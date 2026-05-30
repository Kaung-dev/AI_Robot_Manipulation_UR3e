"""Train diffusion policy for AIR2 pick-and-place.

Reuses AIR2DemoDataset from train_bc.py — same demo format, same frozen encoder.
Trains the noise prediction network with standard DDPM loss (MSE on eps).
Inference uses DDIM (10 steps by default) for fast action generation.

Usage:
    python scripts/train_diffusion.py \
        --demos datasets/air2_demos \
        --seg_ckpt checkpoints/air2_segmentation_unet.pth \
        --backbone unet \
        --out checkpoints/policy_diffusion.pth

    # ResNet-18 backbone:
    python scripts/train_diffusion.py \
        --demos datasets/air2_demos \
        --seg_ckpt checkpoints/air2_segmentation_resnet18.pth \
        --backbone resnet18 \
        --out checkpoints/policy_diffusion_resnet18.pth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab_ext.tasks.air2_franka.diffusion_policy import build_diffusion_policy
from scripts.train_bc import AIR2DemoDataset


def main():
    parser = argparse.ArgumentParser(description="Diffusion policy trainer for AIR2.")
    parser.add_argument("--demos", required=True, help="Root dir with ep_*/ subdirs.")
    parser.add_argument("--seg_ckpt", default=None, help="Segmentation checkpoint (.pth).")
    parser.add_argument("--backbone", default="unet", choices=["unet", "resnet18"],
                        help="Visual encoder backbone — must match seg_ckpt.")
    parser.add_argument("--out", default="checkpoints/policy_diffusion.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--include_failed", action="store_true")
    parser.add_argument("--T", type=int, default=100, help="DDPM diffusion steps.")
    parser.add_argument("--n_inf_steps", type=int, default=10, help="DDIM inference steps.")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log.json")

    print(f"[diffusion] device={args.device}")

    dataset = AIR2DemoDataset(
        Path(args.demos),
        frame_stride=args.frame_stride,
        success_only=not args.include_failed,
    )
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"[diffusion] train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    policy = build_diffusion_policy(
        backbone=args.backbone,
        ckpt_path=args.seg_ckpt,
        num_classes=args.num_classes,
        T=args.T,
        n_inf_steps=args.n_inf_steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)

    # Only noise_pred has gradients — encoder is frozen.
    trainable = [p for p in policy.noise_pred.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[diffusion] trainable params: {n_trainable:,} / {n_total:,} total")
    print(f"[diffusion] backbone={args.backbone}, T={args.T}, n_inf_steps={args.n_inf_steps}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val = math.inf
    log: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        policy.train()
        policy.encoder.eval()  # encoder stays frozen
        t0 = time.time()
        train_loss_sum, train_n = 0.0, 0

        for wrist, board, state, target_one_hot, action, mask in train_loader:
            wrist = wrist.to(args.device, non_blocking=True)
            board = board.to(args.device, non_blocking=True)
            state = state.to(args.device, non_blocking=True)
            target_one_hot = target_one_hot.to(args.device, non_blocking=True)
            action = action.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)

            obs_cond = policy.encode_obs(wrist, board, state, target_one_hot)
            loss = policy.train_loss(obs_cond, action, mask)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()

            b = wrist.size(0)
            train_loss_sum += loss.item() * b
            train_n += b

        train_loss = train_loss_sum / max(1, train_n)

        # --- val ---
        policy.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for wrist, board, state, target_one_hot, action, mask in val_loader:
                wrist = wrist.to(args.device, non_blocking=True)
                board = board.to(args.device, non_blocking=True)
                state = state.to(args.device, non_blocking=True)
                target_one_hot = target_one_hot.to(args.device, non_blocking=True)
                action = action.to(args.device, non_blocking=True)
                mask = mask.to(args.device, non_blocking=True)

                obs_cond = policy.encode_obs(wrist, board, state, target_one_hot)
                loss = policy.train_loss(obs_cond, action, mask)
                val_loss_sum += loss.item() * wrist.size(0)
                val_n += wrist.size(0)

        val_loss = val_loss_sum / max(1, val_n)
        sched.step()
        dt = time.time() - t0

        print(
            f"[diffusion] epoch={epoch:03d}/{args.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s)",
            flush=True,
        )

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "lr": sched.get_last_lr()[0]}
        log.append(record)
        log_path.write_text(json.dumps(log, indent=2))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": policy.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_path)
            print(f"[diffusion]   saved best → {out_path}")

    print(f"[diffusion] done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
