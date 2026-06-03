"""Generate CNN training plots: loss curves + confusion matrix.

Produces two figures:
  1. Training & validation loss / mIoU over epochs (from metrics JSON)
  2. Confusion matrix on the validation set (runs model inference)

Usage:
    python3 scripts/plot_cnn_training.py \
        --ckpt checkpoints/air2_segmentation_v3.pth \
        --metrics checkpoints/air2_segmentation_metrics.json \
        --data datasets/air2_segmentation_v3 \
        --out eval_results/cnn_plots

    # For U-Net newcam (no metrics JSON — only confusion matrix):
    python3 scripts/plot_cnn_training.py \
        --ckpt checkpoints/air2_segmentation_unet_newcam.pth \
        --data datasets/air2_segmentation_newcam \
        --out eval_results/cnn_plots_unet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "isaaclab_ext/tasks/air2_franka/cnn"))

from dataset import AIR2SegmentationDataset, load_class_map
from model import build_model, build_resnet_model

CLASS_NAMES = [
    "background", "brush", "pliers", "scissors",
    "screwdriver", "robot", "basket", "table", "environment",
]


def plot_training_curves(metrics_path: Path, out_dir: Path):
    """Plot train/val loss and mIoU curves from metrics JSON."""
    import matplotlib.pyplot as plt

    with open(metrics_path) as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    mean_iou = [r["mean_iou"] for r in history]
    tool_miou = [r["tool_miou"] for r in history]
    pixel_acc = [r["pixel_accuracy"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (CE + Dice)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # mIoU + pixel accuracy
    ax = axes[1]
    ax.plot(epochs, mean_iou, label="Mean IoU", linewidth=2)
    ax.plot(epochs, tool_miou, label="Tool mIoU (classes 1-4)", linewidth=2)
    ax.plot(epochs, pixel_acc, label="Pixel Accuracy", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved training curves -> {path}")


def compute_confusion_matrix(model, dataloader, num_classes, device):
    """Run model on dataloader, accumulate pixel-level confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            targets = masks.numpy()
            for pred, target in zip(preds, targets):
                for true_c in range(num_classes):
                    mask_t = target == true_c
                    if not mask_t.any():
                        continue
                    pred_in_mask = pred[mask_t]
                    for pred_c in range(num_classes):
                        cm[pred_c, true_c] += (pred_in_mask == pred_c).sum()
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_dir: Path,
                          title: str = "Confusion Matrix"):
    """Plot normalized confusion matrix matching the reference style."""
    import matplotlib.pyplot as plt

    num_classes = cm.shape[0]
    # Normalize per true-class column (each column sums to 1)
    col_sums = cm.sum(axis=0, keepdims=True).astype(np.float64)
    col_sums[col_sums == 0] = 1
    cm_norm = cm.astype(np.float64) / col_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=0.7, aspect="equal")

    # Annotate cells (skip near-zero)
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            if val >= 0.005:
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("True Class", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved confusion matrix -> {path}")

    # Print per-class accuracy
    diag = np.diag(cm_norm)
    print(f"[plot] per-class accuracy:")
    for i, name in enumerate(class_names):
        print(f"  {name:15s} {diag[i]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot CNN segmentation training results.")
    parser.add_argument("--ckpt", required=True, help="Path to trained .pth checkpoint.")
    parser.add_argument("--metrics", default=None,
                        help="Path to metrics JSON (for loss curves). Optional.")
    parser.add_argument("--data", required=True, help="Dataset root (with val.txt, images/, masks/).")
    parser.add_argument("--out", default="eval_results/cnn_plots", help="Output directory for plots.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "unet")
    num_classes = ckpt.get("num_classes", 9)
    class_names = CLASS_NAMES[:num_classes]

    if backbone == "resnet18":
        model = build_resnet_model(num_classes=num_classes, pretrained=False)
    else:
        base_ch = ckpt.get("base_channels", 32)
        model = build_model(num_classes=num_classes, base_channels=base_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[plot] loaded {backbone} model from {args.ckpt} ({num_classes} classes)")

    # Plot training curves if metrics available
    if args.metrics and Path(args.metrics).exists():
        plot_training_curves(Path(args.metrics), out_dir)
    else:
        print("[plot] no metrics JSON — skipping training curves")

    # Load val dataset
    data_root = Path(args.data)
    val_dataset = AIR2SegmentationDataset(data_root, split="val", image_size=args.image_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[plot] val set: {len(val_dataset)} images from {data_root}")

    # Compute and plot confusion matrix
    cm = compute_confusion_matrix(model, val_loader, num_classes, device)
    title = f"Confusion Matrix — {backbone.upper()} ({Path(args.ckpt).stem})"
    plot_confusion_matrix(cm, class_names, out_dir, title=title)

    print(f"[plot] done — all plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
