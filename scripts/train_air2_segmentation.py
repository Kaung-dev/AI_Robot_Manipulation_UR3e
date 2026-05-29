"""Train the AIR2 custom U-Net semantic segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = REPO_ROOT / "isaaclab_ext/tasks/air2_franka/cnn"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CNN_DIR))

from dataset import AIR2SegmentationDataset, load_class_map
from model import build_model


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    union = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return 1.0 - dice.mean()


def metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float | list[float]]:
    pred = logits.argmax(dim=1)
    pixel_accuracy = (pred == target).float().mean().item()
    ious = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
        union = torch.logical_or(pred_mask, target_mask).sum().item()
        ious.append(float(intersection / union) if union else float("nan"))
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "per_class_iou": ious,
    }


def save_overlay(image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, path: Path) -> None:
    palette = np.array(
        [
            [0, 0, 0],         # 0 background
            [76, 175, 80],     # 1 brush
            [255, 152, 0],     # 2 pliers
            [244, 67, 54],     # 3 scissors
            [33, 150, 243],    # 4 screwdriver
            [158, 158, 158],   # 5 robot
            [255, 235, 59],    # 6 basket
            [96, 125, 139],    # 7 table
            [121, 85, 72],     # 8 environment
        ],
        dtype=np.uint8,
    )
    rgb = (image.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    gt = palette[mask.cpu().numpy()]
    pr = palette[pred.cpu().numpy()]
    canvas = np.concatenate([rgb, gt, pr], axis=1)
    Image.fromarray(canvas).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AIR2 U-Net segmentation.")
    parser.add_argument("--data", default="datasets/air2_segmentation")
    parser.add_argument("--output", default="checkpoints/air2_segmentation_unet.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = Path(args.data)
    class_map = load_class_map(data_root / "class_map.json" if (data_root / "class_map.json").exists() else None)
    num_classes = max(class_map) + 1

    train_dataset = AIR2SegmentationDataset(data_root, split="train", image_size=args.image_size)
    val_dataset = AIR2SegmentationDataset(data_root, split="val", image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = build_model(num_classes=num_classes, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_miou = -1.0
    history = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_path.parent / "air2_segmentation_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) + 0.25 * dice_loss(logits, masks, num_classes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        metric_accumulator = []
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks) + 0.25 * dice_loss(logits, masks, num_classes)
                val_loss += loss.item() * images.size(0)
                metric_accumulator.append(metrics_from_logits(logits.cpu(), masks.cpu(), num_classes))
                if batch_idx == 0:
                    pred = logits.argmax(dim=1)
                    save_overlay(images[0].cpu(), masks[0].cpu(), pred[0].cpu(), overlay_dir / f"epoch_{epoch:03d}.png")

        val_loss /= len(val_dataset)
        mean_iou = float(np.mean([item["mean_iou"] for item in metric_accumulator]))
        pixel_accuracy = float(np.mean([item["pixel_accuracy"] for item in metric_accumulator]))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} pixel_acc={pixel_accuracy:.3f} miou={mean_iou:.3f}"
        )

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "base_channels": args.base_channels,
                    "class_map": class_map,
                    "metrics": record,
                },
                output_path,
            )

    (output_path.parent / "air2_segmentation_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved best checkpoint to {output_path}")


if __name__ == "__main__":
    main()
