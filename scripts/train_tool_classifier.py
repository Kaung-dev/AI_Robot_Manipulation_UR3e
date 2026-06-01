"""Train a 4-way tool classifier (CNN router) for the AIR2 pipeline.

Re-uses the visual-BC ResNet-18 encoder (frozen) + a tiny 4-way head.
Trained from the archived image demos — each `ep_NNN/meta.json` has the
`target_object` label (brush / pliers / scissors / screwdriver).

Output: `checkpoints/cnn_tool_classifier.pth` — used by
`scripts/_play_router.py` to pick which per-tool PPO to run at episode start.

Usage:
    isaaclab.bat -p scripts/train_tool_classifier.py \\
        --demos _archive/datasets_pre_stephen/air2_manual_demos \\
        --bc_ckpt checkpoints/policy_bc_visual.pth \\
        --epochs 20 --out checkpoints/cnn_tool_classifier.pth
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torchvision
import torchvision.transforms as T

TARGET_LABELS = ["brush", "pliers", "scissors", "screwdriver"]
LABEL_TO_IDX = {n: i for i, n in enumerate(TARGET_LABELS)}


class ToolClassifierDataset(Dataset):
    """One (wrist_img, board_img, label) per demo episode. Uses frame 0 of each ep."""

    def __init__(self, demos_root: Path, frame_indices: list[int] | None = None):
        self.demos_root = demos_root
        self.frame_indices = frame_indices or [0, 30, 60]
        self.samples: list[tuple[Path, int, int]] = []
        for ep in sorted(demos_root.iterdir()):
            if not ep.is_dir():
                continue
            meta = json.loads((ep / "meta.json").read_text())
            label = LABEL_TO_IDX[meta["target_object"]]
            wrist_dir = ep / "wrist_rgb"
            n_frames = len(list(wrist_dir.glob("t_*.png")))
            for fi in self.frame_indices:
                if fi < n_frames:
                    self.samples.append((ep, fi, label))
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ep, fi, label = self.samples[idx]
        wrist = self.transform(Image.open(ep / "wrist_rgb" / f"t_{fi:04d}.png").convert("RGB"))
        board = self.transform(Image.open(ep / "board_rgb" / f"t_{fi:04d}.png").convert("RGB"))
        return wrist, board, label


class ToolClassifier(nn.Module):
    """ResNet-18 (frozen) per camera → concat features → 4-way softmax head."""

    def __init__(self, num_classes: int = 4, bc_ckpt: str | None = None):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        if bc_ckpt:
            blob = torch.load(bc_ckpt, map_location="cpu", weights_only=False)
            sd = blob.get("state_dict", blob)
            encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
            if encoder_sd:
                backbone.load_state_dict(encoder_sd, strict=False)
                print(f"[classifier] loaded {len(encoder_sd)} encoder tensors from {bc_ckpt}")
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, wrist: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fw = self.backbone(wrist)
            fb = self.backbone(board)
        return self.head(torch.cat([fw, fb], dim=-1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--bc_ckpt", default=None, help="Visual-BC checkpoint to seed the ResNet encoder")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/cnn_tool_classifier.pth")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = ToolClassifierDataset(Path(args.demos))
    n_val = max(1, int(len(full) * args.val_frac))
    val_idx = random.sample(range(len(full)), n_val)
    train_idx = [i for i in range(len(full)) if i not in set(val_idx)]
    print(f"[classifier] device={device}  train={len(train_idx)}  val={n_val}")

    train_loader = DataLoader(torch.utils.data.Subset(full, train_idx),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(full, val_idx),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ToolClassifier(num_classes=len(TARGET_LABELS), bc_ckpt=args.bc_ckpt).to(device)
    opt = torch.optim.Adam([p for p in model.head.parameters() if p.requires_grad], lr=args.lr)
    best_val = 0.0
    log = []

    for ep in range(1, args.epochs + 1):
        model.train(); model.backbone.eval()
        tr_loss = tr_acc = tr_n = 0.0
        for wrist, board, label in train_loader:
            wrist, board, label = wrist.to(device), board.to(device), label.to(device)
            logits = model(wrist, board)
            loss = F.cross_entropy(logits, label)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * label.size(0)
            tr_acc += (logits.argmax(-1) == label).float().sum().item()
            tr_n += label.size(0)

        model.eval()
        vl_acc = vl_n = 0.0
        with torch.no_grad():
            for wrist, board, label in val_loader:
                wrist, board, label = wrist.to(device), board.to(device), label.to(device)
                logits = model(wrist, board)
                vl_acc += (logits.argmax(-1) == label).float().sum().item()
                vl_n += label.size(0)
        vl_acc /= max(1, vl_n); tr_acc /= max(1, tr_n); tr_loss /= max(1, tr_n)
        log.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": vl_acc})
        print(f"  ep {ep:>2}/{args.epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  val_acc={vl_acc:.3f}")

        if vl_acc > best_val:
            best_val = vl_acc
            out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "labels": TARGET_LABELS,
                "best_val_acc": best_val,
                "num_classes": len(TARGET_LABELS),
            }, out)

    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=2))
    print(f"[classifier] best val acc: {best_val:.3f}  →  {args.out}")


if __name__ == "__main__":
    main()
