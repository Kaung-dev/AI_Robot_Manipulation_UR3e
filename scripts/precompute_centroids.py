"""Precompute brush centroid (cx, cy) for every wrist_rgb frame in the demo dataset.

Runs the full U-Net segmentation model on each stored wrist_rgb PNG, extracts
the class-1 (brush) centroid in [0,1] normalised coords, and saves per-episode
centroids.npy files alongside the demo data.

wrist_rgb is used instead of board_rgb because the board camera sees the hooks
edge-on (ring not visible), while the wrist camera is EE-mounted and looks
directly at the object during approach — the green ring is clearly visible.

Output: datasets/air2_manual_demos/ep_XXX/centroids.npy — shape (T, 2), float32
        Rows where the brush is not detected contain (-1, -1).

Usage (via isaaclab.sh so torch is available):
    ./launch_air2.sh precompute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab_ext.tasks.air2_franka.cnn.model import AIR2UNet

UNET_INPUT_SIZE = 224
BRUSH_CLASS_ID = 1
MIN_BRUSH_PIXELS = 32


def load_unet(ckpt_path: Path, num_classes: int = 9, device: str = "cpu") -> AIR2UNet:
    model = AIR2UNet(num_classes=num_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model.to(device)


def image_to_tensor(path: Path, device: str) -> torch.Tensor:
    """Load PNG → (1, 3, H, W) float32 [0,1], resized to 224×224 if needed."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    if t.shape[-2] != UNET_INPUT_SIZE or t.shape[-1] != UNET_INPUT_SIZE:
        t = F.interpolate(t, size=(UNET_INPUT_SIZE, UNET_INPUT_SIZE),
                          mode="bilinear", align_corners=False)
    return t.to(device)


@torch.no_grad()
def centroid_from_logits(logits: torch.Tensor) -> np.ndarray:
    """logits: (1, C, H, W) → (cx, cy) in [0,1] or (-1,-1) if not detected."""
    pred = logits[0].argmax(dim=0)  # (H, W)
    mask = pred == BRUSH_CLASS_ID
    if mask.sum() < MIN_BRUSH_PIXELS:
        return np.array([-1.0, -1.0], dtype=np.float32)
    ys, xs = mask.nonzero(as_tuple=True)
    cx = xs.float().mean().item() / (logits.shape[-1] - 1)
    cy = ys.float().mean().item() / (logits.shape[-2] - 1)
    return np.array([cx, cy], dtype=np.float32)


def process_episode(ep_dir: Path, model: AIR2UNet, device: str) -> int:
    states_path = ep_dir / "states.npz"
    if not states_path.exists():
        return 0

    data = np.load(states_path)
    n = len(data["action"])
    centroids = np.full((n, 2), -1.0, dtype=np.float32)

    wrist_dir = ep_dir / "wrist_rgb"
    detected = 0
    for t in range(n):
        img_path = wrist_dir / f"t_{t:04d}.png"
        if not img_path.exists():
            continue
        img = image_to_tensor(img_path, device)
        logits = model(img)
        c = centroid_from_logits(logits)
        centroids[t] = c
        if c[0] >= 0:
            detected += 1

    out_path = ep_dir / "centroids.npy"
    np.save(str(out_path), centroids)
    return detected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default="datasets/air2_manual_demos")
    parser.add_argument("--unet_ckpt", default="checkpoints/air2_segmentation_unet.pth")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    demos_root = REPO_ROOT / args.demos
    unet_ckpt = REPO_ROOT / args.unet_ckpt

    print(f"[precompute] loading U-Net from {unet_ckpt}")
    model = load_unet(unet_ckpt, num_classes=args.num_classes, device=args.device)

    episodes = sorted(p for p in demos_root.iterdir() if p.is_dir() and p.name.startswith("ep_"))
    print(f"[precompute] found {len(episodes)} episodes in {demos_root}")

    total_frames, total_detected = 0, 0
    for ep in episodes:
        detected = process_episode(ep, model, args.device)
        states = np.load(ep / "states.npz")
        n = len(states["action"])
        total_frames += n
        total_detected += detected
        print(f"  {ep.name}: {detected}/{n} frames with brush detected", flush=True)

    print(f"[precompute] done. {total_detected}/{total_frames} frames had brush "
          f"({100*total_detected/max(1,total_frames):.1f}%)")


if __name__ == "__main__":
    main()
