"""Dataset and mask remapping helpers for AIR2 segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from isaaclab_ext.tasks.air2_franka.objects import OBJECT_SPECS
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from objects import OBJECT_SPECS


AIR2_CLASS_MAP: dict[int, str] = {0: "background", **{spec.class_id: spec.label for spec in OBJECT_SPECS}}
AIR2_CLASS_MAP.update({5: "robot", 6: "basket", 7: "table", 8: "environment"})

# RGB colors -> class IDs. Must match SEMANTIC_MAPPING in segmentation_env_cfg.py.
AIR2_COLOR_TO_CLASS: dict[tuple[int, int, int], int] = {
    **{spec.color_rgb: spec.class_id for spec in OBJECT_SPECS},
    (158, 158, 158): 5,  # robot
    (255, 235, 59): 6,   # basket
    (96, 125, 139): 7,   # table, walls, floor, pegboard, hooks
    (121, 85, 72): 8,    # environment
    (0, 0, 0): 0,        # background
}

_ALIASES = {
    **{spec.label: spec.class_id for spec in OBJECT_SPECS},
    "brush_ring": 1,
    "screw_driver": 4,
    "screw_driver_ring": 4,
    "robot": 5,
    "franka": 5,
    "panda": 5,
    "basket": 6,
    "box": 6,
    "sm_box": 6,
    "boxportable": 6,
    "table": 7,
    "seattlelab": 7,
    "hook": 7,
    "board": 7,
    "pegboard": 7,
    "environment": 8,
}


def load_class_map(path: str | Path | None = None) -> dict[int, str]:
    if path is None:
        return dict(AIR2_CLASS_MAP)
    data = json.loads(Path(path).read_text())
    return {int(key): str(value) for key, value in data.items()}


def class_id_from_text(text: Any) -> int:
    """Infer stable AIR2 class ID from Isaac semantic/path metadata."""
    if text is None:
        return 0
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    text = str(text).lower()
    for token, class_id in _ALIASES.items():
        if token in text:
            return class_id
    return 0


def _normalize_info_mapping(info: dict[str, Any] | None) -> dict[int, str]:
    if not info:
        return {}

    if "idToSemantics" in info:
        mapping = info["idToSemantics"]
    elif "idToLabels" in info:
        mapping = info["idToLabels"]
    else:
        mapping = info

    normalized: dict[int, str] = {}
    for raw_id, label in mapping.items():
        try:
            normalized[int(raw_id)] = str(label)
        except (TypeError, ValueError):
            # Colorized annotators may use color tuples as keys. Training uses
            # non-colorized integer masks, so skip non-integer keys here.
            continue
    return normalized


def remap_isaac_mask(raw_mask: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
    """Convert Isaac annotator IDs to stable AIR2 class IDs."""
    mask = np.asarray(raw_mask)
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        rgb = mask[..., :3].astype(np.uint8)
        remapped = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for color, class_id in AIR2_COLOR_TO_CLASS.items():
            remapped[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
        return remapped
    if mask.ndim == 3:
        mask = mask[..., 0]
    remapped = np.zeros(mask.shape, dtype=np.uint8)
    id_to_text = _normalize_info_mapping(info)

    for raw_id in np.unique(mask):
        class_id = class_id_from_text(id_to_text.get(int(raw_id), raw_id))
        remapped[mask == raw_id] = class_id
    return remapped


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.asarray(Image.open(path))


class AIR2SegmentationDataset(Dataset):
    """Loads samples saved by collect_air2_segmentation_data.py."""

    def __init__(self, root: str | Path, split: str = "train", image_size: int | None = None):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        split_file = self.root / f"{split}.txt"
        if split_file.exists():
            self.sample_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        else:
            self.sample_ids = sorted(path.stem.replace("_rgb", "") for path in (self.root / "images").glob("*_rgb.png"))
        if not self.sample_ids:
            raise FileNotFoundError(f"No AIR2 segmentation samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[index]
        image_path = self.root / "images" / f"{sample_id}_rgb.png"
        mask_path = self.root / "masks" / f"{sample_id}_mask.png"
        if not mask_path.exists():
            mask_path = self.root / "masks" / f"{sample_id}_mask.npy"

        image = Image.open(image_path).convert("RGB")
        mask = _load_mask(mask_path).astype(np.uint8)

        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = np.asarray(Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST))

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor
