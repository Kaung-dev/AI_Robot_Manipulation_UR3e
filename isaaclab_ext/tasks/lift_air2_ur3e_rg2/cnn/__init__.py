"""CNN semantic segmentation utilities for the AIR2 task."""

from .dataset import AIR2SegmentationDataset, AIR2_CLASS_MAP, remap_isaac_mask
from .model import AIR2UNet, build_model
from .postprocess import extract_detections

__all__ = [
    "AIR2SegmentationDataset",
    "AIR2_CLASS_MAP",
    "AIR2UNet",
    "build_model",
    "extract_detections",
    "remap_isaac_mask",
]
