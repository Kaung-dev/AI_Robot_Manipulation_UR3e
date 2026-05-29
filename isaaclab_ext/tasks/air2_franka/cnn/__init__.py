"""CNN semantic segmentation utilities for the AIR2 task."""

# Lazy imports — these pull in torch/PIL which are not available in the
# Isaac Sim runtime. Import explicitly in training/inference scripts instead.

__all__ = [
    "AIR2SegmentationDataset",
    "AIR2_CLASS_MAP",
    "AIR2UNet",
    "build_model",
    "extract_detections",
    "remap_isaac_mask",
]


def __getattr__(name: str):
    if name in ("AIR2SegmentationDataset", "AIR2_CLASS_MAP", "remap_isaac_mask"):
        from .dataset import AIR2SegmentationDataset, AIR2_CLASS_MAP, remap_isaac_mask
        return locals()[name]
    if name in ("AIR2UNet", "build_model"):
        from .model import AIR2UNet, build_model
        return locals()[name]
    if name == "extract_detections":
        from .postprocess import extract_detections
        return extract_detections
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
