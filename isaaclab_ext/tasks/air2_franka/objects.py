"""AIR2 object catalog shared by spawning, segmentation, demos, and policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIR2ObjectSpec:
    scene_key: str
    label: str
    usd_file: str
    class_id: int
    hotkey: str
    color_rgb: tuple[int, int, int]


OBJECT_SPECS: tuple[AIR2ObjectSpec, ...] = (
    AIR2ObjectSpec("object", "brush", "brush_ring.usd", 1, "1", (76, 175, 80)),
    AIR2ObjectSpec("tool_pliers", "pliers", "pliers_ring_orange.usd", 2, "2", (255, 152, 0)),
    AIR2ObjectSpec("tool_scissors", "scissors", "scissors_ring_red.usd", 3, "3", (244, 67, 54)),
    AIR2ObjectSpec("tool_screwdriver", "screwdriver", "screw_driver_ring.usd", 4, "4", (33, 150, 243)),
)

OBJECT_NAMES: tuple[str, ...] = tuple(spec.scene_key for spec in OBJECT_SPECS)
TARGET_LABELS: tuple[str, ...] = tuple(spec.label for spec in OBJECT_SPECS)
TARGET_DIM = len(OBJECT_SPECS)

OBJECT_BY_SCENE_KEY = {spec.scene_key: spec for spec in OBJECT_SPECS}
OBJECT_BY_LABEL = {spec.label: spec for spec in OBJECT_SPECS}
OBJECT_BY_CLASS_ID = {spec.class_id: spec for spec in OBJECT_SPECS}
OBJECT_BY_HOTKEY = {spec.hotkey: spec for spec in OBJECT_SPECS}


def target_one_hot(class_id: int) -> list[float]:
    """Return a one-hot command vector ordered by OBJECT_SPECS."""
    values = [0.0] * TARGET_DIM
    for index, spec in enumerate(OBJECT_SPECS):
        if spec.class_id == class_id:
            values[index] = 1.0
            break
    return values


def catalog_json() -> list[dict[str, object]]:
    """JSON-serializable object catalog for demo metadata."""
    return [
        {
            "scene_key": spec.scene_key,
            "label": spec.label,
            "usd_file": spec.usd_file,
            "class_id": spec.class_id,
            "hotkey": spec.hotkey,
            "color_rgb": list(spec.color_rgb),
        }
        for spec in OBJECT_SPECS
    ]
