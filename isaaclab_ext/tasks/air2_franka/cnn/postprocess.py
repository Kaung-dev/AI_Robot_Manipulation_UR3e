"""Post-processing helpers for AIR2 segmentation predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from .dataset import AIR2_CLASS_MAP
except ImportError:  # Allows direct script imports from the cnn/ directory.
    from dataset import AIR2_CLASS_MAP


def _to_numpy(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix (camera → world)."""
    w, x, y, z = q.astype(float)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def pixel_to_camera(
    centroid_px: tuple[float, float],
    depth: float,
    intrinsic_matrix: np.ndarray | None,
) -> list[float] | None:
    """Unproject a pixel and depth to camera coordinates."""
    if intrinsic_matrix is None or not np.isfinite(depth) or depth <= 0:
        return None
    u, v = centroid_px
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return [float(x), float(y), float(depth)]


def extract_detections(
    logits: torch.Tensor | np.ndarray,
    depth: torch.Tensor | np.ndarray | None = None,
    intrinsic_matrix: torch.Tensor | np.ndarray | None = None,
    pos_w: torch.Tensor | np.ndarray | None = None,
    rot_w_quat: torch.Tensor | np.ndarray | None = None,
    class_map: dict[int, str] | None = None,
    min_area: int = 32,
    min_confidence: float = 0.35,
) -> list[dict[str, Any]]:
    """Convert model logits to robot-facing object records.

    pos_w: camera world position (3,) — from camera.data.pos_w[env_idx].
    rot_w_quat: camera world rotation as quaternion (w,x,y,z) (4,) —
        from camera.data.quat_w_ros[env_idx] in Isaac Lab.
    When both are provided, position_world is populated for each detection.
    """
    class_map = class_map or AIR2_CLASS_MAP
    if isinstance(logits, np.ndarray):
        logits_tensor = torch.from_numpy(logits)
    else:
        logits_tensor = logits.detach().cpu()
    if logits_tensor.ndim == 4:
        logits_tensor = logits_tensor[0]

    probs = torch.softmax(logits_tensor, dim=0).numpy()
    pred = probs.argmax(axis=0).astype(np.uint8)
    depth_np = _to_numpy(depth)
    if depth_np is not None and depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
    intrinsics_np = _to_numpy(intrinsic_matrix)

    pos_w_np = _to_numpy(pos_w)
    rot_w_np = _to_numpy(rot_w_quat)
    R_cam_to_world = _quat_to_rot(rot_w_np) if rot_w_np is not None else None

    detections: list[dict[str, Any]] = []
    for class_id, label in class_map.items():
        if class_id == 0:
            continue
        object_mask = pred == class_id
        area = int(object_mask.sum())
        if area < min_area:
            continue

        class_conf = probs[class_id][object_mask]
        confidence = float(class_conf.mean()) if class_conf.size else 0.0
        if confidence < min_confidence:
            continue

        ys, xs = np.nonzero(object_mask)
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())

        depth_value = None
        position_camera = None
        if depth_np is not None:
            x_i = int(round(centroid_x))
            y_i = int(round(centroid_y))
            y0 = max(0, y_i - 2)
            y1 = min(depth_np.shape[0], y_i + 3)
            x0 = max(0, x_i - 2)
            x1 = min(depth_np.shape[1], x_i + 3)
            local_depth = depth_np[y0:y1, x0:x1]
            finite = local_depth[np.isfinite(local_depth) & (local_depth > 0)]
            if finite.size:
                depth_value = float(np.median(finite))
                position_camera = pixel_to_camera((centroid_x, centroid_y), depth_value, intrinsics_np)

        position_world = None
        if position_camera is not None and R_cam_to_world is not None and pos_w_np is not None:
            p_cam = np.array(position_camera, dtype=float)
            position_world = (pos_w_np + R_cam_to_world @ p_cam).tolist()

        detections.append(
            {
                "class_id": int(class_id),
                "label": label,
                "confidence": confidence,
                "mask_area": area,
                "centroid_px": [centroid_x, centroid_y],
                "depth": depth_value,
                "position_camera": position_camera,
                "position_world": position_world,
            }
        )

    if detections:
        return sorted(detections, key=lambda item: item["confidence"], reverse=True)
    return [
        {
            "class_id": 0,
            "label": "unknown",
            "confidence": 0.0,
            "mask_area": 0,
            "centroid_px": None,
            "depth": None,
            "position_camera": None,
            "position_world": None,
        }
    ]
