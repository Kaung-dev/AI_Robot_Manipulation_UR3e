"""Collect AIR2 RGB/depth/segmentation frames for CNN training.

Example:
    ./IsaacLab/isaaclab.sh -p scripts/collect_air2_segmentation_data.py \
        --task Isaac-Lift-AIR2-UR3e-RG2-Segmentation-v0 \
        --frames 200 --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect AIR2 segmentation training data.")
parser.add_argument("--task", default="Isaac-Lift-AIR2-UR3e-RG2-Segmentation-v0")
parser.add_argument("--output", default="datasets/air2_segmentation")
parser.add_argument("--frames", type=int, default=200)
parser.add_argument("--cameras", nargs="+", default=["board_camera", "wrist_camera"])
parser.add_argument("--settle-steps", type=int, default=5)
parser.add_argument("--val-ratio", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

CNN_DIR = REPO_ROOT / "isaaclab_ext/tasks/lift_air2_ur3e_rg2/cnn"
sys.path.insert(0, str(CNN_DIR))
from dataset import remap_isaac_mask


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _camera_info(camera, data_type: str) -> dict[str, Any]:
    info = getattr(camera.data, "info", {})
    return _jsonable(info.get(data_type, {})) if info else {}


def _squeeze_image(data: torch.Tensor) -> np.ndarray:
    array = data[0].detach().cpu().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _write_split_files(root: Path, sample_ids: list[str], val_ratio: float) -> None:
    val_count = max(1, int(round(len(sample_ids) * val_ratio))) if len(sample_ids) > 1 else 0
    train_ids = sample_ids[:-val_count] if val_count else sample_ids
    val_ids = sample_ids[-val_count:] if val_count else []
    (root / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
    (root / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))


def main() -> None:
    root = Path(args_cli.output)
    for subdir in ["images", "depth", "raw_masks", "masks", "metadata"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    class_map_src = Path(__file__).resolve().parents[1] / "isaaclab_ext/tasks/lift_air2_ur3e_rg2/cnn/class_map.json"
    shutil.copyfile(class_map_src, root / "class_map.json")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    action_dim = env.action_manager.total_action_dim
    dummy_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)

    sample_ids: list[str] = []
    for frame_idx in range(args_cli.frames):
        env.reset()
        for _ in range(args_cli.settle_steps):
            env.step(dummy_action)

        for camera_name in args_cli.cameras:
            camera = env.scene[camera_name]
            rgb = _squeeze_image(camera.data.output["rgb"]).astype(np.uint8)
            depth = _squeeze_image(camera.data.output["distance_to_image_plane"]).astype(np.float32)
            raw_mask = _squeeze_image(camera.data.output["semantic_segmentation"]).astype(np.int32)
            semantic_info = _camera_info(camera, "semantic_segmentation")
            instance_info = _camera_info(camera, "instance_segmentation_fast")
            remapped = remap_isaac_mask(raw_mask, semantic_info)

            sample_id = f"{frame_idx:06d}_{camera_name}"
            Image.fromarray(rgb).save(root / "images" / f"{sample_id}_rgb.png")
            np.save(root / "depth" / f"{sample_id}_depth.npy", depth)
            np.save(root / "raw_masks" / f"{sample_id}_raw_mask.npy", raw_mask)
            Image.fromarray(remapped).save(root / "masks" / f"{sample_id}_mask.png")
            (root / "metadata" / f"{sample_id}.json").write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "camera": camera_name,
                        "frame": frame_idx,
                        "semantic_segmentation": semantic_info,
                        "instance_segmentation_fast": instance_info,
                    },
                    indent=2,
                )
            )
            sample_ids.append(sample_id)
            print(f"[OK] saved {sample_id}")

    _write_split_files(root, sample_ids, args_cli.val_ratio)
    env.close()
    simulation_app.close()
    print(f"\nSaved {len(sample_ids)} samples to {root}")


if __name__ == "__main__":
    main()
