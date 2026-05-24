"""Collect AIR2 RGB/depth/segmentation frames for CNN training.

The robot is driven through the pick-place waypoint sequence (in-front-of-each-
object -> above-basket -> repeat) using the same PickPlaceController used by
collect_air2_demos.py. This gives the CNN diverse viewpoints from BOTH cameras:

  - board_camera (static, third-person): sees the robot in many poses,
    objects on random hooks, basket
  - wrist_camera (attached to panda_hand): sees the gripper approaching each
    object, the basket up close, varying angles as the arm moves

Each saved frame produces:
  images/<id>_rgb.png        : RGB capture (uint8)
  depth/<id>_depth.npy       : depth in meters (float32)
  raw_masks/<id>_raw_mask.npy: Isaac semantic annotator output
  masks/<id>_mask.png        : remapped class-ID mask (uint8, values 0..N-1)
  masks_color/<id>_color.png : COLORIZED mask for human inspection
  overlays/<id>_overlay.png  : RGB + colorized mask side-by-side
  metadata/<id>.json         : info dicts from the annotator

The trainer reads from images/ and masks/. The masks_color/ and overlays/
folders exist purely so you can eyeball the data to verify it looks right.

Example:
    C:/isaac/IsaacLab/isaaclab.bat -p scripts/collect_air2_segmentation_data.py \
        --task Isaac-Lift-AIR2-UR3e-RG2-Segmentation-Play-v0 \
        --frames 200 --enable_cameras --headless
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

parser = argparse.ArgumentParser(description="Collect AIR2 segmentation training data with the robot in motion.")
parser.add_argument("--task", default="Isaac-Lift-AIR2-UR3e-RG2-Segmentation-Play-v0")
parser.add_argument("--output", default="datasets/air2_segmentation")
parser.add_argument("--frames", type=int, default=200, help="Target number of saved frames.")
parser.add_argument("--cameras", nargs="+", default=["board_camera", "wrist_camera"])
parser.add_argument("--save-every-n-steps", type=int, default=3, help="Save a frame every Nth sim step.")
parser.add_argument("--episode-length-s", type=float, default=20.0)
parser.add_argument("--val-ratio", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from PIL import Image

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.scripted_controller import (  # noqa: E402
    PickPlaceController, OBJECT_NAMES, ACTION_DIM,
)

CNN_DIR = REPO_ROOT / "isaaclab_ext/tasks/lift_air2_ur3e_rg2/cnn"
sys.path.insert(0, str(CNN_DIR))
from dataset import remap_isaac_mask, load_class_map  # noqa: E402


# ----- color palette (matches segmentation_env_cfg.SEMANTIC_MAPPING) --------

PALETTE = np.array([
    [0, 0, 0],         # 0 background
    [76, 175, 80],     # 1 toothbrush
    [255, 152, 0],     # 2 pliers
    [244, 67, 54],     # 3 scissors
    [33, 150, 243],    # 4 silicone
    [158, 158, 158],   # 5 robot
    [255, 235, 59],    # 6 basket
    [96, 125, 139],    # 7 table
    [121, 85, 72],     # 8 environment
], dtype=np.uint8)


# ----- helpers --------------------------------------------------------------

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
    # IsaacLab 2.3.x returns `info` as a list of per-env dicts; older versions
    # returned a single dict. Handle both.
    if isinstance(info, list):
        info = info[0] if info else {}
    if not isinstance(info, dict):
        return {}
    return _jsonable(info.get(data_type, {})) if info else {}


def _squeeze_image(data: torch.Tensor) -> np.ndarray:
    array = data[0].detach().cpu().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _colorize(mask: np.ndarray, palette: np.ndarray = PALETTE) -> np.ndarray:
    """Map class IDs to RGB. Class IDs outside palette become black."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    clipped = np.clip(mask, 0, len(palette) - 1).astype(np.int32)
    return palette[clipped]


def _write_split_files(root: Path, sample_ids: list[str], val_ratio: float) -> None:
    val_count = max(1, int(round(len(sample_ids) * val_ratio))) if len(sample_ids) > 1 else 0
    train_ids = sample_ids[:-val_count] if val_count else sample_ids
    val_ids = sample_ids[-val_count:] if val_count else []
    (root / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
    (root / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))


# ----- main -----------------------------------------------------------------

def main() -> None:
    root = Path(args_cli.output)
    for subdir in ["images", "depth", "raw_masks", "masks", "masks_color", "overlays", "metadata"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "isaaclab_ext/tasks/lift_air2_ur3e_rg2/cnn/class_map.json",
                    root / "class_map.json")
    print(f"[seg-collect] output: {root}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device

    controller = PickPlaceController(num_envs=1, device=device)

    def collect_object_positions() -> list[list[torch.Tensor]]:
        per_env = []
        origins = env.unwrapped.scene.env_origins
        for env_idx in range(env.unwrapped.num_envs):
            objs = [
                (env.unwrapped.scene[name].data.root_pos_w[env_idx] - origins[env_idx]).cpu()
                for name in OBJECT_NAMES
            ]
            per_env.append(objs)
        return per_env

    controller.reset(collect_object_positions())
    action = torch.zeros(env.unwrapped.action_space.shape, device=device)

    sample_ids: list[str] = []
    step_counter = 0
    saved_in_episode = 0
    while len(sample_ids) < args_cli.frames and simulation_app.is_running():
        with torch.inference_mode():
            ee = env.unwrapped.scene["ee_frame"]
            ee_local = ee.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
            delta_pos, grip = controller.step(ee_local)
            action.zero_()
            action[:, 0:3] = delta_pos
            action[:, 6] = grip
            env.step(action)
            step_counter += 1

            # Save every Nth step
            if step_counter % args_cli.save_every_n_steps != 0:
                if controller.all_done():
                    # episode finished — reset to next iteration of randomization
                    env.unwrapped.reset()
                    controller.reset(collect_object_positions())
                    saved_in_episode = 0
                continue

            for camera_name in args_cli.cameras:
                camera = env.unwrapped.scene[camera_name]
                rgb = _squeeze_image(camera.data.output["rgb"]).astype(np.uint8)
                depth = _squeeze_image(camera.data.output["distance_to_image_plane"]).astype(np.float32)
                raw_mask = _squeeze_image(camera.data.output["semantic_segmentation"]).astype(np.int32)
                semantic_info = _camera_info(camera, "semantic_segmentation")
                instance_info = _camera_info(camera, "instance_segmentation_fast")
                remapped = remap_isaac_mask(raw_mask, semantic_info).astype(np.uint8)

                sid = f"{len(sample_ids):06d}_{camera_name}"
                Image.fromarray(rgb).save(root / "images" / f"{sid}_rgb.png")
                np.save(root / "depth" / f"{sid}_depth.npy", depth)
                np.save(root / "raw_masks" / f"{sid}_raw_mask.npy", raw_mask)
                Image.fromarray(remapped).save(root / "masks" / f"{sid}_mask.png")

                # Colorized mask (RGB) for human inspection
                colored = _colorize(remapped)
                Image.fromarray(colored).save(root / "masks_color" / f"{sid}_color.png")

                # Overlay: RGB | colorized side-by-side
                h, w = rgb.shape[:2]
                if colored.shape[:2] != (h, w):
                    colored_img = Image.fromarray(colored).resize((w, h), Image.NEAREST)
                    colored_arr = np.array(colored_img)
                else:
                    colored_arr = colored
                overlay = np.concatenate([rgb, colored_arr], axis=1)
                Image.fromarray(overlay).save(root / "overlays" / f"{sid}_overlay.png")

                (root / "metadata" / f"{sid}.json").write_text(json.dumps({
                    "sample_id": sid, "camera": camera_name,
                    "step_in_run": step_counter,
                    "waypoint_idx_when_captured": int(controller.wp_idx[0].item()),
                    "controller_done": bool(controller.done[0].item()),
                    "semantic_segmentation": semantic_info,
                    "instance_segmentation_fast": instance_info,
                }, indent=2))
                sample_ids.append(sid)

            saved_in_episode += 1
            if len(sample_ids) % 25 == 0:
                print(f"[seg-collect] saved {len(sample_ids)}/{args_cli.frames} frames "
                      f"(step {step_counter}, wp_idx={int(controller.wp_idx[0].item())})", flush=True)

            # Episode boundary: reset + re-randomize so we get fresh object layouts
            if controller.all_done():
                env.unwrapped.reset()
                controller.reset(collect_object_positions())
                saved_in_episode = 0

    _write_split_files(root, sample_ids, args_cli.val_ratio)
    print(f"[seg-collect] DONE: saved {len(sample_ids)} samples to {root}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
