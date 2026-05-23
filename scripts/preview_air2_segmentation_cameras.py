"""Save AIR2 segmentation camera previews.

Example:
    ./IsaacLab/isaaclab.sh -p scripts/preview_air2_segmentation_cameras.py \
        --task Isaac-Lift-AIR2-UR3e-RG2-Segmentation-v0 --enable_cameras
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview AIR2 segmentation cameras.")
parser.add_argument("--task", default="Isaac-Lift-AIR2-UR3e-RG2-Segmentation-v0")
parser.add_argument("--output", default="/tmp/air2_camera_preview")
parser.add_argument("--cameras", nargs="+", default=["board_camera", "wrist_camera"])
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


def _save_mask(mask: np.ndarray, path: Path) -> None:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        Image.fromarray(mask.astype(np.uint8)).save(path)
        return
    mask = mask.astype(np.int32)
    if mask.max(initial=0) > 0:
        vis = (mask.astype(np.float32) / float(mask.max()) * 255.0).astype(np.uint8)
    else:
        vis = mask.astype(np.uint8)
    Image.fromarray(vis).save(path)


def main() -> None:
    out = Path(args_cli.output)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    action_dim = env.action_manager.total_action_dim
    dummy = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
    for _ in range(5):
        env.step(dummy)

    for camera_name in args_cli.cameras:
        camera = env.scene[camera_name]
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        semantic = camera.data.output["semantic_segmentation"][0].detach().cpu().numpy()
        Image.fromarray(rgb).save(out / f"{camera_name}_rgb.png")
        _save_mask(semantic, out / f"{camera_name}_semantic.png")
        print(f"[OK] saved {camera_name} previews under {out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
