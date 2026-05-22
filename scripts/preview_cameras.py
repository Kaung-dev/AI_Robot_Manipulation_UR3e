"""Save one frame from each camera to PNG for position verification.

Usage:
    ./launch_preview.sh toothbrush

Output:
    /tmp/preview_wrist_cam.png
    /tmp/preview_table_cam.png
"""

"""Launch Isaac Sim first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Save camera preview frames.")
parser.add_argument("--task", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rasterized rendering — no ray tracing needed for a preview.
import carb as _carb
_s = _carb.settings.get_settings()
_s.set_string("/rtx/rendermode", "RasterizedLighting")
_s.set_bool("/rtx/shadows/enabled", False)
_s.set_bool("/rtx/ambientOcclusion/enabled", False)
_s.set_bool("/rtx/reflections/enabled", False)
_s.set_bool("/rtx/indirectDiffuse/enabled", False)
_s.set_bool("/rtx-transient/resourcemanager/enableTextureStreaming", True)

"""Everything else after app launch."""

import numpy as np
import torch
import gymnasium as gym
from PIL import Image

import isaaclab_tasks  # noqa: F401 — registers tasks including symlinked franka_pegboard
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # Build env config the same way record_demos.py does
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Step a few times so camera sensors are fully populated
    action_dim = env.action_manager.total_action_dim
    dummy = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
    for _ in range(5):
        env.step(dummy)

    saved = []
    for name, path in [("wrist_cam", "/tmp/preview_wrist_cam.png"),
                       ("table_cam",  "/tmp/preview_table_cam.png")]:
        try:
            data = env.scene[name].data.output["rgb"]  # (N, H, W, 3) uint8
            if data is None or data.shape[0] == 0:
                print(f"[WARN] {name}: no data returned")
                continue
            frame = data[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(frame).save(path)
            print(f"[OK] {name} -> {path}  shape={frame.shape}")
            saved.append(path)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    env.close()
    simulation_app.close()

    if saved:
        print(f"\nSaved {len(saved)} image(s). Open with:")
        print(f"  xdg-open /tmp/preview_wrist_cam.png")
    else:
        print("\n[FAIL] No images saved — check errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
