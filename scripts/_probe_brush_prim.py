"""Find the actual rigid-body prim path for the brush so FrameTransformer can target it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Brush-Play-v0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab_tasks  # noqa
import isaaclab_ext.tasks.air2_franka  # noqa
import isaaclab_ext.tasks.air2_robotis_franka  # noqa
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    # Pre-emptively remove the FrameTransformer that's breaking on /Object
    if hasattr(env_cfg.scene, "target_frame"):
        env_cfg.scene.target_frame = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    print("[probe] hierarchy under /World/envs/env_0/Object:", flush=True)
    root = stage.GetPrimAtPath("/World/envs/env_0/Object")
    if not root:
        print("  /Object not found", flush=True)
    else:
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            type_name = prim.GetTypeName()
            has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            print(f"  type={type_name:<20} rb_api={has_rb}  {path}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
