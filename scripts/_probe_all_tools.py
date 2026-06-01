"""Find rigid-body prim paths for all 4 tools (brush + 3 distractors)
so FrameTransformerCfg can target them for per-tool tracking spheres.
"""
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
from pxr import Usd, UsdPhysics

import isaaclab_tasks  # noqa
import isaaclab_ext.tasks.air2_franka  # noqa
import isaaclab_ext.tasks.air2_robotis_franka  # noqa
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    if hasattr(env_cfg.scene, "target_frame"):
        env_cfg.scene.target_frame = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    for tool_prim in ("Object", "ToolPliers", "ToolScissors", "ToolScrewdriver"):
        path = f"/World/envs/env_0/{tool_prim}"
        root = stage.GetPrimAtPath(path)
        print(f"\n[probe] /{tool_prim}:", flush=True)
        if not root:
            print(f"  prim not found at {path}", flush=True)
            continue
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print(f"  RIGID BODY: {prim.GetPath()}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
