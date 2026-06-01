"""Walk the basket-prim subtree to find anything FrameTransformer can target."""
from __future__ import annotations

import argparse, sys
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
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    # nuke our broken markers so the scene loads
    for k in ("basket_marker", "basket_beacon", "target_frame", "brush_frame", "pliers_frame", "scissors_frame", "screwdriver_frame", "ee_tcp_marker"):
        if hasattr(cfg.scene, k):
            setattr(cfg.scene, k, None)
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    print("[probe] full /Environment/SM_BoxPortableD subtree:", flush=True)
    root = stage.GetPrimAtPath("/World/envs/env_0/Environment/SM_BoxPortableD")
    if not root:
        print("  not found", flush=True)
    else:
        for prim in Usd.PrimRange(root):
            t = prim.GetTypeName()
            rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            print(f"  type={t:<18} rb={rb}  {prim.GetPath()}", flush=True)
    print("\n[probe] children of /Environment that look basket/box-like:", flush=True)
    env_prim = stage.GetPrimAtPath("/World/envs/env_0/Environment")
    if env_prim:
        for child in env_prim.GetChildren():
            name = child.GetName().lower()
            if any(t in name for t in ("box", "basket", "portable", "container")):
                rb = child.HasAPI(UsdPhysics.RigidBodyAPI)
                print(f"  rb={rb}  {child.GetPath()}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
