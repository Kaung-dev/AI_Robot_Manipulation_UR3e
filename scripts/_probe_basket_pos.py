"""Diagnostic: walk the AIR2.usd stage and print the world-space position
of every prim whose name looks like a basket so we can verify (or fix)
BASKET_POS_LOCAL in mdp/constants.py.
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
import torch
from pxr import Usd, UsdGeom

import isaaclab_tasks  # noqa
import isaaclab_ext.tasks.air2_franka  # noqa
import isaaclab_ext.tasks.air2_robotis_franka  # noqa
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0].cpu().numpy()
    print(f"[probe] env_origin (env 0) = {env_origin}", flush=True)

    stage = env.unwrapped.scene.stage if hasattr(env.unwrapped.scene, "stage") else None
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()

    print("[probe] walking stage for basket-like prims under /World/envs/env_0:", flush=True)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "env_0" not in path:
            continue
        name = prim.GetName().lower()
        if any(t in name for t in ("basket", "box", "portable", "container", "bin")):
            try:
                m = xform_cache.GetLocalToWorldTransform(prim)
                wpos = m.ExtractTranslation()
                local_pos = (wpos[0] - env_origin[0], wpos[1] - env_origin[1], wpos[2] - env_origin[2])
                candidates.append((path, (wpos[0], wpos[1], wpos[2]), local_pos))
            except Exception as e:
                print(f"  {path}  <error: {e}>", flush=True)

    print(f"[probe] found {len(candidates)} basket-like prims:", flush=True)
    for path, wpos, lpos in candidates:
        print(f"  prim={path}", flush=True)
        print(f"    world_pos = ({wpos[0]:.4f}, {wpos[1]:.4f}, {wpos[2]:.4f})", flush=True)
        print(f"    env_local = ({lpos[0]:.4f}, {lpos[1]:.4f}, {lpos[2]:.4f})", flush=True)

    print("[probe] also printing my current basket marker position from the scene:", flush=True)
    if "basket_marker" in scene.keys() or hasattr(scene, "basket_marker"):
        try:
            marker = scene["basket_marker"]
            mpos = marker.data.root_pos_w[0].cpu().numpy() - env_origin
            print(f"  basket_marker env_local = {tuple(mpos)}", flush=True)
        except Exception as e:
            print(f"  could not read basket_marker: {e}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
