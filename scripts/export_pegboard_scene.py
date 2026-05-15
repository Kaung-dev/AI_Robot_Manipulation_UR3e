"""Export the fully assembled pegboard task scene (robot + table + basket + 6 tools)
to a single USD file you can re-open in Isaac Sim and edit (add cameras, lights, props).

Run from C:\\isaac\\IsaacLab on Windows:
    .\\isaaclab.bat -p <path>\\scripts\\export_pegboard_scene.py [--out PATH] [--task TASK]
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export pegboard scene to USD.")
parser.add_argument("--task", default="Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-Play-v0")
parser.add_argument("--out", default=None,
                    help="Output USD path. Defaults to <repo>/scene/pegboard_full.usd.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

if args.out is None:
    args.out = str(Path(__file__).resolve().parent.parent / "scene" / "pegboard_full.usd")

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401 — registers task IDs
from isaaclab_tasks.utils import parse_env_cfg
import omni.usd

env_cfg = parse_env_cfg(args.task, num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
env.reset()

stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().Export(args.out)
print(f"[export_pegboard_scene] saved -> {args.out}")

env.close()
sim_app.close()
