"""Launch the AIR2 IsaacLab env in GUI. Click any prim in env_0 (e.g. a peg
on the pegboard) and its env-LOCAL position prints to the terminal — exactly
the value to paste into HOOK_POSITIONS in mdp/events.py.

Run:
    conda deactivate
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/click_peg_coords.py

In the Isaac Sim viewport:
    1. Wait for the scene to load (yellow pegboard appears)
    2. Click each visible peg you want to use as a hook
    3. Terminal prints `[click] <prim_path>  local=(X, Y, Z)`
    4. Copy those values back here
    5. Ctrl+C to quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-AIR2-Franka-Segmentation-Play-v0")
AppLauncher.add_app_launcher_args(parser)
# Force GUI mode regardless of CLI defaults.
sys.argv += ["--enable_cameras"]
args_cli = parser.parse_args()
args_cli.headless = False  # always show GUI

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

# -- post-sim imports -------------------------------------------------------

import gymnasium as gym
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom

import isaaclab_tasks  # noqa: F401 -- needed for task-registry side effect
import isaaclab_ext.tasks.air2_franka  # noqa: F401 -- registers AIR2 task IDs
from isaaclab_tasks.utils import parse_env_cfg


env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
env_cfg.episode_length_s = 99999.0
env = gym.make(args_cli.task, cfg=env_cfg)
env.reset()
print(f"[click-peg] env loaded: {args_cli.task}", flush=True)

env_origin = env.unwrapped.scene.env_origins[0].cpu().tolist()
print(f"[click-peg] env_0 origin = ({env_origin[0]:.3f}, {env_origin[1]:.3f}, {env_origin[2]:.3f})",
      flush=True)
print("[click-peg] click any peg in the viewport — env-local coords print here.",
      flush=True)
print("[click-peg] Ctrl+C in this terminal to quit.\n", flush=True)

ctx = omni.usd.get_context()
SEL_EVT = int(omni.usd.StageEventType.SELECTION_CHANGED)
_last: tuple = ()


def _print_selected():
    global _last
    selection = ctx.get_selection()
    paths = tuple(selection.get_selected_prim_paths())
    if paths == _last or not paths:
        return
    _last = paths
    stage = ctx.get_stage()
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        xf = UsdGeom.Xformable(prim)
        if not xf:
            print(f"[click] {path}  (non-xformable)", flush=True)
            continue
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        wx, wy, wz = float(t[0]), float(t[1]), float(t[2])
        lx, ly, lz = wx - env_origin[0], wy - env_origin[1], wz - env_origin[2]
        print(f"[click] {path}", flush=True)
        print(f"        world=({wx:.4f}, {wy:.4f}, {wz:.4f})  "
              f"local=({lx:.4f}, {ly:.4f}, {lz:.4f})", flush=True)


def _on_event(e):
    if e.type == SEL_EVT:
        _print_selected()


sub = ctx.get_stage_event_stream().create_subscription_to_pop(
    _on_event, name="click_peg_print"
)

app = omni.kit.app.get_app()
try:
    while sim_app.is_running():
        app.update()
finally:
    env.close()
    sim_app.close()
