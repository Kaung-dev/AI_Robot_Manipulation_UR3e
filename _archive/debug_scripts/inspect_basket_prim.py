"""One-off: enumerate every prim under SM_BoxPortableD to find the actual rendered geometry."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

OUT = REPO / "_mvp_logs" / "basket_prim_tree.txt"

try:
    cfg = parse_env_cfg("Isaac-Lift-AIR2-UR3e-RG2-Segmentation-Play-v0", device="cuda:0", num_envs=1)
    env = gym.make("Isaac-Lift-AIR2-UR3e-RG2-Segmentation-Play-v0", cfg=cfg).unwrapped
    env.reset()
    env.step(torch.zeros(1, env.action_manager.total_action_dim, device=env.device))

    stage = env.sim.stage
    lines = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "SM_BoxPortable" in path or "boxportable" in path.lower():
            lines.append(f"{prim.GetTypeName():25s} {path}")
    OUT.write_text("\n".join(lines) + f"\n\nTotal: {len(lines)} prims with SM_BoxPortable in path")
finally:
    app.close()
