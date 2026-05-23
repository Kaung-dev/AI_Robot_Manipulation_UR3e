"""Print all prim paths and world transforms in AIR2.usd.

Run with:
  ./IsaacLab/isaaclab.sh -p ur_pick/scripts/inspect_air2_hooks.py

Look for prims named 'hook', 'Hook', 'peg', or similar to find hook positions.
"""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pathlib import Path
from pxr import Usd, UsdGeom, Gf

SCENE = Path(__file__).resolve().parents[1] / "scene" / "AIR2.usd"

print(f"Opening: {SCENE}", flush=True)
stage = Usd.Stage.Open(str(SCENE))
if not stage:
    print("[ERROR] failed to open stage")
    app.close()
    sys.exit(1)

# Print full hierarchy (top 3 levels only to avoid noise)
print("\n=== PRIM HIERARCHY (depth ≤ 3) ===", flush=True)
for prim in stage.Traverse():
    depth = len(prim.GetPath().pathString.split("/")) - 1
    if depth <= 3:
        indent = "  " * depth
        print(f"{indent}{prim.GetPath()}  [{prim.GetTypeName()}]", flush=True)

# Print world transforms for hooks
print("\n=== HOOK / PEG WORLD TRANSFORMS ===", flush=True)
keywords = {"hook", "peg", "Hook", "Peg", "hang", "slot"}
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if any(k.lower() in name for k in keywords):
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            tx = world_xform.ExtractTranslation()
            print(f"  {prim.GetPath()}: xyz = ({tx[0]:.4f}, {tx[1]:.4f}, {tx[2]:.4f})", flush=True)

# Print Franka / robot prim paths
print("\n=== ROBOT PRIMS ===", flush=True)
robot_keywords = {"franka", "panda", "robot", "kafka"}
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if any(k in name for k in robot_keywords):
        xformable = UsdGeom.Xformable(prim)
        pos_str = ""
        if xformable:
            world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            tx = world_xform.ExtractTranslation()
            pos_str = f" xyz = ({tx[0]:.4f}, {tx[1]:.4f}, {tx[2]:.4f})"
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]{pos_str}", flush=True)

print("\nDone.", flush=True)
app.close()
