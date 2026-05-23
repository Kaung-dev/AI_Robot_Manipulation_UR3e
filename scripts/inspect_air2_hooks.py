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

# Basket / box props — SM_BoxPortableD is the cardboard box baked into AIR2.usd
# that the AIR2 task treats as the drop-off receptacle.
print("\n=== BASKET / BOX PRIMS ===", flush=True)
basket_keywords = {"box", "basket", "bin", "sm_", "portable", "container"}
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if any(k in name for k in basket_keywords):
        xformable = UsdGeom.Xformable(prim)
        pos_str = ""
        if xformable:
            world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            tx = world_xform.ExtractTranslation()
            pos_str = f" xyz = ({tx[0]:.4f}, {tx[1]:.4f}, {tx[2]:.4f})"
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]{pos_str}", flush=True)

# Overall bounding box of "interesting" stuff (hooks + robot + basket) so we
# know what the third-person camera has to fit in frame.
print("\n=== SCENE EXTENT (hooks + robot + box) ===", flush=True)
xs, ys, zs = [], [], []
extent_keywords = {"hook", "peg", "franka", "panda", "robot", "box", "basket", "portable"}
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if not any(k in name for k in extent_keywords):
        continue
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        continue
    tx = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    xs.append(tx[0]); ys.append(tx[1]); zs.append(tx[2])
if xs:
    print(f"  x range: [{min(xs):.3f}, {max(xs):.3f}]  center {(min(xs)+max(xs))/2:.3f}", flush=True)
    print(f"  y range: [{min(ys):.3f}, {max(ys):.3f}]  center {(min(ys)+max(ys))/2:.3f}", flush=True)
    print(f"  z range: [{min(zs):.3f}, {max(zs):.3f}]  center {(min(zs)+max(zs))/2:.3f}", flush=True)

print("\nDone.", flush=True)
app.close()
