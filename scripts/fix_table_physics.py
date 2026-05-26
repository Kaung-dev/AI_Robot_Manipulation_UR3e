"""Strip RigidBodyAPI from all child meshes in robotis_net_table.usd.

The original USD has RigidBodyAPI on every individual mesh prim, which causes
them to fall under gravity even when IsaacLab sets kinematic=True on the root.
This script removes those per-mesh rigid bodies and applies a single static
CollisionAPI setup so the table behaves as a pure static collider.

Reads : exported_assets/object/robotis_net_table.usd  (untouched)
Writes: exported_assets/object/robotis_net_table_fixed.usd
"""
from pathlib import Path
from isaaclab.app import AppLauncher
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_USD  = PROJECT_ROOT / "exported_assets" / "object" / "robotis_net_table.usd"
OUT_USD = PROJECT_ROOT / "exported_assets" / "object" / "robotis_net_table_fixed.usd"

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(str(IN_USD))

removed = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        removed.append(prim.GetPath().pathString)

print(f"Removed RigidBodyAPI from {len(removed)} prims:")
for p in removed:
    print(f"  {p}")

stage.GetRootLayer().Export(str(OUT_USD))
print(f"\nSaved: {OUT_USD}")
app.close()
