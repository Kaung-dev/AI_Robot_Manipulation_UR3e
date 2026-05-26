"""Convert robotis_net_table.usd to text (USDA) and show all physics prims."""
from pathlib import Path
from isaaclab.app import AppLauncher
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_USD = PROJECT_ROOT / "exported_assets" / "object" / "robotis_net_table.usd"
OUT_USDA = PROJECT_ROOT / "exported_assets" / "object" / "robotis_net_table.usda"

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(str(IN_USD))

print("=== All prims with physics APIs ===")
for prim in stage.Traverse():
    apis = []
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):   apis.append("RigidBodyAPI")
    if prim.HasAPI(UsdPhysics.CollisionAPI):    apis.append("CollisionAPI")
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI): apis.append("ArticulationRootAPI")
    if apis:
        print(f"  {prim.GetPath()} : {', '.join(apis)}")

# Export as human-readable text
stage.GetRootLayer().Export(str(OUT_USDA))
print(f"\nText USD saved to: {OUT_USDA}")
app.close()
