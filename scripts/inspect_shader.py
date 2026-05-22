"""Inspect shader attributes in a USD file. Run with isaaclab.sh -p."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pathlib import Path
from pxr import Usd, UsdShade

ASSETS = Path(__file__).resolve().parents[1] / "exported_assets" / "object"

for fname in ["tooth_brush.usd", "tooth_brush_green.usd"]:
    path = ASSETS / fname
    print(f"\n{'='*60}", flush=True)
    if not path.exists():
        print(f"[SKIP] {fname} not found", flush=True)
        continue
    print(f"FILE: {fname}", flush=True)
    stage = Usd.Stage.Open(str(path))
    if not stage:
        print(f"  [ERROR] failed to open stage", flush=True)
        continue
    found = False
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            found = True
            print(f"  SHADER: {prim.GetPath()}", flush=True)
            for attr in prim.GetAttributes():
                val = attr.Get()
                if val is not None:
                    print(f"    {attr.GetName()} = {repr(val)}", flush=True)
    if not found:
        print("  (no shader prims found)", flush=True)

print("\nDone.", flush=True)
app.close()
