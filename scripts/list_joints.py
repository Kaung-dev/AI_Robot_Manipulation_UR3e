"""
List every joint in the open stage. Run in Script Editor; output goes to
/home/user/Desktop/ur_pick/scripts/joints_list.txt so we can read it.
"""
from pathlib import Path
import omni.usd
from pxr import UsdPhysics

OUT = Path("/home/user/Desktop/ur_pick/scripts/joints_list.txt")

stage = omni.usd.get_context().get_stage()
lines = []
if stage is None:
    lines.append("[ERROR] no stage open")
else:
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if "Joint" in t or prim.IsA(UsdPhysics.Joint):
            lines.append(f"{prim.GetPath()}  type={t}  name={prim.GetName()}")

OUT.write_text("\n".join(lines))
print(f"Wrote {len(lines)} lines to {OUT}")
print("\n".join(lines[:40]))
