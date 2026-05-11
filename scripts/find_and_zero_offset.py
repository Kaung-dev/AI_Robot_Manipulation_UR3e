"""
The 2 m offset from the Inria test URDF was baked into an xformOp:translate
when the URDF importer merged the fixed bracket_joint into link transforms.
This script:
  (1) lists all xformable prims under /onrobot_rg_test with non-zero translate
  (2) zeroes the largest translate (>= 0.5 m) so the gripper sits at origin

Output also written to scripts/offset_report.txt
"""
from pathlib import Path
import omni.usd
from pxr import Usd, UsdGeom, Gf
PROJECT_ROOT = Path(__file__).resolve().parent.parent


OUT = Path(str(PROJECT_ROOT / "scripts" / "offset_report.txt"))
ROOT = "/onrobot_rg_test"

stage = omni.usd.get_context().get_stage()
lines = []
if stage is None:
    lines.append("[ERROR] no stage")
else:
    root_prim = stage.GetPrimAtPath(ROOT)
    if not root_prim or not root_prim.IsValid():
        lines.append(f"[ERROR] no {ROOT}")
    else:
        # Collect xformable prims with their translate
        candidates = []
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Xformable):
                continue
            xf = UsdGeom.Xformable(prim)
            t_attr = prim.GetAttribute("xformOp:translate")
            if not t_attr or not t_attr.IsValid():
                continue
            v = t_attr.Get()
            if v is None:
                continue
            mag = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
            if mag > 0.01:  # >1cm of offset is interesting
                candidates.append((mag, prim.GetPath().pathString, tuple(v), t_attr))

        candidates.sort(reverse=True)
        lines.append(f"Xformable prims with >1cm translate:")
        for mag, path, v, _ in candidates:
            lines.append(f"  |t|={mag:.3f}  {path}  translate={v}")

        # Zero anything with >0.5 m translate (this is our 2 m bracket offset)
        zeroed = 0
        for mag, path, v, attr in candidates:
            if mag < 0.5:
                continue
            attr.Set(Gf.Vec3d(0, 0, 0))
            lines.append(f"  ZEROED {path}  was {v}")
            zeroed += 1
        lines.append(f"\nZeroed {zeroed} prim(s).")

OUT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\n--> {OUT}")
