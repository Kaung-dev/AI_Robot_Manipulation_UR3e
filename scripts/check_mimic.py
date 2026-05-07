"""
Quick read-out of every gripper mimic joint: gearing, referenceJoint, axis.
Output: scripts/check_mimic.txt
"""
from pathlib import Path
import omni.usd
from pxr import Usd, UsdPhysics

OUT = Path("/home/user/Desktop/ur_pick/scripts/check_mimic.txt")
RG2 = "/World/rg2"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    rg2 = stage.GetPrimAtPath(RG2)
    if not rg2 or not rg2.IsValid():
        return print(f"[ERROR] {RG2} not found")

    lines = []
    for prim in Usd.PrimRange(rg2):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        applied = prim.GetAppliedSchemas()
        has_mimic = any("Mimic" in s for s in applied)
        if not has_mimic:
            continue
        lines.append(prim.GetPath().pathString)
        for attr in prim.GetAttributes():
            n = attr.GetName()
            if n.startswith("physxMimicJoint"):
                lines.append(f"   {n} = {attr.Get()}")
        for rel in prim.GetRelationships():
            n = rel.GetName()
            if n.startswith("physxMimicJoint"):
                tgts = [t.pathString for t in rel.GetTargets()]
                lines.append(f"   {n} -> {tgts}")

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n--> {OUT}")


main()
