"""
Set finite lower/upper limits on every revolute joint that has PhysxMimicJointAPI
applied. PhysX 5's mimic-joint feature requires the follower joints to have
finite limits, otherwise it refuses to enforce the constraint and you see:

    Usd Physics: the revolute joint at prim /.../foo_joint needs a finite limit
    set to be used by the mimic joint feature.

URDF range (rg2_v1.yaml): lower=0.0 rad, upper=1.3 rad
=> in USD degrees: lower=0.0, upper=~74.485

Run in Isaac Sim Script Editor with the imported gripper open. Save (Ctrl+S)
afterwards.
"""

import math

import omni.usd
from pxr import Usd, UsdPhysics

LOWER_RAD = 0.0
UPPER_RAD = 1.3

LOWER_DEG = math.degrees(LOWER_RAD)
UPPER_DEG = math.degrees(UPPER_RAD)


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open.")
        return

    fixed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        applied = prim.GetAppliedSchemas()
        # Skip joints that don't carry the mimic API
        has_mimic = any("PhysxMimicJointAPI" in s for s in applied)
        if not has_mimic:
            continue

        rj = UsdPhysics.RevoluteJoint(prim)
        lo_attr = rj.CreateLowerLimitAttr()
        hi_attr = rj.CreateUpperLimitAttr()

        old_lo = lo_attr.Get()
        old_hi = hi_attr.Get()
        new_lo = LOWER_DEG
        new_hi = UPPER_DEG

        lo_attr.Set(new_lo)
        hi_attr.Set(new_hi)

        print(
            f"[OK] {prim.GetPath()}\n"
            f"     lower {old_lo} -> {new_lo:.3f} deg\n"
            f"     upper {old_hi} -> {new_hi:.3f} deg"
        )
        fixed += 1

    if fixed == 0:
        print("No mimic-tagged revolute joints found. Nothing to do.")
    else:
        print(f"\nDone. Set limits on {fixed} joints. Ctrl+S to save, then Play.")


main()
