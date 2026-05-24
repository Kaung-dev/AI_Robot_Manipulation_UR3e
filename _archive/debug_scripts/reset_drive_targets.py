"""
For every UR3e arm joint, set the drive's target position equal to the
joint's current state position. After this, when you press Play, the drives
will hold the joint at its current rest pose instead of yanking it to a
stale target value left over from previous test runs.

Run once before Play. Save (Ctrl+S) afterwards.
"""
import math
import omni.usd
from pxr import UsdPhysics

UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] no stage")
        return

    for name in UR_JOINT_NAMES:
        prim = None
        for p in stage.Traverse():
            if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint):
                prim = p
                break
        if prim is None:
            print(f"[SKIP] {name} not found")
            continue

        # current angle is in radians on this attribute
        cur_attr = prim.GetAttribute("state:angular:physics:position")
        if not cur_attr or not cur_attr.IsValid() or cur_attr.Get() is None:
            print(f"[SKIP] {name} has no state:angular:physics:position")
            continue
        cur_rad = float(cur_attr.Get())
        cur_deg = math.degrees(cur_rad)

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[SKIP] {name} has no DriveAPI")
            continue
        old_target = drive.GetTargetPositionAttr().Get()
        drive.CreateTargetPositionAttr().Set(cur_deg)
        # also zero target velocity in case it was nonzero
        drive.CreateTargetVelocityAttr().Set(0.0)

        print(f"[OK] {name}: target {old_target:.2f} -> {cur_deg:.2f} deg")

    print("\nDone. Ctrl+S to save, then Play.")


main()
