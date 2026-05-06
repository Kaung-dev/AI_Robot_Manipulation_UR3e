"""
Print every UR3e arm joint's current rest angle vs its drive target.
If they don't match, when you press Play the drives will violently pull
the joint to the target — that's the "insane pose" symptom.

Output: /home/user/Desktop/ur_pick/scripts/drives_report.txt
"""
import math
from pathlib import Path

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

OUT = Path("/home/user/Desktop/ur_pick/scripts/drives_report.txt")


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    rep = []
    for name in UR_JOINT_NAMES:
        # find joint anywhere in stage
        prim = None
        for p in stage.Traverse():
            if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint):
                prim = p
                break
        if prim is None:
            rep.append(f"[MISSING] {name}")
            continue

        rj = UsdPhysics.RevoluteJoint(prim)

        # Current rest angle: from the joint state if available, else 0
        # Use PhysicsJointStateAPI if applied
        jstate = prim.GetAttribute("state:angular:physics:position")
        cur = jstate.Get() if jstate and jstate.IsValid() else None

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            rep.append(f"[{name}]  no DriveAPI")
            continue

        target_pos = drive.GetTargetPositionAttr().Get() if drive.GetTargetPositionAttr() else None
        stiffness = drive.GetStiffnessAttr().Get() if drive.GetStiffnessAttr() else None
        damping = drive.GetDampingAttr().Get() if drive.GetDampingAttr() else None
        max_force = drive.GetMaxForceAttr().Get() if drive.GetMaxForceAttr() else None
        drive_type = drive.GetTypeAttr().Get() if drive.GetTypeAttr() else None

        rep.append(
            f"[{name}]\n"
            f"   path: {prim.GetPath()}\n"
            f"   current state:angular:physics:position = {cur}\n"
            f"   drive target = {target_pos} (deg)\n"
            f"   drive type   = {drive_type}\n"
            f"   stiffness    = {stiffness}\n"
            f"   damping      = {damping}\n"
            f"   maxForce     = {max_force}"
        )

    OUT.write_text("\n".join(rep))
    print("\n".join(rep))


main()
