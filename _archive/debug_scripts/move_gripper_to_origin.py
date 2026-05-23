"""
The Inria test URDF placed the gripper bracket 2 m from the world frame
(xyz="2. 0.05 2." rpy="pi/2 -pi/2 0" in test.urdf.xacro). For inspection in
isolation we want the gripper at world origin. This script zeroes the
bracket_joint local offset and resets its local rotation, so the gripper
sits at (0, 0, 0).

When we later weld this gripper onto the UR3e wrist in scene.usd, our wrist
mount joint takes over anyway and makes the bracket_joint's contribution
moot.

Run in Isaac Sim Script Editor with the imported gripper open. Stop Play
first. Save (Ctrl+S) afterwards.
"""

import omni.usd
from pxr import Gf, UsdPhysics

BRACKET_JOINT_NAME = "rg2_gripper_bracket_joint"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open.")
        return

    bracket = None
    for prim in stage.Traverse():
        if prim.GetName() == BRACKET_JOINT_NAME and prim.IsA(UsdPhysics.Joint):
            bracket = prim
            break
    if bracket is None:
        print(f"[ERROR] joint '{BRACKET_JOINT_NAME}' not found")
        return

    j = UsdPhysics.Joint(bracket)
    old_p0 = j.GetLocalPos0Attr().Get()
    old_r0 = j.GetLocalRot0Attr().Get()
    old_p1 = j.GetLocalPos1Attr().Get()
    old_r1 = j.GetLocalRot1Attr().Get()

    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    print(f"[OK] {bracket.GetPath()}")
    print(f"     localPos0 {old_p0} -> (0,0,0)")
    print(f"     localRot0 {old_r0} -> (1,0,0,0)")
    print(f"     localPos1 {old_p1} -> (0,0,0)")
    print(f"     localRot1 {old_r1} -> (1,0,0,0)")
    print("\nDone. Ctrl+S to save.")


main()
