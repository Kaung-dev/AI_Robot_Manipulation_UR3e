"""
Properly tune gripper drives:
  - rg2_gripper_joint (the master): stiff drive, damped, with maxForce.
  - All other revolute joints in the gripper: ZERO drives so the
    PhysxMimicJointAPI is the sole authority on their motion.
    (If followers have non-zero drives, they fight the mimic.)
"""
import omni.usd
from pxr import Usd, UsdPhysics

RG2 = "/World/rg2"
MASTER = "rg2_gripper_joint"

# Master tuning - strong & well-damped so it can hold against contact at limit
# AND drag the 5 mimic followers along.
M_STIFFNESS = 20000.0
M_DAMPING = 800.0
M_MAX_FORCE = 1000.0


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    rg2 = stage.GetPrimAtPath(RG2)
    if not rg2 or not rg2.IsValid():
        return print(f"[ERROR] {RG2} not found")

    for prim in Usd.PrimRange(rg2):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        if prim.GetName() == MASTER:
            drive.CreateStiffnessAttr().Set(M_STIFFNESS)
            drive.CreateDampingAttr().Set(M_DAMPING)
            drive.CreateMaxForceAttr().Set(M_MAX_FORCE)
            drive.CreateTypeAttr().Set("force")
            print(f"[MASTER] {prim.GetPath()}: stiff={M_STIFFNESS} damp={M_DAMPING} maxF={M_MAX_FORCE}")
        else:
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            drive.CreateMaxForceAttr().Set(0.0)
            drive.CreateTypeAttr().Set("force")
            print(f"[follower] {prim.GetPath()}: drive zeroed (mimic is in charge)")

    print("\nDone. Ctrl+S, then Play.")


main()
