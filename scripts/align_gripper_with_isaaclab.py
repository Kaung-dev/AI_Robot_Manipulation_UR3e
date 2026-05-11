"""
Make scene.usd's gripper behave like Isaac Lab's lift task does:

  Isaac Lab drives BOTH rg2_gripper_joint and rg2_gripper_mirror_joint with
  the SAME target (close = 1.30 rad, open = 0.0 rad). We currently have the
  mirror set up as a mimic follower with gearing -1, so it rotates OPPOSITE
  to the master, which is what makes our close look wrong.

  This script:
    1. Removes PhysxMimicJointAPI:* from /World/rg2/joints/rg2_gripper_mirror_joint
    2. Removes physxMimicJoint:* attributes / relationships
    3. Restores DriveAPI on the mirror joint with same params as the master
       (so the bridge can drive it independently)

  The 4 finger-side mimic followers (truss arms + finger tips) stay as mimics
  of the master - we only un-mimic the OTHER master.

Run from the project root:  python3 scripts/align_gripper_with_isaaclab.py
"""

from pathlib import Path
import sys

try:
    from pxr import Sdf, Usd, UsdPhysics
except ImportError:
    print("[ERROR] pxr.Usd not found. Need usd-core or run from Isaac venv.")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
SCENE = REPO / "scene" / "scene.usd"
MIRROR = "/World/rg2/joints/rg2_gripper_mirror_joint"

# Match the master drive (so both fingers actuate symmetrically)
DRIVE_STIFFNESS = 20000.0
DRIVE_DAMPING = 800.0
DRIVE_MAX_FORCE = 1000.0


def main():
    if not SCENE.exists():
        print(f"[ERROR] {SCENE} not found")
        return

    stage = Usd.Stage.Open(str(SCENE))
    if not stage:
        print(f"[ERROR] could not open {SCENE}")
        return

    prim = stage.GetPrimAtPath(MIRROR)
    if not prim or not prim.IsValid():
        print(f"[ERROR] {MIRROR} not found in stage")
        return

    # 1) Remove every PhysxMimicJointAPI:* applied schema
    for s in list(prim.GetAppliedSchemas()):
        if "PhysxMimicJointAPI" in s:
            try:
                prim.RemoveAppliedSchema(s)
                print(f"[OK] removed schema {s}")
            except Exception as e:
                print(f"[WARN] could not remove {s}: {e}")

    # 2) Remove every physxMimicJoint:* attribute and relationship
    for attr in list(prim.GetAttributes()):
        n = attr.GetName()
        if n.startswith("physxMimicJoint"):
            prim.RemoveProperty(n)
            print(f"[OK] removed attr {n}")
    for rel in list(prim.GetRelationships()):
        n = rel.GetName()
        if n.startswith("physxMimicJoint"):
            prim.RemoveProperty(n)
            print(f"[OK] removed rel {n}")

    # 3) Apply / restore DriveAPI with master-equivalent params
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateStiffnessAttr().Set(DRIVE_STIFFNESS)
    drive.CreateDampingAttr().Set(DRIVE_DAMPING)
    drive.CreateMaxForceAttr().Set(DRIVE_MAX_FORCE)
    drive.CreateTypeAttr().Set("force")
    drive.CreateTargetPositionAttr().Set(0.0)
    print(
        f"[OK] mirror drive set: stiff={DRIVE_STIFFNESS} "
        f"damp={DRIVE_DAMPING} maxF={DRIVE_MAX_FORCE}"
    )

    stage.GetRootLayer().Save()
    print(f"\nSaved {SCENE}")


if __name__ == "__main__":
    main()
