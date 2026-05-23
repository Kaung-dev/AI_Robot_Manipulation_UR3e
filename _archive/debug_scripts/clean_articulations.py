"""
Clean up two issues:
  1) Remove the ArticulationRootAPI I accidentally added to /World/ur3e in
     check_and_fix_ur_anchor.py. The real, correct one is already on
     /World/ur3e/root_joint (URDF importer convention).
  2) Deactivate /World/rg2/root_joint instead of trying to delete it.
     RemovePrim() only edits the local layer; the joint comes back via the
     gripper USD reference. SetActive(False) creates a sticky override that
     makes PhysX ignore it.

Run flow:
  - Stop Play.
  - Run.
  - Ctrl+S, then Play.
"""

import omni.usd
from pxr import UsdPhysics

UR_PATH = "/World/ur3e"
GRIPPER_INTERNAL_ROOT = "/World/rg2/root_joint"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    # 1) Remove ArticulationRootAPI from /World/ur3e (keep the one on root_joint)
    ur = stage.GetPrimAtPath(UR_PATH)
    if ur and ur.HasAPI(UsdPhysics.ArticulationRootAPI):
        try:
            ur.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"[OK] removed ArticulationRootAPI from {UR_PATH}")
        except Exception as e:
            print(f"[WARN] could not remove API from {UR_PATH}: {e}")
    else:
        print(f"[OK] no extra ArticulationRootAPI on {UR_PATH}")

    # 2) Deactivate /World/rg2/root_joint
    rj = stage.GetPrimAtPath(GRIPPER_INTERNAL_ROOT)
    if rj and rj.IsValid():
        rj.SetActive(False)
        print(f"[OK] {GRIPPER_INTERNAL_ROOT} SetActive(False) - now inert")
    else:
        print(f"[OK] {GRIPPER_INTERNAL_ROOT} not present")

    print("\nDone. Ctrl+S, then Play.")


main()
