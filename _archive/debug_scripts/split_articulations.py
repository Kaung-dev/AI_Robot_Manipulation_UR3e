"""
Make the gripper its own articulation again, separate from the UR3e
articulation. The wrist-mount fixed joint /World/wrist_to_rg2 becomes a
'loop' joint between two articulations - PhysX handles that with soft
constraints, NOT with impulse propagation through the UR articulation.

This is the opposite of fix_gripper_integration.py's approach. There we
removed ArticulationRootAPI to MERGE; here we add it back to SPLIT.

After this, the OmniGraph's existing IsaacArticulationController
(/World/RosBridgeGraph/ArtCtrl) will only drive UR joints. The gripper
master 'rg2_gripper_joint' will need a SECOND ArtCtrl (we'll add that
later, after we confirm the arm stays still on Play).

Run flow:
  - Stop Play.
  - Run this in Script Editor.
  - Ctrl+S, then Play.
  - Run record_play.py to verify the arm now stays still.
"""

import omni.usd
from pxr import UsdPhysics

RG2_PATH = "/World/rg2"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    rg2 = stage.GetPrimAtPath(RG2_PATH)
    if not rg2 or not rg2.IsValid():
        return print(f"[ERROR] {RG2_PATH} not found")

    # Apply ArticulationRootAPI to /World/rg2 so it's its own articulation root.
    if rg2.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"[OK] {RG2_PATH} already has ArticulationRootAPI")
    else:
        UsdPhysics.ArticulationRootAPI.Apply(rg2)
        print(f"[OK] applied ArticulationRootAPI to {RG2_PATH}")

    # Also re-enable the OmniGraph if it was disabled by previous diagnostic
    graph = stage.GetPrimAtPath("/World/RosBridgeGraph")
    if graph and graph.IsValid():
        active = graph.GetAttribute("omni:graph:active")
        if active and active.IsValid():
            active.Set(True)
            print(f"[OK] re-enabled /World/RosBridgeGraph")

    print("\nDone. Ctrl+S, then Play. The arm should NOT flail this time.")


main()
