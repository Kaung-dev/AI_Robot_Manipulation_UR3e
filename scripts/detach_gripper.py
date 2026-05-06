"""
Detach the gripper from scene.usd:
  - delete /World/rg2 (the referenced gripper)
  - delete /World/wrist_to_rg2 (the wrist mount fixed joint)

Gripper work is paused until we revisit it with a cleaner strategy.
This lets us finish the arm-only MoveIt -> Isaac demo without the
violent articulation behavior we keep hitting.

Run flow:
  - Stop Play.
  - Run this script in Script Editor.
  - Ctrl+S to save scene.usd.
  - Press Play. Arm should be still and respond cleanly to /isaac_joint_commands.
"""

import omni.usd

PATHS = [
    "/World/wrist_to_rg2",
    "/World/rg2",
]


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] no stage")
        return
    for p in PATHS:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            ok = stage.RemovePrim(p)
            print(f"[{'OK' if ok else 'FAIL'}] removed {p}")
        else:
            print(f"[SKIP not present] {p}")
    print("\nDone. Ctrl+S, then Play.")


main()
