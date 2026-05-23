"""
Flip the gearing on left/right finger_tip mimic joints from +1 to -1, so the
finger_tip rotates opposite to the truss_arm in its local frame, which keeps
the finger pad world-parallel during close (true parallel-jaw behaviour).

Math:
    truss_arm world rotation = +master    (mimic gearing +1)
    finger_tip rotation rel. to truss = -master  (gearing -1)
    => finger_tip world rotation = +master + (-master) = 0  -> stays parallel
"""

import omni.usd

JOINTS = [
    "/World/rg2/joints/rg2_gripper_finger_1_finger_tip_joint",
    "/World/rg2/joints/rg2_gripper_finger_2_finger_tip_joint",
]


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    for path in JOINTS:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            print(f"[SKIP] {path} not found")
            continue

        # find any physxMimicJoint:<axis>:gearing attribute and set to -1.0
        flipped = 0
        for attr in prim.GetAttributes():
            n = attr.GetName()
            if n.startswith("physxMimicJoint:") and n.endswith(":gearing"):
                old = attr.Get()
                attr.Set(-1.0)
                print(f"[OK] {path} {n}: {old} -> -1.0")
                flipped += 1
        if flipped == 0:
            print(f"[WARN] {path}: no physxMimicJoint:*:gearing found")

    print("\nDone. Ctrl+S, then Play. Re-test gripper close in RViz.")


main()
