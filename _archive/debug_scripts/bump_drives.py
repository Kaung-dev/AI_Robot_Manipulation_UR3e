"""
Bump every UR3e arm joint's drive stiffness/damping/maxForce up to values
that can actually hold the arm + gripper load without oscillating.

Why: the URDF importer set drive gains via natural-frequency assuming the
URDF link masses (which are tiny placeholder values like 0.001 kg). The
resulting gains are ~1000x too low for a real arm.

Numbers chosen here: ~critically damped at ~25 Hz natural freq for an
effective inertia of order 1 kg.m^2 per joint:
   k = m * (2*pi*f)^2 ~= 25000
   c = 2 * sqrt(k*m)  ~= 320

Run once before Play. Save (Ctrl+S) afterwards. Re-run reset_drive_targets.py
afterwards if needed (this script doesn't touch targetPosition).
"""

import omni.usd
from pxr import UsdPhysics

# (joint_name, stiffness, damping, maxForce)
SETTINGS = {
    "shoulder_pan_joint":  (25000.0, 320.0, 200.0),
    "shoulder_lift_joint": (25000.0, 320.0, 200.0),
    "elbow_joint":         (15000.0, 240.0, 120.0),
    "wrist_1_joint":       ( 5000.0, 140.0,  60.0),
    "wrist_2_joint":       ( 5000.0, 140.0,  60.0),
    "wrist_3_joint":       ( 2000.0,  90.0,  40.0),
}


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for name, (stiff, damp, mf) in SETTINGS.items():
        prim = None
        for p in stage.Traverse():
            if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint):
                prim = p
                break
        if prim is None:
            print(f"[SKIP missing] {name}")
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[SKIP no drive] {name}")
            continue
        old_k = drive.GetStiffnessAttr().Get() if drive.GetStiffnessAttr() else None
        old_c = drive.GetDampingAttr().Get() if drive.GetDampingAttr() else None
        old_f = drive.GetMaxForceAttr().Get() if drive.GetMaxForceAttr() else None
        drive.CreateStiffnessAttr().Set(stiff)
        drive.CreateDampingAttr().Set(damp)
        drive.CreateMaxForceAttr().Set(mf)
        print(
            f"[OK] {name}: "
            f"stiff {old_k} -> {stiff}, "
            f"damp {old_c} -> {damp}, "
            f"maxF {old_f} -> {mf}"
        )

    print("\nDone. Ctrl+S to save, then Play.")


main()
