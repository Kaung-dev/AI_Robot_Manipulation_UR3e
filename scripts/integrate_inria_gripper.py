"""
Integrate the new (Inria) RG2 USD into scene.usd:

  1) Delete the existing /World/rg2 (the broken closed-chain version).
  2) Define /World/rg2 as an Xform that references the new gripper USD.
  3) Walk under /World/rg2 and remove any PhysicsArticulationRootAPI so the
     gripper merges into the UR3e articulation (single articulation = the
     existing OmniGraph IsaacArticulationController can drive everything).
  4) Add a fixed joint /World/wrist_to_rg2 connecting
        body0 = /World/ur3e/wrist_3_link
        body1 = /World/rg2/world/rg2_gripper_bracket
     so PhysX places the gripper at the wrist on Play, ignoring the cosmetic
     2 m bracket offset baked in the gripper USD.

Run flow:
  - In Isaac Sim, File -> Open scene.usd
  - Window -> Script Editor -> File -> Open this script -> Run
  - Ctrl+S to save scene.usd
  - Press Play

Output: also writes a short report to /home/user/Desktop/ur_pick/scripts/integration_report.txt
"""

from pathlib import Path

import omni.usd
from pxr import Sdf, Gf, Usd, UsdGeom, UsdPhysics

GRIPPER_USD = "/home/user/Desktop/ur_pick/rg2_inria_usd/rg2_inria.usd"
RG2_PATH = "/World/rg2"
WRIST_LINK = "/World/ur3e/wrist_3_link"
BRACKET_REL = "world/rg2_gripper_bracket"  # path *under* RG2_PATH
MOUNT_JOINT_PATH = "/World/wrist_to_rg2"

OUT = Path("/home/user/Desktop/ur_pick/scripts/integration_report.txt")


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open. Open scene.usd first.")
        return

    rep = []

    # 1) remove existing /World/rg2 if present
    existing = stage.GetPrimAtPath(RG2_PATH)
    if existing and existing.IsValid():
        ok = stage.RemovePrim(RG2_PATH)
        rep.append(f"[{'OK' if ok else 'FAIL'}] removed existing {RG2_PATH}")
    else:
        rep.append(f"[OK] no existing {RG2_PATH} to remove")

    # 2) define /World/rg2 as Xform with reference to the new gripper USD
    rg2 = UsdGeom.Xform.Define(stage, RG2_PATH)
    rg2.GetPrim().GetReferences().AddReference(GRIPPER_USD)
    rep.append(f"[OK] referenced {GRIPPER_USD} at {RG2_PATH}")

    # 3) traverse the (now-loaded) gripper subtree, remove ArticulationRootAPI
    removed_roots = []
    for prim in Usd.PrimRange(rg2.GetPrim()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            try:
                ok = prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            except Exception:
                ok = False
            removed_roots.append((prim.GetPath().pathString, ok))
    if removed_roots:
        for p, ok in removed_roots:
            rep.append(f"[{'OK' if ok else 'FAIL'}] removed ArticulationRootAPI from {p}")
    else:
        rep.append("[OK] no ArticulationRootAPI under gripper to remove")

    # 4) Add fixed joint mounting bracket to wrist_3
    wrist = stage.GetPrimAtPath(WRIST_LINK)
    if not wrist or not wrist.IsValid():
        rep.append(f"[ERROR] {WRIST_LINK} not found in scene; aborting joint creation")
        OUT.write_text("\n".join(rep))
        print("\n".join(rep))
        return
    bracket_path = f"{RG2_PATH}/{BRACKET_REL}"
    bracket = stage.GetPrimAtPath(bracket_path)
    if not bracket or not bracket.IsValid():
        rep.append(f"[ERROR] {bracket_path} not found; aborting joint creation")
        OUT.write_text("\n".join(rep))
        print("\n".join(rep))
        return

    # remove any prior mount joint
    if stage.GetPrimAtPath(MOUNT_JOINT_PATH):
        stage.RemovePrim(MOUNT_JOINT_PATH)

    fj = UsdPhysics.FixedJoint.Define(stage, MOUNT_JOINT_PATH)
    fj.CreateBody0Rel().SetTargets([Sdf.Path(WRIST_LINK)])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(bracket_path)])
    # local poses: zero on wrist side, identity on bracket side. The gripper
    # will sit at the wrist origin with its bracket facing along wrist's +Z.
    # Adjust here if you need a tool offset (e.g. 0,0,0.005 to lift it slightly).
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    rep.append(f"[OK] added fixed joint {MOUNT_JOINT_PATH}: {WRIST_LINK} -> {bracket_path}")

    rep.append("\nDone. Ctrl+S to save scene.usd, then Play.")
    OUT.write_text("\n".join(rep))
    print("\n".join(rep))


main()
