"""
Fix the gripper integration in scene.usd:

  1) (Re-)reference the Inria gripper USD at /World/rg2 if missing.
  2) Remove /World/rg2/root_joint (the gripper's own articulation root fixed
     joint). It pins the bracket to (2, 0.05, 2) world from import time and
     fights our /World/wrist_to_rg2 wrist mount. That fight produces the
     violent corrective impulse on Play.
  3) Remove any nested ArticulationRootAPI under /World/rg2 so the gripper
     joins the UR3e articulation cleanly.
  4) Recompute /World/rg2's transform so the bracket lands at wrist_3_link
     rest pose (matrix-order corrected).
  5) (Re-)create the wrist mount fixed joint /World/wrist_to_rg2.

Run flow:
  - Open scene.usd, stop Play.
  - Run this in Script Editor.
  - Ctrl+S, then Play.
"""

from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

GRIPPER_USD = "/home/user/Desktop/ur_pick/rg2_inria_usd/rg2_inria.usd"
RG2_PATH = "/World/rg2"
WRIST_LINK = "/World/ur3e/wrist_3_link"
BRACKET_PATH = "/World/rg2/world/rg2_gripper_bracket"
INTERNAL_ROOT_JOINT = "/World/rg2/root_joint"
MOUNT_JOINT = "/World/wrist_to_rg2"

OUT = Path("/home/user/Desktop/ur_pick/scripts/fix_gripper_report.txt")


def _set_xform_matrix(prim, mat: Gf.Matrix4d):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for a in list(prim.GetAttributes()):
        n = a.GetName()
        if n.startswith("xformOp:") and n != "xformOp:transform":
            prim.RemoveProperty(n)
    op = xf.AddTransformOp()
    op.Set(mat)


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    rep = []

    # 1) ensure /World/rg2 exists with a reference to the gripper USD
    rg2 = stage.GetPrimAtPath(RG2_PATH)
    if not rg2 or not rg2.IsValid():
        rep.append(f"[OK] /World/rg2 missing -> creating Xform with reference")
        rg2 = UsdGeom.Xform.Define(stage, RG2_PATH).GetPrim()
        rg2.GetReferences().AddReference(GRIPPER_USD)
    else:
        rep.append(f"[OK] /World/rg2 already present")

    # 2) Remove /World/rg2/root_joint - the conflicting internal anchor
    rj = stage.GetPrimAtPath(INTERNAL_ROOT_JOINT)
    if rj and rj.IsValid():
        ok = stage.RemovePrim(INTERNAL_ROOT_JOINT)
        rep.append(f"[{'OK' if ok else 'FAIL'}] removed {INTERNAL_ROOT_JOINT}")
    else:
        rep.append(f"[OK] no {INTERNAL_ROOT_JOINT} to remove")

    # 3) Remove any nested ArticulationRootAPI under /World/rg2
    removed = []
    for prim in Usd.PrimRange(rg2):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            try:
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                removed.append(prim.GetPath().pathString)
            except Exception as e:
                rep.append(f"[WARN] could not remove ArticulationRootAPI from {prim.GetPath()}: {e}")
    if removed:
        for p in removed:
            rep.append(f"[OK] removed ArticulationRootAPI from {p}")
    else:
        rep.append(f"[OK] no nested ArticulationRootAPI")

    # 4) Position /World/rg2 so bracket lands at wrist (matrix-order correct)
    _set_xform_matrix(rg2, Gf.Matrix4d(1.0))  # identity first so we read raw bracket
    xc = UsdGeom.XformCache()
    xc.Clear()
    wrist = stage.GetPrimAtPath(WRIST_LINK)
    bracket = stage.GetPrimAtPath(BRACKET_PATH)
    if not wrist or not wrist.IsValid() or not bracket or not bracket.IsValid():
        rep.append(f"[FAIL] wrist or bracket prim missing")
        OUT.write_text("\n".join(rep))
        return print("\n".join(rep))
    wrist_world = xc.GetLocalToWorldTransform(wrist)
    bracket_world = xc.GetLocalToWorldTransform(bracket)
    new_rg2 = bracket_world.GetInverse() * wrist_world
    _set_xform_matrix(rg2, new_rg2)
    xc.Clear()
    bracket_after = xc.GetLocalToWorldTransform(bracket)
    delta = (bracket_after.ExtractTranslation() - wrist_world.ExtractTranslation()).GetLength()
    rep.append(
        f"[OK] /World/rg2 positioned. wrist={tuple(wrist_world.ExtractTranslation())} "
        f"bracket_after={tuple(bracket_after.ExtractTranslation())} "
        f"|delta|={delta:.6f}"
    )

    # 5) (Re-)create the wrist mount fixed joint
    if stage.GetPrimAtPath(MOUNT_JOINT):
        stage.RemovePrim(MOUNT_JOINT)
    fj = UsdPhysics.FixedJoint.Define(stage, MOUNT_JOINT)
    fj.CreateBody0Rel().SetTargets([Sdf.Path(WRIST_LINK)])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(BRACKET_PATH)])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    rep.append(f"[OK] (re)created {MOUNT_JOINT}: {WRIST_LINK} -> {BRACKET_PATH}")

    rep.append("\nDone. Ctrl+S to save scene.usd, then Play.")
    OUT.write_text("\n".join(rep))
    print("\n".join(rep))


main()
