"""
Place /World/rg2 so the gripper bracket lands at /World/ur3e/wrist_3_link
at rest pose. Also ensures the gripper has no nested ArticulationRootAPI
and that the internal /World/rg2/root_joint doesn't fight us.

Key fix vs v1: pxr.Gf is row-major. The correct composition for setting
the parent transform such that a child reaches a target is:

    M_parent_new = M_child_local_to_parent.inverse() * M_target_world

where M_child_local_to_parent equals the child's CURRENT world transform
when the parent is currently identity.

Run flow:
  - Open scene.usd, Play stopped.
  - Run this in Script Editor.
  - Ctrl+S, then Play.

Output: scripts/position_report.txt
"""

from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Usd
PROJECT_ROOT = Path(__file__).resolve().parent.parent


WRIST_LINK = "/World/ur3e/wrist_3_link"
RG2_PATH = "/World/rg2"
BRACKET_PATH = "/World/rg2/world/rg2_gripper_bracket"
INTERNAL_ROOT_JOINT = "/World/rg2/root_joint"

OUT = Path(str(PROJECT_ROOT / "scripts" / "position_report.txt"))


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

    # Sanity checks
    wrist = stage.GetPrimAtPath(WRIST_LINK)
    bracket = stage.GetPrimAtPath(BRACKET_PATH)
    rg2 = stage.GetPrimAtPath(RG2_PATH)
    for p, name in ((wrist, WRIST_LINK), (bracket, BRACKET_PATH), (rg2, RG2_PATH)):
        if not p or not p.IsValid():
            rep.append(f"[ERROR] {name} not found")
            OUT.write_text("\n".join(rep))
            return print("\n".join(rep))

    # 1) Reset /World/rg2 to identity FIRST so XformCache reads the gripper's
    #    own baked-in offsets, not whatever we may have written before.
    _set_xform_matrix(rg2, Gf.Matrix4d(1.0))

    xc = UsdGeom.XformCache()
    xc.Clear()  # invalidate cache after our reset
    wrist_world = xc.GetLocalToWorldTransform(wrist)
    bracket_world = xc.GetLocalToWorldTransform(bracket)

    rep.append(f"wrist_world  translate = {tuple(wrist_world.ExtractTranslation())}")
    rep.append(f"bracket_world translate = {tuple(bracket_world.ExtractTranslation())}")

    # 2) Compute new /World/rg2 transform.
    # In row-major: bracket_world = bracket_local_chain * rg2_world.
    # When rg2_world = I, bracket_local_chain == bracket_world.
    # Want: bracket_world_new = wrist_world
    # =>   bracket_local_chain * rg2_world_new = wrist_world
    # =>   rg2_world_new = bracket_local_chain.inverse() * wrist_world
    new_rg2 = bracket_world.GetInverse() * wrist_world
    rep.append(f"new /World/rg2 translate = {tuple(new_rg2.ExtractTranslation())}")
    _set_xform_matrix(rg2, new_rg2)

    # 3) Verify by re-reading bracket world transform after the change.
    xc.Clear()
    bracket_world_after = xc.GetLocalToWorldTransform(bracket)
    delta = (
        bracket_world_after.ExtractTranslation() - wrist_world.ExtractTranslation()
    ).GetLength()
    rep.append(
        f"bracket_world translate after = {tuple(bracket_world_after.ExtractTranslation())}"
    )
    rep.append(f"|bracket-wrist| after = {delta:.6f}  (should be ~0)")

    # 4) Make sure the gripper's internal articulation root is not fighting
    #    the UR3e articulation. If /World/rg2/root_joint or any prim under
    #    /World/rg2 still carries ArticulationRootAPI, remove it.
    removed = []
    for prim in Usd.PrimRange(rg2):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            try:
                ok = prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            except Exception:
                ok = False
            removed.append((prim.GetPath().pathString, ok))
    if removed:
        for p, ok in removed:
            rep.append(f"removed ArticulationRootAPI from {p} -> ok={ok}")
    else:
        rep.append("no nested ArticulationRootAPI under /World/rg2")

    rep.append("\nDone. Ctrl+S to save, then Play.")
    OUT.write_text("\n".join(rep))
    print("\n".join(rep))


main()
