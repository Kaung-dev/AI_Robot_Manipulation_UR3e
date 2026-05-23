"""
Place /World/rg2 so the gripper bracket sits *exactly* at the wrist_3_link
rest pose. With this set, the fixed joint /World/wrist_to_rg2 is satisfied
at t=0 and PhysX doesn't have to pull the arm/gripper to converge - so the
robot's start pose stays put.

Math:
  bracket_world_NEW = wrist_world          (we want them coincident)
  bracket_world = M_rg2 * bracket_local_chain
  Therefore: M_rg2_NEW = wrist_world * inverse(bracket_local_chain)
  And bracket_local_chain (in world coords, when M_rg2 was identity)
  IS just the current bracket_world, since we haven't touched M_rg2 yet.

Run flow:
  - Open scene.usd in Isaac Sim.
  - Make sure Play is stopped.
  - Run this in Script Editor.
  - Ctrl+S.
  - Press Play.
"""

import omni.usd
from pxr import Gf, UsdGeom

WRIST_LINK = "/World/ur3e/wrist_3_link"
RG2_PATH = "/World/rg2"
BRACKET_PATH = "/World/rg2/world/rg2_gripper_bracket"


def _set_xform_matrix(prim, mat: Gf.Matrix4d):
    """Replace a prim's xform ops with a single xformOp:transform = mat."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    # Remove any leftover op attrs so they don't interfere
    for a in list(prim.GetAttributes()):
        n = a.GetName()
        if n.startswith("xformOp:") and n != "xformOp:transform":
            prim.RemoveProperty(n)
    op = xf.AddTransformOp()
    op.Set(mat)


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] no stage")
        return

    wrist = stage.GetPrimAtPath(WRIST_LINK)
    bracket = stage.GetPrimAtPath(BRACKET_PATH)
    rg2 = stage.GetPrimAtPath(RG2_PATH)
    for p, name in ((wrist, WRIST_LINK), (bracket, BRACKET_PATH), (rg2, RG2_PATH)):
        if not p or not p.IsValid():
            print(f"[ERROR] {name} not found")
            return

    xc = UsdGeom.XformCache()
    wrist_world = xc.GetLocalToWorldTransform(wrist)
    bracket_world = xc.GetLocalToWorldTransform(bracket)

    # Place /World/rg2 so bracket_world becomes wrist_world.
    new_rg2 = wrist_world * bracket_world.GetInverse()

    print("wrist_world translate:", tuple(wrist_world.ExtractTranslation()))
    print("bracket_world translate (before):", tuple(bracket_world.ExtractTranslation()))
    print("new /World/rg2 translate:", tuple(new_rg2.ExtractTranslation()))

    _set_xform_matrix(rg2, new_rg2)
    print("\nDone. Ctrl+S, then Play.")


main()
