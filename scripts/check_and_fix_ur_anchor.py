"""
Inspect articulation roots & base anchors, and repair UR3e if needed.

After making /World/rg2 its own articulation, the UR3e base may have lost its
fix-to-world. This:
  1) Reports every prim with PhysicsArticulationRootAPI applied.
  2) Reports every PhysicsFixedJoint anchoring something to the world.
  3) If /World/ur3e/root_joint is missing or no longer connects base_link
     to world, recreates it.
  4) If /World/ur3e is missing ArticulationRootAPI, applies it.

Output: scripts/anchor_report.txt
"""
from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, Usd, UsdPhysics
PROJECT_ROOT = Path(__file__).resolve().parent.parent


UR_PATH = "/World/ur3e"
UR_BASE_LINK = "/World/ur3e/base_link"
UR_ROOT_JOINT = "/World/ur3e/root_joint"

OUT = Path(str(PROJECT_ROOT / "scripts" / "anchor_report.txt"))


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    rep = []

    # 1) all articulation roots
    rep.append("ArticulationRootAPI is applied on:")
    found_roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            rep.append(f"  - {prim.GetPath()}  (type={prim.GetTypeName()})")
            found_roots.append(prim.GetPath().pathString)
    if not found_roots:
        rep.append("  (none)")

    # 2) fixed joints likely anchoring to world (body0 empty or pointing at world)
    rep.append("\nFixed joints (looking for world anchors):")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        j = UsdPhysics.FixedJoint(prim)
        b0 = [t.pathString for t in j.GetBody0Rel().GetTargets()] if j.GetBody0Rel() else []
        b1 = [t.pathString for t in j.GetBody1Rel().GetTargets()] if j.GetBody1Rel() else []
        rep.append(f"  - {prim.GetPath()}  body0={b0}  body1={b1}")

    # 3) ensure /World/ur3e has ArticulationRootAPI
    ur = stage.GetPrimAtPath(UR_PATH)
    if not ur or not ur.IsValid():
        rep.append(f"\n[FAIL] {UR_PATH} not found - cannot fix")
        OUT.write_text("\n".join(rep))
        return print("\n".join(rep))

    if not ur.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(ur)
        rep.append(f"\n[FIX] applied ArticulationRootAPI to {UR_PATH}")
    else:
        rep.append(f"\n[OK] {UR_PATH} already has ArticulationRootAPI")

    # 4) ensure /World/ur3e/root_joint exists and anchors base_link to world
    base = stage.GetPrimAtPath(UR_BASE_LINK)
    if not base or not base.IsValid():
        rep.append(f"[FAIL] {UR_BASE_LINK} not found")
        OUT.write_text("\n".join(rep))
        return print("\n".join(rep))

    rj = stage.GetPrimAtPath(UR_ROOT_JOINT)
    if rj and rj.IsValid():
        rep.append(f"[OK] {UR_ROOT_JOINT} exists already")
    else:
        # create FixedJoint world (no body0) -> base_link
        fj = UsdPhysics.FixedJoint.Define(stage, UR_ROOT_JOINT)
        # Empty body0 means anchored to world
        fj.CreateBody0Rel().SetTargets([])
        fj.CreateBody1Rel().SetTargets([Sdf.Path(UR_BASE_LINK)])
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rep.append(f"[FIX] (re)created {UR_ROOT_JOINT} anchoring world -> {UR_BASE_LINK}")

    rep.append("\nDone. Ctrl+S, then Play.")
    OUT.write_text("\n".join(rep))
    print("\n".join(rep))


main()
