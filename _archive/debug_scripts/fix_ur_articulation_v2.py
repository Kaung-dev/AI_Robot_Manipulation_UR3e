"""
Try a different articulation-root location: /World/ur3e/base_link (the
root rigid body of the UR articulation). Some Isaac Sim/PhysX paths only
recognize the API when applied to a body prim, not an Xform.

Steps:
  1) Apply ArticulationRootAPI to /World/ur3e/base_link.
  2) Remove ArticulationRootAPI from /World/ur3e and /World/ur3e/root_joint
     (avoid having multiple roots in the same articulation tree).
  3) Repoint PubJS / ArtCtrl inputs:targetPrim -> /World/ur3e/base_link
"""

import omni.usd
from pxr import Sdf, UsdPhysics

UR_BASE = "/World/ur3e/base_link"
PRIMS_TO_CLEAR = ["/World/ur3e", "/World/ur3e/root_joint"]
PUBJS = "/World/RosBridgeGraph/PubJS"
ARTCTRL = "/World/RosBridgeGraph/ArtCtrl"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    # 1) ArticulationRootAPI on base_link
    bl = stage.GetPrimAtPath(UR_BASE)
    if not bl or not bl.IsValid():
        return print(f"[ERROR] {UR_BASE} not found")
    if not bl.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(bl)
        print(f"[OK] applied ArticulationRootAPI to {UR_BASE}")
    else:
        print(f"[OK] {UR_BASE} already has ArticulationRootAPI")

    # 2) Remove from other UR prims to avoid duplicate roots
    for path in PRIMS_TO_CLEAR:
        p = stage.GetPrimAtPath(path)
        if p and p.IsValid() and p.HasAPI(UsdPhysics.ArticulationRootAPI):
            try:
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[OK] removed ArticulationRootAPI from {path}")
            except Exception as e:
                print(f"[WARN] could not remove from {path}: {e}")
        else:
            print(f"[OK] no ArticulationRootAPI on {path}")

    # 3) Repoint OmniGraph nodes
    for path in (PUBJS, ARTCTRL):
        node = stage.GetPrimAtPath(path)
        if not node or not node.IsValid():
            print(f"[SKIP] {path} not found")
            continue
        rel = node.GetRelationship("inputs:targetPrim")
        if not rel or not rel.IsValid():
            rel = node.CreateRelationship("inputs:targetPrim")
        rel.SetTargets([Sdf.Path(UR_BASE)])
        print(f"[OK] {path} inputs:targetPrim -> {UR_BASE}")

    artctrl = stage.GetPrimAtPath(ARTCTRL)
    if artctrl and artctrl.IsValid():
        rp = artctrl.GetAttribute("inputs:robotPath")
        if rp and rp.IsValid():
            rp.Set(UR_BASE)
            print(f"[OK] {ARTCTRL} inputs:robotPath -> {UR_BASE}")

    print("\nDone. Ctrl+S, then Play.")


main()
