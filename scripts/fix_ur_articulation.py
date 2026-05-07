"""
Restore the UR3e articulation root and re-target the OmniGraph nodes.

Symptom: log spams 'Prim /World/ur3e/root_joint is not an articulation'.
Cause: previous cleanup removed ArticulationRootAPI from /World/ur3e/root_joint
(URDF importer puts it there, but it's actually wrong - it should be on
the Xform). The OmniGraph PubJS / ArtCtrl still point at the broken path.

Fix:
  1) Apply ArticulationRootAPI to /World/ur3e (the Xform).
  2) Repoint /World/RosBridgeGraph/PubJS  inputs:targetPrim -> /World/ur3e
  3) Repoint /World/RosBridgeGraph/ArtCtrl inputs:targetPrim -> /World/ur3e
  4) Also clear ArtCtrl inputs:robotPath (string) to /World/ur3e if present.

Run flow:
  - Stop Play.
  - Run.
  - Ctrl+S, then Play.
"""

import omni.usd
from pxr import Sdf, UsdPhysics

UR_XFORM = "/World/ur3e"
PUBJS = "/World/RosBridgeGraph/PubJS"
ARTCTRL = "/World/RosBridgeGraph/ArtCtrl"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    ur = stage.GetPrimAtPath(UR_XFORM)
    if not ur or not ur.IsValid():
        return print(f"[ERROR] {UR_XFORM} not found")

    # 1) ArticulationRootAPI on /World/ur3e
    if not ur.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(ur)
        print(f"[OK] applied ArticulationRootAPI to {UR_XFORM}")
    else:
        print(f"[OK] {UR_XFORM} already has ArticulationRootAPI")

    # 2/3) Update OmniGraph node targets
    for path in (PUBJS, ARTCTRL):
        node = stage.GetPrimAtPath(path)
        if not node or not node.IsValid():
            print(f"[SKIP] {path} not found")
            continue
        rel = node.GetRelationship("inputs:targetPrim")
        if not rel or not rel.IsValid():
            rel = node.CreateRelationship("inputs:targetPrim")
        rel.SetTargets([Sdf.Path(UR_XFORM)])
        print(f"[OK] {path} inputs:targetPrim -> {UR_XFORM}")

    # 4) ArtCtrl might also have a string `inputs:robotPath`
    artctrl = stage.GetPrimAtPath(ARTCTRL)
    if artctrl and artctrl.IsValid():
        rp = artctrl.GetAttribute("inputs:robotPath")
        if rp and rp.IsValid():
            old = rp.Get()
            rp.Set(UR_XFORM)
            print(f"[OK] {ARTCTRL} inputs:robotPath: {old} -> {UR_XFORM}")

    print("\nDone. Ctrl+S, then Play.")


main()
