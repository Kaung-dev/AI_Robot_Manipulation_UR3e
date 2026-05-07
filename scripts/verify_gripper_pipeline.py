"""
Dump the state of the gripper command pipeline, end to end:
  - ArticulationRootAPI on /World/rg2 ?
  - master joint /World/rg2/joints/rg2_gripper_joint exists?
  - master joint drive params
  - /World/RosBridgeGraph/ArtCtrlGripper exists, target & connections
  - /World/RosBridgeGraph/SubJS exists, topic, connections
"""
from pathlib import Path
import omni.usd
from pxr import Sdf, Usd, UsdPhysics

OUT = Path("/home/user/Desktop/ur_pick/scripts/gripper_pipeline.txt")
RG2 = "/World/rg2"
MASTER = "/World/rg2/joints/rg2_gripper_joint"
SUB = "/World/RosBridgeGraph/SubJS"
ART = "/World/RosBridgeGraph/ArtCtrlGripper"
ART_UR = "/World/RosBridgeGraph/ArtCtrl"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    lines = []

    # 1) /World/rg2 articulation root
    rg2 = stage.GetPrimAtPath(RG2)
    if rg2 and rg2.IsValid():
        has_root = rg2.HasAPI(UsdPhysics.ArticulationRootAPI)
        lines.append(f"{RG2}: exists, ArticulationRootAPI={has_root}")
        lines.append(f"    applied schemas: {list(rg2.GetAppliedSchemas())}")
    else:
        lines.append(f"{RG2}: MISSING")

    # 2) master joint + drive
    master = stage.GetPrimAtPath(MASTER)
    if master and master.IsValid():
        rj = UsdPhysics.RevoluteJoint(master)
        lo = rj.GetLowerLimitAttr().Get() if rj.GetLowerLimitAttr() else None
        hi = rj.GetUpperLimitAttr().Get() if rj.GetUpperLimitAttr() else None
        lines.append(f"{MASTER}: limits=[{lo}, {hi}]")
        d = UsdPhysics.DriveAPI.Get(master, "angular")
        if d:
            lines.append(
                "    drive: stiff={}, damp={}, maxF={}, target={}".format(
                    d.GetStiffnessAttr().Get(),
                    d.GetDampingAttr().Get(),
                    d.GetMaxForceAttr().Get(),
                    d.GetTargetPositionAttr().Get(),
                )
            )
        else:
            lines.append("    no DriveAPI")
    else:
        lines.append(f"{MASTER}: MISSING")

    # 3) ArtCtrlGripper node
    for path in (ART, ART_UR):
        node = stage.GetPrimAtPath(path)
        lines.append(f"\n[{path}]")
        if not node or not node.IsValid():
            lines.append("    MISSING")
            continue
        nt = node.GetAttribute("node:type")
        lines.append(f"    node:type = {nt.Get() if nt else None}")
        rp = node.GetAttribute("inputs:robotPath")
        lines.append(f"    inputs:robotPath = {rp.Get() if rp and rp.IsValid() else None}")
        tgt = node.GetRelationship("inputs:targetPrim")
        if tgt and tgt.IsValid():
            lines.append(f"    inputs:targetPrim -> {[t.pathString for t in tgt.GetTargets()]}")
        for n in (
            "inputs:execIn",
            "inputs:positionCommand",
            "inputs:velocityCommand",
            "inputs:effortCommand",
            "inputs:jointNames",
        ):
            a = node.GetAttribute(n)
            if a and a.IsValid():
                conns = a.GetConnections()
                lines.append(f"    {n} <- {[c.pathString for c in conns]}")

    # 4) SubJS
    sub = stage.GetPrimAtPath(SUB)
    lines.append(f"\n[{SUB}]")
    if sub and sub.IsValid():
        nt = sub.GetAttribute("node:type")
        lines.append(f"    node:type = {nt.Get() if nt else None}")
        for n in ("inputs:topicName", "inputs:nodeNamespace"):
            a = sub.GetAttribute(n)
            if a and a.IsValid():
                lines.append(f"    {n} = {a.Get()}")
    else:
        lines.append("    MISSING")

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n--> {OUT}")


main()
