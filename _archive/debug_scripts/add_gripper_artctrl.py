"""
Add a second IsaacArticulationController node in /World/RosBridgeGraph that
targets the gripper articulation /World/rg2. SubJS broadcasts to BOTH the UR
ArtCtrl and this new one; each picks the joints it knows.

Run flow:
  - Stop Play.
  - Run.
  - Ctrl+S.
  - Restart bridge with gripper sending re-enabled (we'll edit it next).
"""

import omni.usd
from pxr import Sdf

GRAPH = "/World/RosBridgeGraph"
SUB = f"{GRAPH}/SubJS"
NEW = f"{GRAPH}/ArtCtrlGripper"
GRIPPER_TARGET = "/World/rg2"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    # If already exists, wipe and recreate so the script is idempotent.
    if stage.GetPrimAtPath(NEW):
        stage.RemovePrim(NEW)

    node = stage.DefinePrim(NEW, "OmniGraphNode")
    # Mark the node type:
    nt = node.CreateAttribute("node:type", Sdf.ValueTypeNames.Token, custom=False)
    nt.Set("isaacsim.core.nodes.IsaacArticulationController")
    nv = node.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int, custom=False)
    nv.Set(1)

    # robotPath (string) and targetPrim (rel)
    rp = node.CreateAttribute("inputs:robotPath", Sdf.ValueTypeNames.String, custom=False)
    rp.Set(GRIPPER_TARGET)
    target_rel = node.CreateRelationship("inputs:targetPrim")
    target_rel.SetTargets([Sdf.Path(GRIPPER_TARGET)])

    # Wire input attributes from SubJS outputs (same as the UR ArtCtrl)
    pairs = [
        ("inputs:execIn", "outputs:execOut"),
        ("inputs:positionCommand", "outputs:positionCommand"),
        ("inputs:velocityCommand", "outputs:velocityCommand"),
        ("inputs:effortCommand", "outputs:effortCommand"),
        ("inputs:jointNames", "outputs:jointNames"),
    ]
    # we need to know the type of each input - just create them and set connections
    type_map = {
        "inputs:execIn": Sdf.ValueTypeNames.UInt,
        "inputs:positionCommand": Sdf.ValueTypeNames.DoubleArray,
        "inputs:velocityCommand": Sdf.ValueTypeNames.DoubleArray,
        "inputs:effortCommand": Sdf.ValueTypeNames.DoubleArray,
        "inputs:jointNames": Sdf.ValueTypeNames.TokenArray,
        "inputs:jointIndices": Sdf.ValueTypeNames.IntArray,
    }
    for in_name, out_name in pairs:
        attr = node.CreateAttribute(in_name, type_map[in_name], custom=False)
        src = Sdf.Path(f"{SUB}.{out_name}")
        attr.SetConnections([src])

    print(f"[OK] created {NEW}")
    print(f"     targetPrim -> {GRIPPER_TARGET}")
    print(f"     robotPath  =  {GRIPPER_TARGET}")
    print(f"     wired from SubJS outputs")
    print("\nDone. Ctrl+S, then Play. Bridge update next.")


main()
