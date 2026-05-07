"""
Configure /World/RosBridgeGraph for the UR3e arm bridge.

Run in Isaac Sim's Script Editor (scene open), then File -> Save the stage.

Fixes:
  1) Connects ros2_context.outputs:context -> SubJS.inputs:context (missing).
  2) Sets PubJS.inputs:jointNames to the 6 arm joints, so we don't publish
     the 6 RG2 mimic joints that have no counterpart in MoveIt's URDF.

Idempotent: safe to re-run.
"""

import omni.usd
from pxr import Sdf, Usd, Vt

GRAPH = "/World/RosBridgeGraph"
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _ensure_attr(prim, name, sdf_type, default=None):
    a = prim.GetAttribute(name)
    if not a or not a.IsValid():
        a = prim.CreateAttribute(name, sdf_type, custom=False)
        if default is not None:
            a.Set(default)
    return a


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open.")
        return

    pub_prim = stage.GetPrimAtPath(f"{GRAPH}/PubJS")
    sub_prim = stage.GetPrimAtPath(f"{GRAPH}/SubJS")
    ctx_prim = stage.GetPrimAtPath(f"{GRAPH}/ros2_context")

    if not all(p and p.IsValid() for p in (pub_prim, sub_prim, ctx_prim)):
        print(
            "[ERROR] Could not find PubJS / SubJS / ros2_context under "
            f"{GRAPH}. Make sure the graph exists."
        )
        return

    # ---- 1) connect ros2_context.outputs:context -> SubJS.inputs:context ----
    sub_ctx = sub_prim.GetAttribute("inputs:context")
    if not sub_ctx or not sub_ctx.IsValid():
        # The attribute should already exist on a ROS2SubscribeJointState
        # node, but create a dangling target if missing.
        sub_ctx = sub_prim.CreateAttribute(
            "inputs:context", Sdf.ValueTypeNames.UInt64
        )

    src = Sdf.Path(f"{GRAPH}/ros2_context.outputs:context")
    existing_conns = list(sub_ctx.GetConnections())
    if src in existing_conns:
        print(f"[OK] SubJS.inputs:context already connected to {src}")
    else:
        sub_ctx.SetConnections([src])
        print(f"[FIX] Connected SubJS.inputs:context -> {src}")

    # ---- 2) set PubJS.inputs:jointNames to the 6 arm joints ----
    jn_attr = pub_prim.GetAttribute("inputs:jointNames")
    if not jn_attr or not jn_attr.IsValid():
        jn_attr = pub_prim.CreateAttribute(
            "inputs:jointNames", Sdf.ValueTypeNames.TokenArray
        )

    desired = Vt.TokenArray(ARM_JOINTS)
    current = jn_attr.Get()
    current_list = list(current) if current is not None else []
    if list(current_list) == ARM_JOINTS:
        print(f"[OK] PubJS.inputs:jointNames already = {ARM_JOINTS}")
    else:
        jn_attr.Set(desired)
        print(f"[FIX] PubJS.inputs:jointNames set to {ARM_JOINTS}")

    print("\nDone. Now: File -> Save (Ctrl+S) to persist into scene.usd, then press Play.")


main()
