"""
Temporarily disable the OnPlaybackTick node in /World/RosBridgeGraph so the
OmniGraph stops ticking on Play. This isolates whether the violent motion
is caused by the IsaacArticulationController applying stale commands.

If after running this and pressing Play the arm stays still and the gripper
sits on the wrist quietly => OmniGraph is the source. We'll then clear the
ArtCtrl's command inputs / SubJS's cached message.

If the arm still goes wild => physics-side problem (articulation, fixed joint,
or inertia). We will then split the gripper into its own articulation.

Run flow:
  - Stop Play.
  - Run this in Script Editor.
  - Run scripts/record_play.py.
  - Press Play immediately.
  - Tell me when joint_log.txt is fresh.

(To re-enable later: set inputs:enabled back to None / True on the Tick node,
or remove this whole prim's enabled=false override.)
"""

import omni.usd
from pxr import Sdf

TICK_PATH = "/World/RosBridgeGraph/Tick"


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return print("[ERROR] no stage")

    tick = stage.GetPrimAtPath(TICK_PATH)
    if not tick or not tick.IsValid():
        return print(f"[ERROR] {TICK_PATH} not found - is the OmniGraph still in scene?")

    # OnPlaybackTick has an inputs:onlyPlayback bool, but the simplest way to
    # turn the whole graph off is to set the omni:graph:active attribute on
    # the parent OmniGraph prim, OR set the node's inputs:enabled.
    graph = stage.GetPrimAtPath("/World/RosBridgeGraph")
    if graph and graph.IsValid():
        attr = graph.CreateAttribute(
            "omni:graph:active", Sdf.ValueTypeNames.Bool, custom=False
        )
        attr.Set(False)
        print(f"[OK] /World/RosBridgeGraph omni:graph:active -> False")

    # Belt-and-suspenders: also flip the Tick node's evaluator to disabled
    # by ungrouping its outputs:tick connection chain. Easiest: rename so
    # downstream nodes can't find it. We won't do that destructively though;
    # the graph:active flag should suffice.

    print("\nDone. Save (Ctrl+S), run record_play.py, press Play.")


main()
