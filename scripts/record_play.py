"""
Arm + gripper joint angle recorder. Subscribes to PhysX step events;
records every joint's angle for 5 seconds, then writes a TSV log.

Usage:
  1) Stop Play if running.
  2) Run this script in the Script Editor.
  3) IMMEDIATELY press Play. The recorder is already armed.
  4) Wait 5 seconds.
  5) The file /home/user/Desktop/ur_pick/scripts/joint_log.txt will be written.
"""
from pathlib import Path

import omni.physx
import omni.usd
from pxr import UsdPhysics

OUT = Path("/home/user/Desktop/ur_pick/scripts/joint_log.txt")
DURATION_S = 5.0

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "rg2_gripper_joint",
]


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] no stage")
        return

    attrs = {}
    for name in JOINT_NAMES:
        for prim in stage.Traverse():
            if prim.GetName() == name and prim.IsA(UsdPhysics.RevoluteJoint):
                attrs[name] = prim.GetAttribute("state:angular:physics:position")
                break
    found = list(attrs.keys())
    missing = [n for n in JOINT_NAMES if n not in attrs]
    if missing:
        print(f"[WARN] missing joints (will record NA): {missing}")
    print(f"[OK] tracking {len(found)} joints. Press Play now.")

    samples = []
    total = [0.0]
    sub_handle = [None]

    def write_log():
        with OUT.open("w") as f:
            f.write("t\t" + "\t".join(JOINT_NAMES) + "\n")
            for s in samples:
                row = [f"{s['t']:.4f}"]
                for n in JOINT_NAMES:
                    v = s.get(n)
                    row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "NA")
                f.write("\t".join(row) + "\n")
        print(f"[DONE] {len(samples)} samples -> {OUT}")

    def on_step(dt):
        total[0] += dt
        s = {"t": total[0]}
        for n, attr in attrs.items():
            try:
                v = attr.Get() if attr and attr.IsValid() else None
                if v is not None:
                    s[n] = float(v)
            except Exception:
                pass
        samples.append(s)
        if total[0] >= DURATION_S:
            sub_handle[0] = None  # unsubscribe -> stops further callbacks
            write_log()

    sub_handle[0] = omni.physx.get_physx_interface().subscribe_physics_step_events(on_step)


main()
