"""
Restore PhysxMimicJointAPI on the 4 inner follower gripper joints in
scene_isaaclab.usd, so the followers physically track the master joint
(rg2_gripper_joint). Master and mirror are NOT mimic'd — they're the two
direct-driven joints from the actuator config.

Why: Isaac Lab's gripper actuator only drives the 2 outer masters. Without
mimic, the 4 inner followers (truss_arm + finger_tip per side) are passive
— the gripper visually closes but the contact pads on the flex_fingers
have no pinch force, so it can't lift the cube. Restoring the mimic
constraint forces the followers to track the master's angle. The master's
drive force then transmits through the constraint into the pads, producing
real grip force.

Run from the project root:  python3 scripts/restore_gripper_mimic.py
"""

from pathlib import Path
import sys

try:
    from pxr import Sdf, Usd, UsdPhysics
except ImportError:
    print("[ERROR] pxr.Usd not found. Need usd-core or run from Isaac venv.")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
SCENE = REPO / "scene" / "scene_isaaclab.usd"

# the 4 followers that carry the contact pads
FOLLOWERS = [
    "/World/ur3e/joints/rg2_gripper_finger_1_truss_arm_joint",
    "/World/ur3e/joints/rg2_gripper_finger_1_finger_tip_joint",
    "/World/ur3e/joints/rg2_gripper_finger_2_truss_arm_joint",
    "/World/ur3e/joints/rg2_gripper_finger_2_finger_tip_joint",
]
# fallback paths in case the scene wasn't collapsed yet (still under /World/rg2)
FALLBACK_FOLLOWERS = [
    "/World/rg2/joints/rg2_gripper_finger_1_truss_arm_joint",
    "/World/rg2/joints/rg2_gripper_finger_1_finger_tip_joint",
    "/World/rg2/joints/rg2_gripper_finger_2_truss_arm_joint",
    "/World/rg2/joints/rg2_gripper_finger_2_finger_tip_joint",
]
MASTER_PATH_OPTIONS = [
    "/World/ur3e/joints/rg2_gripper_joint",
    "/World/rg2/joints/rg2_gripper_joint",
]


def find_master(stage):
    for p in MASTER_PATH_OPTIONS:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            return p
    # Fall back: walk the stage looking for a prim named rg2_gripper_joint
    for prim in stage.Traverse():
        if prim.GetName() == "rg2_gripper_joint":
            return prim.GetPath().pathString
    return None


def apply_mimic(stage, joint_path: str, master_path: str, gearing: float):
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        return False
    base = "physxMimicJoint:rotY:"

    # Apply the multi-apply schema if not already applied
    schema_token = "PhysxMimicJointAPI:rotY"
    if schema_token not in list(prim.GetAppliedSchemas()):
        try:
            prim.ApplyAPI("PhysxMimicJointAPI", "rotY")
        except Exception:
            # Fallback for environments where PhysxSchema isn't registered
            # (e.g. plain usd-core without Isaac Sim plugins). The runtime
            # only needs the schema name in the prim's apiSchemas metadata
            # and the underlying attributes — both of which we can set via Sdf.
            spec = stage.GetEditTarget().GetPrimSpecForScenePath(prim.GetPath())
            if spec is None:
                spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim.GetPath())
            existing = list(spec.GetInfo("apiSchemas").prependedItems) if spec.HasInfo("apiSchemas") else []
            if schema_token not in existing:
                items = Sdf.TokenListOp.Create(prependedItems=existing + [schema_token])
                spec.SetInfo("apiSchemas", items)
                print(f"  [info] added {schema_token} to apiSchemas via Sdf on {joint_path}")

    # gearing
    g = prim.GetAttribute(base + "gearing")
    if not g or not g.IsValid():
        g = prim.CreateAttribute(base + "gearing", Sdf.ValueTypeNames.Float, custom=False)
    g.Set(float(gearing))
    # offset
    o = prim.GetAttribute(base + "offset")
    if not o or not o.IsValid():
        o = prim.CreateAttribute(base + "offset", Sdf.ValueTypeNames.Float, custom=False)
    o.Set(0.0)
    # referenceJointAxis (matches the master's axis token)
    ax = prim.GetAttribute(base + "referenceJointAxis")
    if not ax or not ax.IsValid():
        ax = prim.CreateAttribute(base + "referenceJointAxis", Sdf.ValueTypeNames.Token, custom=False)
    ax.Set("rotY")
    # referenceJoint relationship
    r = prim.GetRelationship(base + "referenceJoint")
    if not r or not r.IsValid():
        r = prim.CreateRelationship(base + "referenceJoint")
    r.SetTargets([Sdf.Path(master_path)])
    return True


def main():
    if not SCENE.exists():
        print(f"[ERROR] {SCENE} not found")
        sys.exit(1)

    stage = Usd.Stage.Open(str(SCENE))
    if not stage:
        print(f"[ERROR] could not open {SCENE}")
        sys.exit(1)

    master_path = find_master(stage)
    if master_path is None:
        print("[ERROR] could not locate rg2_gripper_joint master in stage")
        sys.exit(1)
    print(f"master: {master_path}")

    used_paths = []
    for primary, fallback in zip(FOLLOWERS, FALLBACK_FOLLOWERS):
        if stage.GetPrimAtPath(primary).IsValid():
            used_paths.append(primary)
        elif stage.GetPrimAtPath(fallback).IsValid():
            used_paths.append(fallback)
        else:
            print(f"[WARN] {primary} (and fallback) not found — skipping")

    n_done = 0
    for joint in used_paths:
        ok = apply_mimic(stage, joint, master_path, gearing=-1.0)
        if ok:
            print(f"[OK] mimic restored on {joint}")
            n_done += 1
        else:
            print(f"[FAIL] {joint}")

    if n_done:
        stage.GetRootLayer().Save()
        print(f"\nSaved {SCENE}  ({n_done} joints updated)")
    else:
        print("\nNothing changed.")


if __name__ == "__main__":
    main()
