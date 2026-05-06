"""
Fill in the missing `referenceJoint` relationship on every PhysxMimicJointAPI.

The Isaac Sim 4.5 URDF importer applied the API but left the relationship
empty, which is why PhysX errors with:

    PhysxMimicJointAPI at /.../foo_joint must have exactly 1 "referenceJoint"
    relationship defined.

This script:
  1) Walks the stage for revolute joints with a PhysxMimicJointAPI:* applied.
  2) Finds the master joint by name (`rg2_gripper_joint`) anywhere in the stage.
  3) For each follower, sets the `physxMimicJoint:<axis>:referenceJoint` rel
     to point at the master.
  4) Also writes the `gearing` (=1.0) and `referenceJointAxis` (=rotX) if they
     aren't set, since URDF mimic on these followers is multiplier=1.

Run in Isaac Sim Script Editor with the imported gripper open. Save (Ctrl+S)
afterwards.
"""

import omni.usd
from pxr import Sdf, UsdPhysics

MASTER_NAME = "rg2_gripper_joint"


def _instances(prim):
    """Return the multi-apply instance names for PhysxMimicJointAPI on a prim."""
    out = []
    for s in prim.GetAppliedSchemas():
        if s.startswith("PhysxMimicJointAPI:"):
            out.append(s.split(":", 1)[1])
        elif s == "PhysxMimicJointAPI":
            out.append("")
    return out


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open.")
        return

    # locate the master joint
    master_path = None
    for prim in stage.Traverse():
        if prim.GetName() == MASTER_NAME and prim.IsA(UsdPhysics.RevoluteJoint):
            master_path = prim.GetPath().pathString
            break
    if master_path is None:
        print(f"[ERROR] master joint '{MASTER_NAME}' not found in stage")
        return
    print(f"master = {master_path}")

    fixed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        instances = _instances(prim)
        if not instances:
            continue
        for inst in instances:
            base = f"physxMimicJoint:{inst}:" if inst else "physxMimicJoint:"

            # 1) referenceJoint rel
            ref_rel = prim.GetRelationship(base + "referenceJoint")
            if not ref_rel or not ref_rel.IsValid():
                ref_rel = prim.CreateRelationship(base + "referenceJoint")
            ref_rel.SetTargets([Sdf.Path(master_path)])

            # 2) referenceJointAxis (rotX since these are revolute about local X)
            axis_attr = prim.GetAttribute(base + "referenceJointAxis")
            if not axis_attr or not axis_attr.IsValid():
                axis_attr = prim.CreateAttribute(
                    base + "referenceJointAxis", Sdf.ValueTypeNames.Token, custom=False
                )
            if not axis_attr.Get():
                axis_attr.Set("rotX")

            # 3) gearing = 1.0 (URDF mimic multiplier)
            gear_attr = prim.GetAttribute(base + "gearing")
            if not gear_attr or not gear_attr.IsValid():
                gear_attr = prim.CreateAttribute(
                    base + "gearing", Sdf.ValueTypeNames.Float, custom=False
                )
            if gear_attr.Get() in (None, 0.0):
                gear_attr.Set(1.0)

            # 4) offset = 0
            off_attr = prim.GetAttribute(base + "offset")
            if not off_attr or not off_attr.IsValid():
                off_attr = prim.CreateAttribute(
                    base + "offset", Sdf.ValueTypeNames.Float, custom=False
                )
            if off_attr.Get() is None:
                off_attr.Set(0.0)

            print(
                f"[OK] {prim.GetPath()} (instance={inst!r})\n"
                f"     referenceJoint -> {master_path}\n"
                f"     gearing={gear_attr.Get()}, axis={axis_attr.Get()}, offset={off_attr.Get()}"
            )
            fixed += 1

    print(f"\nFixed {fixed} mimic API instance(s). Ctrl+S to save, then Play.")


main()
