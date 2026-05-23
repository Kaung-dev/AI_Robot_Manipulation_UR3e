"""
Isaac Sim scene inspector.

How to run:
  1. Open your scene (e.g. scene/scene.usd) in Isaac Sim.
  2. Window -> Script Editor.
  3. File -> Open Script... -> select this file. Or copy-paste the contents.
  4. Click Run.

What it does:
  - Walks the open USD stage.
  - Lists every articulation root, every joint (with parent/child link, axis, limits,
    drive APIs), every OmniGraph and the ROS2 / Articulation nodes inside it.
  - Prints a human-readable summary to the Script Editor output.
  - Writes the full structured dump to scripts/scene_inspection.json
    so the next script (graph generator) can consume it.
"""

import json
from pathlib import Path

import omni.usd
from pxr import Usd, UsdPhysics, Sdf
PROJECT_ROOT = Path(__file__).resolve().parent.parent


OUT_PATH = Path(str(PROJECT_ROOT / "scripts" / "scene_inspection.json"))

JOINT_SCHEMAS = {
    "PhysicsRevoluteJoint": UsdPhysics.RevoluteJoint,
    "PhysicsPrismaticJoint": UsdPhysics.PrismaticJoint,
    "PhysicsFixedJoint": UsdPhysics.FixedJoint,
    "PhysicsSphericalJoint": UsdPhysics.SphericalJoint,
    "PhysicsDistanceJoint": UsdPhysics.DistanceJoint,
    "PhysicsJoint": UsdPhysics.Joint,
}


def _targets(rel):
    if rel is None or not rel.IsValid():
        return []
    return [t.pathString for t in rel.GetTargets()]


def _collect_drive_apis(prim):
    drives = []
    for s in prim.GetAppliedSchemas():
        if "DriveAPI" in s:
            # e.g. "PhysicsDriveAPI:angular" -> axis suffix after ':'
            axis = s.split(":", 1)[1] if ":" in s else ""
            d = UsdPhysics.DriveAPI.Get(prim, axis) if axis else None
            entry = {"schema": s, "axis": axis}
            if d:
                for name, attr_get in (
                    ("type", d.GetTypeAttr),
                    ("targetPosition", d.GetTargetPositionAttr),
                    ("targetVelocity", d.GetTargetVelocityAttr),
                    ("damping", d.GetDampingAttr),
                    ("stiffness", d.GetStiffnessAttr),
                    ("maxForce", d.GetMaxForceAttr),
                ):
                    a = attr_get()
                    if a:
                        entry[name] = a.Get()
            drives.append(entry)
    return drives


def _articulation_dof_order(stage, art_root_path):
    """
    Best-effort: walk descendants of the articulation root and return the
    revolute/prismatic joints in stage-traversal order. The PhysX articulation's
    actual DOF order can differ once the sim runs (depends on parent/child
    topology), but for URDF imports the traversal order usually matches.
    """
    order = []
    root = stage.GetPrimAtPath(art_root_path)
    if not root:
        return order
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            order.append(prim.GetName())
    return order


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open. Open your scene.usd first, then re-run.")
        return

    out = {
        "root_layer": stage.GetRootLayer().identifier,
        "default_prim": (
            stage.GetDefaultPrim().GetPath().pathString
            if stage.GetDefaultPrim()
            else None
        ),
        "articulations": [],
        "joints": [],
        "omni_graphs": [],
    }

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        type_name = prim.GetTypeName()

        # 1) Articulation roots
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            out["articulations"].append(
                {
                    "path": path,
                    "name": prim.GetName(),
                    "type": type_name,
                    "dof_order_guess": _articulation_dof_order(stage, path),
                }
            )

        # 2) Joints
        is_joint = any(
            prim.IsA(schema) for schema in JOINT_SCHEMAS.values() if schema is not None
        )
        if is_joint:
            j = UsdPhysics.Joint(prim)
            entry = {
                "path": path,
                "name": prim.GetName(),
                "type": type_name,
                "body0": _targets(j.GetBody0Rel()),
                "body1": _targets(j.GetBody1Rel()),
                "drive_apis": _collect_drive_apis(prim),
            }
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(prim)
                entry["axis"] = rj.GetAxisAttr().Get()
                entry["lower_deg"] = rj.GetLowerLimitAttr().Get()
                entry["upper_deg"] = rj.GetUpperLimitAttr().Get()
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                pj = UsdPhysics.PrismaticJoint(prim)
                entry["axis"] = pj.GetAxisAttr().Get()
                entry["lower"] = pj.GetLowerLimitAttr().Get()
                entry["upper"] = pj.GetUpperLimitAttr().Get()
            out["joints"].append(entry)

        # 3) OmniGraph wrappers + their nodes (look for ROS2/Articulation nodes)
        if type_name == "OmniGraph":
            graph_entry = {"path": path, "nodes": []}
            for node_prim in Usd.PrimRange(prim):
                if node_prim.GetTypeName() == "OmniGraphNode":
                    nt_attr = node_prim.GetAttribute("node:type")
                    nv_attr = node_prim.GetAttribute("node:typeVersion")
                    graph_entry["nodes"].append(
                        {
                            "path": node_prim.GetPath().pathString,
                            "name": node_prim.GetName(),
                            "node_type": nt_attr.Get() if nt_attr else None,
                            "version": nv_attr.Get() if nv_attr else None,
                        }
                    )
            out["omni_graphs"].append(graph_entry)

    # ---- pretty-print summary ----
    bar = "=" * 78
    print(bar)
    print(f"Root layer : {out['root_layer']}")
    print(f"Default prim: {out['default_prim']}")

    print(f"\nArticulations ({len(out['articulations'])}):")
    for a in out["articulations"]:
        print(f"  {a['path']}   ({a['type']})")
        if a["dof_order_guess"]:
            print(f"    DOF order (traversal): {a['dof_order_guess']}")

    print(f"\nJoints ({len(out['joints'])}):")
    for j in out["joints"]:
        b0 = j["body0"][0] if j["body0"] else "?"
        b1 = j["body1"][0] if j["body1"] else "?"
        line = f"  [{j['type']:22s}] {j['name']:24s}  {b0}  ->  {b1}"
        print(line)
        if "axis" in j:
            lo = j.get("lower_deg", j.get("lower"))
            hi = j.get("upper_deg", j.get("upper"))
            print(f"      axis={j['axis']}  range=[{lo}, {hi}]")
        for d in j.get("drive_apis", []):
            print(
                f"      drive[{d.get('axis','')}] type={d.get('type')} "
                f"stiff={d.get('stiffness')} damp={d.get('damping')} "
                f"maxF={d.get('maxForce')}"
            )

    print(f"\nOmniGraphs ({len(out['omni_graphs'])}):")
    for g in out["omni_graphs"]:
        print(f"  {g['path']}  ({len(g['nodes'])} nodes)")
        for n in g["nodes"]:
            print(f"    - {n['node_type']}    @ {n['path']}")

    # ---- structured dump ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n→ Wrote structured dump to: {OUT_PATH}")
    print(bar)


main()
