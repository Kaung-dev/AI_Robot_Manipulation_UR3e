"""
OmniGraph node inspector — dumps every input attribute, relationship target,
and incoming connection for each node in /World/RosBridgeGraph.

Run in Isaac Sim's Script Editor with the scene open.

Outputs:
  - Console summary (per-node attrs, rels, wiring)
  - JSON dump at scripts/graph_inspection.json
"""

import json
from pathlib import Path

import omni.usd
from pxr import Usd, Sdf
PROJECT_ROOT = Path(__file__).resolve().parent.parent


GRAPH_PATH = "/World/RosBridgeGraph"
OUT_PATH = Path(str(PROJECT_ROOT / "scripts" / "graph_inspection.json"))


def _safe(v):
    """Best-effort coerce a USD attribute value to something JSON can serialize."""
    if v is None:
        return None
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        # numpy arrays, Vt arrays, Gf vectors, etc.
        try:
            return list(v)
        except TypeError:
            return repr(v)


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] No stage open.")
        return

    graph_prim = stage.GetPrimAtPath(GRAPH_PATH)
    if not graph_prim or not graph_prim.IsValid():
        print(f"[ERROR] No prim at {GRAPH_PATH}")
        return

    out = {"graph_path": GRAPH_PATH, "nodes": []}

    for prim in Usd.PrimRange(graph_prim):
        if prim.GetTypeName() != "OmniGraphNode":
            continue

        nt = prim.GetAttribute("node:type")
        nv = prim.GetAttribute("node:typeVersion")
        node_entry = {
            "path": prim.GetPath().pathString,
            "name": prim.GetName(),
            "type": nt.Get() if nt else None,
            "version": nv.Get() if nv else None,
            "inputs": {},
            "relationships": {},
            "connections": {},
        }

        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not name.startswith("inputs:"):
                continue
            node_entry["inputs"][name] = _safe(attr.Get())
            conns = attr.GetConnections()
            if conns:
                node_entry["connections"][name] = [c.pathString for c in conns]

        for rel in prim.GetRelationships():
            name = rel.GetName()
            if not name.startswith("inputs:"):
                continue
            targets = [t.pathString for t in rel.GetTargets()]
            if targets:
                node_entry["relationships"][name] = targets

        out["nodes"].append(node_entry)

    bar = "=" * 78
    print(bar)
    print(f"Graph: {GRAPH_PATH}   ({len(out['nodes'])} nodes)")

    for n in out["nodes"]:
        print(f"\n[{n['name']}]   type={n['type']}  v{n['version']}")
        if n["inputs"]:
            print("  inputs:")
            for k in sorted(n["inputs"]):
                short = k[len('inputs:'):]
                v = n["inputs"][k]
                print(f"    {short:24s} = {v!r}")
        if n["relationships"]:
            print("  rel targets:")
            for k in sorted(n["relationships"]):
                short = k[len('inputs:'):]
                print(f"    {short:24s} -> {n['relationships'][k]}")
        if n["connections"]:
            print("  incoming connections:")
            for k in sorted(n["connections"]):
                short = k[len('inputs:'):]
                print(f"    {short:24s} <- {n['connections'][k]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n→ Wrote: {OUT_PATH}")
    print(bar)


main()
