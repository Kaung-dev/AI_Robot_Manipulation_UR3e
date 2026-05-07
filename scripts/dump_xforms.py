"""
Dump every xform op on every Xformable prim under /onrobot_rg_test,
so we can see what kind of transforms each link has and where the bracket
offset is hiding.

Output: /home/user/Desktop/ur_pick/scripts/xform_dump.txt
"""
from pathlib import Path
import omni.usd
from pxr import Usd, UsdGeom

OUT = Path("/home/user/Desktop/ur_pick/scripts/xform_dump.txt")
ROOT = "/onrobot_rg_test"


def fmt(v):
    if v is None:
        return "None"
    try:
        return repr(tuple(v))
    except TypeError:
        return repr(v)


stage = omni.usd.get_context().get_stage()
lines = []
if stage is None:
    lines.append("[ERROR] no stage")
else:
    root = stage.GetPrimAtPath(ROOT)
    if not root or not root.IsValid():
        lines.append(f"[ERROR] {ROOT} not found")
    else:
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Xformable):
                continue
            xf = UsdGeom.Xformable(prim)
            order_attr = prim.GetAttribute("xformOpOrder")
            order = order_attr.Get() if order_attr and order_attr.IsValid() else []
            line = f"{prim.GetPath()}"
            if order:
                line += f"  order={list(order)}"
            else:
                line += "  (no xformOpOrder)"
            lines.append(line)
            # dump each op's value
            for op_name in (order or []):
                op_attr = prim.GetAttribute(op_name)
                if not op_attr or not op_attr.IsValid():
                    lines.append(f"    [missing attr] {op_name}")
                    continue
                lines.append(f"    {op_name} = {fmt(op_attr.Get())}")
            # also list any xformOp:* attributes that exist (in case order is stale)
            extra = [a.GetName() for a in prim.GetAttributes()
                     if a.GetName().startswith("xformOp:") and a.GetName() not in (order or [])]
            for n in extra:
                a = prim.GetAttribute(n)
                lines.append(f"    [unused] {n} = {fmt(a.Get())}")

OUT.write_text("\n".join(lines))
print(f"Wrote {len(lines)} lines to {OUT}")
print("\n".join(lines[:50]))
