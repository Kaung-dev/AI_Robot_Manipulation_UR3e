"""CLI scene inspector. Writes output to scene_inspection.json + .txt"""
import argparse, json, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="/home/user/Desktop/ur_pick/scene/scene.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

OUT_TXT = Path("/home/user/Desktop/ur_pick/scripts/scene_inspection.txt")
OUT_JSON = Path("/home/user/Desktop/ur_pick/scripts/scene_inspection.json")
OUT_TXT.write_text("starting AppLauncher...\n")

app = AppLauncher(args).app

OUT_TXT.write_text(OUT_TXT.read_text() + "AppLauncher started, importing pxr\n")

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(args.usd)
lines = [f"=== {args.usd} ==="]
if stage is None:
    lines.append("FAILED to open stage")
    OUT_TXT.write_text("\n".join(lines))
    app.close(); sys.exit(1)

dp = stage.GetDefaultPrim()
lines.append(f"defaultPrim: {dp.GetPath() if dp else None}")

arts, joints, bodies = [], [], []
for prim in stage.Traverse():
    p = prim.GetPath().pathString
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        arts.append(p)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        bodies.append(p)
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        entry = {
            "path": p, "name": prim.GetName(), "type": prim.GetTypeName(),
            "body0": [t.pathString for t in j.GetBody0Rel().GetTargets()],
            "body1": [t.pathString for t in j.GetBody1Rel().GetTargets()],
        }
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            entry["axis"] = str(rj.GetAxisAttr().Get())
            entry["lower"] = rj.GetLowerLimitAttr().Get()
            entry["upper"] = rj.GetUpperLimitAttr().Get()
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            pj = UsdPhysics.PrismaticJoint(prim)
            entry["axis"] = str(pj.GetAxisAttr().Get())
            entry["lower"] = pj.GetLowerLimitAttr().Get()
            entry["upper"] = pj.GetUpperLimitAttr().Get()
        joints.append(entry)

lines.append(f"\nArticulation roots ({len(arts)}):")
for a in arts: lines.append(f"  {a}")

lines.append(f"\nRigid bodies / links ({len(bodies)}):")
for b in bodies: lines.append(f"  {b}")

lines.append(f"\nJoints ({len(joints)}):")
for j in joints:
    b0 = j["body0"][0] if j["body0"] else "?"
    b1 = j["body1"][0] if j["body1"] else "?"
    extra = f" axis={j['axis']} range=[{j.get('lower')}, {j.get('upper')}]" if "axis" in j else ""
    lines.append(f"  [{j['type']:22s}] {j['name']:30s}  {b0} -> {b1}{extra}")

OUT_TXT.write_text("\n".join(lines))
OUT_JSON.write_text(json.dumps({"articulations": arts, "bodies": bodies, "joints": joints}, indent=2, default=str))
app.close()
