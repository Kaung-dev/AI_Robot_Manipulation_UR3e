"""Standalone viewer for the exported robotis_lab assets.

Spawns the table at the origin, then places each object on a slot from the
FFW-SG2 task layout. Open the Isaac Sim viewport and fly around to inspect.

Asset location resolution (in order):
    1. CLI arg:                /isaac-sim/python.sh view.py /path/to/exported_assets
    2. ASSETS_DIR env var:     ASSETS_DIR=/path /isaac-sim/python.sh view.py
    3. Sibling of this script: <view.py dir>/exported_assets
    4. One level up:           <view.py dir>/../exported_assets

This means the script works whether it sits next to the assets or whether the
launcher (view.sh) drops them next to it inside the container.
"""
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

import os
import sys
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics


SCRIPT_DIR = Path(__file__).resolve().parent

candidates = []
if len(sys.argv) >= 2:
    candidates.append(Path(sys.argv[1]))
if os.environ.get("ASSETS_DIR"):
    candidates.append(Path(os.environ["ASSETS_DIR"]))
candidates.append(SCRIPT_DIR / "exported_assets")
candidates.append(SCRIPT_DIR.parent / "exported_assets")

assets_root = next((p.resolve() for p in candidates if (p / "object").is_dir()), None)
if assets_root is None:
    print("Error: could not locate exported_assets. Tried:", flush=True)
    for c in candidates:
        print(f"  - {c}", flush=True)
    sim_app.close()
    sys.exit(1)

object_dir = assets_root / "object"
print(f"Using assets at: {assets_root}", flush=True)

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Root xform
root = UsdGeom.Xform.Define(stage, "/World")

# Big buried cube as a ground reference
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.CreateSizeAttr(50.0)
UsdGeom.XformCommonAPI(ground).SetTranslate(Gf.Vec3d(0.0, 0.0, -25.0))

# Dome light for nice ambient
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(2000.0)
dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

# Distant sun light for direction
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(45.0, 0.0, 30.0))


def make_kinematic(root_prim: Usd.Prim):
    """Mark every rigid body under root_prim as kinematic (frozen, no gravity).

    Without this, hitting Play makes the referenced rigid bodies fall under
    gravity. For a viewer we want them locked in place.
    """
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr().Set(True)
            rb.CreateRigidBodyEnabledAttr().Set(True)


def spawn(prim_path: str, usd_path: Path, translate=(0.0, 0.0, 0.0), kinematic: bool = False):
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd_path))
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*translate))
    if kinematic:
        make_kinematic(prim)
    return prim


# Names whose USDs should be locked in place. Everything else stays dynamic
# so you can pick it up.
STATIC_NAMES = {"Table", "Basket"}


# Slot positions in the table's local frame (from FFW-SG2 task)
LEFT_SLOTS = [
    (0.555,  0.260, 0.95),
    (0.555,  0.087, 0.95),
    (0.555,  0.260, 1.235),
    (0.555,  0.087, 1.235),
]
RIGHT_SLOTS = [
    (0.555, -0.087, 0.95),
    (0.555, -0.260, 0.95),
    (0.555, -0.087, 1.235),
    (0.555, -0.260, 1.235),
]

scene = [
    # (prim name, USD filename, world position)
    ("Table",        "robotis_net_table.usd", (0.0, 0.0, 0.0)),
    ("Basket",       "plastic_basket2.usd",   (0.41, 0.0, 0.72)),
    ("Brush",        "brush_ring.usd",        LEFT_SLOTS[0]),
    ("Scissors",     "scissors_ring.usd",     LEFT_SLOTS[1]),
    ("SiliconeTube", "silicone_tube_ring.usd", LEFT_SLOTS[2]),
    ("ToothBrush",   "tooth_brush.usd",       LEFT_SLOTS[3]),
    ("Pliers",       "pliers_ring.usd",       RIGHT_SLOTS[0]),
    ("ScrewDriver",  "screw_driver_ring.usd", RIGHT_SLOTS[1]),
]

for name, fname, pos in scene:
    usd_path = object_dir / fname
    if not usd_path.exists():
        print(f"  ! missing: {usd_path}", flush=True)
        continue
    is_static = name in STATIC_NAMES
    spawn(f"/World/{name}", usd_path, translate=pos, kinematic=is_static)
    kind = "static " if is_static else "dynamic"
    print(f"  spawned {kind} {name:14s} @ {pos}", flush=True)

print(f"\nLoaded {len(scene)} assets. Use the viewport to fly around.", flush=True)
print("Close the window or Ctrl+C to exit.", flush=True)

app = omni.kit.app.get_app()
while sim_app.is_running():
    app.update()
sim_app.close()
