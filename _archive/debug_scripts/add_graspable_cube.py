"""
Add a properly graspable cube to scene/scene.usd.

A "graspable" cube needs all of:
  - UsdGeom.Cube                       (the visual mesh)
  - UsdPhysics.RigidBodyAPI            (so physics moves it)
  - UsdPhysics.CollisionAPI            (so other objects can touch it)
  - UsdPhysics.MassAPI (or auto)       (so it has weight)
  - A high-friction PhysicsMaterial    (so the gripper doesn't slip)

Without RigidBody + Collision the gripper just phases through it.

Run with Isaac Sim's Python (any version that has pxr):
    ~/isaacsim_env/bin/python scripts/add_graspable_cube.py

Or, equivalently:
    source ~/isaacsim_env/bin/activate
    python scripts/add_graspable_cube.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema


REPO = Path(__file__).resolve().parents[1]
DEFAULT_USD = REPO / "scene" / "scene.usd"


def add_graspable_cube(
    stage_path: Path,
    prim_path: str = "/World/GraspCube",
    size_m: float = 0.04,            # 4 cm — fits between RG2 fingers
    mass_kg: float = 0.05,           # 50 g — light enough for RG2
    pos_xyz: tuple[float, float, float] = (0.4, 0.0, 0.05),  # in front of UR3e
    static_friction: float = 1.2,
    dynamic_friction: float = 1.1,
    restitution: float = 0.0,
    color_rgb: tuple[float, float, float] = (0.85, 0.15, 0.15),
) -> None:
    """Add (or replace) a fully physical cube in the given USD stage."""

    if not stage_path.exists():
        raise FileNotFoundError(f"USD stage not found: {stage_path}")

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    # Wipe an existing prim at the same path so re-runs are idempotent.
    if stage.GetPrimAtPath(prim_path).IsValid():
        print(f"[info] removing existing prim at {prim_path}")
        stage.RemovePrim(prim_path)

    # 1. Geometry — a cube primitive (NOT a Mesh, so we don't need to
    #    deal with mesh-collider approximation choices).
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(size_m)
    cube.AddTranslateOp().Set(Gf.Vec3f(*pos_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])

    prim = cube.GetPrim()

    # 2. Rigid body — physics will move this prim.
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)        # crucial: NOT kinematic
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)   # enables solver tuning

    # 3. Collision — other prims can hit it.
    col = UsdPhysics.CollisionAPI.Apply(prim)
    col.CreateCollisionEnabledAttr(True)

    # 4. Mass — give it weight so the gripper has something to react against.
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(mass_kg)

    # 5. Friction material — RG2 fingers slip on the default low-friction
    #    material; bump it.
    mat_path = "/World/Looks/CubeFriction"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        mat = UsdPhysics.MaterialAPI.Apply(
            UsdGeom.Scope.Define(stage, mat_path).GetPrim()
        )
        mat.CreateStaticFrictionAttr(static_friction)
        mat.CreateDynamicFrictionAttr(dynamic_friction)
        mat.CreateRestitutionAttr(restitution)

    # Bind the material to the cube's collider.
    binding = UsdPhysics.MaterialBindingAPI.Apply(prim) \
        if hasattr(UsdPhysics, "MaterialBindingAPI") else None
    if binding is None:
        # Fallback: use the generic UsdShade material binding for "physics"
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)),
            materialPurpose="physics",
        )

    # 6. Tweak PhysX solver iteration count for stable grasping.
    physx = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx.CreateSolverPositionIterationCountAttr(16)
    physx.CreateSolverVelocityIterationCountAttr(1)
    physx.CreateMaxDepenetrationVelocityAttr(5.0)

    stage.GetRootLayer().Save()
    print(f"[ok] added graspable cube at {prim_path}")
    print(f"     position {pos_xyz}, size {size_m} m, mass {mass_kg} kg")
    print(f"     friction static={static_friction} dynamic={dynamic_friction}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usd", type=Path, default=DEFAULT_USD,
                   help=f"path to scene.usd (default: {DEFAULT_USD})")
    p.add_argument("--path", default="/World/GraspCube",
                   help="prim path inside the stage")
    p.add_argument("--size", type=float, default=0.04,
                   help="cube edge length in meters")
    p.add_argument("--mass", type=float, default=0.05,
                   help="cube mass in kilograms")
    p.add_argument("--x", type=float, default=0.4)
    p.add_argument("--y", type=float, default=0.0)
    p.add_argument("--z", type=float, default=0.05)
    args = p.parse_args()

    add_graspable_cube(
        stage_path=args.usd,
        prim_path=args.path,
        size_m=args.size,
        mass_kg=args.mass,
        pos_xyz=(args.x, args.y, args.z),
    )


if __name__ == "__main__":
    main()
