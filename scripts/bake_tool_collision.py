"""One-shot: open each tool USD, set every Mesh's MeshCollisionAPI
approximation to convexDecomposition (preserves the ring hole), save in place.

After running this, Isaac Lab loads the tools with the proper collision
baked into the file — the runtime fix that crashed PhysX is no longer
needed.

Backs up each .usd to .usd.bak first.

Run:
    C:/isaacsim/python.bat scripts/bake_tool_collision.py
"""
import sys
from pathlib import Path
import shutil

# pxr lives in Isaac Sim's bundled python; add its site-packages to sys.path
# so we can import it WITHOUT booting a SimulationApp (which is taking 1+ hr
# on this VM). This script only needs pxr.Usd and pxr.UsdPhysics — no sim.
_ISAAC_SITE = Path(r"C:\isaac\IsaacLab\_isaac_sim\kit\python\Lib\site-packages")
if _ISAAC_SITE.exists() and str(_ISAAC_SITE) not in sys.path:
    sys.path.insert(0, str(_ISAAC_SITE))

from pxr import Usd, UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "exported_assets" / "object"

TOOLS = [
    "tooth_brush_green.usd",
    "pliers_ring_orange.usd",
    "scissors_ring_red.usd",
    "silicone_tube_ring_blue.usd",
]

for fname in TOOLS:
    usd_path = ASSETS / fname
    if not usd_path.exists():
        print(f"[skip] {usd_path} missing")
        continue

    # Backup once
    backup = usd_path.with_suffix(usd_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(usd_path, backup)
        print(f"[backup] {fname} -> {backup.name}")

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"[error] could not open {usd_path}")
        continue

    n_meshes = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        # Make sure CollisionAPI is enabled, then set approximation.
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mca.CreateApproximationAttr().Set("convexDecomposition")
        n_meshes += 1

    stage.GetRootLayer().Save()
    print(f"[done] {fname}: convexDecomposition on {n_meshes} meshes -> saved in place")

print("\nAll tool USDs updated. Run teleop now — rings should hang on pegs.")
