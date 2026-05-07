"""Convert your scene.usd into an Isaac-Lab-friendly variant.

Two transformations:
  1. Strip ArticulationRootAPI from /World/rg2 so the arm + gripper form a
     single articulation rooted at /World/ur3e/base_link.
  2. Remove every OmniGraph (Action Graph) sub-tree. The original scene has
     ROS/MoveIt graphs that call legacy PhysX `setDriveTarget()`, which is
     illegal under Isaac Lab's GPU pipeline (PxSceneFlag::eENABLE_DIRECT_GPU_API)
     and causes the controller to fight Isaac Lab's actions.

Reads : scene/scene.usd
Writes: scene/scene_isaaclab.usd  (original is untouched)
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--in_usd",  default=str(PROJECT_ROOT / "scene" / "scene.usd"))
parser.add_argument("--out_usd", default=str(PROJECT_ROOT / "scene" / "scene_isaaclab.usd"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

LOG = Path(str(PROJECT_ROOT / "scripts" / "fix_scene.log"))
LOG.write_text("starting\n")

app = AppLauncher(args).app

from pxr import Usd, UsdPhysics, Sdf  # noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parent.parent


stage = Usd.Stage.Open(args.in_usd)
log = [f"opened {args.in_usd}"]
roots_before = [p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
log.append(f"articulation roots before: {roots_before}")

# (1) Single articulation root.
target_path = "/World/rg2"
prim = stage.GetPrimAtPath(target_path)
if prim and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    log.append(f"removed ArticulationRootAPI from {target_path}")

# (2) Strip every OmniGraph sub-tree (ROS bridge, Action Graphs, etc.).
GRAPH_TYPES = {"OmniGraph"}
graph_paths = []
for prim in stage.Traverse():
    if prim.GetTypeName() in GRAPH_TYPES:
        graph_paths.append(prim.GetPath())

# Also catch top-level prims commonly used as Action Graph roots.
for candidate in ("/World/ActionGraph", "/World/RosBridge", "/World/Graphs", "/World/ROS"):
    p = stage.GetPrimAtPath(candidate)
    if p and p.IsValid():
        graph_paths.append(p.GetPath())

# Deduplicate and sort deepest-first so we don't invalidate parents while removing.
graph_paths = sorted(set(graph_paths), key=lambda p: -len(p.pathString.split("/")))
for p in graph_paths:
    if stage.RemovePrim(p):
        log.append(f"removed prim {p}")
    else:
        log.append(f"FAILED to remove prim {p}")

# Belt-and-braces: also remove any prim whose typeName starts with "OmniGraph".
extra = []
for prim in stage.Traverse():
    tn = prim.GetTypeName()
    if tn and str(tn).startswith("OmniGraph"):
        extra.append(prim.GetPath())
extra = sorted(set(extra), key=lambda p: -len(p.pathString.split("/")))
for p in extra:
    stage.RemovePrim(p)
    log.append(f"removed OmniGraph-typed prim {p}")

# Save.
stage.GetRootLayer().Export(args.out_usd)
log.append(f"wrote {args.out_usd}")

# Verify.
stage2 = Usd.Stage.Open(args.out_usd)
roots_after = [p.GetPath().pathString for p in stage2.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
remaining_graphs = [p.GetPath().pathString for p in stage2.Traverse() if str(p.GetTypeName()).startswith("OmniGraph")]
log.append(f"articulation roots after: {roots_after}")
log.append(f"remaining OmniGraph-typed prims: {remaining_graphs}")

LOG.write_text("\n".join(log) + "\n")
app.close()
