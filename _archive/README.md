# `_archive/` — historical / one-shot files

Files in this folder were used during the **initial scene build** and **gripper integration** phases of the project, but are not part of the current operational pipeline (CNN segmentation, BC training, PPO fine-tune).

They are kept here for reference and for anyone rebuilding from scratch via [`../REBUILD_GUIDE.md`](../REBUILD_GUIDE.md). When you see a step in `REBUILD_GUIDE.md` reference `_archive/debug_scripts/...`, copy that script back to `scripts/` temporarily, run it, then delete the copy.

## Layout

| Folder | What's inside | When you'd touch it |
|---|---|---|
| [`debug_scripts/`](debug_scripts/) | 35 one-shot Python scripts used to fix mimic joints, articulations, drives, and gripper attachment in the source `scene.usd`. Includes both v1 and v2 variants of `fix_ur_articulation` and `position_gripper`. | Rebuilding `scene.usd` from scratch (the .usd files in `scene/` are already committed, so most rebuilds skip these). |
| [`diagnostic_outputs/`](diagnostic_outputs/) | 13 `.txt` and `.json` reports produced by the scripts above (anchor positions, joint lists, gripper state, etc.). | Never — these were one-shot debugging output. |
| [`alt_robot_configs/`](alt_robot_configs/) | `rg2_omni/` (Omniverse default gripper config — superseded by the Inria URDF in `../rg2_inria/`) and `rg2_only/` (UR3e-only variant without gripper integration). | If you want to experiment with the original Omniverse RG2 or a gripper-less UR3e. |
| [`docker_tools/`](docker_tools/) | `asset_viewer/` — a small Docker-based asset viewer for `robotis_lab` containers. Not used in the Windows-native pipeline. | Only if running the project inside the Docker container variant. |
| [`scene_backups/`](scene_backups/) | `scene_isaaclab.usd.bak.before_mimic` (pre-mimic-joint snapshot) and `thumbs_cache/` (Omniverse-regenerated thumbnails). | If a `scene_isaaclab.usd` regeneration goes wrong and you need to roll back. |

## What stays in the main `scripts/` folder

The 19 actively-used scripts: see [`../scripts/`](../scripts/). Highlights:

- **Pipeline (CNN → BC → PPO)**: `collect_air2_segmentation_data.py`, `train_air2_segmentation.py`, `collect_air2_demos.py`, `train_bc.py`, `eval_bc.py`, `bc_to_ppo.py`
- **Operations**: `run_teleop_windows.py` (VR/keyboard teleop), `fix_scene_for_isaaclab.py`, `find_isaaclab.sh`, `inspect_air2_hooks.py`, `preview_air2_segmentation_cameras.py`, `run_air2_segmentation_inference.py`
- **Older auto-pick demos (still useful)**: `pick_cube_auto.py`, `pick_pegboard_auto.py`
- **ROS bridge (Path A)**: `moveit_to_isaac_bridge.py`
- **Scene tools**: `export_pegboard_scene.py`, `preview_cameras.py`, `reset_drive_targets.py`, `verify_gripper_pipeline.py`

## How to recover an archived script

```powershell
# Just copy it back temporarily:
Copy-Item _archive\debug_scripts\fix_ur_articulation_v2.py scripts\
# ...run it...
# Then either delete the copy or leave it if you'll keep using it.
```

All files were moved here on 2026-05-23 as part of a codebase cleanup. Nothing was deleted — everything is recoverable from git history if needed.
