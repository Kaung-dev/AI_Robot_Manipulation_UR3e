#!/usr/bin/env bash
# Bootstrap script: links the contents of isaaclab_ext/ into a local IsaacLab install
# so Isaac Lab's auto-discovery picks up the UR3e+RG2 robot config and both lift tasks.
# Also produces a clean scene_isaaclab.usd from scene/scene.usd.
#
# Usage:
#   ./setup_isaaclab.sh                    # uses bundled IsaacLab inside Isaac Sim 5.1 standalone
#   ISAACLAB_PATH=/path/to/IsaacLab ./setup_isaaclab.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/IsaacLab}"

if [[ ! -d "$ISAACLAB_PATH" ]]; then
  echo "[ERROR] IsaacLab not found at $ISAACLAB_PATH"
  echo "        Set ISAACLAB_PATH=/path/to/IsaacLab and re-run."
  exit 1
fi

echo "[INFO] Repo:      $REPO_DIR"
echo "[INFO] IsaacLab:  $ISAACLAB_PATH"

# 1) robot config
ROBOT_DST="$ISAACLAB_PATH/source/isaaclab_assets/isaaclab_assets/robots/ur3e_rg2.py"
ROBOT_SRC="$REPO_DIR/isaaclab_ext/robots/ur3e_rg2.py"
ln -sfn "$ROBOT_SRC" "$ROBOT_DST"
echo "[INFO] linked  $ROBOT_DST  ->  $ROBOT_SRC"

# 2) lift-cube task
CUBE_TASK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur3e_rg2"
CUBE_TASK_SRC="$REPO_DIR/isaaclab_ext/tasks/lift_cube_ur3e_rg2"
ln -sfn "$CUBE_TASK_SRC" "$CUBE_TASK_DST"
echo "[INFO] linked  $CUBE_TASK_DST  ->  $CUBE_TASK_SRC"

# 3) lift-pegboard task (UR3e)
PEGBOARD_TASK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur3e_rg2_pegboard"
PEGBOARD_TASK_SRC="$REPO_DIR/isaaclab_ext/tasks/lift_pegboard_ur3e_rg2"
ln -sfn "$PEGBOARD_TASK_SRC" "$PEGBOARD_TASK_DST"
echo "[INFO] linked  $PEGBOARD_TASK_DST  ->  $PEGBOARD_TASK_SRC"

# 4) lift-pegboard task (Franka)
FRANKA_TASK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka_pegboard"
FRANKA_TASK_SRC="$REPO_DIR/isaaclab_ext/tasks/lift_pegboard_franka"
ln -sfn "$FRANKA_TASK_SRC" "$FRANKA_TASK_DST"
echo "[INFO] linked  $FRANKA_TASK_DST  ->  $FRANKA_TASK_SRC"

# 5) Apply IsaacLab patches (VR teleoperation + recording controls)
echo "[INFO] applying isaaclab_patches/ ..."
PATCHES="$REPO_DIR/isaaclab_patches"
cp -f "$PATCHES/scripts/tools/vr_camera_screens.py"                                                     "$ISAACLAB_PATH/scripts/tools/"
cp -f "$PATCHES/scripts/tools/vr_gesture_detector.py"                                                   "$ISAACLAB_PATH/scripts/tools/"
cp -f "$PATCHES/scripts/tools/record_demos.py"                                                          "$ISAACLAB_PATH/scripts/tools/"
cp -f "$PATCHES/scripts/environments/teleoperation/teleop_se3_agent.py"                                 "$ISAACLAB_PATH/scripts/environments/teleoperation/"
cp -f "$PATCHES/source/isaaclab/isaaclab/devices/openxr/openxr_device.py"                              "$ISAACLAB_PATH/source/isaaclab/isaaclab/devices/openxr/"
cp -f "$PATCHES/source/isaaclab/isaaclab/devices/openxr/retargeters/manipulator/se3_rel_retargeter.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/devices/openxr/retargeters/manipulator/"
echo "[INFO] patches applied"

# 6) Generate scene_isaaclab.usd if missing.
if [[ ! -f "$REPO_DIR/scene/scene_isaaclab.usd" ]]; then
  echo "[INFO] producing scene/scene_isaaclab.usd ..."
  "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_DIR/scripts/fix_scene_for_isaaclab.py" \
      --in_usd "$REPO_DIR/scene/scene.usd" \
      --out_usd "$REPO_DIR/scene/scene_isaaclab.usd"
fi

echo
echo "[DONE] You can now run, for example:"
echo "  $ISAACLAB_PATH/isaaclab.sh -p $ISAACLAB_PATH/scripts/environments/teleoperation/teleop_se3_agent.py \\"
echo "      --task Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device keyboard"
