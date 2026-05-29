#!/usr/bin/env bash
# Bootstrap script: links the contents of isaaclab_ext/ into a local IsaacLab install
# so Isaac Lab's auto-discovery picks up the robot config and all lift tasks.
# Also produces a clean scene_isaaclab.usd from scene/scene.usd.
#
# Usage:
#   ./setup_isaaclab.sh                    # uses bundled IsaacLab inside Isaac Sim 5.1 standalone
#   ISAACLAB_PATH=/path/to/IsaacLab ./setup_isaaclab.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve ISAACLAB_PATH (env var > .env > auto-detect)
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/find_isaaclab.sh"
_resolve_isaaclab_path "$REPO_DIR" || exit 1

if [[ ! -d "$ISAACLAB_PATH" ]]; then
  echo "[ERROR] IsaacLab not found at $ISAACLAB_PATH"
  echo "        Set ISAACLAB_PATH=/path/to/IsaacLab and re-run."
  exit 1
fi

echo "[INFO] Repo:      $REPO_DIR"
echo "[INFO] IsaacLab:  $ISAACLAB_PATH"

LIFT_CONFIG="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config"

# 1) robot config
ROBOT_DST="$ISAACLAB_PATH/source/isaaclab_assets/isaaclab_assets/robots/ur3e_rg2.py"
ROBOT_SRC="$REPO_DIR/isaaclab_ext/robots/ur3e_rg2.py"
ln -sfn "$ROBOT_SRC" "$ROBOT_DST"
echo "[INFO] linked  $ROBOT_DST"

# 2) AIR2 task (Franka)
ln -sfn "$REPO_DIR/isaaclab_ext/tasks/air2_franka"         "$LIFT_CONFIG/air2_franka"
echo "[INFO] linked  $LIFT_CONFIG/air2_franka"

# 3) AIR2 Robotis variant (Franka)
ln -sfn "$REPO_DIR/isaaclab_ext/tasks/air2_robotis_franka" "$LIFT_CONFIG/air2_robotis_franka"
echo "[INFO] linked  $LIFT_CONFIG/air2_robotis_franka"

# 4) Pegboard task (Franka) — original/baseline env
ln -sfn "$REPO_DIR/isaaclab_ext/tasks/pegboard_franka"     "$LIFT_CONFIG/pegboard_franka"
echo "[INFO] linked  $LIFT_CONFIG/pegboard_franka"

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
echo "      --task Isaac-AIR2-Franka-Play-v0 --num_envs 1 --teleop_device keyboard --enable_cameras"
