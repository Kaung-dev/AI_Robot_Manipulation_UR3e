#!/usr/bin/env bash
# Bootstrap script: links the contents of isaaclab_ext/ into a local IsaacLab clone
# so Isaac Lab's auto-discovery picks up the UR3e+RG2 robot config and lift task.
# Also produces a clean scene_isaaclab.usd from scene/scene.usd.
#
# Usage:
#   ./setup_isaaclab.sh                    # uses ~/IsaacLab as the clone path
#   ISAACLAB_PATH=/path/to/IsaacLab ./setup_isaaclab.sh
#
# Pre-reqs:
#   - IsaacLab cloned (./isaaclab.sh --install already run, Isaac Sim 4.5 in venv)
#   - This repo cloned at any path; the script uses its own location.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"

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

# 2) lift-cube task variant
TASK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur3e_rg2"
TASK_SRC="$REPO_DIR/isaaclab_ext/tasks/lift_cube_ur3e_rg2"
ln -sfn "$TASK_SRC" "$TASK_DST"
echo "[INFO] linked  $TASK_DST  ->  $TASK_SRC"

# 3) lift-pegboard-franka task variant
TASK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/pegboard_franka"
TASK_SRC="$REPO_DIR/isaaclab_ext/tasks/lift_pegboard_franka"
ln -sfn "$TASK_SRC" "$TASK_DST"
echo "[INFO] linked  $TASK_DST  ->  $TASK_SRC"

# 4) lift-pegboard-ur3e-rg2 task variant
TASK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/pegboard_ur3e_rg2"
TASK_SRC="$REPO_DIR/isaaclab_ext/tasks/lift_pegboard_ur3e_rg2"
ln -sfn "$TASK_SRC" "$TASK_DST"
echo "[INFO] linked  $TASK_DST  ->  $TASK_SRC"

# 5) stack-cube-franka task (reference)
STACK_DST="$ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cube_franka"
STACK_SRC="$REPO_DIR/isaaclab_ext/tasks/stack_cube_franka"
if [[ ! -e "$STACK_DST" ]]; then
  ln -sfn "$STACK_SRC" "$STACK_DST"
  echo "[INFO] linked  $STACK_DST  ->  $STACK_SRC"
fi

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
echo "      --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device keyboard"
