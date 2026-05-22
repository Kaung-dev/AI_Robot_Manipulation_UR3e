#!/usr/bin/env bash
# Launch VR teleoperation + demo recording for the Franka pegboard task.
#
# Everything is always active (cameras, screens, hand marker).
# Recording starts paused — enable with open palm gesture in headset.
#
# Usage:
#   ./launch_teleop.sh [device] [object] [dataset_file]
#
#   device:       handtracking | keyboard         (default: handtracking)
#   object:       toothbrush | scissors | silicone | pliers
#                                                  (default: toothbrush)
#   dataset_file: path to output HDF5              (default: ~/datasets/franka_<object>_<timestamp>.hdf5)
#
# Examples:
#   ./launch_teleop.sh
#   ./launch_teleop.sh handtracking scissors
#   ./launch_teleop.sh keyboard toothbrush ~/datasets/test.hdf5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Resolve ISAACLAB_PATH (env var > .env > auto-detect)
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/find_isaaclab.sh"
_resolve_isaaclab_path "$REPO_ROOT" || exit 1

DEVICE="${1:-handtracking}"
OBJECT="${2:-toothbrush}"

case "$OBJECT" in
    toothbrush) TASK="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Toothbrush-v0" ;;
    scissors)   TASK="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Scissors-v0" ;;
    silicone)   TASK="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Silicone-v0" ;;
    pliers)     TASK="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Pliers-v0" ;;
    *)
        echo "[ERROR] Unknown object: $OBJECT"
        echo "        Choose: toothbrush | scissors | silicone | pliers"
        exit 1
        ;;
esac

DATASET_DIR="$HOME/datasets"
mkdir -p "$DATASET_DIR"
DATASET_FILE="${3:-$DATASET_DIR/franka_${OBJECT}_$(date +%Y%m%d_%H%M%S).hdf5}"

echo "[INFO] Object:  $OBJECT"
echo "[INFO] Task:    $TASK"
echo "[INFO] Device:  $DEVICE"
echo "[INFO] Dataset: $DATASET_FILE"

if [[ "$DEVICE" == "handtracking" ]]; then
    export XR_RUNTIME_JSON="$HOME/.config/openxr/1/active_runtime.json"
    export XR_LOADER_DEBUG=all
    echo "[INFO] XR_RUNTIME_JSON: $XR_RUNTIME_JSON"
fi

"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/tools/record_demos.py" \
    --task "$TASK" --teleop_device "$DEVICE" \
    --dataset_file "$DATASET_FILE" --step_hz 30 --enable_cameras
