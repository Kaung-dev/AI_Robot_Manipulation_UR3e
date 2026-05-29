#!/usr/bin/env bash
# Save one frame from wrist_cam and table_cam to PNG for position verification.
# Does NOT require VR. Output written to /tmp/preview_*.png
#
# Usage:
#   ./launch_preview.sh [object]
#   ./launch_preview.sh toothbrush

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Resolve ISAACLAB_PATH (env var > .env > auto-detect)
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/find_isaaclab.sh"
_resolve_isaaclab_path "$REPO_ROOT" || exit 1

OBJECT="${1:-toothbrush}"

case "$OBJECT" in
    toothbrush) TASK="Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Toothbrush-v0" ;;
    scissors)   TASK="Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Scissors-v0" ;;
    silicone)   TASK="Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Silicone-v0" ;;
    pliers)     TASK="Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Pliers-v0" ;;
    *)
        echo "[ERROR] Unknown object: $OBJECT"
        echo "        Choose: toothbrush | scissors | silicone | pliers"
        exit 1
        ;;
esac

echo "[INFO] Saving camera preview frames for: $OBJECT"

"$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/preview_cameras.py" \
    --task "$TASK" --enable_cameras

# Python script prints its own success/failure message above.
