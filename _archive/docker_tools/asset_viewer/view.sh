#!/usr/bin/env bash
# Copy this asset_viewer/ directory and the sibling exported_assets/ into the
# robotis_lab container, then launch the Isaac Sim viewer. The viewer finds
# its assets via a path relative to its own location, so no arg-passing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$(cd "$HERE/.." && pwd)/exported_assets"
CONTAINER="robotis_lab"
DEST="/tmp/asset_viewer"   # in-container destination

if [ ! -d "$ASSETS" ]; then
    echo "Error: expected assets at $ASSETS but it doesn't exist."
    exit 1
fi

if [ -z "$(docker ps -q -f name="^${CONTAINER}$")" ]; then
    echo "Error: the '${CONTAINER}' container is not running."
    echo "Start it first:  ../robotis_lab/docker/container.sh start"
    exit 1
fi

# Allow root inside the container to reach the host X server (idempotent)
xhost +SI:localuser:root >/dev/null 2>&1 || true

echo "[1/3] Resetting destination ${DEST} in container..."
docker exec "$CONTAINER" rm -rf "$DEST"
docker exec "$CONTAINER" mkdir -p "$DEST"

echo "[2/3] Copying viewer + assets..."
docker cp "$HERE/view.py" "${CONTAINER}:${DEST}/view.py"
docker cp "$ASSETS" "${CONTAINER}:${DEST}/exported_assets"

echo "[3/3] Launching Isaac Sim viewer..."
# Use -it only when stdin is a real TTY (so this script also works backgrounded)
EXEC_FLAGS=""
[ -t 0 ] && [ -t 1 ] && EXEC_FLAGS="-it"

docker exec $EXEC_FLAGS -e DISPLAY="${DISPLAY:-:0}" "$CONTAINER" bash -lc \
    "/isaac-sim/python.sh ${DEST}/view.py"
