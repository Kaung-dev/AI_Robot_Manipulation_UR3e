#!/usr/bin/env bash
# Resolve ISAACLAB_PATH and export it.
# Sourced by launch_teleop.sh, launch_preview.sh, and setup_isaaclab.sh.
#
# Resolution order:
#   1. Already set in the environment
#   2. Set in .env in the repo root
#   3. Auto-detected: first isaaclab.sh found under ~/isaac-sim/

_resolve_isaaclab_path() {
    local repo_root="$1"

    # 1. Already set — use it
    if [[ -n "${ISAACLAB_PATH:-}" ]]; then
        return 0
    fi

    # 2. Load .env if present
    if [[ -f "$repo_root/.env" ]]; then
        # shellcheck disable=SC1091
        source "$repo_root/.env"
        if [[ -n "${ISAACLAB_PATH:-}" ]]; then
            return 0
        fi
    fi

    # 3. Auto-detect: search ~/isaac-sim/ for isaaclab.sh (Isaac Sim standalone default location)
    if [[ -d "$HOME/isaac-sim" ]]; then
        local found
        found=$(find "$HOME/isaac-sim" -maxdepth 4 -name "isaaclab.sh" 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            ISAACLAB_PATH="$(dirname "$found")"
            export ISAACLAB_PATH
            echo "[INFO] Auto-detected IsaacLab at: $ISAACLAB_PATH"
            return 0
        fi
    fi

    # 4. Give up
    echo "[ERROR] Could not find IsaacLab."
    echo "        Options:"
    echo "          a) Install Isaac Sim to ~/isaac-sim/ (default location)"
    echo "          b) Set ISAACLAB_PATH in your shell: export ISAACLAB_PATH=/path/to/IsaacLab"
    echo "          c) Add it to .env in the repo root:  echo 'ISAACLAB_PATH=/path/to/IsaacLab' >> .env"
    return 1
}
