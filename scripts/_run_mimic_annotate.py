"""Wrapper around IsaacLab's annotate_demos.py that pre-imports our tasks
so the gym IDs (`Isaac-AIR2-Robotis-Franka-Brush-Mimic-v0` etc.) are
registered before the script's task lookup runs.

The import has to happen AFTER the AppLauncher is created (IsaacLab convention),
but the upstream script handles that. We just need our task package on
sys.path with auto-imports, which `isaaclab_ext` already provides.

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\_run_mimic_annotate.py [annotate_demos.py args...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Pre-import our task package so the gym.register() calls fire and the new
# Mimic task IDs are available to annotate_demos.py's parse_env_cfg().
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401

# Hand control to the upstream script. runpy executes it as __main__ so its
# CLI parser sees the args we forwarded.
ANNOTATE_SCRIPT = Path(r"C:\isaac\IsaacLab\scripts\imitation_learning\isaaclab_mimic\annotate_demos.py")
runpy.run_path(str(ANNOTATE_SCRIPT), run_name="__main__")
