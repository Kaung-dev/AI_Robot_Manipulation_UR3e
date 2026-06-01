"""Wrapper around IsaacLab's generate_dataset.py — same pattern as
_run_mimic_annotate.py: pre-imports our task package before delegating.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401

GENERATE_SCRIPT = Path(r"C:\isaac\IsaacLab\scripts\imitation_learning\isaaclab_mimic\generate_dataset.py")
runpy.run_path(str(GENERATE_SCRIPT), run_name="__main__")
