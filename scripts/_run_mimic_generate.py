"""Wrapper around IsaacLab's generate_dataset.py — same monkey-patch
pattern as _run_mimic_annotate.py so our task imports fire AFTER Isaac
boots (otherwise the pxr chain crashes before AppLauncher initialises).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import isaaclab.app as _isaaclab_app

_original_init = _isaaclab_app.AppLauncher.__init__


def _post_init_register_tasks(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    import isaaclab_ext.tasks.air2_franka          # noqa: F401
    import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401


_isaaclab_app.AppLauncher.__init__ = _post_init_register_tasks

GENERATE_SCRIPT = Path(r"C:\isaac\IsaacLab\scripts\imitation_learning\isaaclab_mimic\generate_dataset.py")
runpy.run_path(str(GENERATE_SCRIPT), run_name="__main__")
