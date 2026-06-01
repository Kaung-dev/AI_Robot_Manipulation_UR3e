"""Wrapper around IsaacLab's annotate_demos.py that registers our task IDs
AFTER Isaac's AppLauncher boots.

The upstream `annotate_demos.py` creates `AppLauncher` near the top of the
file; that's what initialises Isaac Sim's USD plugins (and makes `pxr`
importable). Our `isaaclab_ext` task package imports modules that need
`pxr` at module-load time (e.g. via `isaaclab.markers`, `isaaclab.sensors`).

If we pre-import our tasks BEFORE `AppLauncher` runs, the `pxr` chain
crashes. Workaround: monkey-patch `AppLauncher.__init__` so our task
imports fire RIGHT AFTER it finishes — `pxr` is then on sys.path and the
imports succeed before annotate_demos.py tries to `gym.make` our task.

Usage:
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\_run_mimic_annotate.py [annotate_demos.py args...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Patch AppLauncher so our task package is imported AFTER it boots — at
# that point pxr/Usd is available and our cfg modules can load cleanly.
import isaaclab.app as _isaaclab_app

_original_init = _isaaclab_app.AppLauncher.__init__


def _post_init_register_tasks(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # Trigger gym.register() side-effects now that pxr is importable.
    import isaaclab_ext.tasks.air2_franka          # noqa: F401
    import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401


_isaaclab_app.AppLauncher.__init__ = _post_init_register_tasks

# Hand control to the upstream script (uses runpy so its __main__ block runs).
ANNOTATE_SCRIPT = Path(r"C:\isaac\IsaacLab\scripts\imitation_learning\isaaclab_mimic\annotate_demos.py")
runpy.run_path(str(ANNOTATE_SCRIPT), run_name="__main__")
