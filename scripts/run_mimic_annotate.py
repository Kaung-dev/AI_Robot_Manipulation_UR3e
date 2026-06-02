"""Pre-import our tasks then hand off to Isaac Lab's annotate_demos.py.

The task imports must happen AFTER AppLauncher starts Isaac Sim (pxr etc. aren't
available before that). We inject them via sys.modules post-boot by monkey-patching
the script's globals after runpy loads it, OR more simply: we add our repo root to
sys.path and let annotate_demos.py's own import of isaaclab_tasks trigger our
__init__ registrations automatically via the installed extension path.

Simplest working approach: patch sys.argv with our task, add repo to path, and
let annotate_demos.py boot Isaac Sim then import everything.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Hand control to upstream script — it boots Isaac Sim first via AppLauncher,
# then our tasks register when isaaclab_ext is imported post-boot.
import runpy

ISAACLAB_PATH = Path(os.environ.get("ISAACLAB_PATH", "/mnt/extra/IsaacLab"))
SCRIPT = ISAACLAB_PATH / "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"

# Patch the script to import our tasks after AppLauncher starts.
# We do this by injecting into the script's source.
src = SCRIPT.read_text()
inject = """
import sys as _sys
_sys.path.insert(0, r"{repo}")
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
""".format(repo=str(REPO_ROOT))

# Insert after the AppLauncher boot line
marker = "simulation_app = app_launcher.app"
src = src.replace(marker, marker + "\n" + inject, 1)

exec(compile(src, str(SCRIPT), "exec"), {"__name__": "__main__", "__file__": str(SCRIPT)})
