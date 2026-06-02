"""Pre-import our tasks then hand off to Isaac Lab's generate_dataset.py."""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ISAACLAB_PATH = Path(os.environ.get("ISAACLAB_PATH", "/mnt/extra/IsaacLab"))
SCRIPT = ISAACLAB_PATH / "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"

src = SCRIPT.read_text()
inject = """
import sys as _sys
_sys.path.insert(0, r"{repo}")
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
""".format(repo=str(REPO_ROOT))

marker = "simulation_app = app_launcher.app"
src = src.replace(marker, marker + "\n" + inject, 1)

exec(compile(src, str(SCRIPT), "exec"), {"__name__": "__main__", "__file__": str(SCRIPT)})
