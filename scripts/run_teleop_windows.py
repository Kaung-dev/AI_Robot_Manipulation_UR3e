"""Windows launcher for Isaac Lab teleop.

Preloads h5py so its bundled HDF5 DLLs are loaded before Isaac Sim sensor
extensions add older HDF5 DLLs to the process.
"""

from __future__ import annotations

import runpy
import sys
import os
from pathlib import Path

import h5py  # noqa: F401


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", r"D:\IsaacLab"))
    os.environ.setdefault("UR3E_RG2_USD", str(repo_root / "scene" / "scene_isaaclab.usd"))
    teleop_script = isaaclab_root / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
    if not teleop_script.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab teleop script: {teleop_script}")

    sys.argv[0] = str(teleop_script)
    runpy.run_path(str(teleop_script), run_name="__main__")


if __name__ == "__main__":
    main()
