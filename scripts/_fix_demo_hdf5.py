"""Strip the extra leading batch dim from `initial_state/*` tensors and
rewrite the per-tool source HDF5s under a uniform filename.

The FreshDatas HDF5s have `initial_state/articulation/<robot>/root_pose` shape
`(1, 1, 7)` but IsaacLab's `InteractiveScene.reset_to` does
`root_pose[:, :3] += env_origins[env_ids]` which expects `(num_envs, 7)` — the
extra `1` makes broadcasting fail with: "tensor a (7) must match tensor b (3)
at non-singleton dimension 2". Squeezing the leading dim fixes it.

This also normalises the source filename to `air2_mimic_source.hdf5` across
tools (the brush zip already had this name; the others were named
`air2_mimic_<tool>.hdf5`).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np


SOURCES = {
    "brush":       "datasets/air2_manual_demos_brush/air2_mimic_source.hdf5",
    "pliers":      "datasets/air2_manual_demos_pliers/air2_mimic_pliers.hdf5",
    "scissors":    "datasets/air2_manual_demos_scissors/air2_mimic_scissors.hdf5",
    "screwdriver": "datasets/air2_manual_demos_screwdriver/air2_mimic_screwdriver.hdf5",
}
OUTPUT_NAME = "air2_mimic_source.hdf5"


def _maybe_squeeze(arr: np.ndarray) -> np.ndarray:
    """If the leading dim is 1 AND the array is 3D, drop it.

    Targets (1, num_envs, D) -> (num_envs, D). Leaves already-2D arrays alone.
    """
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _copy_with_squeeze(src: h5py.Group, dst: h5py.Group, path: str = "") -> None:
    """Recursively copy datasets, squeezing 3D-with-leading-1 shapes."""
    for key, item in src.items():
        full = f"{path}/{key}" if path else key
        if isinstance(item, h5py.Dataset):
            data = item[()]
            new = _maybe_squeeze(data) if hasattr(data, "ndim") else data
            ds = dst.create_dataset(key, data=new, compression=item.compression)
            for ak, av in item.attrs.items():
                ds.attrs[ak] = av
        elif isinstance(item, h5py.Group):
            sub = dst.create_group(key)
            for ak, av in item.attrs.items():
                sub.attrs[ak] = av
            _copy_with_squeeze(item, sub, full)


def fix_one(tool: str, src_path: str) -> tuple[str, bool, str]:
    src = Path(src_path)
    if not src.exists():
        return tool, False, f"source missing: {src_path}"
    dst = src.parent / OUTPUT_NAME
    backup = src.parent / (src.stem + ".original.hdf5")

    # Use a temp file then atomically move so we never half-write.
    tmp = src.parent / "_fix_tmp.hdf5"
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        for ak, av in fin.attrs.items():
            fout.attrs[ak] = av
        _copy_with_squeeze(fin, fout)
    # If the dst exists (e.g. brush already named "source"), back up the
    # original first.
    if dst.exists() and dst.resolve() == src.resolve():
        # Same file: just move tmp over it after backing up.
        shutil.copy2(src, backup)
        tmp.replace(dst)
    else:
        if dst.exists():
            dst.unlink()
        tmp.replace(dst)
        # Rename the differently-named original to .original.hdf5 so it's
        # archived but obvious which file is now live.
        if src != dst:
            src.rename(backup)
    return tool, True, f"{src.name} -> {dst.name} (orig backed up as {backup.name})"


if __name__ == "__main__":
    print("[fixer] squeezing initial_state extra batch dim from per-tool source HDF5s\n")
    for tool, path in SOURCES.items():
        tool_, ok, msg = fix_one(tool, path)
        flag = "ok " if ok else "FAIL"
        print(f"  [{flag}] {tool_:<12}  {msg}")
    print("\n[fixer] done — re-run annotate against the new air2_mimic_source.hdf5 in each folder")
