"""Rewrite friend's HDF5 demos so the initial_state + states groups use
`tool_screwdriver` instead of `tool_silicone` (Stephen's rename in the
air2_robotis_franka task cfg).

Recursively walks the HDF5 and renames any group named exactly `tool_silicone`
under any `initial_state/rigid_object/...` or `states/rigid_object/...` path.

Friend recorded against `Isaac-Lift-AIR2-Robotis-v0` whose 4th object was
`tool_silicone` (silicone_tube_ring_blue.usd). Current scene has the screwdriver
in that slot under the name `tool_screwdriver`. The state replay reads names
from the recorded initial_state and fails KeyError on the missing key.

Usage:
    python scripts\\_rename_silicone_to_screwdriver.py <input.hdf5> <output.hdf5>
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py


def rename_keys(group: h5py.Group, depth: int = 0) -> int:
    """Recursively rename any child group named 'tool_silicone' -> 'tool_screwdriver'. Returns count."""
    n = 0
    # snapshot keys because we mutate during iteration
    for name in list(group.keys()):
        item = group[name]
        if name == "tool_silicone" and isinstance(item, h5py.Group):
            group.move("tool_silicone", "tool_screwdriver")
            n += 1
            # after move, recurse into the renamed group
            n += rename_keys(group["tool_screwdriver"], depth + 1)
        elif isinstance(item, h5py.Group):
            n += rename_keys(item, depth + 1)
    return n


def main(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        raise SystemExit("refusing to rewrite the source file in place; pick a different output")
    # copy first so the rename happens on a fresh file
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rename] copy {src} -> {dst}", flush=True)
    dst.write_bytes(src.read_bytes())
    with h5py.File(dst, "a") as f:
        n = rename_keys(f["data"])
    print(f"[rename] done — {n} 'tool_silicone' group(s) renamed to 'tool_screwdriver'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
