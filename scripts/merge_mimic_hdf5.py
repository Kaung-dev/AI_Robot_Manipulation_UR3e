"""Merge multiple Mimic-format HDF5 files into one.

Usage:
    python scripts/merge_mimic_hdf5.py \
        --inputs datasets/screwdriver_s1.hdf5 datasets/screwdriver_s2.hdf5 \
        --output datasets/air2_mimic_screwdriver_merged.hdf5
"""

import argparse
import h5py
import shutil
from pathlib import Path


def merge(input_paths: list[str], output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Copy first file as base
    shutil.copy2(input_paths[0], output_path)

    with h5py.File(output_path, "a") as f_out:
        data = f_out["data"]
        existing = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        next_idx = int(existing[-1].split("_")[1]) + 1 if existing else 0
        total_existing = len(existing)
        print(f"[merge] base: {input_paths[0]} — {total_existing} demos")

        for inp in input_paths[1:]:
            with h5py.File(inp, "r") as f_in:
                demos = sorted(f_in["data"].keys(), key=lambda k: int(k.split("_")[1]))
                print(f"[merge] adding {inp} — {len(demos)} demos")
                for demo_key in demos:
                    new_key = f"data/demo_{next_idx}"
                    f_in.copy(f"data/{demo_key}", f_out, name=new_key)
                    next_idx += 1

        # Update total count
        all_demos = [k for k in f_out["data"].keys() if k.startswith("demo_")]
        total = len(all_demos)
        if "data" in f_out.attrs:
            pass
        f_out["data"].attrs["num_demos"] = total
        # Update env_args total if present
        if "total" in data.attrs:
            data.attrs["total"] = total

    print(f"[merge] done → {output_path} ({total} demos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Mimic HDF5 demo files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 files to merge.")
    parser.add_argument("--output", required=True, help="Output merged HDF5 file.")
    args = parser.parse_args()
    merge(args.inputs, args.output)
