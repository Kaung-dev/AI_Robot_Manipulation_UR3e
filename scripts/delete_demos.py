#!/usr/bin/env python3
"""Delete specific demos from an HDF5 file and re-index sequentially.

Usage:
    python3 scripts/delete_demos.py datasets/demos_franka.hdf5 demo_3 demo_7 demo_12
    python3 scripts/delete_demos.py datasets/demos_franka.hdf5 3 7 12
"""
import argparse
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Delete demos from HDF5 file")
    parser.add_argument("file", help="HDF5 file to modify")
    parser.add_argument("demos", nargs="+", help="Demo keys or indices to delete (e.g. demo_3 or just 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without doing it")
    args = parser.parse_args()

    # Normalize keys: accept both "demo_3" and "3"
    to_delete = set()
    for d in args.demos:
        if d.startswith("demo_"):
            to_delete.add(d)
        else:
            to_delete.add(f"demo_{d}")

    with h5py.File(args.file, "a") as f:
        existing = sorted(f["data"].keys())
        print(f"Before: {len(existing)} demos")

        found = to_delete & set(existing)
        not_found = to_delete - set(existing)

        if not_found:
            print(f"  Not found (skipping): {sorted(not_found)}")
        if not found:
            print("  Nothing to delete.")
            return

        print(f"  Deleting: {sorted(found)}")

        if args.dry_run:
            print("  (dry run — no changes made)")
            return

        for key in found:
            del f[f"data/{key}"]

        # Re-index remaining demos sequentially
        remaining = sorted(f["data"].keys())
        temp_keys = []
        for i, old_key in enumerate(remaining):
            temp = f"__temp_{i}"
            f.move(f"data/{old_key}", f"data/{temp}")
            temp_keys.append(temp)
        for i, temp in enumerate(temp_keys):
            f.move(f"data/{temp}", f"data/demo_{i}")

        # Rebuild train/valid mask
        final = sorted(f["data"].keys())
        n_train = max(1, int(len(final) * 0.8))
        if "mask" in f:
            del f["mask"]
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(final[:n_train], dtype="S"))
        mask.create_dataset("valid", data=np.array(final[n_train:], dtype="S"))

        print(f"After: {len(final)} demos (train={n_train}, valid={len(final)-n_train})")


if __name__ == "__main__":
    main()
