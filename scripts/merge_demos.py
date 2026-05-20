#!/usr/bin/env python3
"""Merge multiple HDF5 demo files into one, with sequential demo_0, demo_1, ...
Also creates the train/valid mask (80/20 split) required by robomimic.

Usage:
    python3 scripts/merge_demos.py datasets/demos_franka_merged.hdf5 datasets/session_*.hdf5
"""
import argparse
import json
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge HDF5 demo files")
    parser.add_argument("output", help="Output merged HDF5 file")
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files to merge")
    parser.add_argument("--task", default=None, help="Task ID for env_args metadata (auto-detected from inputs if omitted)")
    args = parser.parse_args()

    # Pre-check: verify at least one input file is readable
    valid_inputs = []
    for path in sorted(args.inputs):
        try:
            f = h5py.File(path, "r")
            if "data" in f and len(f["data"].keys()) > 0:
                valid_inputs.append(path)
            f.close()
        except Exception:
            pass
    if not valid_inputs:
        print("ERROR: No valid input files with demos found. Output file NOT modified.")
        sys.exit(1)

    total = 0
    env_name = args.task
    with h5py.File(args.output, "w") as out_f:
        data_grp = out_f.create_group("data")

        for path in sorted(args.inputs):
            try:
                in_f = h5py.File(path, "r")
            except Exception as e:
                print(f"  SKIP {path}: {e}")
                continue

            if "data" not in in_f:
                print(f"  SKIP {path}: no 'data' group")
                in_f.close()
                continue

            # Try to pick up env_args from the first input that has it
            if env_name is None and "data" in in_f and "env_args" in in_f["data"].attrs:
                env_name = json.loads(in_f["data"].attrs["env_args"]).get("env_name")

            demos = sorted(in_f["data"].keys())
            print(f"  {path}: {len(demos)} demos")
            for demo_key in demos:
                new_key = f"demo_{total}"
                in_f.copy(f"data/{demo_key}", data_grp, name=new_key)
                total += 1
            in_f.close()

        # Create train/valid mask (80/20 split)
        all_demos = sorted(data_grp.keys())
        n_train = max(1, int(len(all_demos) * 0.8))
        if "mask" in out_f:
            del out_f["mask"]
        mask = out_f.create_group("mask")
        mask.create_dataset("train", data=np.array(all_demos[:n_train], dtype="S"))
        mask.create_dataset("valid", data=np.array(all_demos[n_train:], dtype="S"))

        # Add env_args metadata required by robomimic
        if env_name:
            env_args = {"env_name": env_name, "type": 1, "env_kwargs": {}}
            data_grp.attrs["env_args"] = json.dumps(env_args)

    print(f"\nMerged {total} demos into {args.output}")
    print(f"  Train: {n_train}, Valid: {total - n_train}")


if __name__ == "__main__":
    main()
