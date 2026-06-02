"""Replace R0 demos in an existing generated screwdriver HDF5 with fresh ones.

Usage:
    python scripts/merge_generated_r0.py \
        --base  datasets/air2_mimic_generated_screwdriver.hdf5 \
        --new_r0 datasets/air2_mimic_generated_screwdriver_r0.hdf5 \
        --out   datasets/air2_mimic_generated_screwdriver_merged.hdf5
"""
import argparse
import numpy as np
import h5py

SLOTS = {
    "R0": np.array([-0.0315, -0.6749, 0.5713]),
    "R1": np.array([-0.2045, -0.6749, 0.5713]),
    "R2": np.array([-0.0315, -0.6749, 0.2863]),
    "R3": np.array([-0.2045, -0.6749, 0.2863]),
}


def detect_slot(demo) -> str:
    pos = demo["obs"]["object_position"][0]
    return min(SLOTS, key=lambda s: np.linalg.norm(SLOTS[s] - pos))


def copy_demo(src_demo, dst_group, demo_key):
    dst = dst_group.create_group(demo_key)
    for k in src_demo.keys():
        if isinstance(src_demo[k], h5py.Group):
            src_demo.copy(k, dst)
        else:
            dst.create_dataset(k, data=src_demo[k][:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",   default="datasets/air2_mimic_generated_screwdriver.hdf5")
    parser.add_argument("--new_r0", default="datasets/air2_mimic_generated_screwdriver_r0.hdf5")
    parser.add_argument("--out",    default="datasets/air2_mimic_generated_screwdriver_merged.hdf5")
    args = parser.parse_args()

    counts = {"kept": {}, "dropped_R0": 0, "added_R0": 0}

    with h5py.File(args.base, "r") as base, \
         h5py.File(args.new_r0, "r") as new_r0, \
         h5py.File(args.out, "w") as out:

        dst = out.create_group("data")
        idx = 0

        # Copy non-R0 demos from base dataset
        demos = sorted(base["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for dk in demos:
            slot = detect_slot(base["data"][dk])
            if slot == "R0":
                counts["dropped_R0"] += 1
                continue
            copy_demo(base["data"][dk], dst, f"demo_{idx}")
            counts["kept"][slot] = counts["kept"].get(slot, 0) + 1
            idx += 1

        # Stitch in new R0 demos
        new_demos = sorted(new_r0["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for dk in new_demos:
            copy_demo(new_r0["data"][dk], dst, f"demo_{idx}")
            counts["added_R0"] += 1
            idx += 1

    print(f"Done → {args.out}")
    print(f"  Dropped R0 from base : {counts['dropped_R0']}")
    print(f"  Kept from base       : {counts['kept']}")
    print(f"  Added new R0         : {counts['added_R0']}")
    print(f"  Total demos          : {idx}")


if __name__ == "__main__":
    main()
