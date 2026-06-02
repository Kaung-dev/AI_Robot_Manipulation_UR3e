"""Filter a Mimic-generated HDF5 to only the CLEAN demos — ones where the object
actually ended up in the basket. Mimic keeps some demos that grasp but don't
deliver; training on those adds noise + skews obs normalization, which makes the
BC roll out badly (freeze / fly-away). Keeping only deliveries makes a small
dataset behave like the brush set (which was ~97% clean and worked).

Usage:
  ~/isaacsim/python.sh scripts/state_bc_v2/filter_clean_demos.py \
      --in datasets/air2_mimic_generated_pliers_v2.hdf5 \
      --out datasets/air2_mimic_generated_pliers_v2_clean.hdf5 \
      --radius 0.18
No Isaac Sim needed (pure h5py).
"""
from __future__ import annotations
import argparse
import numpy as np
import h5py

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", dest="out", required=True)
p.add_argument("--radius", type=float, default=0.18,
               help="keep demo if object END is within this 3D dist of the basket (robot-root frame)")
args = p.parse_args()

# basket in robot-root frame ≈ world basket (-3.941,-5.785,1.140) - robot base (-4.2405,-5.2851,1.0397)
BASKET = np.array([0.2995, -0.4999, 0.1003], dtype=np.float32)

with h5py.File(args.inp, "r") as fin, h5py.File(args.out, "w") as fout:
    din = fin["data"]
    dout = fout.create_group("data")
    # copy top-level data attrs (mimic stores env metadata there)
    for k, v in din.attrs.items():
        dout.attrs[k] = v
    demos = sorted(din.keys(), key=lambda x: int(x.split("_")[1]))
    kept = 0
    for dk in demos:
        end = din[dk]["obs"]["object_position"][-1]
        if np.linalg.norm(np.asarray(end, dtype=np.float32) - BASKET) <= args.radius:
            fin.copy(din[dk], dout, name=f"demo_{kept}")  # renumber contiguously
            kept += 1
    dout.attrs["num_demos"] = kept
    print(f"[filter] kept {kept}/{len(demos)} clean demos ({100*kept/len(demos):.0f}%) -> {args.out}")
