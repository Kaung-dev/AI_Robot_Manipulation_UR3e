"""Quick peek at an annotated mimic HDF5 — confirm subtask_term_signals are present."""
import sys
from pathlib import Path
import h5py
import numpy as np

fp = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/mimic/brush_annotated.hdf5")
with h5py.File(fp, "r") as f:
    data = f["data"]
    demos = sorted(data.keys(), key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else 0)
    print(f"{fp.name}: {len(demos)} demos, env_args: {dict(data.attrs).get('env_args', '?')[:120]}")
    for d in demos[:3]:
        grp = data[d]
        print(f"\n--- {d} ---  num_samples={grp.attrs.get('num_samples')}  success={grp.attrs.get('success')}")
        print(f"  top-level keys: {list(grp.keys())}")
        if "obs" in grp:
            print(f"  obs/ keys: {list(grp['obs'].keys())}")
            if "datagen_info" in grp["obs"]:
                di = grp["obs/datagen_info"]
                print(f"    datagen_info keys: {list(di.keys()) if hasattr(di, 'keys') else 'array'}")
                if "subtask_term_signals" in di:
                    sts = di["subtask_term_signals"]
                    if hasattr(sts, "keys"):
                        for k in sts.keys():
                            arr = sts[k][:]
                            on_steps = int(arr.sum())
                            print(f"      {k}: total True steps = {on_steps}/{len(arr)} (first True at step {int(arr.argmax()) if on_steps else 'never'})")
