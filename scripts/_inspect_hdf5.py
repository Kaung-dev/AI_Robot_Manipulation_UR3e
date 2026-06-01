"""One-off — does friend's HDF5 contain `initial_state` (needed by mimic)?"""
import h5py
from pathlib import Path

fp = Path("_archive/v2_friend_hdf5/teleop_air2_robotis.hdf5")
with h5py.File(fp, "r") as f:
    print(f"top-level keys: {list(f.keys())}")
    print(f"data attrs: {dict(f['data'].attrs)}")
    demo0 = f["data/demo_0"]
    print(f"\ndemo_0 keys: {list(demo0.keys())}")
    print(f"demo_0 attrs: {dict(demo0.attrs)}")
    if "initial_state" in demo0:
        ini = demo0["initial_state"]
        print(f"\ninitial_state: type={type(ini).__name__}")
        if hasattr(ini, "keys"):
            for k in ini.keys():
                print(f"  {k}")
        else:
            print(f"  shape={ini.shape} dtype={ini.dtype}")
    else:
        print("\ninitial_state: MISSING")
    if "obs" in demo0:
        print(f"\nobs/ keys: {list(demo0['obs'].keys())}")
    if "states" in demo0:
        print(f"states/ keys: {list(demo0['states'].keys()) if hasattr(demo0['states'], 'keys') else 'array'}")
