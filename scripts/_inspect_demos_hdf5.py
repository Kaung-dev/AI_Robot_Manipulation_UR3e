"""Inspect the structure of the FreshDatas source HDF5 files so we can
see what state shape they recorded vs what our Mimic env expects to reset.
"""
import h5py

paths = {
    'brush':       'datasets/air2_manual_demos_brush/air2_mimic_source.hdf5',
    'pliers':      'datasets/air2_manual_demos_pliers/air2_mimic_pliers.hdf5',
    'scissors':    'datasets/air2_manual_demos_scissors/air2_mimic_scissors.hdf5',
    'screwdriver': 'datasets/air2_manual_demos_screwdriver/air2_mimic_screwdriver.hdf5',
}

for tool, path in paths.items():
    print(f'=== {tool} === {path}')
    try:
        with h5py.File(path, 'r') as f:
            demos = list(f['data'].keys())
            print(f'  total demos: {len(demos)}')
            if not demos:
                continue
            demo = f['data'][demos[0]]
            print(f'  demo[0] keys: {list(demo.keys())}')
            for k in demo.keys():
                v = demo[k]
                if hasattr(v, 'shape'):
                    print(f'    {k}: shape={v.shape} dtype={v.dtype}')
                elif hasattr(v, 'keys'):
                    sub = list(v.keys())
                    print(f'    {k}/: {sub[:8]}{"..." if len(sub) > 8 else ""}')
                    for k2 in sub[:8]:
                        v2 = v[k2]
                        if hasattr(v2, 'shape'):
                            print(f'      {k2}: shape={v2.shape}')
                        elif hasattr(v2, 'keys'):
                            print(f'      {k2}/: {list(v2.keys())}')
            # Try reading 'initial_state' specifically since that's what reset_to consumes
            if 'initial_state' in demo:
                ist = demo['initial_state']
                print(f'  --- initial_state deep dive ---')
                if hasattr(ist, 'keys'):
                    for k in ist.keys():
                        v = ist[k]
                        print(f'    initial_state/{k}: keys={list(v.keys()) if hasattr(v, "keys") else "(dataset shape="+str(v.shape)+")"}')
                        if hasattr(v, 'keys'):
                            for k2 in v.keys():
                                v2 = v[k2]
                                print(f'      initial_state/{k}/{k2}: keys={list(v2.keys()) if hasattr(v2, "keys") else "shape="+str(v2.shape)}')
                                if hasattr(v2, 'keys'):
                                    for k3 in list(v2.keys())[:5]:
                                        v3 = v2[k3]
                                        if hasattr(v3, 'shape'):
                                            print(f'        initial_state/{k}/{k2}/{k3}: shape={v3.shape}')
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}')
    print()
