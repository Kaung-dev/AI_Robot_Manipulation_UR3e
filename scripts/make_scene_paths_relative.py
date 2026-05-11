"""
Rewrite all asset references in scene/scene.usd and scene/scene_isaaclab.usd
from absolute paths (/home/user/Desktop/ur_pick/...) to repo-relative paths
(./../rg2_inria_usd/rg2_inria.usd etc.), so the scene works regardless of
where the repo is cloned.

This is run as a STANDALONE pxr.Usd script (not inside Isaac Sim's Script
Editor), so we don't need omni.usd. Just `python3 scripts/make_scene_paths_relative.py`
from the repo root.

Requires:
    pip install usd-core   # or use a Python with pxr already on path (Isaac venv)
"""

from pathlib import Path
import sys

try:
    from pxr import Usd, Sdf
except ImportError:
    print("[ERROR] pxr.Usd not found. Run with the Isaac venv:")
    print("    ~/isaacsim_env/bin/python scripts/make_scene_paths_relative.py")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
SENTINEL = "/home/user/Desktop/ur_pick"


def fix_layer(layer_path: Path):
    """Open a layer, rewrite every assetPath that contains SENTINEL to a
    relative path. Save in place."""
    if not layer_path.exists():
        print(f"[SKIP] {layer_path} not found")
        return

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        print(f"[ERROR] could not open {layer_path}")
        return

    layer_dir = layer_path.parent.resolve()
    n_changed = 0

    def rewrite_str(s):
        nonlocal n_changed
        if not isinstance(s, str) or SENTINEL not in s:
            return s
        # extract the part after SENTINEL: "/home/user/Desktop/ur_pick/foo/bar.usd" -> "foo/bar.usd"
        idx = s.index(SENTINEL)
        prefix = s[:idx]
        rest = s[idx + len(SENTINEL):].lstrip("/")
        target_abs = (REPO / rest).resolve()
        rel = Path("./") / Path(target_abs).resolve().relative_to(REPO)
        # Actually relative-from layer dir, not from REPO:
        try:
            rel = Path(target_abs).resolve().relative_to(layer_dir)
        except ValueError:
            # not under layer dir, use os.path.relpath
            import os
            rel = Path(os.path.relpath(target_abs, layer_dir))
        new = prefix + str(rel)
        n_changed += 1
        return new

    # Walk every prim spec and every reference / payload
    def visit_prim_spec(prim_spec):
        for ref in list(prim_spec.referenceList.GetAddedOrExplicitItems()):
            if SENTINEL in ref.assetPath:
                new_path = rewrite_str(ref.assetPath)
                # Sdf reference is immutable; rebuild
                new_ref = Sdf.Reference(
                    assetPath=new_path,
                    primPath=ref.primPath,
                    layerOffset=ref.layerOffset,
                    customData=ref.customData,
                )
                prim_spec.referenceList.Remove(ref)
                prim_spec.referenceList.Add(new_ref)
        for ref in list(prim_spec.payloadList.GetAddedOrExplicitItems()):
            if SENTINEL in ref.assetPath:
                new_path = rewrite_str(ref.assetPath)
                new_ref = Sdf.Payload(
                    assetPath=new_path,
                    primPath=ref.primPath,
                    layerOffset=ref.layerOffset,
                )
                prim_spec.payloadList.Remove(ref)
                prim_spec.payloadList.Add(new_ref)
        for child in prim_spec.nameChildren:
            visit_prim_spec(child)

    if layer.pseudoRoot:
        for child in layer.pseudoRoot.nameChildren:
            visit_prim_spec(child)

    # Also walk every attribute that might hold an SdfAssetPath
    for prim_path in layer.rootPrims.values():
        pass  # references are the main thing we care about

    if n_changed:
        layer.Save()
        print(f"[OK] {layer_path}: {n_changed} reference(s) rewritten")
    else:
        print(f"[OK] {layer_path}: no absolute paths found")


def main():
    fix_layer(REPO / "scene" / "scene.usd")
    fix_layer(REPO / "scene" / "scene_isaaclab.usd")
    print("\nDone. Verify with:  strings scene/*.usd | grep '/home/user'")


if __name__ == "__main__":
    main()
