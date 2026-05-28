"""Convert IsaacLab record_demos.py HDF5 -> repo's PNG+npz demo layout.

HDF5 layout (written by ActionStateRecorderManagerCfg, EXPORT_SUCCEEDED_ONLY):
    data/
      attrs: total=N, env_args=<json>
      demo_0/
        actions            (T, 7)            raw EE delta + gripper toggle
        obs/joint_pos      (T, 9)
        obs/joint_vel      (T, 9)
        obs/wrist_camera   (T, H, W, 3) uint8     (only if --enable_cameras was passed)
        obs/object_position(T, 3)
        ...
      demo_1/ ...

Output layout (consumed by scripts/train_bc.py and scripts/train_state_bc.py):
    demos_root/
      ep_000/
        wrist_rgb/t_0000.png ...     (only if HDF5 has obs/wrist_camera)
        states.npz                    joint_pos, joint_vel, action, gripper, wp_idx
        meta.json                     {task, num_frames, success, source_hdf5, source_demo}

Runs in plain Python — no SimulationApp boot. Requires h5py + numpy + pillow.

Usage:
    python scripts/hdf5_to_demos.py \
        --hdf5 datasets/teleop_air2_robotis.hdf5 \
        --out  demos_air2_robotis_teleop

Notes
-----
* Robotis task has wrist_camera but no board_camera. train_bc.py requires both;
  for visual BC on this dataset either add a board camera to the cfg or use
  state-only BC against states.npz (see train_state_bc.py for the MLP shape).
* All exported HDF5 episodes are successes (recorder filters during export),
  so meta["success"] is always True.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _decode_env_name(data_group) -> str:
    raw = data_group.attrs.get("env_args", None)
    if raw is None:
        return ""
    try:
        return json.loads(raw).get("env_name", "")
    except (TypeError, json.JSONDecodeError):
        return ""


def convert_episode(demo_grp: h5py.Group, ep_dir: Path, *, save_every: int) -> dict:
    """Write one HDF5 demo to a single ep_XXX directory. Returns its meta dict."""
    ep_dir.mkdir(parents=True, exist_ok=True)

    actions = demo_grp["actions"][...]  # (T, 7) — IK delta(0:6) + gripper(6)
    obs = demo_grp["obs"]
    if "joint_pos" not in obs or "joint_vel" not in obs:
        raise KeyError(
            f"{demo_grp.name}: missing obs/joint_pos or obs/joint_vel — was the demo recorded with the wrong cfg?"
        )
    joint_pos = obs["joint_pos"][...]   # (T, 9)
    joint_vel = obs["joint_vel"][...]   # (T, 9)
    T = actions.shape[0]

    # Optional wrist camera.
    have_wrist = "wrist_camera" in obs
    if have_wrist:
        wrist = obs["wrist_camera"]    # (T, H, W, 3) uint8
        wrist_dir = ep_dir / "wrist_rgb"
        wrist_dir.mkdir(exist_ok=True)
        for t in range(0, T, save_every):
            Image.fromarray(np.asarray(wrist[t], dtype=np.uint8)).save(
                wrist_dir / f"t_{t:04d}.png"
            )

    np.savez_compressed(
        ep_dir / "states.npz",
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        action=actions.astype(np.float32),
        gripper=actions[:, 6].astype(np.float32),
        # Teleop has no scripted-waypoint indices — zero column keeps the schema.
        wp_idx=np.zeros(T, dtype=np.int32),
    )

    return {"num_frames": T, "has_wrist_rgb": have_wrist}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", required=True, help="Input HDF5 from record_demos.py.")
    ap.add_argument("--out", required=True, help="Output demos root directory.")
    ap.add_argument("--save_every", type=int, default=1,
                    help="Save every Nth frame's PNG (1=all). State arrays are always full-rate.")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="First ep_XXX index in the output (use to append to an existing demos dir).")
    args = ap.parse_args()

    src = Path(args.hdf5).resolve()
    if not src.exists():
        raise SystemExit(f"hdf5 not found: {src}")

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as f:
        data_grp = f["data"]
        env_name = _decode_env_name(data_grp)
        # Sort demo_N by N so output ep numbering is deterministic.
        demo_names = sorted(data_grp.keys(), key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else 0)
        print(f"[hdf5_to_demos] {src.name}: env={env_name!r}, {len(demo_names)} demos")

        for i, demo_name in enumerate(demo_names):
            ep_idx = args.start_idx + i
            ep_dir = out_root / f"ep_{ep_idx:03d}"
            info = convert_episode(data_grp[demo_name], ep_dir, save_every=args.save_every)
            meta = {
                "task": env_name,
                "env_idx_in_batch": 0,
                "num_frames": info["num_frames"],
                "success": True,                  # recorder only exports successes
                "controller_completed": True,
                "source_hdf5": str(src),
                "source_demo": demo_name,
                "has_wrist_rgb": info["has_wrist_rgb"],
            }
            (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[hdf5_to_demos] {demo_name} -> ep_{ep_idx:03d}  ({info['num_frames']} frames)")

    print(f"[hdf5_to_demos] done: wrote {len(demo_names)} episodes to {out_root}")


if __name__ == "__main__":
    main()
