"""Collect AIR2 pick-and-place demos using a scripted waypoint controller.

For each episode:
  - Reset the env (objects randomly placed on 4 of 8 hooks)
  - Run a per-env waypoint controller that picks each of the 4 objects in turn
    and drops it into the basket
  - At every env step, record (wrist_rgb, board_rgb, joint_pos, joint_vel,
    action, state) to memory
  - At episode end, dump frames as PNGs + state/action arrays as .npz

Output layout:
  datasets/air2_demos/
    ep_000/
      meta.json
      states.npz                  # joint_pos, joint_vel, action, gripper, waypoint_idx
      wrist_rgb/t_0000.png ...
      board_rgb/t_0000.png ...
    ep_001/ ...

No HDF5 (avoids the h5py DLL conflict on Windows). Convert to HDF5 later if
robomimic-style training is wanted.

Usage (Windows):
  conda deactivate
  C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/collect_air2_demos.py ^
      --task Isaac-Lift-AIR2-UR3e-RG2-Play-v0 ^
      --enable_cameras --headless --num_envs 4 --num_episodes 20 ^
      --output C:\\Users\\Administrator\\Desktop\\air2_demos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Windows workaround: load h5py's bundled HDF5 DLLs before Isaac Sim extensions.
import h5py  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect AIR2 demos via scripted controller.")
parser.add_argument("--task", default="Isaac-Lift-AIR2-Robotis-Segmentation-Play-v0",
                    help="Segmentation variants have both wrist_camera + board_camera; non-segmentation only have wrist_camera.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20, help="Total episodes to collect across all envs.")
parser.add_argument("--max_episode_steps", type=int, default=1500, help="Hard cap on steps per episode.")
parser.add_argument("--episode_length_s", type=float, default=30.0,
                    help="Override env's episode_length_s. Default Lift env is ~5s which is too short for 4-object pick-place.")
parser.add_argument("--output", default="datasets/air2_demos")
parser.add_argument("--save_every_n_steps", type=int, default=2, help="Save every Nth frame to disk (1=all frames).")
parser.add_argument("--single_object", action="store_true",
                    help="Per-episode, scripted controller picks ONE random object instead of all 4. "
                         "Produces cleaner single-pick-place demos for BC training.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import gymnasium as gym
from PIL import Image

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
import isaaclab_ext.tasks.lift_air2_robotis  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.scripted_controller import (
    PickPlaceController, OBJECT_NAMES,
)
from isaaclab_ext.tasks.lift_air2_ur3e_rg2.objects import (
    OBJECT_BY_CLASS_ID, catalog_json, target_one_hot,
)


# -- per-env episode buffer -------------------------------------------------

class EpisodeBuffer:
    """In-memory record of a single env's episode."""

    def __init__(self):
        self.wrist_rgb: list[np.ndarray] = []
        self.board_rgb: list[np.ndarray] = []
        self.joint_pos: list[np.ndarray] = []
        self.joint_vel: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.gripper: list[float] = []
        self.wp_idx: list[int] = []
        self.target_class_id: list[int] = []
        self.target_one_hot: list[list[float]] = []
        self.target_scene_key: list[str] = []
        self.target_valid: list[bool] = []

    def append(self, wrist, board, jp, jv, act, grip, wp, target_class_id):
        if wrist is not None:
            self.wrist_rgb.append(wrist)
        if board is not None:
            self.board_rgb.append(board)
        self.joint_pos.append(jp)
        self.joint_vel.append(jv)
        self.action.append(act)
        self.gripper.append(grip)
        self.wp_idx.append(wp)
        self.target_class_id.append(int(target_class_id))
        self.target_one_hot.append(target_one_hot(int(target_class_id)))
        spec = OBJECT_BY_CLASS_ID.get(int(target_class_id))
        self.target_scene_key.append(spec.scene_key if spec else "")
        self.target_valid.append(spec is not None)

    def save(self, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir = out_dir / "wrist_rgb"; wrist_dir.mkdir(exist_ok=True)
        board_dir = out_dir / "board_rgb"; board_dir.mkdir(exist_ok=True)
        for t, img in enumerate(self.wrist_rgb):
            Image.fromarray(img).save(wrist_dir / f"t_{t:04d}.png")
        for t, img in enumerate(self.board_rgb):
            Image.fromarray(img).save(board_dir / f"t_{t:04d}.png")
        np.savez_compressed(
            out_dir / "states.npz",
            joint_pos=np.array(self.joint_pos, dtype=np.float32),
            joint_vel=np.array(self.joint_vel, dtype=np.float32),
            action=np.array(self.action, dtype=np.float32),
            gripper=np.array(self.gripper, dtype=np.float32),
            wp_idx=np.array(self.wp_idx, dtype=np.int32),
            target_class_id=np.array(self.target_class_id, dtype=np.int32),
            target_one_hot=np.array(self.target_one_hot, dtype=np.float32),
            target_scene_key=np.array(self.target_scene_key),
            target_valid=np.array(self.target_valid, dtype=np.bool_),
        )
        return len(self.action)


# -- main loop --------------------------------------------------------------

def to_local(world_pos: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    """Convert world-frame positions (num_envs, 3) to env-local."""
    return world_pos - env_origins


def main():
    out_root = Path(args_cli.output)
    out_root.mkdir(parents=True, exist_ok=True)
    heartbeat_path = out_root / "_heartbeat.log"
    def heartbeat(msg: str):
        with open(heartbeat_path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)
    heartbeat(f"[collect] output dir: {out_root}")

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    heartbeat(f"[collect] episode_length_s = {env_cfg.episode_length_s} (override)")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    heartbeat(f"[collect] launched {args_cli.task} with {num_envs} envs")
    heartbeat(f"[collect] action shape: {env.unwrapped.action_space.shape}")
    heartbeat(f"[collect] scene keys: {list(env.unwrapped.scene.keys())}")

    controller = PickPlaceController(num_envs, device, single_object=args_cli.single_object)
    if args_cli.single_object:
        heartbeat("[collect] single_object mode: each episode picks ONE random object")
    action = torch.zeros(env.unwrapped.action_space.shape, device=device)

    saved_episodes = 0
    episode_buffers: list[EpisodeBuffer] = [EpisodeBuffer() for _ in range(num_envs)]
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def collect_object_positions() -> list[list[torch.Tensor]]:
        per_env = []
        origins = env.unwrapped.scene.env_origins  # (num_envs, 3)
        for env_idx in range(num_envs):
            objs = []
            for name in OBJECT_NAMES:
                w = env.unwrapped.scene[name].data.root_pos_w[env_idx]
                objs.append((w - origins[env_idx]).cpu())
            per_env.append(objs)
        return per_env

    controller.reset(collect_object_positions())

    heartbeat("[collect] entering main loop")
    step_counter = 0
    while saved_episodes < args_cli.num_episodes and simulation_app.is_running():
        step_counter += 1
        if step_counter % 50 == 0:
            heartbeat(f"[collect] step {step_counter}, saved {saved_episodes} eps, controller.done={controller.done.tolist()}, wp_idx={controller.wp_idx.tolist()}")
        with torch.inference_mode():
            # Current EE pose (frame transformer publishes target_pos_w)
            ee = env.unwrapped.scene["ee_frame"]
            ee_world = ee.data.target_pos_w[..., 0, :]
            origins = env.unwrapped.scene.env_origins
            ee_local = ee_world - origins

            target_ids = controller.current_class_ids()
            delta_pos, grip = controller.step(ee_local)

            # Build action: AIR2 uses IK-Rel pose (6) + binary gripper (1).
            #   action[:, 0:3] = position delta
            #   action[:, 3:6] = axis-angle rotation delta (zero — keep current orientation)
            #   action[:, 6]   = gripper command (+1 open, -1 close)
            action.zero_()
            action[:, 0:3] = delta_pos
            action[:, 6]   = grip

            obs, rew, term, trunc, info = env.step(action)
            episode_step_count += 1

            # Capture cameras + state per env every N steps
            do_save = ((episode_step_count % args_cli.save_every_n_steps) == 0).cpu()
            wrist_buf = env.unwrapped.scene["wrist_camera"].data.output["rgb"].cpu().numpy()
            board_buf = env.unwrapped.scene["board_camera"].data.output["rgb"].cpu().numpy()
            jp_buf = env.unwrapped.scene["robot"].data.joint_pos.cpu().numpy()
            jv_buf = env.unwrapped.scene["robot"].data.joint_vel.cpu().numpy()
            act_np = action.cpu().numpy()
            grip_np = grip.cpu().numpy()
            wp_np = controller.wp_idx.cpu().numpy()
            target_np = target_ids.cpu().numpy()
            done_np = controller.done.cpu().numpy()
            for i in range(num_envs):
                if done_np[i] or not bool(do_save[i].item()):
                    if not done_np[i]:
                        # still record state vectors even on skipped frames? No, keep aligned.
                        continue
                    continue
                episode_buffers[i].append(
                    wrist_buf[i], board_buf[i], jp_buf[i], jv_buf[i], act_np[i],
                    float(grip_np[i]), int(wp_np[i]), int(target_np[i])
                )

            # Episode termination: any env done OR all controllers done OR cap hit
            cap_hit = (episode_step_count >= args_cli.max_episode_steps)
            episode_finished = controller.done | term | trunc | cap_hit

            if episode_finished.any():
                finished_idx = episode_finished.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                for i in finished_idx:
                    if saved_episodes >= args_cli.num_episodes:
                        break
                    ep_dir = out_root / f"ep_{saved_episodes:03d}"
                    n_frames = episode_buffers[i].save(ep_dir)
                    success = bool(controller.done[i].item()) and not bool(cap_hit[i].item())
                    meta = {
                        "task": args_cli.task,
                        "env_idx_in_batch": i,
                        "num_frames": n_frames,
                        "success": success,
                        "controller_completed": bool(controller.done[i].item()),
                        "object_count": len(OBJECT_NAMES),
                        "waypoints_per_object": 7,
                        "object_catalog": catalog_json(),
                    }
                    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                    print(f"[collect] saved ep_{saved_episodes:03d} ({n_frames} frames, success={success})", flush=True)
                    saved_episodes += 1
                    episode_buffers[i] = EpisodeBuffer()
                    episode_step_count[i] = 0

                # Reset env + controller for finished envs only
                # (Isaac Lab auto-resets on done flags, so we just rebuild waypoints next iter)
                controller.reset(collect_object_positions())

    print(f"[collect] done — saved {saved_episodes} episodes to {out_root}", flush=True)
    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        err_path = Path(args_cli.output) / "_error.log"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with open(err_path, "w") as f:
            f.write(f"{e!r}\n\n{traceback.format_exc()}")
        print(f"[collect] ERROR — see {err_path}", flush=True)
        raise
    finally:
        simulation_app.close()
