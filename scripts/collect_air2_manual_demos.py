"""Collect AIR2 manual pick-and-place demos with object hotkey annotations.

The output format matches scripts/collect_air2_demos.py so train_bc.py can use
manual and scripted demos through the same PNG/NPZ dataset path.

Controls:
  1/2/3/4     set current target: brush / pliers / scissors / screwdriver
  L           pause/resume recording frames
  Enter       accept and save current episode
  Backspace   discard current episode and reset
  R           reset current episode without saving
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Windows workaround: load h5py's bundled HDF5 DLLs before Isaac Sim extensions
# place older HDF5 DLLs on the process search path.
import h5py  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect AIR2 manual demos with object annotations.")
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Manual recording supports 1 env; kept for CLI parity.")
parser.add_argument("--teleop_device", default="keyboard", choices=["keyboard", "spacemouse", "gamepad", "handtracking"])
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--output", default="datasets/air2_manual_demos")
parser.add_argument("--save_every_n_steps", type=int, default=2)
parser.add_argument("--episode_length_s", type=float, default=60.0)
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--default_target", default=None,
                    choices=["brush", "pliers", "scissors", "screwdriver"],
                    help="Pre-select this object at the start of each episode. "
                         "Still overridable with 1/2/3/4 hotkeys.")
parser.add_argument("--hdf5_output", default=None,
                    help="If set, also save demos in HDF5 format (for Mimic pipeline) at this path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.utils.datasets.episode_data import EpisodeData

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka  # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.air2_franka.objects import (
    OBJECT_BY_CLASS_ID,
    OBJECT_BY_HOTKEY,
    OBJECT_BY_LABEL,
    catalog_json,
    target_one_hot,
)


# ---------------------------------------------------------------------------
# Mimic pose helpers (pure numpy — no Isaac Lab import needed at save time)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float32)


def _poses_from_pos_quat(positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    """Build (T, 4, 4) homogeneous matrices from (T, 3) positions and (T, 4) wxyz quats."""
    T = len(positions)
    mats = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for t in range(T):
        mats[t, :3, :3] = _quat_wxyz_to_matrix(quats_wxyz[t])
        mats[t, :3,  3] = positions[t]
    return mats


def _target_eef_poses(eef_pos: np.ndarray, eef_quat: np.ndarray,
                      actions: np.ndarray) -> np.ndarray:
    """Compute target EEF 4×4 matrices from current pose + raw IK delta action.

    Matches mimic_env.action_to_target_eef_pose: uses raw action (no IK scale).
    """
    from scipy.spatial.transform import Rotation
    T = len(actions)
    t_pos = eef_pos + actions[:, :3]           # (T, 3)
    t_mats = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for t in range(T):
        curr_rot = _quat_wxyz_to_matrix(eef_quat[t])
        rv = actions[t, 3:6]
        angle = float(np.linalg.norm(rv))
        if angle > 1e-6:
            delta_rot = Rotation.from_rotvec(rv).as_matrix().astype(np.float32)
        else:
            delta_rot = np.eye(3, dtype=np.float32)
        t_mats[t, :3, :3] = delta_rot @ curr_rot
        t_mats[t, :3,  3] = t_pos[t]
    return t_mats


class EpisodeBuffer:
    def __init__(self):
        self.wrist_rgb: list[np.ndarray] = []
        self.board_rgb: list[np.ndarray] = []
        self.joint_pos: list[np.ndarray] = []
        self.joint_vel: list[np.ndarray] = []
        self.eef_pos: list[np.ndarray] = []
        self.eef_quat: list[np.ndarray] = []
        self.object_position: list[np.ndarray] = []
        self.object_quat: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.gripper: list[float] = []
        self.target_class_id: list[int] = []
        self.target_one_hot: list[list[float]] = []
        self.target_scene_key: list[str] = []
        self.target_valid: list[bool] = []
        self.annotation_events: list[dict[str, object]] = []
        self.initial_state: dict | None = None  # scene state at episode start (for Mimic)

    def add_annotation_event(self, step: int, class_id: int) -> None:
        spec = OBJECT_BY_CLASS_ID.get(class_id)
        self.annotation_events.append(
            {
                "step": int(step),
                "class_id": int(class_id),
                "label": spec.label if spec else "unknown",
                "scene_key": spec.scene_key if spec else "",
            }
        )

    def append(self, wrist, board, jp, jv, act, grip, target_class_id: int,
               eef_pos=None, eef_quat=None, object_position=None, object_quat=None) -> None:
        self.wrist_rgb.append(wrist)
        self.board_rgb.append(board)
        self.joint_pos.append(jp)
        self.joint_vel.append(jv)
        if eef_pos is not None:
            self.eef_pos.append(eef_pos)
        if eef_quat is not None:
            self.eef_quat.append(eef_quat)
        if object_position is not None:
            self.object_position.append(object_position)
        if object_quat is not None:
            self.object_quat.append(object_quat)
        self.action.append(act)
        self.gripper.append(float(grip))
        self.target_class_id.append(int(target_class_id))
        self.target_one_hot.append(target_one_hot(int(target_class_id)))
        spec = OBJECT_BY_CLASS_ID.get(int(target_class_id))
        self.target_scene_key.append(spec.scene_key if spec else "")
        self.target_valid.append(spec is not None)

    def save(self, out_dir: Path, meta: dict[str, object], hdf5_handler: "HDF5DatasetFileHandler | None" = None) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.wrist_rgb:
            wrist_dir = out_dir / "wrist_rgb"
            board_dir = out_dir / "board_rgb"
            wrist_dir.mkdir(exist_ok=True)
            board_dir.mkdir(exist_ok=True)
            for t, img in enumerate(self.wrist_rgb):
                Image.fromarray(img).save(wrist_dir / f"t_{t:04d}.png")
            for t, img in enumerate(self.board_rgb):
                Image.fromarray(img).save(board_dir / f"t_{t:04d}.png")
        npz_kwargs = dict(
            joint_pos=np.array(self.joint_pos, dtype=np.float32),
            joint_vel=np.array(self.joint_vel, dtype=np.float32),
            action=np.array(self.action, dtype=np.float32),
            gripper=np.array(self.gripper, dtype=np.float32),
            target_class_id=np.array(self.target_class_id, dtype=np.int32),
            target_one_hot=np.array(self.target_one_hot, dtype=np.float32),
            target_scene_key=np.array(self.target_scene_key),
            target_valid=np.array(self.target_valid, dtype=np.bool_),
        )
        if self.eef_pos:
            npz_kwargs["eef_pos"] = np.array(self.eef_pos, dtype=np.float32)
        if self.eef_quat:
            npz_kwargs["eef_quat"] = np.array(self.eef_quat, dtype=np.float32)
        if self.object_position:
            npz_kwargs["object_position"] = np.array(self.object_position, dtype=np.float32)
        if self.object_quat:
            npz_kwargs["object_quat"] = np.array(self.object_quat, dtype=np.float32)
        np.savez_compressed(out_dir / "states.npz", **npz_kwargs)
        meta = {
            **meta,
            "num_frames": len(self.action),
            "unannotated_frames": int(np.count_nonzero(np.logical_not(self.target_valid))),
            "annotation_events": self.annotation_events,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Write Mimic-compatible HDF5 if handler provided.
        if hdf5_handler is not None and self.eef_pos and self.object_quat:
            act_arr  = np.array(self.action,          dtype=np.float32)
            ep_arr   = np.array(self.eef_pos,         dtype=np.float32)
            eq_arr   = np.array(self.eef_quat,        dtype=np.float32)
            op_arr   = np.array(self.object_position, dtype=np.float32)
            oq_arr   = np.array(self.object_quat,     dtype=np.float32)
            jp_arr   = np.array(self.joint_pos,       dtype=np.float32)

            # 4×4 pose matrices for datagen_info
            eef_poses    = _poses_from_pos_quat(ep_arr, eq_arr)         # (T, 4, 4)
            obj_poses    = _poses_from_pos_quat(op_arr, oq_arr)         # (T, 4, 4)
            tgt_poses    = _target_eef_poses(ep_arr, eq_arr, act_arr)   # (T, 4, 4)

            # Grasp signal: EEF within 15 cm of object AND fingers mostly closed
            eef_obj_dist = np.linalg.norm(ep_arr - op_arr, axis=-1)
            proximity    = eef_obj_dist < 0.15
            fingers_ok   = (jp_arr[:, 7] < 0.07) & (jp_arr[:, 8] < 0.07)
            grasp_signal = (proximity & fingers_ok).astype(np.float32)
            # Ensure at least one 0→1 transition so Mimic annotation doesn't fail.
            if grasp_signal.max() < 0.5:
                print("[manual] WARNING: no grasp detected — forcing signal at last frame", flush=True)
                grasp_signal[-1] = 1.0

            ep = EpisodeData()
            ep.success = True
            if self.initial_state is not None:
                ep.add("initial_state", self.initial_state)
            actions_t = torch.tensor(act_arr)
            T = len(self.action)
            for i in range(T):
                ep.add("actions",                                           actions_t[i])
                ep.add("obs/joint_pos",                                     torch.tensor(jp_arr[i]))
                ep.add("obs/joint_vel",                                     torch.tensor(self.joint_vel[i]))
                ep.add("obs/eef_pos",                                       torch.tensor(ep_arr[i]))
                ep.add("obs/eef_quat",                                      torch.tensor(eq_arr[i]))
                ep.add("obs/object_position",                               torch.tensor(op_arr[i]))
                ep.add("obs/datagen_info/eef_pose/franka",                  torch.tensor(eef_poses[i]))
                ep.add("obs/datagen_info/object_pose/object",               torch.tensor(obj_poses[i]))
                ep.add("obs/datagen_info/target_eef_pose/franka",           torch.tensor(tgt_poses[i]))
                ep.add("obs/datagen_info/subtask_term_signals/grasp_brush", torch.tensor([grasp_signal[i]]))
            hdf5_handler.write_episode(ep)
            hdf5_handler.flush()

        return len(self.action)


def make_teleop_interface(env_cfg, callbacks):
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    sensitivity = args_cli.sensitivity
    if args_cli.teleop_device == "keyboard":
        interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
        # Remap WASD: D=-X, A=+X, W=-Y, S=+Y (rotated 90° from Isaac Lab default, negated)
        s = 0.05 * sensitivity
        interface._INPUT_KEY_MAPPING["W"] = np.asarray([0.0, -1.0, 0.0]) * s
        interface._INPUT_KEY_MAPPING["S"] = np.asarray([0.0,  1.0, 0.0]) * s
        interface._INPUT_KEY_MAPPING["A"] = np.asarray([1.0,  0.0, 0.0]) * s
        interface._INPUT_KEY_MAPPING["D"] = np.asarray([-1.0, 0.0, 0.0]) * s
        # C/V yaw at 1.5x rotation sensitivity
        r = 0.05 * sensitivity * 1.5
        interface._INPUT_KEY_MAPPING["C"] = np.asarray([0.0, 0.0,  1.0]) * r
        interface._INPUT_KEY_MAPPING["V"] = np.asarray([0.0, 0.0, -1.0]) * r
    elif args_cli.teleop_device == "spacemouse":
        interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
    elif args_cli.teleop_device == "gamepad":
        interface = Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity))
    else:
        raise RuntimeError(
            "handtracking requires an environment teleop_devices config. "
            "Use keyboard/spacemouse/gamepad for this AIR2 task unless handtracking is added to the env cfg."
        )
    for key, callback in callbacks.items():
        try:
            interface.add_callback(key, callback)
        except Exception:
            pass
    return interface


def subscribe_global_keyboard(callback_by_name: dict[str, object]):
    """Subscribe to app-window keyboard events when available.

    This lets number-key annotations work while the robot is driven by a
    spacemouse or gamepad. If the API is unavailable, teleop-device callbacks
    still cover keyboard teleop.
    """
    try:
        import carb.input
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()

        def on_keyboard_event(event, *args, **kwargs):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True
            raw_name = str(getattr(event.input, "name", event.input)).upper()
            key_name = raw_name.replace("KEY_", "").replace("NUMPAD_", "")
            callback = callback_by_name.get(key_name)
            if callback is not None:
                callback()
            return True

        return input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    except Exception as exc:
        print(f"[manual] global keyboard annotation unavailable: {exc}", flush=True)
        return None


def main() -> None:
    out_root = Path(args_cli.output)
    out_root.mkdir(parents=True, exist_ok=True)
    if args_cli.num_envs != 1:
        raise ValueError("collect_air2_manual_demos.py records one manual environment; use --num_envs 1.")

    default_class_id = 0
    if args_cli.default_target:
        spec = OBJECT_BY_LABEL[args_cli.default_target]
        default_class_id = spec.class_id
        print(f"[manual] default target locked to: {spec.label} (class_id={spec.class_id})", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    device = env.device

    state = {
        "recording": True,
        "accept": False,
        "discard": False,
        "reset": False,
        "target_class_id": default_class_id,
        "default_target_class_id": default_class_id,
        "step": 0,
    }
    buffer = EpisodeBuffer()

    def set_target(hotkey: str):
        spec = OBJECT_BY_HOTKEY[hotkey]
        state["target_class_id"] = spec.class_id
        buffer.add_annotation_event(state["step"], spec.class_id)
        print(f"[manual] target={spec.label} class_id={spec.class_id}", flush=True)

    def toggle_recording():
        state["recording"] = not bool(state["recording"])
        print(f"[manual] recording={'on' if state['recording'] else 'paused'}", flush=True)

    callbacks = {
        "1": lambda: set_target("1"),
        "2": lambda: set_target("2"),
        "3": lambda: set_target("3"),
        "4": lambda: set_target("4"),
        "L": toggle_recording,
        "ENTER": lambda: state.__setitem__("accept", True),
        "RETURN": lambda: state.__setitem__("accept", True),
        "BACKSPACE": lambda: state.__setitem__("discard", True),
        "R": lambda: state.__setitem__("reset", True),
        "RESET": lambda: state.__setitem__("reset", True),
    }
    teleop_interface = make_teleop_interface(env_cfg, callbacks)
    keyboard_subscription = subscribe_global_keyboard(callbacks)
    teleop_interface.reset()

    action = torch.zeros(env.action_space.shape, device=device)
    existing = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("ep_")]
    saved = max((int(p.name.split("_")[1]) + 1 for p in existing), default=0)
    if saved > 0:
        print(f"[manual] Resuming from ep_{saved:03d} ({len(existing)} existing episodes)", flush=True)
    print("[manual] Press 1/2/3/4 to annotate the object, Enter to save, Backspace to discard.", flush=True)

    # HDF5 handler for Mimic pipeline (optional).
    hdf5_handler = None
    if args_cli.hdf5_output:
        hdf5_path = Path(args_cli.hdf5_output)
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        hdf5_handler = HDF5DatasetFileHandler()
        if hdf5_path.exists():
            hdf5_handler.open(str(hdf5_path), mode="a")
            print(f"[manual] appending to HDF5: {hdf5_path} ({hdf5_handler.get_num_episodes()} existing)", flush=True)
        else:
            hdf5_handler.create(str(hdf5_path), env_name=args_cli.task)
            print(f"[manual] creating HDF5: {hdf5_path}", flush=True)

    # Check which cameras are available.
    has_cameras = "wrist_camera" in env.scene.sensors and "board_camera" in env.scene.sensors
    first_step_of_episode = True

    while saved < args_cli.num_demos and simulation_app.is_running():
        with torch.inference_mode():
            action_cmd = teleop_interface.advance()
            action[:] = action_cmd.repeat(env.num_envs, 1)
            env.step(action)
            state["step"] += 1

            # Capture initial scene state for Mimic after first physics step.
            if first_step_of_episode and hdf5_handler is not None:
                buffer.initial_state = env.scene.get_state(is_relative=True)
                first_step_of_episode = False

            if state["recording"] and state["step"] % args_cli.save_every_n_steps == 0:
                wrist = env.scene["wrist_camera"].data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8) if has_cameras else np.zeros((1,1,3), dtype=np.uint8)
                board = env.scene["board_camera"].data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8) if has_cameras else np.zeros((1,1,3), dtype=np.uint8)
                jp = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
                jv = env.scene["robot"].data.joint_vel[0].detach().cpu().numpy()
                act = action[0].detach().cpu().numpy()

                # Extra fields for Mimic.
                eef_pos  = env.scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
                eef_quat = env.scene["ee_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
                obj_pos  = env.scene["object"].data.root_pos_w[0].detach().cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
                obj_quat = env.scene["object"].data.root_quat_w[0].detach().cpu().numpy()

                buffer.append(wrist, board, jp, jv, act, float(act[6]), int(state["target_class_id"]),
                               eef_pos=eef_pos, eef_quat=eef_quat,
                               object_position=obj_pos, object_quat=obj_quat)

            if state["accept"]:
                if not buffer.action:
                    print("[manual] no recorded frames yet; press L to record or wait a moment before accepting.", flush=True)
                    state["accept"] = False
                    continue
                if not any(buffer.target_valid):
                    print("[manual] no target labels recorded; press 1/2/3/4 before accepting.", flush=True)
                    state["accept"] = False
                    continue
                ep_dir = out_root / f"ep_{saved:03d}"
                n_frames = buffer.save(
                    ep_dir,
                    {
                        "task": args_cli.task,
                        "teleop_device": args_cli.teleop_device,
                        "success": True,
                        "object_catalog": catalog_json(),
                    },
                    hdf5_handler=hdf5_handler,
                )
                print(f"[manual] saved ep_{saved:03d} ({n_frames} frames)", flush=True)
                saved += 1
                state["accept"] = False
                state["reset"] = True

            if state["discard"]:
                print("[manual] discarded current episode", flush=True)
                state["discard"] = False
                state["reset"] = True

            if state["reset"]:
                buffer = EpisodeBuffer()
                state["target_class_id"] = state["default_target_class_id"]
                state["step"] = 0
                first_step_of_episode = True
                env.reset()
                teleop_interface.reset()
                state["reset"] = False
                print("[manual] reset. Select target with 1/2/3/4.", flush=True)

    if keyboard_subscription is not None:
        del keyboard_subscription
    env.close()
    print(f"[manual] done. saved {saved} demos to {out_root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
