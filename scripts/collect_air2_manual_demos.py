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
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Windows workaround: load h5py's bundled HDF5 DLLs before Isaac Sim extensions
# place older HDF5 DLLs on the process search path.
import h5py  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect AIR2 manual demos with object annotations.")
parser.add_argument("--task", default="Isaac-Lift-AIR2-Robotis-Segmentation-Play-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Manual recording supports 1 env; kept for CLI parity.")
parser.add_argument("--teleop_device", default="keyboard", choices=["keyboard", "spacemouse", "gamepad", "handtracking"])
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--output", default="datasets/air2_manual_demos")
parser.add_argument("--save_every_n_steps", type=int, default=2)
parser.add_argument("--episode_length_s", type=float, default=60.0)
parser.add_argument("--sensitivity", type=float, default=1.0)
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

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.lift_air2_ur3e_rg2  # noqa: F401
import isaaclab_ext.tasks.lift_air2_robotis  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.objects import (
    OBJECT_BY_CLASS_ID,
    OBJECT_BY_HOTKEY,
    catalog_json,
    target_one_hot,
)


class EpisodeBuffer:
    def __init__(self):
        self.wrist_rgb: list[np.ndarray] = []
        self.board_rgb: list[np.ndarray] = []
        self.joint_pos: list[np.ndarray] = []
        self.joint_vel: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.gripper: list[float] = []
        self.target_class_id: list[int] = []
        self.target_one_hot: list[list[float]] = []
        self.target_scene_key: list[str] = []
        self.target_valid: list[bool] = []
        self.annotation_events: list[dict[str, object]] = []

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

    def append(self, wrist, board, jp, jv, act, grip, target_class_id: int) -> None:
        self.wrist_rgb.append(wrist)
        self.board_rgb.append(board)
        self.joint_pos.append(jp)
        self.joint_vel.append(jv)
        self.action.append(act)
        self.gripper.append(float(grip))
        self.target_class_id.append(int(target_class_id))
        self.target_one_hot.append(target_one_hot(int(target_class_id)))
        spec = OBJECT_BY_CLASS_ID.get(int(target_class_id))
        self.target_scene_key.append(spec.scene_key if spec else "")
        self.target_valid.append(spec is not None)

    def save(self, out_dir: Path, meta: dict[str, object]) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir = out_dir / "wrist_rgb"
        board_dir = out_dir / "board_rgb"
        wrist_dir.mkdir(exist_ok=True)
        board_dir.mkdir(exist_ok=True)
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
            target_class_id=np.array(self.target_class_id, dtype=np.int32),
            target_one_hot=np.array(self.target_one_hot, dtype=np.float32),
            target_scene_key=np.array(self.target_scene_key),
            target_valid=np.array(self.target_valid, dtype=np.bool_),
        )
        meta = {
            **meta,
            "num_frames": len(self.action),
            "unannotated_frames": int(np.count_nonzero(np.logical_not(self.target_valid))),
            "annotation_events": self.annotation_events,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return len(self.action)


def make_teleop_interface(env_cfg, callbacks):
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    sensitivity = args_cli.sensitivity
    if args_cli.teleop_device == "keyboard":
        interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
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


def subscribe_global_keyboard(callback_by_name: dict[str, object], allowed_keys: set[str] | None = None):
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
            if key_name.startswith("KEYBOARDINPUT."):
                key_name = key_name.split(".", 1)[1]
            if allowed_keys is not None and key_name not in allowed_keys:
                return True
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
        "target_class_id": 0,
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

    last_callback_time: dict[str, float] = {}

    def once(key: str, func, cooldown_s: float = 0.25):
        now = time.monotonic()
        if now - last_callback_time.get(key, 0.0) < cooldown_s:
            return
        last_callback_time[key] = now
        func()

    callbacks = {
        "1": lambda: once("1", lambda: set_target("1")),
        "KEY_1": lambda: once("1", lambda: set_target("1")),
        "NUMPAD_1": lambda: once("1", lambda: set_target("1")),
        "2": lambda: once("2", lambda: set_target("2")),
        "KEY_2": lambda: once("2", lambda: set_target("2")),
        "NUMPAD_2": lambda: once("2", lambda: set_target("2")),
        "3": lambda: once("3", lambda: set_target("3")),
        "KEY_3": lambda: once("3", lambda: set_target("3")),
        "NUMPAD_3": lambda: once("3", lambda: set_target("3")),
        "4": lambda: once("4", lambda: set_target("4")),
        "KEY_4": lambda: once("4", lambda: set_target("4")),
        "NUMPAD_4": lambda: once("4", lambda: set_target("4")),
        "L": lambda: once("L", toggle_recording),
        "ENTER": lambda: once("ENTER", lambda: state.__setitem__("accept", True)),
        "RETURN": lambda: once("ENTER", lambda: state.__setitem__("accept", True)),
        "BACKSPACE": lambda: once("BACKSPACE", lambda: state.__setitem__("discard", True)),
        "R": lambda: once("R", lambda: state.__setitem__("reset", True)),
        "RESET": lambda: once("R", lambda: state.__setitem__("reset", True)),
    }
    teleop_interface = make_teleop_interface(env_cfg, callbacks)
    # Se3Keyboard does not consistently expose number keys as additional
    # callbacks, so use a global listener only for annotation/accept/discard
    # keys. Movement keys plus L/R remain owned by the teleop device.
    keyboard_subscription = subscribe_global_keyboard(
        callbacks,
        allowed_keys={"1", "2", "3", "4", "ENTER", "RETURN", "BACKSPACE"},
    )
    teleop_interface.reset()

    action = torch.zeros(env.action_space.shape, device=device)
    saved = 0
    print("[manual] Press 1/2/3/4 to annotate the object, Enter to save, Backspace to discard.", flush=True)

    while saved < args_cli.num_demos and simulation_app.is_running():
        with torch.inference_mode():
            action_cmd = teleop_interface.advance()
            action[:] = action_cmd.repeat(env.num_envs, 1)
            env.step(action)
            state["step"] += 1

            if state["recording"] and state["step"] % args_cli.save_every_n_steps == 0:
                wrist = env.scene["wrist_camera"].data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
                board = env.scene["board_camera"].data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
                jp = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
                jv = env.scene["robot"].data.joint_vel[0].detach().cpu().numpy()
                act = action[0].detach().cpu().numpy()
                buffer.append(wrist, board, jp, jv, act, float(act[6]), int(state["target_class_id"]))

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
                state["target_class_id"] = 0
                state["step"] = 0
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
