# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to record teleoperated demos and run mimic dataset generation in real-time.
"""

# Launching Isaac Sim Simulator first.

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Record demonstrations and run mimic dataset generation for Isaac Lab environments."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=5,
    help=(
        "Number of environments to instantiate to test recording and generating datasets. The environment specified by"
        " `teleop_env_index` will be used for teleoperation and recording while the remaining environments will be used"
        " for real-time data generation. Default is 5."
    ),
)
parser.add_argument(
    "--teleop_env_index",
    type=int,
    default=0,
    help="Index of the environment to be used for teleoperation. Set -1 for disabling the teleop robot. Default is 0.",
)
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--step_hz", type=int, default=0, help="Environment stepping rate in Hz. Set to 0 for maximum speed."
)
parser.add_argument("--input_file", type=str, default=None, help="File path to the source demo dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded episodes.",
)
parser.add_argument(
    "--generated_output_file",
    type=str,
    default=None,
    help="File path to export generated episodes by mimic.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import contextlib
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka  # noqa: F401 — register our task IDs
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# global variable to keep track of the data generation statistics
num_recorded = 0
num_success = 0
num_failures = 0
num_attempts = 0
_prev_grip_sign = [None]
_current_demo_grip_events = []


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.unwrapped.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


async def run_teleop_robot(
    env, env_id, env_action_queue, shared_datagen_info_pool, success_term, exported_dataset_path, teleop_interface=None
):
    """Run teleop robot."""
    global num_recorded, _prev_grip_sign, _current_demo_grip_events
    should_reset_teleop_instance = False
    should_accept_teleop_instance = False
    # create controller if needed
    if teleop_interface is None:
        if args_cli.teleop_device.lower() == "keyboard":
            s, r = 0.1, 0.15  # pos and yaw sensitivity matching collect_air2_manual_demos.py
            teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=s, rot_sensitivity=s))
            teleop_interface._INPUT_KEY_MAPPING["W"] = np.asarray([0.0, -1.0, 0.0]) * s
            teleop_interface._INPUT_KEY_MAPPING["S"] = np.asarray([0.0,  1.0, 0.0]) * s
            teleop_interface._INPUT_KEY_MAPPING["A"] = np.asarray([1.0,  0.0, 0.0]) * s
            teleop_interface._INPUT_KEY_MAPPING["D"] = np.asarray([-1.0, 0.0, 0.0]) * s
            teleop_interface._INPUT_KEY_MAPPING["C"] = np.asarray([0.0, 0.0,  1.0]) * r
            teleop_interface._INPUT_KEY_MAPPING["V"] = np.asarray([0.0, 0.0, -1.0]) * r
        elif args_cli.teleop_device.lower() == "spacemouse":
            teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
        else:
            raise ValueError(
                f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'."
            )

    def reset_teleop_instance():
        nonlocal should_reset_teleop_instance
        should_reset_teleop_instance = True

    def accept_teleop_instance():
        nonlocal should_accept_teleop_instance
        should_accept_teleop_instance = True

    teleop_interface.add_callback("R", reset_teleop_instance)
    teleop_interface.add_callback("ENTER", accept_teleop_instance)

    teleop_interface.reset()
    print(teleop_interface)
    print("[collect] ENTER = save demo | R = discard and reset", flush=True)

    recorded_episode_dataset_file_handler = HDF5DatasetFileHandler()
    recorded_episode_dataset_file_handler.create(exported_dataset_path, env_name=env.unwrapped.cfg.env_name)
    _grip_log_path = exported_dataset_path.replace(".hdf5", "_gripper_log.txt")
    _grip_log = open(_grip_log_path, "w")

    env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=env.unwrapped.device)
    num_recorded = 0
    while True:
        if should_reset_teleop_instance:
            env.unwrapped.recorder_manager.reset(env_id_tensor)
            env.unwrapped.reset(env_ids=env_id_tensor)
            should_reset_teleop_instance = False
            _prev_grip_sign[0] = None
            _current_demo_grip_events.clear()

        if should_accept_teleop_instance:
            should_accept_teleop_instance = False
            env.unwrapped.recorder_manager.set_success_to_episodes(
                env_id_tensor, torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
            )
            teleop_episode = env.unwrapped.recorder_manager.get_episode(env_id)
            teleop_episode.pre_export()
            recorded_episode_dataset_file_handler.write_episode(teleop_episode)
            recorded_episode_dataset_file_handler.flush()
            env.unwrapped.recorder_manager.reset(env_id_tensor)
            num_recorded += 1
            _grip_log.write(f"=== demo {num_recorded} ===\n")
            for event in _current_demo_grip_events:
                _grip_log.write(event + "\n")
            _grip_log.flush()
            _prev_grip_sign[0] = None
            _current_demo_grip_events.clear()
            should_reset_teleop_instance = True
            print(f"[collect] saved {num_recorded} / {args_cli.num_demos}", flush=True)

        # get keyboard command — advance() returns a single (1, action_dim) tensor in IsaacLab 2.3.2
        action_cmd = teleop_interface.advance()
        teleop_action = action_cmd.repeat(env.unwrapped.num_envs, 1).to(env.unwrapped.device)

        await env_action_queue.put((env_id, teleop_action))
        await env_action_queue.join()


async def run_data_generator(
    env, env_id, env_action_queue, shared_datagen_info_pool, success_term, pause_subtask=False, export_demo=True
):
    """Run data generator."""
    global num_success, num_failures, num_attempts
    data_generator = DataGenerator(env=env.unwrapped, src_demo_datagen_info_pool=shared_datagen_info_pool)
    idle_action = torch.zeros(env.unwrapped.action_space.shape)[0]
    while True:
        while data_generator.src_demo_datagen_info_pool.num_datagen_infos < 1:
            await env_action_queue.put((env_id, idle_action))
            await env_action_queue.join()

        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_action_queue=env_action_queue,
            select_src_per_subtask=env.unwrapped.cfg.datagen_config.generation_select_src_per_subtask,
            transform_first_robot_pose=env.unwrapped.cfg.datagen_config.generation_transform_first_robot_pose,
            interpolate_from_last_target_pose=env.unwrapped.cfg.datagen_config.generation_interpolate_from_last_target_pose,
            pause_subtask=pause_subtask,
            export_demo=export_demo,
        )
        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1


def env_loop(env, env_action_queue, shared_datagen_info_pool, asyncio_event_loop):
    """Main loop for the environment."""
    global num_recorded, num_success, num_failures, num_attempts
    prev_num_attempts = 0
    prev_num_recorded = 0

    rate_limiter = None
    if args_cli.step_hz > 0:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # simulate environment -- run everything in inference mode
    is_first_print = True
    _basket_printed = False
    global _prev_grip_sign, _current_demo_grip_events
    _prev_grip_sign = [None] * env.unwrapped.num_envs
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            actions = torch.zeros(env.unwrapped.action_space.shape)

            # get actions from all the data generators
            for i in range(env.unwrapped.num_envs):
                # an async-blocking call to get an action from a data generator
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            # perform action on environment
            env.step(actions)

            # One-shot: print basket prim world position so we can update BASKET_POS_LOCAL
            if not _basket_printed:
                try:
                    from pxr import UsdGeom, Usd
                    stage = env.unwrapped.sim.stage
                    origin = env.unwrapped.scene.env_origins[0].cpu()
                    for prim in stage.Traverse():
                        if "sm_boxportabled" in prim.GetName().lower():
                            t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                                Usd.TimeCode.Default()).ExtractTranslation()
                            local = (t[0] - float(origin[0]), t[1] - float(origin[1]), t[2] - float(origin[2]))
                            msg = (f"[basket] world=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})  "
                                   f"env-local=({local[0]:.4f}, {local[1]:.4f}, {local[2]:.4f})")
                            print(msg, flush=True)
                            break
                except Exception as e:
                    print(f"[basket] could not read prim: {e}", flush=True)
                _basket_printed = True

            # Accumulate EE + object position on gripper open/close transitions (env 0 only)
            grip = float(actions[0, 6].item())
            grip_sign = 1 if grip >= 0 else -1
            if _prev_grip_sign[0] is not None and grip_sign != _prev_grip_sign[0]:
                origin = env.unwrapped.scene.env_origins[0].cpu()
                ee_w  = env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0].cpu()
                obj_w = env.unwrapped.scene["object"].data.root_pos_w[0].cpu()
                ee_local  = (ee_w - origin).tolist()
                obj_local = (obj_w - origin).tolist()
                state_str = "CLOSE" if grip_sign < 0 else "OPEN"
                msg = (f"[gripper→{state_str}]  "
                       f"ee=({ee_local[0]:.3f}, {ee_local[1]:.3f}, {ee_local[2]:.3f})  "
                       f"obj=({obj_local[0]:.3f}, {obj_local[1]:.3f}, {obj_local[2]:.3f})")
                print(msg, flush=True)
                _current_demo_grip_events.append(msg)
            _prev_grip_sign[0] = grip_sign

            # mark done so the data generators can continue with the step results
            for i in range(env.unwrapped.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != num_attempts or prev_num_recorded != num_recorded:
                prev_num_attempts = num_attempts
                prev_num_recorded = num_recorded
                generated_sucess_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
                if is_first_print:
                    is_first_print = False
                else:
                    print("\r", "\033[F" * 5, end="")
                print("")
                print("*" * 50, "\033[K")
                print(f"{num_recorded} teleoperated demos recorded\033[K")
                print(
                    f"{num_success}/{num_attempts} ({generated_sucess_rate:.1f}%) successful demos generated by"
                    " mimic\033[K"
                )
                print("*" * 50, "\033[K")

                if args_cli.num_demos > 0 and num_recorded >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env.unwrapped)
    env.close()


def main():
    num_envs = args_cli.num_envs

    # create output directory for recorded episodes if it does not exist
    recorded_output_dir = os.path.dirname(args_cli.output_file)
    if not os.path.exists(recorded_output_dir):
        os.makedirs(recorded_output_dir)

    # check if the given input dataset file exists
    if args_cli.input_file and not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")

    # get the environment name
    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    elif args_cli.input_file:
        # if the environment name is not specified, try to get it from the dataset file
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(args_cli.input_file)
        env_name = dataset_file_handler.get_env_name()
    else:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # parse configuration
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)
    env_cfg.env_name = env_name

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # data generator is in charge of resetting the environment
    env_cfg.terminations = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = MimicRecorderManagerCfg()

    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    if args_cli.generated_output_file:
        # create output directory for generated episodes if it does not exist
        generated_output_dir = os.path.dirname(args_cli.generated_output_file)
        if not os.path.exists(generated_output_dir):
            os.makedirs(generated_output_dir)
        generated_output_file_name = os.path.splitext(os.path.basename(args_cli.generated_output_file))[0]
        env_cfg.recorders.dataset_export_dir_path = generated_output_dir
        env_cfg.recorders.dataset_filename = generated_output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Re-enable slot randomization for collection — the Mimic cfg disables it
    # to prevent it overwriting reset_to() during annotation/generation, but
    # collection needs objects to randomize each episode for demo diversity.
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab_ext.tasks.air2_robotis_franka.mdp import reset_objects_on_slots
    env_cfg.events.reset_object_position = EventTerm(
        func=reset_objects_on_slots, mode="reset"
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    if not isinstance(env.unwrapped, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # check if the mimic API env.unwrapped.get_subtask_term_signals() is implemented
    if env.unwrapped.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
        raise NotImplementedError(
            "The environment does not implement the get_subtask_term_signals method required to run this script."
        )

    # set seed for generation
    random.seed(env.unwrapped.cfg.datagen_config.seed)
    np.random.seed(env.unwrapped.cfg.datagen_config.seed)
    torch.manual_seed(env.unwrapped.cfg.datagen_config.seed)

    # reset before starting
    env.reset()

    # Set up asyncio stuff
    asyncio_event_loop = asyncio.get_event_loop()
    env_action_queue = asyncio.Queue()

    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(
        env.unwrapped, env.unwrapped.cfg, env.unwrapped.device, asyncio_lock=shared_datagen_info_pool_lock
    )
    if args_cli.input_file:
        shared_datagen_info_pool.load_from_dataset_file(args_cli.input_file)
        print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")

    # make data generator object
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        if args_cli.teleop_env_index is not None and i == args_cli.teleop_env_index:
            data_generator_asyncio_tasks.append(
                asyncio_event_loop.create_task(
                    run_teleop_robot(
                        env, i, env_action_queue, shared_datagen_info_pool, success_term, args_cli.output_file
                    )
                )
            )
            continue
        data_generator_asyncio_tasks.append(
            asyncio_event_loop.create_task(
                run_data_generator(
                    env,
                    i,
                    env_action_queue,
                    shared_datagen_info_pool,
                    success_term,
                    export_demo=bool(args_cli.generated_output_file),
                )
            )
        )

    try:
        asyncio.ensure_future(asyncio.gather(*data_generator_asyncio_tasks))
    except asyncio.CancelledError:
        print("Tasks were cancelled.")

    env_loop(env, env_action_queue, shared_datagen_info_pool, asyncio_event_loop)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
