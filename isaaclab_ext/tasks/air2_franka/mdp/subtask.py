"""Subtask term signals for IsaacLab Mimic data generation.

A subtask term signal is a boolean per env, per step that fires when a logical
sub-phase of the task completes. Mimic uses these signals to split each demo
into reusable segments (approach, grasp, lift, carry, release) that can be
recombined into thousands of synthetic trajectories.

For the AIR2 pick-place task we expose:
  grasp_<target>   — fires when the target object is in the gripper
                      (EE within grasp_radius AND fingers nearly closed)

Future tasks can append `lift_<target>` / `carry_<target>` signals here if
finer subtask splits become useful.
"""
from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv


def _ee_local_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.scene.env_origins


def _obj_local_pos(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    asset: RigidObject = env.scene[key]
    return asset.data.root_pos_w - env.scene.env_origins


def grasped(
    env: ManagerBasedRLEnv,
    target_key: str = "object",
    grasp_radius: float = 0.08,
    finger_threshold: float = 0.04,
) -> torch.Tensor:
    """Per-env bool: target is in the gripper.

    True when EE is within `grasp_radius` m of target AND fingers are
    closer than `finger_threshold` (sum of left+right finger positions).
    Matches `rewards.target_in_hand` so the signal aligns with what PPO
    optimises for.
    """
    dist = torch.linalg.norm(_obj_local_pos(env, target_key) - _ee_local_pos(env), dim=-1)
    robot = env.scene["robot"]
    finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)
    return (dist < grasp_radius) & (finger_sum < finger_threshold)
