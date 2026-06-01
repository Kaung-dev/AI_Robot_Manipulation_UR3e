"""IsaacLab Mimic env wrapper for AIR2-Robotis-Franka pick-place tasks.

Subclasses `ManagerBasedRLMimicEnv` to provide the EEF pose getters/setters
and subtask term signal getters that Mimic's data generation pipeline calls.

Action format (matches `DifferentialInverseKinematicsActionCfg` + binary
gripper): `[delta_pos(3), delta_axis_angle(3), gripper(1)]` = 7-D.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class AIR2RobotisFrankaMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic env for the AIR2 Robotis pegboard pick-place task."""

    # --- EEF pose accessors ------------------------------------------------

    def get_robot_eef_pose(
        self,
        eef_name: str,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Current EE pose as (len(env_ids), 4, 4) homogeneous matrix."""
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]  # wxyz
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EE pose -> env action (delta pose + gripper)."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action = torch.clamp(pose_action + noise, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of target_eef_pose_to_action — extract target EE pose from a recorded action."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[zero_angle] = torch.zeros_like(delta_rotation_axis)[zero_angle]
        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper component from a (num_envs, T, 7) action sequence."""
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    # --- subtask term signals ---------------------------------------------

    def get_subtask_term_signals(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pull subtask term signals out of the observation buffer.

        Mimic expects a dict[subtask_name -> bool tensor of shape (len(env_ids),)].
        The actual signals are computed by ObsTerm functions in the
        `subtask_terms` observation group — see mimic_env_cfg.py.
        """
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {name: tensor[env_ids] for name, tensor in subtask_terms.items()}
