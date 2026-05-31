"""Mimic env config for the AIR2-Robotis-Franka pick-place task (brush target).

Wraps the per-target task cfg with the extra observation terms and subtask
configs that IsaacLab Mimic's `annotate_demos.py` / `generate_dataset.py`
pipeline requires.

Adds two observation groups:
  policy        — existing 35-D + eef_pos (3) + eef_quat (4) = 42-D
  subtask_terms — boolean signals (e.g. grasp_brush) for mimic to detect
                  subtask boundaries during annotation

Subtask layout for brush pick-place (one EE, one object):
  0) Approach + grasp brush  — terminates on `grasp_brush`
  1) Carry brush to basket   — terminal subtask, no termination signal needed
"""
from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

# Stack task's mdp module has the standard ee_frame_pos / ee_frame_quat funcs
# we need — reuse those instead of defining new ones.
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from isaaclab_ext.tasks.air2_franka import mdp as air2_mdp

from .joint_pos_env_cfg import AIR2RobotisBrushEnvCfg


@configclass
class SubtaskTermsObsCfg(ObsGroup):
    """Subtask term signals — booleans Mimic reads to detect phase boundaries."""

    grasp_brush = ObsTerm(
        func=air2_mdp.grasped,
        params={"target_key": "object"},
    )

    def __post_init__(self):
        # Subtask term signals are NOT concatenated into a single tensor —
        # the mimic env reads them by name from obs_buf["subtask_terms"].
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class AIR2RobotisBrushMimicEnvCfg(AIR2RobotisBrushEnvCfg, MimicEnvCfg):
    """AIR2 brush task wrapped for Mimic data generation.

    Multiple-inheritance: takes the base AIR2 brush task and mixes in the
    Mimic-required cfg fields (datagen_config, subtask_configs).
    """

    def __post_init__(self):
        # Calls AIR2RobotisBrushEnvCfg.__post_init__ then MimicEnvCfg defaults.
        super().__post_init__()

        # Augment the policy obs with EE pose — the mimic env's
        # get_robot_eef_pose() reads these out of obs_buf at runtime.
        self.observations.policy.eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        self.observations.policy.eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        # mimic_env.get_robot_eef_pose does `obs_buf["policy"]["eef_pos"]` —
        # that dict-style indexing only works when the group is NOT a single
        # concatenated tensor. Mimic data gen doesn't feed obs into a policy
        # network, so disabling concatenation here is safe.
        self.observations.policy.concatenate_terms = False

        # Register the subtask-terms obs group on the env's observations cfg.
        self.observations.subtask_terms = SubtaskTermsObsCfg()

        # IsaacLab Mimic's annotate_demos.py / generate_dataset.py look for a
        # termination named exactly "success" on the env cfg. Our cfg names it
        # "task_success" — alias it so mimic finds it.
        if hasattr(self.terminations, "task_success"):
            self.terminations.success = self.terminations.task_success

        # Mimic data-gen only replays recorded actions — it doesn't need
        # camera renders. The wrist_camera in the parent cfg crashes
        # omni.syntheticdata.plugin on this T4 setup. Strip cameras to skip.
        if hasattr(self.scene, "wrist_camera"):
            self.scene.wrist_camera = None
        if hasattr(self.scene, "board_camera"):
            self.scene.board_camera = None

        # Data-gen knobs (defaults are mostly fine; tune later if needed).
        self.datagen_config.name = "demo_src_air2_brush_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 200
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 0

        # Subtask layout for a single-target pick-place:
        # 1) approach + grasp brush  -> terminator = grasp_brush
        # 2) carry brush to basket   -> terminal subtask (no signal needed)
        subtasks = [
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="grasp_brush",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach and grasp the brush",
                next_subtask_description="Carry brush to basket",
            ),
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,                 # final — no terminator
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Carry brush to basket",
            ),
        ]
        # The mimic env wrapper keys subtask_configs by EEF name (we use "franka").
        self.subtask_configs["franka"] = subtasks
