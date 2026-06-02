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
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

# Stack task's mdp module has the standard ee_frame_pos / ee_frame_quat funcs
# we need — reuse those instead of defining new ones.
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from isaaclab_ext.tasks.air2_franka import mdp as air2_mdp

from .joint_pos_env_cfg import (
    AIR2RobotisBrushEnvCfg,
    AIR2RobotisPliersFrankaEnvCfg,
    AIR2RobotisScissorsFrankaEnvCfg,
    AIR2RobotisScrewdriverFrankaEnvCfg,
)


def _subtask_terms_obs_group(target_key: str, signal_name: str) -> type:
    """Build a per-tool SubtaskTermsObsCfg class whose grasp signal points at
    the right rigid body. Factory because Mimic looks up the signal by NAME
    (e.g. `grasp_brush`) and that name must match the subtask config.
    """
    @configclass
    class _PerToolSubtaskCfg(ObsGroup):
        # Dynamically named field — added below via setattr.
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    setattr(
        _PerToolSubtaskCfg,
        signal_name,
        ObsTerm(func=air2_mdp.gripper_closed),
    )
    return _PerToolSubtaskCfg


@configclass
class SubtaskTermsObsCfg(ObsGroup):
    """Subtask term signals for the BRUSH target (legacy class kept for
    backward compatibility with existing checkpoints / logs)."""

    grasp_brush = ObsTerm(func=air2_mdp.gripper_closed)

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
        # Override task_success with the LOOSER `target_reached_basket` (legacy
        # euclidean dist < 0.30 m) for Mimic annotation. The strict drop-in
        # rule used at PPO time requires the brush to settle inside the basket
        # AFTER physics — but manual demos stop right after the release command,
        # before fingers physically open and before the object falls. Verified
        # empirically: brush demos end at ~7 cm from basket center with z=1.10
        # (held above rim), which would satisfy the loose check but NOT strict.
        self.terminations.task_success = TerminationTermCfg(
            func=air2_mdp.target_reached_basket,
            params={"target_key": "object", "radius": 0.30},
        )
        if hasattr(self.terminations, "task_success"):
            self.terminations.success = self.terminations.task_success

        # Mimic data-gen only replays recorded actions — it doesn't need
        # camera renders. The wrist_camera in the parent cfg crashes
        # omni.syntheticdata.plugin on this T4 setup. Strip cameras to skip.
        if hasattr(self.scene, "wrist_camera"):
            self.scene.wrist_camera = None
        if hasattr(self.scene, "board_camera"):
            self.scene.board_camera = None
        # Drop the GUI-only marker spheres (FrameTransformer-based). They
        # don't affect state obs but slow Isaac boot and would contaminate
        # any downstream camera-based pretraining.
        _strip_visual_markers(self)
        _disable_object_randomization(self)

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


def _build_mimic_subtasks(target_key: str, signal_name: str, tool_name: str) -> list:
    """Two-stage subtask list (approach+grasp, then carry to basket)
    parameterised by which tool we're targeting."""
    return [
        SubTaskConfig(
            object_ref=target_key,
            subtask_term_signal=signal_name,
            subtask_term_offset_range=(5, 15),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.03,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description=f"Approach and grasp the {tool_name}",
            next_subtask_description=f"Carry {tool_name} to basket",
        ),
        SubTaskConfig(
            object_ref=target_key,
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.03,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description=f"Carry {tool_name} to basket",
        ),
    ]


def _disable_object_randomization(self) -> None:
    """Mimic replays the demo's recorded actions starting from the demo's
    recorded initial state. If the env's per-reset randomization event fires
    AFTER `reset_to`, it overwrites the demo's brush/tool spawn positions with
    random peg-slot placements, and the recorded actions then miss the target.
    Disable both randomization events (AIR2 pegboard + Robotis cylinder slots)
    while in Mimic mode.

    OPT-OUT for generation: set env var MIMIC_KEEP_RANDOMIZATION=1 to KEEP the
    object-slot randomization ON. Needed at *generation* time so each trial
    spawns the object at a different slot and Mimic re-solves IK to it, giving a
    multi-slot dataset. Without this, every generated demo is the single default
    slot (zero slot diversity) and the BC can only grasp one slot. Leave it
    UNSET for annotation/replay, where randomization must stay off.
    """
    import os
    if os.environ.get("MIMIC_KEEP_RANDOMIZATION") == "1":
        return
    if hasattr(self.events, "reset_object_position"):
        self.events.reset_object_position = None
    if hasattr(self.events, "randomize_hook_objects"):
        self.events.randomize_hook_objects = None


_MARKER_SCENE_KEYS = (
    "basket_frame", "basket_beacon_frame",
    "brush_frame", "pliers_frame", "scissors_frame", "screwdriver_frame",
    "ee_tcp_marker",
)


def _strip_visual_markers(self) -> None:
    """Remove the colored debug spheres from the scene.

    The markers are visualization-only for GUI eval. For Mimic data gen
    and CNN/visual training they should be off because:
      - Mimic doesn't render cameras (markers waste init time)
      - CNN training reads camera RGB and the spheres would teach the
        network to cheat by detecting them instead of the real tools.
    """
    for key in _MARKER_SCENE_KEYS:
        if hasattr(self.scene, key):
            setattr(self.scene, key, None)


def _mimic_cfg_common_init(self, target_key: str, tool_name: str) -> None:
    """Shared __post_init__ body for all per-tool Mimic env cfgs.

    The class hierarchy already calls the parent task config; this helper just
    layers on the Mimic-specific obs groups, success alias, datagen knobs,
    subtask list, and camera + marker strip.
    """
    self.observations.policy.eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
    self.observations.policy.eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
    self.observations.policy.concatenate_terms = False
    _strip_visual_markers(self)
    _disable_object_randomization(self)

    signal_name = f"grasp_{tool_name}"
    SubtaskGroup = _subtask_terms_obs_group(target_key, signal_name)
    self.observations.subtask_terms = SubtaskGroup()

    # Override task_success with the looser legacy rule — same reasoning
    # as in AIR2RobotisBrushMimicEnvCfg.__post_init__ above: manual demos
    # end before physics finishes the drop, so the strict drop-in rule
    # rejects all valid demos. Use euclidean-distance < 0.30 m here.
    self.terminations.task_success = TerminationTermCfg(
        func=air2_mdp.target_reached_basket,
        params={"target_key": target_key, "radius": 0.30},
    )
    if hasattr(self.terminations, "task_success"):
        self.terminations.success = self.terminations.task_success

    if hasattr(self.scene, "wrist_camera"):
        self.scene.wrist_camera = None
    if hasattr(self.scene, "board_camera"):
        self.scene.board_camera = None

    self.datagen_config.name = f"demo_src_air2_{tool_name}_D0"
    self.datagen_config.generation_guarantee = True
    self.datagen_config.generation_keep_failed = False
    self.datagen_config.generation_num_trials = 200
    self.datagen_config.generation_select_src_per_subtask = True
    self.datagen_config.generation_transform_first_robot_pose = False
    self.datagen_config.generation_interpolate_from_last_target_pose = True
    self.datagen_config.generation_relative = True
    self.datagen_config.max_num_failures = 50
    self.datagen_config.seed = 0

    self.subtask_configs["franka"] = _build_mimic_subtasks(target_key, signal_name, tool_name)


@configclass
class AIR2RobotisPliersMimicEnvCfg(AIR2RobotisPliersFrankaEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _mimic_cfg_common_init(self, target_key="object", tool_name="pliers")


@configclass
class AIR2RobotisScissorsMimicEnvCfg(AIR2RobotisScissorsFrankaEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _mimic_cfg_common_init(self, target_key="object", tool_name="scissors")


@configclass
class AIR2RobotisScrewdriverMimicEnvCfg(AIR2RobotisScrewdriverFrankaEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _mimic_cfg_common_init(self, target_key="object", tool_name="screwdriver")
