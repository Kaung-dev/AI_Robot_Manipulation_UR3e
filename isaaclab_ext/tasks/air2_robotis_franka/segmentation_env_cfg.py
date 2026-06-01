"""Robotis AIR2 task variant with RGB, depth, and semantic cameras."""

from isaaclab.utils import configclass

from .joint_pos_env_cfg import AIR2RobotisFrankaEnvCfg, AIR2RobotisFrankaEnvCfg_PLAY
from .mimic_env_cfg import _strip_visual_markers
from isaaclab_ext.tasks.air2_franka.segmentation_env_cfg import _apply_segmentation_cameras


@configclass
class AIR2RobotisSegmentationEnvCfg(AIR2RobotisFrankaEnvCfg):
    """Robotis pegboard variant for visual/manual object-annotated demos."""

    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)
        # The colored debug spheres inherited from AIR2RobotisFrankaEnvCfg
        # would appear in every camera frame and teach the CNN to cheat by
        # matching their colors instead of the actual tool geometry.
        _strip_visual_markers(self)


@configclass
class AIR2RobotisSegmentationEnvCfg_PLAY(AIR2RobotisFrankaEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)
        _strip_visual_markers(self)
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
