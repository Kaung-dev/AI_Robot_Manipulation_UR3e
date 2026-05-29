"""Robotis AIR2 task variant with RGB, depth, and semantic cameras."""

from isaaclab.utils import configclass

from .joint_pos_env_cfg import AIR2RobotisFrankaEnvCfg, AIR2RobotisFrankaEnvCfg_PLAY
from isaaclab_ext.tasks.air2_franka.segmentation_env_cfg import _apply_segmentation_cameras


@configclass
class AIR2RobotisSegmentationEnvCfg(AIR2RobotisFrankaEnvCfg):
    """Robotis pegboard variant for visual/manual object-annotated demos."""

    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)


@configclass
class AIR2RobotisSegmentationEnvCfg_PLAY(AIR2RobotisFrankaEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
