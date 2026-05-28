"""Robotis AIR2 task variant with RGB, depth, and semantic cameras."""

from isaaclab.utils import configclass

from .joint_pos_env_cfg import FrankaAIR2RobotisLiftEnvCfg, FrankaAIR2RobotisLiftEnvCfg_PLAY
from isaaclab_ext.tasks.lift_air2_ur3e_rg2.segmentation_env_cfg import _apply_segmentation_cameras


@configclass
class AIR2RobotisSegmentationEnvCfg(FrankaAIR2RobotisLiftEnvCfg):
    """Robotis pegboard variant for visual/manual object-annotated demos."""

    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)


@configclass
class AIR2RobotisSegmentationEnvCfg_PLAY(FrankaAIR2RobotisLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
