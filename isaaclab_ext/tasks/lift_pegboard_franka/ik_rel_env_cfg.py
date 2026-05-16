"""Franka Panda pegboard task — relative differential-IK control variant.

This is the variant used for SE(3) teleoperation and imitation learning.
The 6-DoF EE pose delta is the action; an internal damped-least-squares IK
solver maps that to Franka joint targets every step.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaPegboardLiftEnvCfg(joint_pos_env_cfg.FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Stiffer PD makes IK tracking accurate.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Preserve the pegboard-clearing mount position from the parent cfg.
        self.scene.robot.init_state.pos = (-0.2, 0.0, 0.0)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Replace joint-position arm action with relative differential IK.
        # body_offset places the IK target at the TCP (~10.7 cm in +Z from
        # panda_hand), matching the FrameTransformer in the parent cfg.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaPegboardLiftEnvCfg_PLAY(FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
