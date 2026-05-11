"""UR3e + RG2 cube-lift task — relative differential-IK control variant.

This is the variant used for SE(3) teleoperation and imitation learning.
The 6-DoF EE pose delta is the action; an internal damped-least-squares IK
solver maps that to UR3e joint targets every step.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

from isaaclab_assets.robots.ur3e_rg2 import UR3E_RG2_HIGH_PD_CFG  # isort: skip


@configclass
class UR3eRG2PegboardLiftEnvCfg(joint_pos_env_cfg.UR3eRG2PegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Stiffer PD makes IK tracking accurate.
        self.scene.robot = UR3E_RG2_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Replace joint-position arm action with relative differential IK.
        # body_offset places the IK target at the gripper TCP (~18 cm in +Z
        # from wrist_3_link), matching the FrameTransformer in the parent cfg.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.18]),
        )


@configclass
class UR3eRG2PegboardLiftEnvCfg_PLAY(UR3eRG2PegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
