"""UR3e + RG2 cube-lift task — absolute differential-IK control variant.

This is the variant used by the auto-pick state machine (scripts/pick_cube_auto.py).
The action is a 7-D vector: (x, y, z, qw, qx, qy, qz) — an absolute end-effector
pose in the robot base frame. The IK solver maps it to UR3e joint targets.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

from isaaclab_assets.robots.ur3e_rg2 import UR3E_RG2_HIGH_PD_CFG  # isort: skip


@configclass
class UR3eRG2CubeLiftEnvCfg(joint_pos_env_cfg.UR3eRG2CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Stiff PD for accurate IK tracking, gravity off on arm bodies.
        self.scene.robot = UR3E_RG2_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Absolute end-effector pose action (no use_relative_mode=True).
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.18]),
        )


@configclass
class UR3eRG2CubeLiftEnvCfg_PLAY(UR3eRG2CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
