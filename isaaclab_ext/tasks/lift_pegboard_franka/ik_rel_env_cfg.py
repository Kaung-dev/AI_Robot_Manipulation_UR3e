"""Franka Panda pegboard task — relative differential-IK control variant.

This is the variant used for SE(3) teleoperation and imitation learning.
The 6-DoF EE pose delta is the action; an internal damped-least-squares IK
solver maps that to Franka joint targets every step.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
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

        # XR anchor: operator appears ~80 cm behind the robot, eyes at pegboard height.
        self.xr = XrCfg(anchor_pos=(-1.1, 1.0, -0.5), anchor_rot=(1.0, 0.0, 0.0, 0.0))

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=False,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            delta_pos_scale_factor=20.0,
                            delta_rot_scale_factor=15.0,
                            alpha_rot=0.3,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


@configclass
class FrankaPegboardLiftEnvCfg_PLAY(FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
