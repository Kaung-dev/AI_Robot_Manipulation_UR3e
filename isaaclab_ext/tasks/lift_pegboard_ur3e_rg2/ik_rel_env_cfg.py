"""UR3e + RG2 cube-lift task — relative differential-IK control variant.

This is the variant used for SE(3) teleoperation and imitation learning.
The 6-DoF EE pose delta is the action; an internal damped-least-squares IK
solver maps that to UR3e joint targets every step.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
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

        # XR anchor: where the operator's head appears in the sim. (-0.8, 0, 0.3)
        # puts them ~80 cm behind the robot pedestal, eyes roughly level with
        # the pegboard work surface. Adjust if your headset is mis-anchored.
        self.xr = XrCfg(anchor_pos=(-0.8, 0.0, 0.3), anchor_rot=(1.0, 0.0, 0.0, 0.0))

        # VR teleop device. Only consulted when teleop_se3_agent.py is run
        # with --teleop_device handtracking. Other devices fall through to
        # the script's manual creation path, so this addition is purely
        # additive and cannot break keyboard/gamepad/spacemouse teleop.
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            # Scaling tuned for the UR3e's ~50 cm reach.
                            # Upstream Franka uses 10.0; halve it here because
                            # the pegboard work area is smaller and the
                            # operator's hand motion needs less amplification.
                            delta_pos_scale_factor=5.0,
                            delta_rot_scale_factor=5.0,
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
class UR3eRG2PegboardLiftEnvCfg_PLAY(UR3eRG2PegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
