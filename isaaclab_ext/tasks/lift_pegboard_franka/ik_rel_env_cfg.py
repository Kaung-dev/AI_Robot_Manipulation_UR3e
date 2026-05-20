"""Franka Panda pegboard task — relative differential-IK control variant.

This is the variant used for SE(3) teleoperation and imitation learning.
The 6-DoF EE pose delta is the action; an internal damped-least-squares IK
solver maps that to Franka joint targets every step.

All four per-object configs (Toothbrush, Scissors, Silicone, Pliers) live here.
They each inherit the correct object setup from joint_pos_env_cfg and overlay
the same IK + VR hand tracking configuration via _apply_ik_vr.
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


def _apply_ik_vr(cfg) -> None:
    """Apply IK-Rel arm action and VR hand tracking to any Franka pegboard cfg."""
    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.init_state.pos = (-0.2, 0.0, 0.0)
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    cfg.xr = XrCfg(anchor_pos=(-1.1, 1.0, -0.5), anchor_rot=(1.0, 0.0, 0.0, 0.0))
    cfg.teleop_devices = DevicesCfg(
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
                        sim_device=cfg.sim.device,
                    ),
                    GripperRetargeterCfg(
                        bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                        sim_device=cfg.sim.device,
                    ),
                ],
                sim_device=cfg.sim.device,
                xr_cfg=cfg.xr,
            ),
        }
    )


@configclass
class FrankaPegboardLiftEnvCfg(joint_pos_env_cfg.FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_ik_vr(self)


@configclass
class FrankaPegboardLiftEnvCfg_PLAY(FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaPegboardLiftScissorsEnvCfg(joint_pos_env_cfg.FrankaPegboardLiftScissorsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_ik_vr(self)


@configclass
class FrankaPegboardLiftSiliconeEnvCfg(joint_pos_env_cfg.FrankaPegboardLiftSiliconeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_ik_vr(self)


@configclass
class FrankaPegboardLiftPliersEnvCfg(joint_pos_env_cfg.FrankaPegboardLiftPliersEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_ik_vr(self)
