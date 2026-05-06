"""UR3e + RG2 cube-lift task — joint-position control variant."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.ur3e_rg2 import UR3E_RG2_CFG  # isort: skip


@configclass
class UR3eRG2CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = UR3E_RG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Arm action: joint position control on the 6 UR3e revolute joints
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            scale=0.5,
            use_default_offset=True,
        )

        # Gripper action: binary (open=0 rad, close=~74°) on RG2 driver joint + mirror.
        # If gripper appears inverted at runtime, swap open/close values.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["rg2_gripper_joint", "rg2_gripper_mirror_joint"],
            open_command_expr={"rg2_gripper.*": 0.0},
            close_command_expr={"rg2_gripper.*": 1.30},  # ~74.5° in radians
        )

        # End-effector body for the goal-pose command. wrist_3_link is the arm
        # flange; the FrameTransformer below adds the TCP offset.
        self.commands.object_pose.body_name = "wrist_3_link"

        # Cube object (same dex_cube as Franka task)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # EE frame transformer: from base_link, target wrist_3_link, with a
        # ~18 cm offset along +Z to land at the RG2 finger midpoint.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur3e/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur3e/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.18]),
                ),
            ],
        )


@configclass
class UR3eRG2CubeLiftEnvCfg_PLAY(UR3eRG2CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
