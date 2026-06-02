"""AIR2 scene pick task — Franka already in the scene USD, 4 objects on 8 hooks."""

from pathlib import Path

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from . import mdp
from .objects import OBJECT_SPECS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSETS = _REPO_ROOT / "exported_assets" / "object"
_SCENE  = _REPO_ROOT / "scene"

_TOOL_RIGID = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
    linear_damping=0.5,
    angular_damping=0.5,
)
_TOOL_MASS = MassPropertiesCfg(mass=0.05)


@configclass
class AIR2FrankaEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Spawn a fresh controllable Franka at the same position as the one
        # baked into AIR2.usd (which is frozen/kinematic as part of the scene).
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Moved 20 cm closer to the pegboard (y=-5.2851 -> -5.4851) so the
        # restricted hook subset (hooks 3-8) is comfortably within the
        # Franka's ~85 cm reach. AIR2.usd's baked Franka visual stays at the
        # old pose; the controllable Franka floats ~20 cm forward of it.
        self.scene.robot.init_state.pos = (-4.2405, -5.2851, 1.0397)
        self.scene.robot.init_state.joint_pos["panda_joint1"] = -1.5708  # 90° rotation at spawn
        self.scene.robot.init_state.joint_pos["panda_joint4"] = -2.26892803   # wrist horizontal

        self.sim.physx.gpu_max_rigid_patch_count = 524288
        self.sim.physx.gpu_max_rigid_contact_count = 2_097_152
        self.sim.physx.gpu_found_lost_pairs_capacity = 524288
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 524288

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls",
                ik_params={"lambda_val": 0.1},
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = lift_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )

        # NOTE: An OpenXR/handtracking DevicesCfg block lived here. It was removed
        # because Se3RelRetargeterCfg/GripperRetargeterCfg defaults carry a lambda
        # that IsaacLab's official rsl_rl/train.py cannot serialize through Hydra
        # ("ValueError: Could not resolve the input string 'lambda headpose' into
        #  callable object"), which silently crashed PPO before the first iter.
        # Keyboard/spacemouse teleop in collect_air2_manual_demos.py and the
        # built-in record_demos.py both work via their own fallback paths, so this
        # block was only useful for VR — if you need it back, gate it behind a
        # CLI flag and construct it OUTSIDE the configclass __post_init__ so
        # Hydra never sees it.

        self.commands.object_pose.body_name = "panda_hand"
        self.commands.object_pose.debug_vis = False

        # AIR2.usd loaded at origin — hooks and Franka at their USD world positions.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Environment",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(usd_path=str(_SCENE / "AIR2.usd"),
                             rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True)),
        )

        # Main pick target at hook_01.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-5.0900, -2.5800, 1.6800], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=str(_ASSETS / OBJECT_SPECS[0].usd_file),
                rigid_props=_TOOL_RIGID,
                mass_props=_TOOL_MASS,
            ),
        )

        # Three distractor objects at hooks 2–4.
        for spec, pos in [
            (OBJECT_SPECS[1], [-5.2250, -2.5800, 1.7800]),
            (OBJECT_SPECS[2], [-5.4000, -2.5800, 1.7200]),
            (OBJECT_SPECS[3], [-5.7056, -2.5800, 1.4700]),
        ]:
            setattr(
                self.scene, spec.scene_key,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{spec.scene_key.title().replace('_', '')}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
                    spawn=UsdFileCfg(
                        usd_path=str(_ASSETS / spec.usd_file),
                        rigid_props=_TOOL_RIGID,
                        mass_props=_TOOL_MASS,
                            ),
                ),
            )

        # Disable the base class single-object reset; our event handles all 4.
        self.events.reset_object_position = None

        # Shuffle all 4 objects across 8 hooks on every episode reset.
        self.events.randomize_hook_objects = EventTerm(
            func=mdp.reset_objects_on_hooks,
            mode="reset",
        )

        # Goal pose near the board.
        self.commands.object_pose.ranges.pos_x = (-5.8, -5.0)
        self.commands.object_pose.ranges.pos_y = (-3.0, -2.0)
        self.commands.object_pose.ranges.pos_z = (1.0, 1.5)

        # Viewport camera — positioned in front of the pegboard, elevated.
        # Robot is at (-4.24, -5.29, 1.04); hooks at Y≈-5.9.
        self.viewer.eye = (-3.5, -2.5, 2.5)
        self.viewer.lookat = (-4.0, -5.5, 1.2)

        # --- AIR2-scene reward overrides ---------------------------------
        # Disable broken base lift rewards (see mdp/rewards.py for diagnosis).
        # Target-specific rewards are applied by _apply_target_rewards() in
        # per-object subclasses (air2_robotis_franka/joint_pos_env_cfg.py).
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.rewards.reaching_object = None

        self.rewards.action_rate.weight = -1e-3
        self.rewards.joint_vel.weight = -1e-3

        _pinhole = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1e5),
        )

        # Wrist camera — same hand-mounted view used by the Franka pegboard visuomotor task.
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=_pinhole,
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.6964, 0.1233, 0.1233, -0.6964),
                convention="ros",
            ),
        )

        # EE frame placeholder — points to Franka hand inside AIR2.usd.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1]),
                ),
            ],
        )

        # 42-D obs layout (LiftEnvCfg target_object_position kept active):
        #   joint_pos(9) + joint_vel(9) + object_position(3) + target_object_position(7)
        #   + actions=last_action(7) + eef_pos(3) + eef_quat(4) = 42
        self.observations.policy.eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        self.observations.policy.eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)


@configclass
class AIR2FrankaEnvCfg_PLAY(AIR2FrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
