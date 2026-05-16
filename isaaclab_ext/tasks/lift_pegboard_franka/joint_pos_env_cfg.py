"""Franka Panda pick-and-place task — joint-position control variant.

Same pegboard scene as ``lift_pegboard_ur3e_rg2`` but the UR3e + RG2 has been
swapped out for the Franka Panda + Panda hand. The pedestal mount, slot
positions, basket, and tool layout are unchanged so all the geometry tuning
(min_separation, basket lip, peg z) carries over verbatim.
"""

from pathlib import Path

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# Locate ur_pick/exported_assets regardless of whether this file is being
# read from the repo or from its hardlink under IsaacLab/source/...
_HERE = Path(__file__).resolve()
_REPO_CANDIDATES = [
    _HERE.parents[3],
    Path("/home/user/Desktop/ur_pick"),
]
_REPO_ROOT = next((p for p in _REPO_CANDIDATES if (p / "exported_assets").exists()), _REPO_CANDIDATES[-1])
_ASSETS = _REPO_ROOT / "exported_assets" / "object"


@configclass
class FrankaPegboardLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Mounted on the work surface (z=0 after the -0.696 table shift) and
        # pulled back to x=-0.2 so the upright Franka clears the pegboard
        # (pegboard front face is around x≈0.55). Self-collisions off during
        # exploration so random PPO motion can't NaN PhysX.
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-0.2, 0.0, 0.0)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Same PhysX buffer bumps as the UR3e variant for multi-env training.
        self.sim.physx.gpu_max_rigid_patch_count = 524288
        self.sim.physx.gpu_max_rigid_contact_count = 2_097_152
        self.sim.physx.gpu_found_lost_pairs_capacity = 524288
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 524288

        # Arm action: joint position control on Franka's 7 revolute joints.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # Gripper action: binary on the two Panda finger joints.
        # Franka uses 0.04 (open) / 0.0 (close) — see FRANKA_PANDA_CFG.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # End-effector body for the goal-pose command.
        self.commands.object_pose.body_name = "panda_hand"

        # Same tightened reset as UR3e — the toothbrush must stay on L1.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
        }
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg(
            "object", body_names="tooth_brush"
        )
        # Goal-pose target above the basket — unchanged from UR3e variant.
        self.commands.object_pose.ranges.pos_x = (0.28, 0.36)
        self.commands.object_pose.ranges.pos_y = (-0.22, -0.14)
        self.commands.object_pose.ranges.pos_z = (0.34, 0.42)

        # Table — KINEMATIC, lowered by 0.696 m so the wooden base sits on
        # the ground plane. After translation: work surface ≈ z=0, pegboard
        # top ≈ z=1.11. Slot positions below already reflect this shift.
        _kinematic_rigid = RigidBodyPropertiesCfg(kinematic_enabled=True)
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.696]),
            spawn=UsdFileCfg(usd_path=str(_ASSETS / "robotis_net_table.usd"),
                             rigid_props=_kinematic_rigid),
        )

        # Common physics props for the 6 graspable tools (dynamic).
        _tool_rigid = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        _tool_mass = MassPropertiesCfg(mass=0.05)  # 50 g each

        # Pegboard slot positions — z shifted by -0.696 to match lowered table.
        _Z_SHIFT = -0.696
        LEFT_SLOTS = [
            (0.555,  0.260, 0.95 + _Z_SHIFT),    # L0
            (0.555,  0.087, 0.95 + _Z_SHIFT),    # L1 — toothbrush
            (0.555,  0.260, 1.235 + _Z_SHIFT),   # L2
            (0.555,  0.087, 1.235 + _Z_SHIFT),   # L3
        ]
        RIGHT_SLOTS = [
            (0.555, -0.087, 0.95 + _Z_SHIFT),    # R0
            (0.555, -0.260, 0.95 + _Z_SHIFT),    # R1
            (0.555, -0.087, 1.235 + _Z_SHIFT),   # R2
            (0.555, -0.260, 1.235 + _Z_SHIFT),   # R3
        ]
        _tool_specs = [
            ("brush",        "brush_ring.usd",         list(LEFT_SLOTS[0])),
            ("silicone",     "silicone_tube_ring.usd", list(LEFT_SLOTS[2])),
            ("scissors",     "scissors_ring.usd",      list(LEFT_SLOTS[3])),
            ("pliers",       "pliers_ring.usd",        list(RIGHT_SLOTS[0])),
            ("screwdriver",  "screw_driver_ring.usd",  list(RIGHT_SLOTS[1])),
        ]

        # ToothBrush — the env "object", on L1 so it both hangs and is pickable.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=list(LEFT_SLOTS[1]), rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=str(_ASSETS / "tooth_brush.usd"),
                rigid_props=_tool_rigid,
                mass_props=_tool_mass,
            ),
        )

        # Basket — KINEMATIC, sits on the (lowered) work surface at z≈0.
        self.scene.basket = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.32, -0.18, 0.02]),
            spawn=UsdFileCfg(usd_path=str(_ASSETS / "plastic_basket2.usd"),
                             rigid_props=_kinematic_rigid),
        )

        for name, fname, pos in _tool_specs:
            setattr(
                self.scene,
                f"tool_{name}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Tool_{name}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
                    spawn=UsdFileCfg(
                        usd_path=str(_ASSETS / fname),
                        rigid_props=_tool_rigid,
                        mass_props=_tool_mass,
                    ),
                ),
            )

        # EE frame transformer. Franka's TCP offset from panda_hand is 0.107 m
        # (matches the upstream stack-cube IK body_offset).
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.107]),
                ),
            ],
        )


@configclass
class FrankaPegboardLiftEnvCfg_PLAY(FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
