"""UR3e + RG2 pick-and-place task — joint-position control variant."""

from pathlib import Path

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.ur3e_rg2 import UR3E_RG2_CFG  # isort: skip

# Locate ur_pick/exported_assets regardless of whether this file is being
# read from the repo or from its hardlink under IsaacLab/source/...
_HERE = Path(__file__).resolve()
_REPO_CANDIDATES = [
    _HERE.parents[3],  # original repo layout: ur_pick/isaaclab_ext/tasks/lift_cube_ur3e_rg2/joint_pos_env_cfg.py
    Path("/home/user/Desktop/ur_pick"),
]
_REPO_ROOT = next((p for p in _REPO_CANDIDATES if (p / "exported_assets").exists()), _REPO_CANDIDATES[-1])
_ASSETS = _REPO_ROOT / "exported_assets" / "object"


@configclass
class UR3eRG2PegboardLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot pedestal-mounted in front of the table. (0.10, 0, 0.20) puts
        # the toothbrush at L1 (0.555, 0.087, 0.254) at distance 0.47 m —
        # within UR3e's 500 mm reach.
        # Self-collisions OFF during PPO: random exploration slams the
        # gripper into itself and NaNs PhysX without it (value loss -> inf).
        self.scene.robot = UR3E_RG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.10, 0.0, 0.20)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Bump PhysX GPU buffers for multi-env training. The pegboard scene
        # has many rigid bodies per env (table + 6 tools + basket + robot),
        # so the default buffer overflows past ~512 envs without these.
        self.sim.physx.gpu_max_rigid_patch_count = 524288
        self.sim.physx.gpu_max_rigid_contact_count = 2_097_152
        self.sim.physx.gpu_found_lost_pairs_capacity = 524288
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 524288

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
            # 0.75 rad ≈ 43° — past the previous 0.60. The mimic constraint
            # keeps both fingers symmetric, so this just commands a harder
            # squeeze; PhysX clamps once the pads contact the cube.
            close_command_expr={"rg2_gripper.*": 0.75},
        )

        # End-effector body for the goal-pose command. wrist_3_link is the arm
        # flange; the FrameTransformer below adds the TCP offset.
        self.commands.object_pose.body_name = "wrist_3_link"

        # Tighten the reset randomization. The toothbrush hangs on the L1
        # peg — too much xy randomization and it spawns off the peg (no peg
        # at that location → falls to the floor). Keep ±1 cm so it always
        # hangs on L1.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
        }
        # The standard lift task expects a body called "Object" (DexCube's
        # rigid body name). The toothbrush USD's body is called "tooth_brush"
        # — point the reset event at it. ".*" would also match.
        from isaaclab.managers import SceneEntityCfg
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg(
            "object", body_names="tooth_brush"
        )
        # Goal-pose target = ABOVE the basket so the lift reward drives the
        # toothbrush from its peg into the basket. Basket is at (0.32, -0.18,
        # 0.02) and is ~0.28 m tall, so its lip is at z≈0.30. Place the goal
        # 5–15 cm above the lip with some xy randomization within the basket
        # opening footprint.
        self.commands.object_pose.ranges.pos_x = (0.28, 0.36)
        self.commands.object_pose.ranges.pos_y = (-0.22, -0.14)
        self.commands.object_pose.ranges.pos_z = (0.34, 0.42)

        # Table — KINEMATIC, lowered by 0.696 m so its wooden base sits on
        # the Isaac Lab ground plane (z=0) instead of floating at z=0.696.
        # After translation: wooden work surface at z≈0, pegboard rises to
        # z≈1.11. Slot positions shift down by 0.696 too.
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

        # Pegboard slot positions — same as view.py but with z shifted down
        # by 0.696 to match the lowered table.
        # NOTE: tools must spawn with their ring already positioned around
        # the peg (the peg threads through the ring horizontally). Any
        # vertical drop clearance means the ring falls *past* the peg
        # before settling — confirmed worse with even 8 cm of drop. Spawn
        # exactly at slot z so the peg is inside the ring on the first step.
        _Z_SHIFT = -0.696
        LEFT_SLOTS = [
            (0.555,  0.260, 0.95 + _Z_SHIFT),    # L0 lower-far-left
            (0.555,  0.087, 0.95 + _Z_SHIFT),    # L1 lower-near-left
            (0.555,  0.260, 1.235 + _Z_SHIFT),   # L2 upper-far-left
            (0.555,  0.087, 1.235 + _Z_SHIFT),   # L3 upper-near-left
        ]
        RIGHT_SLOTS = [
            (0.555, -0.087, 0.95 + _Z_SHIFT),    # R0
            (0.555, -0.260, 0.95 + _Z_SHIFT),    # R1
            (0.555, -0.087, 1.235 + _Z_SHIFT),   # R2
            (0.555, -0.260, 1.235 + _Z_SHIFT),   # R3
        ]
        # All 6 tools DYNAMIC and pickable. The 4 lower-row tools (z=0.254)
        # catch on the z=0.434 pegs and hang. The 2 upper-row tools (z=0.539)
        # don't reliably catch on upper pegs — they fall to the work surface
        # (z≈0) but are still pickable from there.
        # Layout:
        #   L0 (lower far)  : brush       — hangs on peg
        #   L1 (lower near) : toothbrush  — hangs on peg, env "object"
        #   L2 (upper far)  : silicone    — falls to work surface
        #   L3 (upper near) : scissors    — falls to work surface
        #   R0 (lower near) : pliers      — hangs on peg
        #   R1 (lower far)  : screwdriver — hangs on peg
        _tool_specs = [
            ("brush",        "brush_ring.usd",         list(LEFT_SLOTS[0])),
            ("silicone",     "silicone_tube_ring.usd", list(LEFT_SLOTS[2])),
            ("scissors",     "scissors_ring.usd",      list(LEFT_SLOTS[3])),
            ("pliers",       "pliers_ring.usd",        list(RIGHT_SLOTS[0])),
            ("screwdriver",  "screw_driver_ring.usd",  list(RIGHT_SLOTS[1])),
        ]

        # ToothBrush — the env "object". Moved from L3 (upper, would be
        # kinematic) to L1 (lower) so it actually hangs AND can be picked.
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

        # EE frame transformer: from base_link, target wrist_3_link, with a
        # ~18 cm offset along +Z to land at the RG2 finger midpoint.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur3e/base_link",
            debug_vis=True,
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
class UR3eRG2PegboardLiftEnvCfg_PLAY(UR3eRG2PegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
