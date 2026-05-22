"""Franka Panda pick-and-place task — joint-position control variant.

Same pegboard scene as ``lift_pegboard_ur3e_rg2`` but the UR3e + RG2 has been
swapped out for the Franka Panda + Panda hand. The pedestal mount, slot
positions, basket, and tool layout are unchanged so all the geometry tuning
(min_separation, basket lip, peg z) carries over verbatim.

Four pick targets are supported via subclasses:
    FrankaPegboardLiftEnvCfg          — toothbrush (default)
    FrankaPegboardLiftScissorsEnvCfg  — scissors
    FrankaPegboardLiftSiliconeEnvCfg  — silicone tube
    FrankaPegboardLiftPliersEnvCfg    — pliers
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

# ---------------------------------------------------------------------------
# Asset path resolution
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_CANDIDATES = [
    _HERE.parents[3],
    Path("/home/user/Desktop/ur_pick"),
]
_REPO_ROOT = next((p for p in _REPO_CANDIDATES if (p / "exported_assets").exists()), _REPO_CANDIDATES[-1])
_ASSETS = _REPO_ROOT / "exported_assets" / "object"

# ---------------------------------------------------------------------------
# Pegboard slot positions — z shifted by -0.696 to match lowered table
# ---------------------------------------------------------------------------

_Z_SHIFT = -0.696

LEFT_SLOTS = [
    (0.555,  0.260, 0.95 + _Z_SHIFT),    # L0
    (0.555,  0.087, 0.95 + _Z_SHIFT),    # L1 — toothbrush default
    (0.555,  0.260, 1.235 + _Z_SHIFT),   # L2 — silicone
    (0.555,  0.087, 1.235 + _Z_SHIFT),   # L3 — scissors
]
RIGHT_SLOTS = [
    (0.555, -0.087, 0.95 + _Z_SHIFT),    # R0 — pliers
    (0.555, -0.260, 0.95 + _Z_SHIFT),    # R1
    (0.555, -0.087, 1.235 + _Z_SHIFT),   # R2
    (0.555, -0.260, 1.235 + _Z_SHIFT),   # R3
]

# ---------------------------------------------------------------------------
# Shared physics properties
# ---------------------------------------------------------------------------

_KINEMATIC_RIGID = RigidBodyPropertiesCfg(kinematic_enabled=True)
_TOOL_RIGID = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_TOOL_MASS = MassPropertiesCfg(mass=0.05)  # 50 g each


# ---------------------------------------------------------------------------
# Base environment config — toothbrush as default pick target
# ---------------------------------------------------------------------------

@configclass
class FrankaPegboardLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot — upright Franka, pulled back so it clears the pegboard.
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
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.commands.object_pose.body_name = "panda_hand"

        # Goal-pose target: above the basket so RL / command-following carries
        # the object to the drop zone.
        self.commands.object_pose.ranges.pos_x = (0.28, 0.36)
        self.commands.object_pose.ranges.pos_y = (-0.22, -0.14)
        self.commands.object_pose.ranges.pos_z = (0.34, 0.42)

        # Table — KINEMATIC, lowered so work surface ≈ z=0.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.696]),
            spawn=UsdFileCfg(usd_path=str(_ASSETS / "robotis_net_table.usd"),
                             rigid_props=_KINEMATIC_RIGID),
        )

        # Basket — KINEMATIC, on the work surface.
        self.scene.basket = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.32, -0.18, 0.02]),
            spawn=UsdFileCfg(usd_path=str(_ASSETS / "plastic_basket2.usd"),
                             rigid_props=_KINEMATIC_RIGID),
        )

        # EE frame transformer — TCP 0.107 m ahead of panda_hand.
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

        # Object and background tools — overridden by per-object subclasses.
        self._setup_scene_objects()

    # ------------------------------------------------------------------
    # Per-object hooks — override in subclasses to change the pick target
    # ------------------------------------------------------------------

    def _pick_object_spec(self):
        """Return (usd_filename, peg_position)."""
        return ("tooth_brush_green.usd", list(LEFT_SLOTS[1]))

    def _background_tool_specs(self):
        """Return [(scene_name, usd_filename, peg_position), ...] for static props."""
        return [
            ("silicone", "silicone_tube_ring_blue.usd", list(LEFT_SLOTS[2])),
            ("scissors", "scissors_ring_red.usd",       list(LEFT_SLOTS[3])),
            ("pliers",   "pliers_ring_orange.usd",      list(RIGHT_SLOTS[0])),
        ]

    def _setup_scene_objects(self):
        usd_fname, peg_pos = self._pick_object_spec()

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=peg_pos, rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=str(_ASSETS / usd_fname),
                rigid_props=_TOOL_RIGID,
                mass_props=_TOOL_MASS,
            ),
        )

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
        }
        # Single-body RigidObjects — no body_names filter needed.
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")

        for name, fname, pos in self._background_tool_specs():
            setattr(
                self.scene,
                f"tool_{name}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Tool_{name}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
                    spawn=UsdFileCfg(
                        usd_path=str(_ASSETS / fname),
                        rigid_props=_TOOL_RIGID,
                        mass_props=_TOOL_MASS,
                    ),
                ),
            )


@configclass
class FrankaPegboardLiftEnvCfg_PLAY(FrankaPegboardLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Per-object subclasses
# ---------------------------------------------------------------------------

@configclass
class FrankaPegboardLiftScissorsEnvCfg(FrankaPegboardLiftEnvCfg):
    def _pick_object_spec(self):
        return ("scissors_ring_red.usd", list(LEFT_SLOTS[3]))

    def _background_tool_specs(self):
        return [
            ("toothbrush", "tooth_brush_green.usd",       list(LEFT_SLOTS[1])),
            ("silicone",   "silicone_tube_ring_blue.usd",  list(LEFT_SLOTS[2])),
            ("pliers",     "pliers_ring_orange.usd",       list(RIGHT_SLOTS[0])),
        ]


@configclass
class FrankaPegboardLiftScissorsEnvCfg_PLAY(FrankaPegboardLiftScissorsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaPegboardLiftSiliconeEnvCfg(FrankaPegboardLiftEnvCfg):
    def _pick_object_spec(self):
        return ("silicone_tube_ring_blue.usd", list(LEFT_SLOTS[2]))

    def _background_tool_specs(self):
        return [
            ("toothbrush", "tooth_brush_green.usd",    list(LEFT_SLOTS[1])),
            ("scissors",   "scissors_ring_red.usd",    list(LEFT_SLOTS[3])),
            ("pliers",     "pliers_ring_orange.usd",   list(RIGHT_SLOTS[0])),
        ]


@configclass
class FrankaPegboardLiftSiliconeEnvCfg_PLAY(FrankaPegboardLiftSiliconeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaPegboardLiftPliersEnvCfg(FrankaPegboardLiftEnvCfg):
    def _pick_object_spec(self):
        return ("pliers_ring_orange.usd", list(RIGHT_SLOTS[0]))

    def _background_tool_specs(self):
        return [
            ("toothbrush", "tooth_brush_green.usd",       list(LEFT_SLOTS[1])),
            ("silicone",   "silicone_tube_ring_blue.usd",  list(LEFT_SLOTS[2])),
            ("scissors",   "scissors_ring_red.usd",        list(LEFT_SLOTS[3])),
        ]


@configclass
class FrankaPegboardLiftPliersEnvCfg_PLAY(FrankaPegboardLiftPliersEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
