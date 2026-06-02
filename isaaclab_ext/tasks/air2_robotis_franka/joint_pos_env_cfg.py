"""AIR2 scene with robotis_net_table as functional pegboard.

Mirrors pegboard_franka: objects spawn directly at cylinder slot
positions rather than relying on hook alignment.

Table world pose in the AIR2 scene (read from Isaac Sim):
    pos = (-4.18548, -5.4202, 0.3063)
    rot = Z -90°  →  quaternion (w=0.7071, x=0, y=0, z=-0.7071)

Slot world positions computed from table transform + internal
robotis_net_table_2 offset (-0.0153, 0, 0.06981):
    local→world with Z=-90°: wx = tx + ly,  wy = ty - lx,  wz = tz + lz
"""

from pathlib import Path

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, TerminationTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CylinderCfg, SphereCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass


def _sphere_marker(visuals_name: str, color: tuple[float, float, float], radius: float = 0.05) -> VisualizationMarkersCfg:
    """Build a one-marker VisualizationMarkersCfg with a custom-color sphere."""
    return VisualizationMarkersCfg(
        prim_path=f"/Visuals/{visuals_name}",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )

from . import mdp
import isaaclab_ext.tasks.air2_franka.mdp as air2_mdp

from isaaclab_ext.tasks.air2_franka.joint_pos_env_cfg import (
    AIR2FrankaEnvCfg,
    _TOOL_RIGID,
    _TOOL_MASS,
)
from isaaclab_ext.tasks.air2_franka.objects import OBJECT_SPECS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSETS = _REPO_ROOT / "exported_assets" / "object"

_TABLE_POS = [-4.18548, -5.4202, 0.3063]
_TABLE_ROT = [0.7071, 0.0, 0.0, -0.7071]  # Z = -90°

# Tool rotation matches table rotation so rings align with rotated pegs.
_TOOL_ROT = [0.7071, 0.0, 0.0, -0.7071]

# World slot positions (lower row z=1.611, upper row z=1.611).
# NOTE: these are computed estimates — tune if tools don't hang on first run.
_SLOTS = {
    "L0": [-3.925, -5.960, 1.611],
    "L1": [-4.098, -5.960, 1.611],
    "L2": [-3.925, -5.960, 1.326],
    "L3": [-4.098, -5.960, 1.326],
    "R0": [-4.272, -5.960, 1.611],
    "R1": [-4.445, -5.960, 1.611],
    "R2": [-4.272, -5.960, 1.326],
    "R3": [-4.445, -5.960, 1.326],
}


@configclass
class AIR2RobotisFrankaEnvCfg(AIR2FrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Functional pegboard — kinematic, tools hang on its cylinder pegs.
        self.scene.robotis_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RobotisTable",
            init_state=AssetBaseCfg.InitialStateCfg(pos=_TABLE_POS, rot=_TABLE_ROT),
            spawn=UsdFileCfg(
                usd_path=str(_ASSETS / "robotis_net_table.usda"),
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )

        # Main pick target at slot R0 (right side only).
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_SLOTS["R0"], rot=_TOOL_ROT),
            spawn=UsdFileCfg(
                usd_path=str(_ASSETS / OBJECT_SPECS[0].usd_file),
                rigid_props=_TOOL_RIGID,
                mass_props=_TOOL_MASS,
            ),
        )

        # 3 distractors — all right-side slots so Mimic collection matches PPO.
        # R3 goes to pliers or scissors (screwdriver can't reach R3).
        for spec, slot in [
            (OBJECT_SPECS[1], "R1"),   # pliers
            (OBJECT_SPECS[2], "R3"),   # scissors (can reach R3)
            (OBJECT_SPECS[3], "R2"),   # screwdriver (cannot reach R3)
        ]:
            setattr(
                self.scene, spec.scene_key,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{spec.scene_key.title().replace('_', '')}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=_SLOTS[slot], rot=_TOOL_ROT),
                    spawn=UsdFileCfg(
                        usd_path=str(_ASSETS / spec.usd_file),
                        rigid_props=_TOOL_RIGID,
                        mass_props=_TOOL_MASS,
                    ),
                ),
            )

        # Disable AIR2 hook randomization — spawn at fixed cylinder positions.
        self.events.randomize_hook_objects = None

        # Each reset: place 4 objects on 4 randomly chosen slots out of 8.
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_objects_on_slots,
            mode="reset",
        )

        # Goal: lift away from the peg toward the work area.
        self.commands.object_pose.ranges.pos_x = (-5.0, -3.5)
        self.commands.object_pose.ranges.pos_y = (-6.5, -5.0)
        self.commands.object_pose.ranges.pos_z = (1.8, 2.2)

        # Strict drop-in success needs more time per episode than the 5 s base:
        # approach + grasp + lift + carry + release + wait-for-drop ≈ 6–8 s.
        self.episode_length_s = 10.0

        # Basket marker: use a FrameTransformerCfg pointing at the basket
        # mesh's rigid body (same mechanism as the tool spheres — confirmed
        # to render reliably). AssetBaseCfg + SphereCfg/CylinderCfg failed
        # to display in earlier runs; switching to FrameTransformer fixes it.
        # The mesh prim was probed via scripts/_probe_basket_prims.py.
        BASKET_CENTER = (-3.918, -5.787, 1.042)  # kept for the success rule + the beacon offset
        self.scene.basket_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            # Push the source-frame marker 10 m below the floor so the stack
            # of overlapping source-spheres at the robot base disappears.
            # Only affects the debug visualizer — relative-pose data from
            # this FrameTransformer is not consumed elsewhere.
            source_frame_offset=OffsetCfg(pos=[0.0, 0.0, -10.0]),
            visualizer_cfg=_sphere_marker("Tracker_basket", (0.1, 1.0, 0.1), radius=0.12),
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Environment/SM_BoxPortableD/SM_BoxPortableD",
                name="basket_center",
            )],
        )
        # A second FrameTransformer with a 25 cm upward offset acts as a
        # floating beacon so the drop target is visible from any angle.
        self.scene.basket_beacon_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            # Push the source-frame marker 10 m below the floor so the stack
            # of overlapping source-spheres at the robot base disappears.
            # Only affects the debug visualizer — relative-pose data from
            # this FrameTransformer is not consumed elsewhere.
            source_frame_offset=OffsetCfg(pos=[0.0, 0.0, -10.0]),
            visualizer_cfg=_sphere_marker("Tracker_basket_beacon", (0.3, 1.0, 0.3), radius=0.10),
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Environment/SM_BoxPortableD/SM_BoxPortableD",
                name="basket_beacon",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.25]),
            )],
        )
        # Per-tool tracking spheres — one FrameTransformerCfg per object,
        # each with debug_vis=True so a colored sphere follows the rigid
        # body across episode randomization. Distinct colors per tool make
        # the GUI readable when multiple tools are visible. Rigid-body prim
        # paths were probed via scripts/_probe_all_tools.py.
        _TOOL_SPHERES = (
            # (scene_key,        rigid_body_prim,                      color (rgb), name)
            ("brush_frame",      "Object/brush_ring",                  (1.0, 0.0, 0.0), "brush"),         # red
            ("pliers_frame",     "ToolPliers/pliers_ring",             (1.0, 1.0, 0.0), "pliers"),       # yellow
            ("scissors_frame",   "ToolScissors/scissors_ring",         (0.0, 1.0, 1.0), "scissors"),     # cyan
            ("screwdriver_frame","ToolScrewdriver/screw_driver",       (1.0, 0.0, 1.0), "screwdriver"),  # magenta
        )
        for scene_key, rb_path, color, name in _TOOL_SPHERES:
            setattr(
                self.scene, scene_key,
                FrameTransformerCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                    debug_vis=True,
                    visualizer_cfg=_sphere_marker(f"Tracker_{name}", color, radius=0.05),
                    target_frames=[FrameTransformerCfg.FrameCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{rb_path}",
                        name=f"{name}_target",
                    )],
                ),
            )

        # Blue sphere on the end-effector TCP — the midpoint between the
        # two fingers, NOT the wrist. The base ee_frame already uses this
        # offset (pos=[0,0,0.1] in panda_hand local frame); copy it here
        # so the visual marker matches the actual TCP that reward and
        # grasp-assist code read from `ee_frame.data.target_pos_w`.
        self.scene.ee_tcp_marker = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            # Push the source-frame marker 10 m below the floor so the stack
            # of overlapping source-spheres at the robot base disappears.
            # Only affects the debug visualizer — relative-pose data from
            # this FrameTransformer is not consumed elsewhere.
            source_frame_offset=OffsetCfg(pos=[0.0, 0.0, -10.0]),
            visualizer_cfg=_sphere_marker("Tracker_ee_tcp", (0.0, 0.4, 1.0), radius=0.04),
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1]),
            )],
        )

        # (Basket marker handled above via FrameTransformerCfg; the old
        # AssetBaseCfg + SphereCfg/CylinderCfg block was invisible in-scene.)


@configclass
class AIR2RobotisFrankaEnvCfg_PLAY(AIR2RobotisFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Per-object task configs — one per pick target
# ---------------------------------------------------------------------------

def _apply_target_rewards(cfg, target_key: str) -> None:
    """Replace generic AIR2 rewards with target-specific ones."""
    # PPO tasks use GT positions for rewards — no camera needed.
    cfg.scene.wrist_camera = None

    # Disable the base lift curriculum — it ramps action_rate/joint_vel weights
    # from -0.0001 to -0.1 over 10k steps, which overwhelms task rewards.
    cfg.curriculum.action_rate = None
    cfg.curriculum.joint_vel = None

    # Remove old generic rewards from AIR2FrankaEnvCfg
    cfg.rewards.objects_to_basket = None
    cfg.rewards.objects_off_hook = None
    cfg.rewards.objects_in_basket = None
    cfg.rewards.ee_to_object = None
    cfg.rewards.reaching_object = None

    cfg.rewards.ee_to_target = RewTerm(
        func=air2_mdp.ee_to_target,
        params={"target_key": target_key, "std": 0.5},
        weight=2.0,
    )
    cfg.rewards.target_off_slot = RewTerm(
        func=air2_mdp.target_off_slot,
        params={"target_key": target_key},
        weight=5.0,
    )
    cfg.rewards.target_in_hand = RewTerm(
        func=air2_mdp.target_in_hand,
        params={"target_key": target_key, "grasp_radius": 0.15},
        weight=30.0,
    )
    # v3 reward design: build a continuous gradient from "near target" through
    # "closing on it" through "lifted" through "near basket" through "in basket".
    # v2 had no signal between "grasped" (binary) and "in basket" (binary), so
    # PPO learned to grasp briefly but never carried.

    # v4 reward rebalance: grasp_shaping dropped 5→2, lift_progress 3→10,
    # target_to_basket 3→6. Previous 5.0 grasp weight made PPO converge to
    # "grasp and stop" — lift_progress signal was drowned out and task_success
    # decayed iter 400 → 800 (1.56% → 0%). Inverting the gradient: grasp is now
    # a touch-and-go bonus, lift is the dominant ongoing signal.
    cfg.rewards.grasp_shaping = RewTerm(
        func=air2_mdp.grasp_shaping,
        params={"target_key": target_key, "near_radius": 0.15},
        weight=2.0,                       # v3=5.0, v4=2.0 (de-emphasize)
    )
    cfg.rewards.lift_progress = RewTerm(
        func=air2_mdp.lift_progress,
        params={"target_key": target_key, "base_z": 1.61, "max_lift": 0.30},
        weight=10.0,                      # v3=3.0, v4=10.0 (dominant carry signal)
    )
    cfg.rewards.target_to_basket = RewTerm(
        func=air2_mdp.target_to_basket,
        params={"target_key": target_key, "std": 0.5},
        weight=2.0,                       # 2026-06-01: cut 6 → 2. The Gaussian fires
                                          # regardless of whether the brush is in hand,
                                          # so PPO kept finding the "hover EE near basket
                                          # without grasping" local minimum. Lower weight
                                          # means grasp-related signals (target_in_hand,
                                          # lift_progress) dominate.
    )
    cfg.rewards.target_in_basket = RewTerm(
        func=air2_mdp.target_in_basket,
        params={"target_key": target_key, "radius": 0.30},
        weight=20.0,
    )
    # Pairs with the strict drop-in success rule: rewards opening the gripper
    # while the object is over the basket footprint (above rim). Without this
    # PPO gets stuck hovering the object inside the basket bounds with the
    # gripper closed — high carry reward, zero termination signal.
    cfg.rewards.release_above_basket = RewTerm(
        func=air2_mdp.release_above_basket,
        params={
            "target_key": target_key,
            "xy_radius": 0.18,
            "height_above_rim": 0.10,
            "finger_open_thresh": 0.03,
        },
        weight=10.0,
    )
    cfg.rewards.wrong_object_moved = RewTerm(
        func=air2_mdp.wrong_object_moved,
        params={"target_key": target_key},
        weight=-3.0,  # bumped from -1.0 — collisions with distractors are now a stronger penalty
    )
    # New safety / robustness penalties (per user request 2026-06-01):
    cfg.rewards.tool_fell_to_floor = RewTerm(
        func=air2_mdp.tool_fell_to_floor,
        params={"target_key": target_key, "floor_z": 0.5},
        weight=-10.0,  # 2026-06-01: harder penalty; threshold lowered to 0.5 m so
                       # a tool successfully landing in the basket (z ~ 1.04) does
                       # NOT trigger this — only real-floor drops do.
    )
    cfg.rewards.joint_near_limit = RewTerm(
        func=air2_mdp.joint_near_limit,
        params={"margin_frac": 0.05},
        weight=-0.5,
    )
    cfg.rewards.arm_stuck = RewTerm(
        func=air2_mdp.arm_stuck,
        params={"jvel_threshold": 0.01, "action_threshold": 0.1},
        weight=-1.0,
    )
    # Soft workspace-bounds penalty: continuous distance-from-box, keeps
    # the EE between the pegboard (y > -5.95 = no collision), table
    # (z > 0.95), basket-side (x < -3.5 = don't fly off right), ceiling
    # (z < 2.2). Distance is in meters → weight tunes severity.
    cfg.rewards.ee_out_of_workspace = RewTerm(
        func=air2_mdp.ee_out_of_workspace,
        params={"x_min": -5.5, "x_max": -3.5,
                "y_min": -6.0, "y_max": -5.0,
                "z_min": 0.95, "z_max": 2.2},
        weight=-2.0,
    )
    # Soft "no sky-pointing" penalty: discourages the gripper from being
    # oriented straight up. Fires only when forward axis's world-z > 0.3
    # (~17° above horizontal).
    cfg.rewards.gripper_sky_pointing = RewTerm(
        func=air2_mdp.gripper_sky_pointing,
        params={"sky_thresh": 0.3},
        weight=-2.0,
    )
    cfg.rewards.object_slipped = RewTerm(
        func=air2_mdp.object_slipped,
        params={"target_key": target_key},
        weight=-3.0,
    )
    cfg.rewards.grasp_lost = RewTerm(
        func=air2_mdp.grasp_lost,
        params={"target_key": target_key},
        weight=-3.0,
    )
    # Reduced from -0.5 to -0.02. At -0.5, with ~95% no-progress steps per
    # episode the penalty was ~-95 reward, which dominated everything else
    # and caused the policy to "freeze" to minimize action_rate / joint_vel
    # / progress_stall instead of exploring grasping.
    cfg.rewards.progress_stall = RewTerm(
        func=air2_mdp.progress_stall,
        params={"target_key": target_key},
        weight=-0.02,
    )
    # Strict drop-in: success only fires when the object has actually
    # been released inside the basket (gripper open, object below rim).
    # Replaces the loose "distance < 0.30 m" rule that fired while still
    # gripped — the policy could no longer just hover the object over the
    # basket and call it done.
    cfg.terminations.task_success = TerminationTermCfg(
        func=air2_mdp.target_dropped_in_basket,
        params={
            "target_key": target_key,
            "xy_radius": 0.18,
            "rim_z_offset": 0.15,
            "finger_open_thresh": 0.03,
        },
    )

    # The base lift_env_cfg curriculum ramps action_rate / joint_vel weights
    # from -1e-3 to -1e-1 over 10000 env steps (=~25 PPO iters at 16 envs ×
    # 24 steps). By iter 400 these are at -1e-1, dominating positive reward
    # and forcing policy collapse. Disable the curriculum entirely — keep
    # the small base weights (-1e-3) which were already enough smoothness.
    if hasattr(cfg, "curriculum"):
        cfg.curriculum.action_rate = None
        cfg.curriculum.joint_vel = None


@configclass
class AIR2RobotisBrushEnvCfg(AIR2RobotisFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_target_rewards(self, "object")


@configclass
class AIR2RobotisBrushEnvCfg_PLAY(AIR2RobotisBrushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class AIR2RobotisPliersFrankaEnvCfg(AIR2RobotisFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.usd_path = str(_ASSETS / OBJECT_SPECS[1].usd_file)
        # Replace tool_pliers distractor with brush so there's no duplicate pliers
        self.scene.tool_pliers.spawn.usd_path = str(_ASSETS / OBJECT_SPECS[0].usd_file)
        _apply_target_rewards(self, "object")


@configclass
class AIR2RobotisPliersFrankaEnvCfg_PLAY(AIR2RobotisPliersFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class AIR2RobotisScissorsFrankaEnvCfg(AIR2RobotisFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.usd_path = str(_ASSETS / OBJECT_SPECS[2].usd_file)
        # Replace tool_scissors distractor with brush so there's no duplicate scissors
        self.scene.tool_scissors.spawn.usd_path = str(_ASSETS / OBJECT_SPECS[0].usd_file)
        _apply_target_rewards(self, "object")


@configclass
class AIR2RobotisScissorsFrankaEnvCfg_PLAY(AIR2RobotisScissorsFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class AIR2RobotisScrewdriverFrankaEnvCfg(AIR2RobotisFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.usd_path = str(_ASSETS / OBJECT_SPECS[3].usd_file)
        # Replace tool_screwdriver distractor with brush so there's no duplicate screwdriver
        self.scene.tool_screwdriver.spawn.usd_path = str(_ASSETS / OBJECT_SPECS[0].usd_file)
        _apply_target_rewards(self, "object")


@configclass
class AIR2RobotisScrewdriverFrankaEnvCfg_PLAY(AIR2RobotisScrewdriverFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
