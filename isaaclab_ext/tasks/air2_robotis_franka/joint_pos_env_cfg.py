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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

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

        # Main pick target at slot L1.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_SLOTS["L1"], rot=_TOOL_ROT),
            spawn=UsdFileCfg(
                usd_path=str(_ASSETS / OBJECT_SPECS[0].usd_file),
                rigid_props=_TOOL_RIGID,
                mass_props=_TOOL_MASS,
            ),
        )

        # 3 distractors + main object = 4 total; reset event randomizes slots.
        for spec, slot in [
            (OBJECT_SPECS[1], "L0"),
            (OBJECT_SPECS[2], "R0"),
            (OBJECT_SPECS[3], "R1"),
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
        weight=6.0,                       # v3=3.0, v4=6.0 (stronger pull)
    )
    cfg.rewards.target_in_basket = RewTerm(
        func=air2_mdp.target_in_basket,
        params={"target_key": target_key, "radius": 0.30},
        weight=20.0,
    )
    cfg.rewards.wrong_object_moved = RewTerm(
        func=air2_mdp.wrong_object_moved,
        params={"target_key": target_key},
        weight=-1.0,
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
    cfg.terminations.task_success = TerminationTermCfg(
        func=air2_mdp.target_reached_basket,
        params={"target_key": target_key, "radius": 0.30},
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
        _apply_target_rewards(self, "tool_pliers")


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
        _apply_target_rewards(self, "tool_scissors")


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
        _apply_target_rewards(self, "tool_screwdriver")


@configclass
class AIR2RobotisScrewdriverFrankaEnvCfg_PLAY(AIR2RobotisScrewdriverFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
