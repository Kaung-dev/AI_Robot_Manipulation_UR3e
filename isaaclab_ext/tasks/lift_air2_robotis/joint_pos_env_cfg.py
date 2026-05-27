"""AIR2 scene with robotis_net_table as functional pegboard.

Mirrors lift_pegboard_ur3e_rg2: objects spawn directly at cylinder slot
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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp

from isaaclab_ext.tasks.lift_air2_ur3e_rg2.joint_pos_env_cfg import (
    FrankaAIR2LiftEnvCfg,
    _TOOL_RIGID,
    _TOOL_MASS,
)

_HERE = Path(__file__).resolve()
_REPO_CANDIDATES = [
    _HERE.parents[3],
    Path("/home/user/Desktop/ur_pick"),
]
_REPO_ROOT = next((p for p in _REPO_CANDIDATES if (p / "exported_assets").exists()), _REPO_CANDIDATES[-1])
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
class FrankaAIR2RobotisLiftEnvCfg(FrankaAIR2LiftEnvCfg):
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
                usd_path=str(_ASSETS / "brush_ring.usd"),
                rigid_props=_TOOL_RIGID,
                mass_props=_TOOL_MASS,
            ),
        )

        # 3 distractors + main object = 4 total; reset event randomizes slots.
        for name, fname, slot in [
            ("tool_pliers",   "pliers_ring_orange.usd",      "L0"),
            ("tool_scissors", "scissors_ring_red.usd",       "R0"),
            ("tool_silicone", "screw_driver_ring.usd",       "R1"),
        ]:
            setattr(
                self.scene, name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name.title().replace('_', '')}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=_SLOTS[slot], rot=_TOOL_ROT),
                    spawn=UsdFileCfg(
                        usd_path=str(_ASSETS / fname),
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
class FrankaAIR2RobotisLiftEnvCfg_PLAY(FrankaAIR2RobotisLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
