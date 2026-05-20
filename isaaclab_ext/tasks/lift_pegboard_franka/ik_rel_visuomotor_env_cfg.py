"""Franka Panda pegboard task — IK-Rel + wrist & overhead cameras.

Visuomotor variant for VR-based imitation learning data collection.

Changes vs the base IK-Rel config:
- 60 s episode length so the operator can complete a full pick→carry→drop.
- Goal command does not resample within an episode (resampling_time_range > episode_length_s).
- wrist_cam on panda_hand (84x84 RGB+depth, top-down view of fingers).
- table_cam facing the pegboard + basket (84x84 RGB+depth).
- rerender_on_reset and DLSS disabled to avoid ghosting after env resets.

Four per-object variants:
    FrankaPegboardLiftVisuomotorToothbrushEnvCfg  — L1 toothbrush
    FrankaPegboardLiftVisuomotorScissorsEnvCfg    — L3 scissors
    FrankaPegboardLiftVisuomotorSiliconeEnvCfg    — L2 silicone tube
    FrankaPegboardLiftVisuomotorPliersEnvCfg      — R0 pliers
"""

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import image as mdp_image
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import ik_rel_env_cfg


@configclass
class ObservationsCfg:
    """Policy obs = low-dim terms from the lift env + 2 RGB cameras."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        table_cam = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _apply_visuomotor(cfg) -> None:
    """Add cameras, extend episode length, and fix goal resampling."""
    cfg.episode_length_s = 60.0
    # Prevent goal from resampling mid-episode — operator decides where to place.
    cfg.commands.object_pose.resampling_time_range = (65.0, 65.0)

    cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )

    cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.1, 0.0, 0.7),
            rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
    )

    cfg.rerender_on_reset = True
    cfg.sim.render.antialiasing_mode = "OFF"
    cfg.image_obs_list = ["table_cam", "wrist_cam"]


# ---------------------------------------------------------------------------
# Toothbrush (original default, now explicit)
# ---------------------------------------------------------------------------

@configclass
class FrankaPegboardLiftVisuomotorToothbrushEnvCfg(ik_rel_env_cfg.FrankaPegboardLiftEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_visuomotor(self)


@configclass
class FrankaPegboardLiftVisuomotorToothbrushEnvCfg_PLAY(FrankaPegboardLiftVisuomotorToothbrushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


# ---------------------------------------------------------------------------
# Scissors
# ---------------------------------------------------------------------------

@configclass
class FrankaPegboardLiftVisuomotorScissorsEnvCfg(ik_rel_env_cfg.FrankaPegboardLiftScissorsEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_visuomotor(self)


@configclass
class FrankaPegboardLiftVisuomotorScissorsEnvCfg_PLAY(FrankaPegboardLiftVisuomotorScissorsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


# ---------------------------------------------------------------------------
# Silicone tube
# ---------------------------------------------------------------------------

@configclass
class FrankaPegboardLiftVisuomotorSiliconeEnvCfg(ik_rel_env_cfg.FrankaPegboardLiftSiliconeEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_visuomotor(self)


@configclass
class FrankaPegboardLiftVisuomotorSiliconeEnvCfg_PLAY(FrankaPegboardLiftVisuomotorSiliconeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


# ---------------------------------------------------------------------------
# Pliers
# ---------------------------------------------------------------------------

@configclass
class FrankaPegboardLiftVisuomotorPliersEnvCfg(ik_rel_env_cfg.FrankaPegboardLiftPliersEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_visuomotor(self)


@configclass
class FrankaPegboardLiftVisuomotorPliersEnvCfg_PLAY(FrankaPegboardLiftVisuomotorPliersEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


# ---------------------------------------------------------------------------
# Backwards-compatible alias for the original single-object visuomotor task
# ---------------------------------------------------------------------------

FrankaPegboardLiftVisuomotorEnvCfg = FrankaPegboardLiftVisuomotorToothbrushEnvCfg
FrankaPegboardLiftVisuomotorEnvCfg_PLAY = FrankaPegboardLiftVisuomotorToothbrushEnvCfg_PLAY
