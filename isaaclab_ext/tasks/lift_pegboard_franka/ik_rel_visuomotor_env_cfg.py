"""Franka Panda pegboard task — IK-Rel + wrist camera.

This is the visuomotor variant for camera-based imitation learning.
``wrist_cam`` is mounted on ``panda_hand`` and gives a top-down view of the
gripper fingers — useful for learning precise grasp alignment.

The image observation term is added to the policy group. ``concatenate_terms``
is turned off because the RGB tensor can't be concatenated with the existing
low-dim observations into a single flat vector — the BC pipeline reads them
as a dict.
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
    """Policy obs = lift env's low-dim terms + wrist RGB camera."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Low-dim terms from the parent lift env.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        # Wrist RGB camera (84x84 to match upstream BC config defaults).
        wrist_cam = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        # Fixed table camera — full workspace coverage.
        table_cam = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            # Off because RGB tensors can't be flat-concatenated with low-dim.
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaPegboardLiftVisuomotorEnvCfg(ik_rel_env_cfg.FrankaPegboardLiftEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Wrist camera on panda_hand. Offset copied from upstream Franka
        # stack-cube visuomotor cfg — it gives a good top-down view of the
        # gripper fingers, which is what we want when picking from a peg.
        self.scene.wrist_cam = CameraCfg(
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
                pos=(0.15, 0.0, -0.3),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # Fixed table camera — overhead view of full workspace.
        self.scene.table_cam = CameraCfg(
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
                pos=(0.5, 0.25, 1.1),
                rot=(0.7071, 0.0, 0.7071, 0.0),
                convention="world",
            ),
        )

        # Required for camera observations to stay in sync after env resets,
        # and to disable DLSS (which introduces frame-to-frame ghosting).
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.image_obs_list = ["wrist_cam", "table_cam"]


@configclass
class FrankaPegboardLiftVisuomotorEnvCfg_PLAY(FrankaPegboardLiftVisuomotorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
