"""Franka Panda pegboard task — IK-Rel + wrist & overhead cameras.

This is the visuomotor variant for camera-based imitation learning. Modeled
after the upstream ``stack_ik_rel_visuomotor_env_cfg.py`` but adapted to the
pegboard scene:

* ``wrist_cam`` is mounted on ``panda_hand`` (same prim as upstream Franka).
* ``table_cam`` is positioned to frame the pegboard front + the basket; the
  upstream values were tuned for a flat table with 3 cubes and are not
  useful here.

Two image observation terms are added to the policy group. ``concatenate_terms``
is turned off because RGB tensors can't be concatenated with the existing
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
    """Policy obs = lift env's low-dim terms + 2 RGB cameras."""

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

        # RGB cameras (84x84 to match upstream BC config defaults).
        table_cam = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
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
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # Overhead/front "table" camera. Placed in front of the pegboard,
        # ~70 cm up, looking back and down toward the robot + pegboard.
        # Tune these in CameraCfg.OffsetCfg if the framing is off — pos is in
        # the env's local frame (the env origin is between robot and pegboard).
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
                pos=(1.1, 0.0, 0.7),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        # Required for camera observations to stay in sync after env resets,
        # and to disable DLSS (which introduces frame-to-frame ghosting).
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.image_obs_list = ["table_cam", "wrist_cam"]


@configclass
class FrankaPegboardLiftVisuomotorEnvCfg_PLAY(FrankaPegboardLiftVisuomotorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
