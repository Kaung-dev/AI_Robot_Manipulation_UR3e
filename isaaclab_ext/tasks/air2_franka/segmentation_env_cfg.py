"""AIR2 segmentation task variant with RGB, depth, and semantic cameras."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
from .mdp import events as _air2_events
from .objects import OBJECT_SPECS


SEGMENTATION_DATA_TYPES = [
    "rgb",
    "distance_to_image_plane",
    "semantic_segmentation",
    "instance_segmentation_fast",
]

SEMANTIC_MAPPING = {
    **{f"class:{spec.label}": (*spec.color_rgb, 255) for spec in OBJECT_SPECS},
    "class:robot": (158, 158, 158, 255),
    "class:basket": (255, 235, 59, 255),     # bright yellow
    "class:table": (96, 125, 139, 255),      # blue-grey
    "class:environment": (121, 85, 72, 255),
    "class:BACKGROUND": (0, 0, 0, 255),
    "class:UNLABELLED": (0, 0, 0, 255),
}


def _set_air2_semantics(cfg) -> None:
    """Attach semantic labels to spawned AIR2 assets.

    Note: cfg.scene.table (the whole AIR2.usd payload — walls, floor, pegboard,
    basket) is NOT tagged at spawn time. Tagging it as 'table' here would
    recursively apply 'table' to the basket child too, and the post-load
    Semantics API doesn't reliably overwrite that for the colorize annotator.
    Instead, the startup event tag_basket_and_table() in mdp/events.py walks
    the loaded stage and tags basket prims as 'basket' and other AIR2 prims
    as 'table' — clean slate, no inheritance conflict.
    """
    cfg.scene.robot.spawn.semantic_tags = [("class", "robot")]
    # cfg.scene.table.spawn.semantic_tags intentionally NOT set — handled by event.
    for spec in OBJECT_SPECS:
        getattr(cfg.scene, spec.scene_key).spawn.semantic_tags = [("class", spec.label)]


def _apply_segmentation_cameras(cfg) -> None:
    """Enable segmentation annotators and add a board-facing camera."""
    _set_air2_semantics(cfg)

    # Keep the wrist camera for close object/gripper views, but request all
    # annotators needed for supervised segmentation datasets.
    cfg.scene.wrist_camera.data_types = list(SEGMENTATION_DATA_TYPES)
    # Isaac Sim 5.1 can return RGBA segmentation buffers even when the
    # non-colorized annotator is requested. Keep colorized output stable and
    # remap known colors back to class IDs in the dataset collector.
    cfg.scene.wrist_camera.colorize_semantic_segmentation = True
    cfg.scene.wrist_camera.colorize_instance_segmentation = True
    cfg.scene.wrist_camera.semantic_segmentation_mapping = SEMANTIC_MAPPING

    # Wider 18 mm lens (≈60° HFOV) so the whole work cell fits in one frame:
    # robot, pegboard with hooks, tools, and the basket.
    pinhole_wide = sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )

    # Third-person 3/4 view aimed at the actual action zone confirmed by
    # inspect_air2_hooks.py:
    #   - pegboard wall     y = -6.01 (back)
    #   - object spawn line y = -5.90 (in front of the wall, where reset places them)
    #   - basket            (-3.56, -5.37, 1.04)
    #   - Franka base       (-4.24, -5.29, 1.04)
    # Look-at = (-3.9, -5.7, 1.3) (centre of the strip).
    # Camera position is offset to +X and +Y of the cluster, ~3 m away, raised
    # to 2.2 m so the operator can see the arm reach back into the board.
    cfg.scene.board_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/board_camera",
        update_period=0.0,
        height=360,
        width=640,
        data_types=list(SEGMENTATION_DATA_TYPES),
        colorize_semantic_segmentation=True,
        colorize_instance_segmentation=True,
        semantic_segmentation_mapping=SEMANTIC_MAPPING,
        spawn=pinhole_wide,
        offset=CameraCfg.OffsetCfg(
            pos=(-2.0, -3.5, 2.2),
            rot=(-0.2068, 0.2807, 0.7545, -0.5560),
            convention="ros",
        ),
    )

    cfg.rerender_on_reset = True
    cfg.sim.render.antialiasing_mode = "OFF"
    cfg.image_obs_list = ["board_camera", "wrist_camera"]

    # Hide the goal_pose / body_pose / EE-frame debug arrows so they don't
    # block the camera views during dataset collection.
    if hasattr(cfg.commands, "object_pose"):
        cfg.commands.object_pose.debug_vis = False
    if hasattr(cfg.scene, "ee_frame") and cfg.scene.ee_frame is not None:
        cfg.scene.ee_frame.debug_vis = False
    if hasattr(cfg.actions, "arm_action") and cfg.actions.arm_action is not None:
        if hasattr(cfg.actions.arm_action, "debug_vis"):
            cfg.actions.arm_action.debug_vis = False

    # Startup event: override the broad 'environment' semantic tag on the basket
    # and table prims so they get their own colors in the segmentation mask.
    cfg.events.tag_basket_and_table = EventTerm(
        func=_air2_events.tag_basket_and_table,
        mode="startup",
    )


@configclass
class AIR2SegmentationEnvCfg(joint_pos_env_cfg.AIR2FrankaEnvCfg):
    """AIR2 task variant for collecting CNN segmentation data."""

    def __post_init__(self):
        super().__post_init__()
        _apply_segmentation_cameras(self)


@configclass
class AIR2SegmentationEnvCfg_PLAY(AIR2SegmentationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
