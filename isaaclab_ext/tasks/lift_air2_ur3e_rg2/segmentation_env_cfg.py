"""AIR2 segmentation task variant with RGB, depth, and semantic cameras."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


SEGMENTATION_DATA_TYPES = [
    "rgb",
    "distance_to_image_plane",
    "semantic_segmentation",
    "instance_segmentation_fast",
]

SEMANTIC_MAPPING = {
    "class:toothbrush": (76, 175, 80, 255),
    "class:pliers": (255, 152, 0, 255),
    "class:scissors": (244, 67, 54, 255),
    "class:silicone": (33, 150, 243, 255),
    "class:robot": (158, 158, 158, 255),
    "class:environment": (121, 85, 72, 255),
    "class:BACKGROUND": (0, 0, 0, 255),
    "class:UNLABELLED": (0, 0, 0, 255),
}


def _set_air2_semantics(cfg) -> None:
    """Attach semantic labels to spawned AIR2 assets."""
    cfg.scene.robot.spawn.semantic_tags = [("class", "robot")]
    cfg.scene.table.spawn.semantic_tags = [("class", "environment")]
    cfg.scene.object.spawn.semantic_tags = [("class", "toothbrush")]
    cfg.scene.tool_pliers.spawn.semantic_tags = [("class", "pliers")]
    cfg.scene.tool_scissors.spawn.semantic_tags = [("class", "scissors")]
    cfg.scene.tool_silicone.spawn.semantic_tags = [("class", "silicone")]


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

    pinhole = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )

    # Fixed scene camera aimed at the AIR2 hooks. The pose is deliberately
    # separate from the wrist camera so data collection has a stable full-board
    # view even while the gripper moves.
    cfg.scene.board_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/board_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=list(SEGMENTATION_DATA_TYPES),
        colorize_semantic_segmentation=True,
        colorize_instance_segmentation=True,
        semantic_segmentation_mapping=SEMANTIC_MAPPING,
        spawn=pinhole,
        offset=CameraCfg.OffsetCfg(
            pos=(-4.10, -4.20, 1.85),
            rot=(0.50, -0.50, -0.50, 0.50),
            convention="ros",
        ),
    )

    cfg.rerender_on_reset = True
    cfg.sim.render.antialiasing_mode = "OFF"
    cfg.image_obs_list = ["board_camera", "wrist_camera"]


@configclass
class AIR2SegmentationEnvCfg(joint_pos_env_cfg.FrankaAIR2LiftEnvCfg):
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
