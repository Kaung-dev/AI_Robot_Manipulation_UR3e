"""AIR2 scene with robotis_net_table as functional pegboard."""

import gymnasium as gym

gym.register(
    id="Isaac-Lift-AIR2-Robotis-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaAIR2RobotisLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-AIR2-Robotis-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaAIR2RobotisLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-AIR2-Robotis-Segmentation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2RobotisSegmentationEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-AIR2-Robotis-Segmentation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2RobotisSegmentationEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
