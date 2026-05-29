"""AIR2 scene with robotis_net_table as functional pegboard."""

import gymnasium as gym

gym.register(
    id="Isaac-AIR2-Robotis-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisFrankaEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisFrankaEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Segmentation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2RobotisSegmentationEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2RobotisSegmentationEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

# --- Per-object tasks (4 pick targets) ------------------------------------

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Brush-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisBrushEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Brush-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisBrushEnvCfg_PLAY"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Pliers-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisPliersFrankaEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Pliers-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisPliersFrankaEnvCfg_PLAY"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Scissors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScissorsFrankaEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Scissors-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScissorsFrankaEnvCfg_PLAY"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Screwdriver-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScrewdriverFrankaEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Screwdriver-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScrewdriverFrankaEnvCfg_PLAY"},
    disable_env_checker=True,
)
