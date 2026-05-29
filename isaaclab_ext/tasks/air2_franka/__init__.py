"""Pick task — UR3e + RG2 in the AIR2 scene with 8-hook randomization."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-AIR2-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2FrankaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3eRG2AIR2LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2FrankaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3eRG2AIR2LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Franka-Segmentation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2SegmentationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3eRG2AIR2LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Franka-Segmentation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2SegmentationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3eRG2AIR2LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)
