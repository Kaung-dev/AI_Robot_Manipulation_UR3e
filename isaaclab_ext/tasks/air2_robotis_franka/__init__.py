"""AIR2 scene with robotis_net_table as functional pegboard."""

import gymnasium as gym
from . import agents
from .mimic_env import AIR2RobotisFrankaMimicEnv

# Reuse the parent AIR2 task's PPO runner config (same actor/critic shape,
# same algorithm hyperparams) — Stephen's air2_franka.agents.rsl_rl_ppo_cfg.
_PPO_ENTRY = "isaaclab_ext.tasks.air2_franka.agents.rsl_rl_ppo_cfg:UR3eRG2AIR2LiftPPORunnerCfg"

gym.register(
    id="Isaac-AIR2-Robotis-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisFrankaEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_ENTRY,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisFrankaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _PPO_ENTRY,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Segmentation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2RobotisSegmentationEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_ENTRY,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.segmentation_env_cfg:AIR2RobotisSegmentationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _PPO_ENTRY,
    },
    disable_env_checker=True,
)

# --- Per-object tasks (4 pick targets) ------------------------------------

_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:AIR2RobotisPPORunnerCfg"

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Brush-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisBrushEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Brush-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisBrushEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Pliers-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisPliersFrankaEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Pliers-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisPliersFrankaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Scissors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScissorsFrankaEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Scissors-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScissorsFrankaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Screwdriver-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScrewdriverFrankaEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-AIR2-Robotis-Franka-Screwdriver-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AIR2RobotisScrewdriverFrankaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
    disable_env_checker=True,
)

# --- Mimic tasks (one per pick target) -------------------------------------

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Brush-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:AIR2RobotisFrankaMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:AIR2RobotisBrushMimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Pliers-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:AIR2RobotisFrankaMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:AIR2RobotisPliersMimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Scissors-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:AIR2RobotisFrankaMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:AIR2RobotisScissorsMimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-AIR2-Robotis-Franka-Screwdriver-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:AIR2RobotisFrankaMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:AIR2RobotisScrewdriverMimicEnvCfg",
    },
    disable_env_checker=True,
)
