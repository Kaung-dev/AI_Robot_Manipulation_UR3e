"""Pegboard pick-and-place task IDs — UR3e + RG2 against the
robotis_net_table pegboard scene from exported_assets/."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Pegboard-UR3e-RG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3eRG2PegboardLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3eRG2LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-UR3e-RG2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3eRG2PegboardLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3eRG2LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:UR3eRG2PegboardLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:UR3eRG2PegboardLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-UR3e-RG2-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:UR3eRG2PegboardLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-UR3e-RG2-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:UR3eRG2PegboardLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
