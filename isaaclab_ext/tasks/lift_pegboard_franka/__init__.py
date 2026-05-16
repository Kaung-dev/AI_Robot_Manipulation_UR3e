"""Pegboard pick-and-place task IDs — Franka Panda against the
robotis_net_table pegboard scene from exported_assets/."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Pegboard-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaPegboardLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegboardLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaPegboardLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegboardLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaPegboardLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaPegboardLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
