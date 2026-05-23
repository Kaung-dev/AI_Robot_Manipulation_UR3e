"""Pegboard pick-and-place task IDs — Franka Panda against the
robotis_net_table pegboard scene from exported_assets/."""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# Joint-position (RL training)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# IK-Rel (teleop / BC without cameras)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaPegboardLiftEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn.json",
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

# ---------------------------------------------------------------------------
# IK-Rel + cameras (VR demo recording) — one task per object
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image.json",
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

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Toothbrush-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorToothbrushEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Scissors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorScissorsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Silicone-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorSiliconeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Pliers-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_visuomotor_env_cfg:FrankaPegboardLiftVisuomotorPliersEnvCfg",
    },
    disable_env_checker=True,
)
