# isaaclab_ext

Custom Isaac Lab assets and tasks for the UR3e + OnRobot RG2 setup.

## Layout

```
isaaclab_ext/
├── robots/
│   └── ur3e_rg2.py                  # UR3E_RG2_CFG and UR3E_RG2_HIGH_PD_CFG
└── tasks/
    └── lift_cube_ur3e_rg2/          # registers Isaac-Lift-Cube-UR3e-RG2-{v0, IK-Rel-v0, …}
        ├── __init__.py
        ├── joint_pos_env_cfg.py
        ├── ik_rel_env_cfg.py
        └── agents/
            └── rsl_rl_ppo_cfg.py
```

These files are **symlinked** into a local IsaacLab clone by `../setup_isaaclab.sh`,
so Isaac Lab's auto-discovery picks them up at runtime. Edit the files here in the
repo — changes are reflected in IsaacLab via the symlinks.

## Registered task IDs

| ID | Action space | Use |
|----|--------------|-----|
| `Isaac-Lift-Cube-UR3e-RG2-v0` | joint position (6 arm + 1 gripper) | RL training |
| `Isaac-Lift-Cube-UR3e-RG2-Play-v0` | joint position | small-batch RL eval |
| `Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0` | SE(3) pose delta + gripper | teleop / imitation learning |
| `Isaac-Lift-Cube-UR3e-RG2-IK-Rel-Play-v0` | SE(3) pose delta + gripper | teleop / IL eval |
