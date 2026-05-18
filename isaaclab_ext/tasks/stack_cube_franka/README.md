# Franka cube-stack — reference copy

This is a verbatim copy of upstream Isaac Lab's `isaaclab_tasks/manager_based/manipulation/stack/`, included here as a study reference for the **camera + IL + domain-randomization** pattern. The same task is already registered by your installed Isaac Lab — you don't need to re-register from this copy.

Original location:
`<IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack`

## Why it's in our repo

To serve as the template we'll adapt for our UR3e + RG2 tasks. See [`CAMERA_AND_TELEOP.md`](../../../CAMERA_AND_TELEOP.md) Section 3 ("what the upstream does better than ours") and Section 4 ("how CNN + IL + domain-rand + RL fit together") — those sections reference this code by file:line.

## Layout

```
stack_cube_franka/
├── __init__.py                              # parent package marker (empty)
├── stack_env_cfg.py                         # base scene + obs + actions (robot-agnostic)
├── stack_instance_randomize_env_cfg.py      # variant: multiple cube instances per env
├── mdp/
│   ├── __init__.py                          # re-exports observations + terminations
│   ├── observations.py                      # object_grasped, object_stacked, object_obs, etc.
│   ├── terminations.py                      # cubes_stacked success
│   └── franka_stack_events.py               # domain-rand events: joint noise, cube pose
└── config/
    └── franka/
        ├── __init__.py                      # registers 7 gym task IDs
        ├── stack_joint_pos_env_cfg.py       # base + EventCfg domain randomization
        ├── stack_ik_rel_env_cfg.py          # IK-Rel control (proprio obs only)
        ├── stack_ik_abs_env_cfg.py          # IK-Abs (for state-machine demos)
        ├── stack_ik_rel_visuomotor_env_cfg.py        # ★ IK-Rel + wrist & table cameras
        ├── stack_ik_rel_instance_randomize_env_cfg.py
        ├── stack_ik_rel_blueprint_env_cfg.py
        ├── stack_joint_pos_instance_randomize_env_cfg.py
        └── agents/
            ├── __init__.py
            └── robomimic/
                ├── bc_rnn_image_84.json     # ★ CNN + LSTM config (ResNet18 + SpatialSoftmax)
                └── bc_rnn_low_dim.json      # MLP + LSTM config (no cameras)
```

★ = the two files referenced most heavily in CAMERA_AND_TELEOP.md.

## Reading order

1. **`stack_env_cfg.py`** — `ObservationsCfg` shows the 3-group pattern (`policy`, `rgb_camera`, `subtask_terms`). `TerminationsCfg` shows the per-cube `cube_X_dropping` early-stop.
2. **`config/franka/stack_joint_pos_env_cfg.py`** — `EventCfg` is the in-sim domain randomization (joint Gaussian noise, cube pose ranges, min_separation). This is the parent that the visuomotor variant inherits from.
3. **`config/franka/stack_ik_rel_visuomotor_env_cfg.py`** — the actual camera definitions (`wrist_cam`, `table_cam`), the override that adds `mdp.image` obs terms, the two render flags (`rerender_on_reset`, `antialiasing_mode="OFF"`).
4. **`config/franka/__init__.py`** — gym task registrations. Note `robomimic_bc_cfg_entry_point` pointing at the JSON below.
5. **`config/franka/agents/robomimic/bc_rnn_image_84.json`** —
   - lines 149–169 (`observation.modalities`): which obs keys are low-dim vs RGB.
   - lines 171–204 (`observation.encoder.rgb`): the CNN config (ResNet18Conv + SpatialSoftmax + 64-D feature + CropRandomizer augmentation).
   - lines 123–133 (`algo.rnn`): the LSTM policy head (hidden_dim=1000, 2 layers, horizon=10).
6. **`mdp/franka_stack_events.py`** — implementation of the domain-rand event functions referenced by `EventCfg`.
7. **`mdp/observations.py`** — `object_grasped`, `object_stacked`, `object_obs` implementations. Useful for adding subtask booleans to our pegboard.

## Running this task

Since the upstream Isaac Lab install already has these files registered, run them through the normal task IDs (no extra setup):

```powershell
cd <IsaacLab>

# Teleop with both cameras visible
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py `
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 `
    --num_envs 1 --teleop_device keyboard --enable_cameras

# Record demos (HDF5 with images + actions)
.\isaaclab.bat -p scripts\tools\record_demos.py `
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 `
    --teleop_device keyboard --enable_cameras `
    --dataset_file <repo>\datasets\stack_demos.hdf5

# Train BC-RNN with CNN encoder
.\isaaclab.bat -p scripts\imitation_learning\robomimic\train.py `
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --algo bc `
    --dataset <repo>\datasets\stack_demos.hdf5

# Play the trained policy
.\isaaclab.bat -p scripts\imitation_learning\robomimic\play.py `
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 `
    --checkpoint <path-to-model.pth> --enable_cameras
```

On Linux substitute `./isaaclab.sh` for `.\isaaclab.bat`. `<IsaacLab>` is wherever you cloned IsaacLab (`~/IsaacLab` is the default in `setup_isaaclab.sh`); `<repo>` is the root of this repo.

## What we'd change to port this to UR3e + RG2 pegboard

1. Replace `FRANKA_PANDA_HIGH_PD_CFG` with `UR3E_RG2_HIGH_PD_CFG`.
2. Change `panda_joint.*` joint regex to UR3e's 6 joints, gripper to `rg2_gripper.*`.
3. Change `panda_hand` mount point to `wrist_3_link` and adjust camera offsets (UR3e flange is in a different pose).
4. Replace the 3 cubes with our 6 tools (already done in our pegboard task — we'd add the cameras + obs split here).
5. Edit `bc_rnn_image_84.json` `observation.modalities.obs.low_dim` to reference UR3e-specific keys.
6. Re-tune `EventCfg` pose ranges for the pegboard layout.
