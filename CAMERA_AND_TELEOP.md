# Adding cameras + moving the robot

Two things this doc covers:

1. **How Isaac Lab wires cameras into a manipulation task** — reading from the upstream `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0` example so we can copy the pattern into our UR3e tasks.
2. **How to move the robot once the task is running** — keyboard / gamepad / spacemouse / VR controls.

---

## 1. Cameras

Reference example (full reference copy mirrored into this repo at [`isaaclab_ext/tasks/stack_cube_franka/`](isaaclab_ext/tasks/stack_cube_franka/); the original lives in your IsaacLab clone at `<IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py`):

### 1a. Where the cameras live

Two cameras, both spawned **inside the per-env namespace** (`{ENV_REGEX_NS}` is replaced with `/World/envs/env_0`, `env_1`, …).

| Camera | Mount point | Purpose |
|---|---|---|
| `wrist_cam` | child of `Robot/panda_hand` — moves with the gripper | first-person "what the gripper sees" |
| `table_cam` | child of the per-env root — fixed overhead view | third-person scene view |

Camera config block (line 119–146 of the upstream file):

```python
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

self.scene.wrist_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",   # mount under the hand link
    update_period=0.0,                                       # update every render step
    height=84, width=84,                                     # match robomimic input size
    data_types=["rgb", "distance_to_image_plane"],           # RGB + depth
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 2),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.13, 0.0, -0.15),                              # 13 cm fwd, 15 cm down from hand origin
        rot=(-0.70614, 0.03701, 0.03701, -0.70614),          # quaternion (w,x,y,z)
        convention="ros",                                    # ROS = +Z forward, +X right, +Y down
    ),
)

self.scene.table_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/table_cam",                    # mount at env root, not on the robot
    update_period=0.0,
    height=84, width=84,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 2),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(1.0, 0.0, 0.4),                                 # 1 m back, 40 cm up from env origin
        rot=(0.35355, -0.61237, -0.61237, 0.35355),
        convention="ros",
    ),
)
```

Key things to know:

- **`prim_path`** decides what the camera is attached to. Put it under a robot link → it moves with that link. Put it under the env root → it's fixed in the world.
- **`offset.pos` / `offset.rot`** are relative to the parent prim. Tune by hand or by dragging in Isaac Sim, then copying back.
- **`height` / `width`** are the rendered resolution. 84×84 is the robomimic default; bump to 224×224 for larger pretrained backbones at higher GPU cost.
- **`data_types`** controls what tensors come out per step. `"rgb"` is required for vision; `"distance_to_image_plane"` is the depth image. Other options: `"semantic_segmentation"`, `"instance_segmentation"`, `"normals"`.

### 1b. Wiring cameras into observations

In the same file (lines 46–51), the camera tensors are added to the observation dict via `mdp.image`:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from ... import mdp

class PolicyCfg(ObsGroup):
    # ... other obs (joint_pos, eef_pos, gripper_pos, etc.)
    table_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
    )
    wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False   # IMPORTANT — images stay as separate tensors
```

`concatenate_terms = False` matters: image obs are kept as their own (H, W, C) tensors instead of being flattened+concatenated with the proprio vector.

### 1c. Two render-pipeline flags you must set

Same file, lines 149–150:

```python
self.rerender_on_reset = True               # re-render after reset() so first obs has fresh pixels
self.sim.render.antialiasing_mode = "OFF"   # disable DLSS — DLSS causes ghosting in small renders
```

### 1d. Launching anything with cameras requires `--enable_cameras`

Cameras are off by default because they add ~20–40 s to scene init and a render pass per step. Always add the flag:

```powershell
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py `
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 `
    --num_envs 1 --teleop_device keyboard `
    --enable_cameras
```

On a Tesla T4, expect 60–90 seconds before the viewport appears with cameras enabled.

### 1e. Porting this to the UR3e tasks

To add cameras to our pegboard task ([`isaaclab_ext/tasks/lift_pegboard_ur3e_rg2/joint_pos_env_cfg.py`](isaaclab_ext/tasks/lift_pegboard_ur3e_rg2/joint_pos_env_cfg.py)):

1. Add `from isaaclab.sensors import CameraCfg` at the top.
2. Inside `__post_init__`, after the existing scene setup, add:
   - `self.scene.wrist_cam` with `prim_path="{ENV_REGEX_NS}/Robot/ur3e/wrist_3_link/wrist_cam"` and an offset that points the lens toward the gripper tips (try `pos=(0.0, 0.0, 0.15)` to look down from the flange).
   - `self.scene.table_cam` with `prim_path="{ENV_REGEX_NS}/table_cam"`, `pos=(0.6, 0.0, 0.6)` and a rotation pointing back at the pegboard.
3. Set `self.rerender_on_reset = True` and `self.sim.render.antialiasing_mode = "OFF"`.
4. Override the observation manager (subclass `ObservationsCfg` or duplicate the visuomotor pattern), adding `wrist_cam` and `table_cam` as `mdp.image` obs terms.
5. Register a new visuomotor task ID in `__init__.py`.

---

## 2. Moving the robot

You can move the robot via four channels — pick whichever fits the hardware you have.

### 2a. Keyboard (default, no extra hardware)

Launch (one of the IK-Rel tasks — joint-pos won't accept SE(3) deltas):

```powershell
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py `
    --task Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-v0 `
    --num_envs 1 --teleop_device keyboard
```

Then **click the 3D viewport** (focus must be on the Isaac Sim window) and press **L** to start. Controls:

| key | action |
|---|---|
| `W` / `S` | translate +X / −X (forward / back) |
| `A` / `D` | translate +Y / −Y (left / right) |
| `Q` / `E` | translate +Z / −Z (up / down) |
| `Z` / `X` | rotate roll |
| `T` / `G` | rotate pitch |
| `C` / `V` | rotate yaw |
| `K` | toggle gripper open / close |
| `L` | start teleop / reset env |

Holding a key produces continuous motion. The action is a 6-D SE(3) delta + 1 gripper bit; the env's IK solver maps it to UR3e joint targets every step.

### 2b. Gamepad

```powershell
... --teleop_device gamepad
```

Plug the controller in **before** launching. Left stick = XY translate, right stick = pitch/yaw, triggers = ±Z, bumpers = roll, A/B = gripper.

### 2c. SpaceMouse (3Dconnexion)

```powershell
... --teleop_device spacemouse
```

Native 6-DoF input — push/pull/twist the puck for EE pose deltas. Side buttons toggle gripper.

### 2d. VR (Meta Quest 2 via OpenXR)

```powershell
... --teleop_device handtracking
```

Requires Monado + WiVRn set up (see `REBUILD_GUIDE.md` Step 9). Right-hand wrist motion drives the EE; thumb+index pinch toggles the gripper.

⚠️ Currently disabled in our IsaacLab copies — the installed `isaaclab-0.40.6` lacks `DevicesCfg`. The repo originals under `isaaclab_ext/` still have the VR block; needs an IsaacLab upgrade to use.

### 2e. Programmatic (state-machine auto-pick)

For non-interactive runs, use the absolute-IK variants with a Python state machine:

```powershell
.\isaaclab.bat -p <repo>\scripts\pick_pegboard_auto.py --num_envs 1
```

This drives the `*-IK-Abs-v0` task with a warp FSM (rest → approach → grasp → lift → release). Good for generating demo data without a human in the loop.

### 2f. Tweaking responsiveness

Two places control how aggressively your input maps to robot motion:

- **Keyboard sensitivity** — the `teleop_se3_agent.py` script has `--sensitivity` flag (default 1.0); raise to make each key press move further.
- **IK scale** — in [`ik_rel_env_cfg.py`](isaaclab_ext/tasks/lift_pegboard_ur3e_rg2/ik_rel_env_cfg.py), `DifferentialInverseKinematicsActionCfg(... scale=0.5)`. Higher = bigger joint moves per action; lower = smoother but slower.

---

## 3. What the upstream Franka visuomotor task does better than ours

The teleop script is the same — `teleop_se3_agent.py`. What's different is the **task definition** wired around it. Reading `stack_env_cfg.py` + `stack_ik_rel_visuomotor_env_cfg.py`, here's what they have that our `lift_pegboard_ur3e_rg2` does not, and is worth copying:

### 3a. Three observation groups (clean separation)

```python
policy:        PolicyCfg          # proprio: joint pos/vel, eef pose, gripper, last_action
rgb_camera:    RGBCameraPolicyCfg # images only — kept separate so the policy net can route them through a CNN
subtask_terms: SubtaskCfg         # boolean success flags per sub-step
```

Why it matters: robomimic / hierarchical policies want images on a separate group so a CNN backbone handles them while the proprio goes through an MLP. Concatenating everything into one vector (what we currently do) forces flattening and breaks image inputs.

### 3b. Subtask boolean observations

`mdp.object_grasped` and `mdp.object_stacked` produce per-step booleans like "is cube_2 currently grasped" / "is cube_2 stacked on cube_1". They're:

- **Free success labels for BC** — robomimic can filter demos by which subtasks were completed.
- **Useful for hierarchical RL** — sub-policy gating.
- **Free debug overlay** in teleop — you immediately see whether a grasp registered.

We could add `mdp.object_grasped` for the toothbrush + `mdp.object_in_basket` for the place subtask.

### 3c. Consolidated `object_obs`

`mdp.object_obs` returns a single tensor with position/orientation/velocity of every relevant object. Ours has `cube` only and re-derives it ad-hoc. Switching to `mdp.object_obs` gives the policy a uniform view of all 6 pegboard tools, which would let one trained policy pick whichever tool it sees.

### 3d. Per-object termination on "dropped"

```python
cube_1_dropping = DoneTerm(func=mdp.root_height_below_minimum,
                           params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")})
```

Episode ends early if a cube falls off the table. We currently rely on `time_out` only — wasted physics steps on already-failed episodes. Adding `tool_dropping` per tool would speed up RL.

### 3e. Image-conditioned BC config already wired

In the task registration:

```python
"robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
```

That JSON spells out a BC-RNN with image encoder, hidden size, recurrent length, etc. — ready to train against teleop demos by:

```powershell
.\isaaclab.bat -p scripts\imitation_learning\robomimic\train.py `
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --algo bc `
    --dataset <demos.hdf5>
```

For our pegboard, we'd need an equivalent JSON under `isaaclab_ext/tasks/lift_pegboard_ur3e_rg2/agents/robomimic/` and register it.

### 3f. Two render-pipeline flags

```python
self.rerender_on_reset = True
self.sim.render.antialiasing_mode = "OFF"
```

First one guarantees the first obs after `reset()` has fresh pixels (otherwise you get a stale frame from the previous episode → corrupts BC targets). Second one disables DLSS, which causes temporal ghosting in 84×84 renders.

### 3g. Action-space ergonomics

Their `DifferentialInverseKinematicsActionCfg` uses `scale=0.5` and `body_offset=[0, 0, 0.107]` (the Franka TCP). Ours uses `scale=0.5` and `[0, 0, 0.18]` (RG2 TCP) — that's fine. But they tune the `xr.anchor_pos` and per-key sensitivity carefully for the Franka workspace. We should re-tune for the UR3e's smaller reach: the current 5× VR scale and default keyboard sensitivity feel sluggish.

---

### Adoption priority for us

If we want to climb the most-impact rungs first:

1. **Add wrist + table cameras** to `joint_pos_env_cfg.py` (Section 1e above).
2. **Split observations** into `policy` / `rgb_camera` / `subtask_terms` groups.
3. **Add subtask booleans** — `tool_grasped`, `tool_in_basket`.
4. **Add per-tool drop terminations**.
5. **Write `bc_rnn_image_84.json` adaptation** for UR3e + register it.
6. **Set the two render flags.**

Items 1, 2, 6 are mechanical copy-paste. Items 3, 4 need a few lines of MDP code per term. Item 5 is a JSON edit + one line of registration.

---

## 4. How CNN + IL + domain-rand + RL fit together in this task

There are four moving parts. Here's exactly where each lives.

### 4a. The CNN — lives in robomimic, not in the task config

The task config only says "feed these two image tensors to the policy". The CNN that turns images into features is defined in `bc_rnn_image_84.json` ([isaaclab_ext/tasks/stack_cube_franka/config/franka/agents/robomimic/bc_rnn_image_84.json](isaaclab_ext/tasks/stack_cube_franka/config/franka/agents/robomimic/bc_rnn_image_84.json)):

```json
"rgb": {
    "core_class": "VisualCore",
    "core_kwargs": {
        "feature_dimension": 64,            // each camera → 64-D feature vector
        "backbone_class": "ResNet18Conv",   // ResNet-18 trunk (no pretrained weights)
        "pool_class": "SpatialSoftmax",     // 32 learned keypoints, smoother than avg-pool
        "pool_kwargs": {"num_kp": 32}
    },
    "obs_randomizer_class": "CropRandomizer",   // training-time augmentation
    "obs_randomizer_kwargs": {"crop_height": 76, "crop_width": 76, "num_crops": 1}
}
```

Flow per step:

```
  table_cam (84×84×3) ─┐
                       ├─→ random-crop to 76×76  ─→ ResNet18Conv ─→ SpatialSoftmax(32 keypoints) ─→ 64-D feat
  wrist_cam (84×84×3) ─┘                                                                              │
                                                                                                       ▼
  eef_pos, eef_quat, gripper_pos (low_dim) ─→ identity (no encoder) ─→ proprio vec ─────────────────→ concat ─→ LSTM
```

The CNN weights are **trained jointly** with the policy head — it's not a frozen feature extractor. Pretrained weights are off (`"pretrained": false`) because robomimic doesn't ship them; if you wanted ImageNet init you'd flip that flag.

### 4b. Imitation Learning — BC-RNN over teleop demos

Same JSON, the policy head:

```json
"rnn": {
    "enabled": true,
    "horizon": 10,                  // 10-step rollout window during training
    "hidden_dim": 1000,
    "rnn_type": "LSTM",
    "num_layers": 2
}
```

How a training run goes end-to-end:

1. **Collect demos** with teleop (`record_demos.py`):
   ```powershell
   .\isaaclab.bat -p scripts\tools\record_demos.py `
       --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --teleop_device keyboard `
       --dataset_file <path>\demos.hdf5 --enable_cameras
   ```
   Each successful episode appends an entry to the HDF5 with: `obs/eef_pos`, `obs/eef_quat`, `obs/gripper_pos`, `obs/table_cam`, `obs/wrist_cam`, `actions`, and the `subtask_terms` booleans.

2. **Train BC-RNN**:
   ```powershell
   .\isaaclab.bat -p scripts\imitation_learning\robomimic\train.py `
       --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --algo bc `
       --dataset <path>\demos.hdf5
   ```
   The task-registration `robomimic_bc_cfg_entry_point` tells `train.py` which JSON to use — that's how the CNN config above gets picked up automatically. No CLI override needed.

3. **Play the trained policy**:
   ```powershell
   .\isaaclab.bat -p scripts\imitation_learning\robomimic\play.py `
       --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 `
       --checkpoint <path>\model.pth --enable_cameras
   ```

The LSTM matters: most pick-and-place demos have multi-step structure (approach → grasp → lift → carry). A feed-forward policy at the same horizon stalls; the recurrence carries "I'm in the grasping phase" forward.

### 4c. Domain randomization — Isaac Lab `EventCfg` + robomimic `CropRandomizer`

Randomization happens in **two places**:

**(i) In sim, per reset** ([`isaaclab_ext/tasks/stack_cube_franka/config/franka/stack_joint_pos_env_cfg.py:29-57`](isaaclab_ext/tasks/stack_cube_franka/config/franka/stack_joint_pos_env_cfg.py#L29-L57)):

```python
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="startup",                                    # runs once on env build
        params={"default_pose": [...]},
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",                                      # runs every episode reset
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )
```

What gets randomized:
- **Robot start joints** — small Gaussian noise (std=0.02 rad) on each joint at reset.
- **All 3 cube xy + yaw** — within a 20×20 cm patch, with 10 cm minimum separation so they don't overlap.

What's **not** randomized here (you'd add if you want sim-to-real robustness): lighting, materials, camera pose, friction, cube colors. The `stack_instance_randomize_env_cfg.py` variant adds instance variety (different cube meshes per episode).

**(ii) At training time, per minibatch** (the robomimic `CropRandomizer` from §4a) — every image fed to the CNN gets a different 76×76 random crop from the 84×84 raw frame. Cheap regularizer, prevents the policy from overfitting to absolute pixel positions.

### 4d. Reinforcement Learning — separate, not yet wired for the visuomotor variant

Checking the task registration ([`isaaclab_ext/tasks/stack_cube_franka/config/franka/__init__.py`](isaaclab_ext/tasks/stack_cube_franka/config/franka/__init__.py)): there is **no** `rsl_rl_cfg_entry_point` for any of the stack tasks. The upstream stack task ships **BC-only**, no PPO config.

The convention in Isaac Lab is:

| Task type | Trained how |
|---|---|
| **State-based** (`joint_pos`, no cameras) | PPO via rsl_rl — wired through `rsl_rl_cfg_entry_point` |
| **Visuomotor** (`Visuomotor` variants with cameras) | BC via robomimic — wired through `robomimic_bc_cfg_entry_point` |

Why this split: PPO from scratch with image observations on a 16 GB T4 isn't practical (rendering thousands of envs at 84×84 each step is GPU-bound), so visuomotor tasks default to imitation learning where demos make the problem tractable.

If you want **camera-based RL anyway**, three options exist in the wild (you'd have to plumb them yourself):

1. **State-based PPO + image-conditioned distillation** — train PPO on the joint-pos task, roll out, save trajectories with images, BC-train the visuomotor policy on those. Robust, well-understood, two stages.
2. **PPO from a BC warm-start** — BC-pretrain on demos, then fine-tune with PPO using the visuomotor obs. Needs care: PPO can erase BC structure quickly. Implementations like Robomimic-RL exist.
3. **End-to-end pixel PPO** — possible but expensive. Isaac Lab's `TiledCamera` (different from `CameraCfg`) renders many envs into one big GPU buffer to amortize render cost; that's the path if you really want pure RL with vision.

For our pegboard: the realistic short path is **BC with cameras** (option in §4b's recipe). If you want RL, do **state-based PPO** on the already-registered `Isaac-Lift-Pegboard-UR3e-RG2-v0` (no cameras), which already has a `rsl_rl_ppo_cfg.py` agent config in `lift_pegboard_ur3e_rg2/agents/`.

### 4e. The combined picture

```
                    ┌───── teleop demos (HDF5 with images + actions)
                    │
   DOMAIN RAND      ▼                              CNN (ResNet18 + SpatialSoftmax)
   ───────────  ┌───────┐    obs (proprio +    ┌──────────────────────┐
   reset:       │  ENV  │ ─→ images, subtask) ─┼─→ visual core (64-D) │ ─→ LSTM ─→ action
   - joints     │ (sim) │                      │   proprio (identity)  │  hidden_dim=1000
   - cube xy    └───────┘                      └──────────────────────┘
   train:           ▲                                       BC-RNN policy
   - random crop    │                                            │
                    └─── action (6-D EE delta + gripper) ────────┘
```

**RL** is not in this diagram because the upstream stack task doesn't use it. If you bolt it on (state-based PPO on the joint-pos variant), it lives in a parallel pipeline that doesn't see images.
