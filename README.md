# AI Robot Manipulation — Franka Panda + AIR2 Scene

Franka Panda pick-and-place in Isaac Sim / Isaac Lab. 4 objects on a Robotis pegboard, each picked into a basket. Pipeline: segmentation data → CNN → BC/diffusion policy → PPO fine-tuning.

> **First time?** See [REBUILD_GUIDE.md](REBUILD_GUIDE.md) for from-scratch setup. This README is the operate-it doc.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Isaac Sim standalone | 5.1.0 | Not pip, not 4.x |
| Isaac Lab | 2.3.2 | Bundled inside Isaac Sim 5.1 |
| WiVRn | v26.x | VR only — tested on v26.2.3 |
| Python | 3.11 | Isaac Sim's bundled runtime |

---

## Setup

```bash
git clone git@github.com:Kaung-dev/AI_Robot_Manipulation_UR3e.git
cd AI_Robot_Manipulation_UR3e

# Set Isaac Lab path (skip if installed at default ~/isaac-sim/...)
cp .env.example .env   # edit ISAACLAB_PATH inside

source .env
./setup_isaaclab.sh
```

`setup_isaaclab.sh` symlinks task configs into IsaacLab, copies VR patches, and generates `scene/scene_isaaclab.usd`. Run it once after cloning and again if `isaaclab_patches/` changes in a pull.

**After pulling:** task config changes under `isaaclab_ext/` take effect immediately (symlinked). If `isaaclab_patches/` changed, re-run `./setup_isaaclab.sh`.

---

## Repo layout

| Path | What |
|---|---|
| `isaaclab_ext/tasks/air2_franka/` | Main AIR2 task — Franka + AIR2 scene (hooks) |
| `isaaclab_ext/tasks/air2_robotis_franka/` | Robotis variant — slot-based placement, stable for demos |
| `isaaclab_ext/tasks/pegboard_franka/` | Original pegboard env — Franka + Robotis table, VR recording |
| `isaaclab_ext/robots/ur3e_rg2.py` | UR3e + RG2 robot config (legacy) |
| `exported_assets/object/` | Tool USDs: brush, pliers, scissors, screwdriver + pegboard + basket |
| `scene/` | Scene USD files (AIR.usd, AIR2.usd, AI_Robotics.usd) |
| `isaaclab_patches/` | Modified IsaacLab files — VR rendering, hand tracking, recording controls |
| `scripts/` | Data collection, training, evaluation scripts |
| `logs/` | Component logs — CNN, PPO, diffusion, env (add entries here) |
| `setup_isaaclab.sh` | Bootstrap script |
| `launch_air2.sh` | Main pipeline launcher |
| `launch_teleop.sh` | VR/keyboard teleop launcher for pegboard env |

---

## Task IDs

**AIR2 Robotis — main data collection env (slot-based, stable):**
| Task ID | Use |
|---|---|
| `Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0` | Segmentation data + manual demos |
| `Isaac-AIR2-Robotis-Franka-Brush-v0` | PPO training — brush |
| `Isaac-AIR2-Robotis-Franka-Pliers-v0` | PPO training — pliers |
| `Isaac-AIR2-Robotis-Franka-Scissors-v0` | PPO training — scissors |
| `Isaac-AIR2-Robotis-Franka-Screwdriver-v0` | PPO training — screwdriver |
| `Isaac-AIR2-Robotis-Franka-Brush-Play-v0` | Inspection / eval (1 env) |
| *(same -Play-v0 for each object)* | |

**AIR2 — hook-based (WIP, objects currently fall):**
| Task ID | Use |
|---|---|
| `Isaac-AIR2-Franka-Play-v0` | Teleop inspection only |

**Pegboard Franka — original VR recording env:**
| Task ID | Use |
|---|---|
| `Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Toothbrush-v0` | VR demo recording |
| *(Scissors, Silicone, Pliers variants)* | |

---

## Pipeline

```
1. collect-seg    → segmentation training data (scripted, no input)
2. train-seg      → train U-Net CNN on segmentation data
3. collect-demos  → manual keyboard/VR demos with object annotation
4. train-bc       → train BC policy (frozen CNN encoder + MLP)
5. eval           → evaluate policy rollouts
6. ppo            → PPO fine-tuning (per-object task)
```

All steps via `launch_air2.sh`:

```bash
source .env

./launch_air2.sh collect-seg 500        # collect segmentation frames
./launch_air2.sh train-seg 30           # train CNN
./launch_air2.sh collect-demos 20       # record manual demos
./launch_air2.sh train-bc 50            # train BC policy
./launch_air2.sh eval brush             # evaluate brush policy
./launch_air2.sh ppo brush              # PPO training — brush
./launch_air2.sh ppo-teleop brush       # inspect brush task in teleop
./launch_air2.sh teleop                 # free teleop (no recording)
```

Smoke test before a full run:
```bash
./launch_air2.sh collect-seg 50     # verify pipeline runs
./launch_air2.sh train-seg 2
./launch_air2.sh collect-demos 2
```

---

## Manual demo recording (keyboard)

```bash
source .env
./launch_air2.sh collect-demos 20
```

| Key | Action |
|---|---|
| `1` | Select target: brush |
| `2` | Select target: pliers |
| `3` | Select target: scissors |
| `4` | Select target: screwdriver |
| `W/A/S/D/Q/E` | Move EE (X/Y/Z) |
| `Z/X T/G C/V` | Rotate EE |
| `K` | Toggle gripper |
| `L` | Pause / resume recording |
| `Enter` | Save episode + reset |
| `Backspace` | Discard episode + reset |

**Rule:** press the object key **before** moving toward it. Frames without a target key pressed are tagged `target_valid=False` and filtered during training.

Output: `datasets/air2_manual_demos/ep_000/` ... each episode has `states.npz` + `wrist_rgb/` + `board_rgb/`.

---

## VR teleoperation (Meta Quest 2)

```bash
# 1. Start WiVRn server
flatpak run io.github.wivrn.wivrn

# 2. On Quest 2: open WiVRn app → confirm "Connected to server"

# 3. Launch
source .env
./launch_teleop.sh handtracking toothbrush   # or: scissors | silicone | pliers
```

**Right hand — arm control:**

| Input | Action |
|---|---|
| Right wrist movement | EE translation (20× scale) |
| Right wrist orientation | EE rotation (10× scale, full 6-DoF) |
| Right thumb + index < 3 cm | Close gripper |
| Right thumb + index > 5 cm | Open gripper |

**Left hand — recording control (hold 3 seconds to trigger):**

| Gesture | Action |
|---|---|
| Open palm | Toggle recording pause / resume |
| Thumb + middle finger pinch | Accept episode — save + reset |
| Fist | Discard episode — reset without saving |

> Left thumb+index is not used — triggers Quest 2 system menu.

Green sphere at left wrist = recording active. Red = paused.

**Anchor position** (where you spawn relative to scene): edit `xr = XrCfg(anchor_pos=(...))` in `isaaclab_ext/tasks/air2_franka/joint_pos_env_cfg.py`. Current: `(-4.2405, -4.75, 1.25)` — tune in-headset.

**GPU requirement:** RTX 3070 minimum. Ray tracing is force-disabled; rasterised only.

---

## WiVRn config

`~/.config/wivrn/config.json`:
```json
{ "encoder": "nvenc", "codec": "h264", "bitrate": 35000000, "scale": 0.75 }
```

`~/.config/openxr/1/active_runtime.json` must point to WiVRn's `.so`. See [REBUILD_GUIDE.md](REBUILD_GUIDE.md) if missing.

---

## Editing configs

All task files under `isaaclab_ext/` are symlinked into IsaacLab — edits take effect immediately.

| What to change | File |
|---|---|
| Robot spawn position / joint angles | `air2_franka/joint_pos_env_cfg.py` |
| VR anchor, scale factors | `air2_franka/joint_pos_env_cfg.py` lines 86–98 |
| Reward weights | `air2_robotis_franka/joint_pos_env_cfg.py` (`_apply_target_rewards`) |
| Slot positions | `air2_robotis_franka/joint_pos_env_cfg.py` `_SLOTS` dict |
| Hook positions (AIR2 scene) | `air2_franka/mdp/events.py` `HOOK_POSITIONS` |
| Basket position | `air2_franka/mdp/constants.py` `BASKET_POS_LOCAL` |
| VR gesture hold time | `isaaclab_patches/scripts/tools/vr_gesture_detector.py` `HOLD_FRAMES` |
| Camera positions | `air2_franka/segmentation_env_cfg.py` |
| PPO hyperparameters | `air2_robotis_franka/agents/rsl_rl_ppo_cfg.py` |

---

## Component logs

See `logs/` for ongoing notes on each component. Add entries when you try something, fix something, or find a bug.

| Log | Component |
|---|---|
| `logs/env.md` | Scene, task configs, Isaac Sim issues |
| `logs/cnn.md` | Segmentation CNN pipeline |
| `logs/ppo.md` | Rewards, PPO training |
| `logs/diffusion.md` | Diffusion / BC policy |

---

## Known issues

- **AIR2 hook env objects fall** — `HOOK_POSITIONS` in `air2_franka/mdp/events.py` are calibrated for an older scene. Use the Robotis env (`Isaac-AIR2-Robotis-Franka-*`) for all data collection until hooks are re-calibrated. See `logs/env.md`.
- **CNN `position_world` always None** — camera extrinsics not plumbed through `extract_detections()`. Blocks CNN-based reward confirmation. See `logs/cnn.md`.
- **No data augmentation in segmentation training** — model may overfit on small datasets. See `logs/cnn.md`.
