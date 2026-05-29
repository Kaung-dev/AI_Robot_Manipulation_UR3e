# AIR2 Object Demo Guide

This guide covers the full workflow for teaching the robot to recognize and pick a requested object from the Robotis pegboard environment:

1. Collect segmentation data.
2. Train the segmentation model.
3. Record manual object-annotated demonstrations.
4. Train the behavior cloning policy.
5. Run the trained policy with a target object command.

The target objects are:

| Key | Object |
| --- | --- |
| `1` | `brush` |
| `2` | `pliers` |
| `3` | `scissors` |
| `4` | `screwdriver` |

## Environment

Use the Robotis AIR2 task family so Windows and Linux users launch the same environment:

```text
Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0
```

Use this task for segmentation, manual demo recording, training evaluation, and command-conditioned policy rollout.

## Robot Controls

Keyboard teleop controls:

| Key | Action |
| --- | --- |
| `W` / `S` | Move along X axis |
| `A` / `D` | Move along Y axis |
| `Q` / `E` | Move up / down along Z axis |
| `Z` / `X` | Rotate around X axis |
| `T` / `G` | Rotate around Y axis |
| `C` / `V` | Rotate around Z axis |
| `K` | Open / close gripper |

Manual demo recording controls:

| Key | Action |
| --- | --- |
| `1` | Select target `brush` |
| `2` | Select target `pliers` |
| `3` | Select target `scissors` |
| `4` | Select target `screwdriver` |
| `L` | Pause / resume recording |
| `R` | Reset environment |
| `Enter` | Save current episode |
| `Backspace` | Discard current episode |

Press `1`, `2`, `3`, or `4` before moving toward the object you are about to pick.

## 1. Collect Segmentation Data

This collects RGB images and segmentation masks so the vision model can learn object identity.

### Windows

```powershell
cd .
conda deactivate

$ISAACLAB_PATH/isaaclab.sh -p scripts\collect_air2_segmentation_data.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --enable_cameras `
  --frames 500 `
  --output datasets\air2_segmentation
```

### Linux

```bash
cd ~/ai_in_robotics_project/ur_pick

$ISAACLAB_PATH/isaaclab.sh -p scripts/collect_air2_segmentation_data.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --enable_cameras \
  --frames 500 \
  --output datasets/air2_segmentation
```

Check these folders after collection:

```text
datasets/air2_segmentation/images
datasets/air2_segmentation/masks
datasets/air2_segmentation/overlays
```

Open images in `overlays` and confirm the masks line up with the brush, pliers, scissors, screwdriver, basket, robot, and table.

## 2. Train Segmentation Model

Train the U-Net segmentation model.

### Windows

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\train_air2_segmentation.py `
  --data datasets\air2_segmentation `
  --epochs 30 `
  --output checkpoints\air2_segmentation_unet.pth
```

### Linux

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/train_air2_segmentation.py \
  --data datasets/air2_segmentation \
  --epochs 30 \
  --output checkpoints/air2_segmentation_unet.pth
```

Expected outputs:

```text
checkpoints/air2_segmentation_unet.pth
checkpoints/air2_segmentation_metrics.json
checkpoints/air2_segmentation_overlays/
```

## 3. Record Manual Object-Annotated Demos

This records your teleoperation actions, camera images, robot states, and the selected target object label.

### Windows

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\collect_air2_manual_demos.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --num_envs 1 `
  --teleop_device keyboard `
  --enable_cameras `
  --num_demos 20 `
  --output datasets\air2_manual_demos
```

### Linux

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/collect_air2_manual_demos.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras \
  --num_demos 20 \
  --output datasets/air2_manual_demos
```

Recommended recording pattern:

```text
Press 1 -> pick brush -> place in basket
Press 2 -> pick pliers -> place in basket
Press 3 -> pick scissors -> place in basket
Press 4 -> pick screwdriver -> place in basket
Press Enter to save the episode
```

You can also record one object per episode. The important rule is that the correct target key must be selected before the approach and pick.

Saved demo layout:

```text
datasets/air2_manual_demos/
  ep_000/
    meta.json
    states.npz
    wrist_rgb/
    board_rgb/
  ep_001/
    ...
```

Aim for at least:

```text
20-50 successful demos
Balanced examples for all 4 objects
500-2000 segmentation frames
```

## 4. Train Behavior Cloning Policy

This trains the robot policy from manual demonstrations. The policy input is:

```text
wrist camera + board camera + robot state + selected target object
```

### Windows

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\train_bc.py `
  --demos datasets\air2_manual_demos `
  --unet_ckpt checkpoints\air2_segmentation_unet.pth `
  --epochs 50 `
  --batch_size 32 `
  --out checkpoints\policy_bc.pth
```

### Linux

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/train_bc.py \
  --demos datasets/air2_manual_demos \
  --unet_ckpt checkpoints/air2_segmentation_unet.pth \
  --epochs 50 \
  --batch_size 32 \
  --out checkpoints/policy_bc.pth
```

Expected outputs:

```text
checkpoints/policy_bc.pth
checkpoints/policy_bc.log.json
```

## 5. Run Trained Demo

Run the trained policy by specifying which object to pick.

Valid target objects:

```text
brush
pliers
scissors
screwdriver
```

### Windows

Brush:

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\eval_bc.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --bc_ckpt checkpoints\policy_bc.pth `
  --unet_ckpt checkpoints\air2_segmentation_unet.pth `
  --target_object brush `
  --enable_cameras `
  --num_envs 1 `
  --num_episodes 5
```

Screwdriver:

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\eval_bc.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --bc_ckpt checkpoints\policy_bc.pth `
  --unet_ckpt checkpoints\air2_segmentation_unet.pth `
  --target_object screwdriver `
  --enable_cameras `
  --num_envs 1 `
  --num_episodes 5
```

### Linux

Brush:

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/eval_bc.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --bc_ckpt checkpoints/policy_bc.pth \
  --unet_ckpt checkpoints/air2_segmentation_unet.pth \
  --target_object brush \
  --enable_cameras \
  --num_envs 1 \
  --num_episodes 5
```

Screwdriver:

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/eval_bc.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --bc_ckpt checkpoints/policy_bc.pth \
  --unet_ckpt checkpoints/air2_segmentation_unet.pth \
  --target_object screwdriver \
  --enable_cameras \
  --num_envs 1 \
  --num_episodes 5
```

## Quick Smoke Test

Before doing a long collection, test the full pipeline with small numbers.

### Windows

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\collect_air2_segmentation_data.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --enable_cameras `
  --frames 50

$ISAACLAB_PATH/isaaclab.sh -p scripts\train_air2_segmentation.py `
  --data datasets\air2_segmentation `
  --epochs 2 `
  --output checkpoints\air2_segmentation_unet.pth

$ISAACLAB_PATH/isaaclab.sh -p scripts\collect_air2_manual_demos.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --num_envs 1 `
  --teleop_device keyboard `
  --enable_cameras `
  --num_demos 2

$ISAACLAB_PATH/isaaclab.sh -p scripts\train_bc.py `
  --demos datasets\air2_manual_demos `
  --unet_ckpt checkpoints\air2_segmentation_unet.pth `
  --epochs 2 `
  --batch_size 4 `
  --out checkpoints\policy_bc.pth
```

### Linux

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/collect_air2_segmentation_data.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --enable_cameras \
  --frames 50

$ISAACLAB_PATH/isaaclab.sh -p scripts/train_air2_segmentation.py \
  --data datasets/air2_segmentation \
  --epochs 2 \
  --output checkpoints/air2_segmentation_unet.pth

$ISAACLAB_PATH/isaaclab.sh -p scripts/collect_air2_manual_demos.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras \
  --num_demos 2

$ISAACLAB_PATH/isaaclab.sh -p scripts/train_bc.py \
  --demos datasets/air2_manual_demos \
  --unet_ckpt checkpoints/air2_segmentation_unet.pth \
  --epochs 2 \
  --batch_size 4 \
  --out checkpoints/policy_bc.pth
```

## Troubleshooting

If normal `python` cannot import PyTorch, use Isaac Lab Python:

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\train_air2_segmentation.py ...
```

or:

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/train_air2_segmentation.py ...
```

On Windows, if Isaac crashes with HDF5 or `h5py` DLL errors, start from a clean shell:

```powershell
conda deactivate
```

Then rerun the Isaac command.

If cameras crash due to RTX sensor DLL errors, first verify normal teleop without cameras:

```powershell
$ISAACLAB_PATH/isaaclab.sh -p scripts\run_teleop_windows.py `
  --task Isaac-AIR2-Robotis-Franka-v0 `
  --num_envs 1 `
  --teleop_device keyboard
```

Use the segmentation task only when collecting camera-based data.
