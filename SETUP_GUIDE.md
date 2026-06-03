# Setup Guide — Running Locally After Git Pull

This guide explains how to set up paths and dependencies after cloning the repo so you can run evaluation and training scripts on your own machine.

## 1. Prerequisites

- **Isaac Sim** installed (tested with 4.5+)
- **Isaac Lab** cloned and built (the `isaaclab.sh` launcher)
- Python 3.10+ with PyTorch, torchvision, OpenCV (`cv2`), matplotlib, numpy

## 2. Clone and Check Paths

```bash
git clone https://github.com/Kaung-dev/AI_Robot_Manipulation_UR3e.git
cd AI_Robot_Manipulation_UR3e
```

All scripts use `REPO_ROOT = Path(__file__).resolve().parents[1]` to auto-detect the repo root, so **relative paths inside the repo resolve automatically**. You only need to set one external path: where Isaac Lab lives.

## 3. Required Checkpoints

These must exist under `checkpoints/` (included in repo via Git LFS or manual download):

| File | Purpose |
|------|---------|
| `policy_state_bc_mimic_v2.pth` | BC policy — brush |
| `policy_state_bc_mimic_pliers_v2.pth` | BC policy — pliers |
| `policy_state_bc_mimic_screwdriver_v2.pth` | BC policy — screwdriver |
| `air2_segmentation_unet_newcam.pth` | U-Net segmentation model (recommended) |
| `air2_segmentation_v3.pth` | ResNet-18 segmentation model (alternative) |

Verify they exist:
```bash
ls checkpoints/policy_state_bc_mimic_v2.pth \
   checkpoints/policy_state_bc_mimic_pliers_v2.pth \
   checkpoints/policy_state_bc_mimic_screwdriver_v2.pth \
   checkpoints/air2_segmentation_unet_newcam.pth
```

## 4. Isaac Lab Path

Scripts are launched through Isaac Lab's Python environment:

```bash
# Set this to YOUR Isaac Lab install location
export ISAACLAB_PATH=/mnt/extra/IsaacLab   # <-- CHANGE THIS
```

All run commands use:
```bash
cd $ISAACLAB_PATH && ./isaaclab.sh -p <script_path> [args]
```

## 5. Running Evaluation Scripts

### Multi-object CNN pipeline (eval_multi_object_bc.py)

Picks objects closest-to-basket first using CNN detection + per-tool BC policies.

```bash
cd $ISAACLAB_PATH && ./isaaclab.sh -p \
    /path/to/AI_Robot_Manipulation_UR3e/scripts/eval_multi_object_bc.py \
    --seg_ckpt /path/to/AI_Robot_Manipulation_UR3e/checkpoints/air2_segmentation_unet_newcam.pth \
    --num_rounds 3 --enable_cameras \
    --out /path/to/AI_Robot_Manipulation_UR3e/eval_results/multi_object_unet.json
```

**Key arguments to update with your paths:**
- `--seg_ckpt` — path to segmentation model (.pth)
- `--ckpt_dir` — directory containing BC policy checkpoints (defaults to `checkpoints/` in repo)
- `--out` — output JSON for results

### Sequential pipeline (eval_sequential.py)

Interactive pipeline with manual tool selection or `auto` mode.

```bash
cd $ISAACLAB_PATH && ./isaaclab.sh -p \
    /path/to/AI_Robot_Manipulation_UR3e/scripts/eval_sequential.py \
    --enable_cameras
```

**Key arguments to update with your paths (if defaults don't match):**
- `--brush_ckpt` — default: `checkpoints/policy_state_bc_mimic_v2.pth`
- `--pliers_ckpt` — default: `checkpoints/policy_state_bc_mimic_pliers_v2.pth`
- `--screwdriver_ckpt` — default: `checkpoints/policy_state_bc_mimic_screwdriver_v2.pth`
- `--seg_ckpt` — default: `checkpoints/air2_segmentation_unet_newcam.pth`

### Single-object evaluation (eval_state_bc_cnn.py)

```bash
cd $ISAACLAB_PATH && ./isaaclab.sh -p \
    /path/to/AI_Robot_Manipulation_UR3e/scripts/eval_state_bc_cnn.py \
    --seg_ckpt /path/to/AI_Robot_Manipulation_UR3e/checkpoints/air2_segmentation_unet_newcam.pth \
    --enable_cameras
```

## 6. Training Scripts

### Train segmentation model (U-Net)

```bash
python3 scripts/train_air2_segmentation.py \
    --data datasets/air2_segmentation_newcam \
    --backbone unet --lr 1e-3 --epochs 60 \
    --output checkpoints/air2_segmentation_unet_newcam.pth
```

### Train segmentation model (ResNet-18)

```bash
python3 scripts/train_air2_segmentation.py \
    --data datasets/air2_segmentation_newcam \
    --backbone resnet18 --lr 1e-4 --epochs 60 \
    --output checkpoints/air2_segmentation_v3.pth
```

### Train BC policy (per-tool, from robomimic HDF5)

```bash
cd $ISAACLAB_PATH && ./isaaclab.sh -p \
    /path/to/AI_Robot_Manipulation_UR3e/scripts/train_state_bc_from_hdf5.py \
    --data /path/to/demos.hdf5 \
    --out checkpoints/policy_state_bc_mimic_<tool>_v2.pth
```

## 7. Generate Plots

### CNN confusion matrix + training curves

```bash
python3 scripts/plot_cnn_training.py \
    --metrics checkpoints/air2_segmentation_unet_newcam.metrics.json \
    --seg_ckpt checkpoints/air2_segmentation_unet_newcam.pth \
    --data datasets/air2_segmentation_newcam \
    --out_dir eval_results/cnn_plots_unet_newcam
```

## 8. Common Issues

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: checkpoints/...` | Ensure all `.pth` files are downloaded. Check `ls checkpoints/` |
| `KeyError: 'main_camera'` | You're using an old scene USD. Pull latest `scene/scene_isaaclab.usd` |
| `cv2.error: The function is not implemented` | Isaac Sim's Python lacks GTK GUI. Scripts save PNGs to disk instead of `cv2.imshow()` |
| Isaac Sim freezes on `env.reset()` | Known issue when cameras are active between rounds. Scripts use `go_home()` instead |
| Controller not detected | Plug in gamepad BEFORE launching. Verify with `ls /dev/input/js0` |
