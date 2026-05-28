# AIR2 CNN Segmentation

This folder contains the AIR2-specific semantic segmentation pipeline.

The object classes are shared with the task object catalog:
`brush`, `pliers`, `scissors`, and `screwdriver`.

## Data Collection

Run from the repo root with Isaac Lab:

```bash
./IsaacLab/isaaclab.sh -p scripts/collect_air2_segmentation_data.py \
  --task Isaac-Lift-AIR2-UR3e-RG2-Segmentation-v0 \
  --frames 200 \
  --enable_cameras
```

The collector saves RGB frames, depth arrays, raw Isaac masks, remapped class
masks, and annotator metadata under `datasets/air2_segmentation/`.

## Training

Run in an environment with PyTorch:

```bash
python scripts/train_air2_segmentation.py \
  --data datasets/air2_segmentation \
  --epochs 30
```

The best checkpoint is saved to `checkpoints/air2_segmentation_unet.pth`.

## Inference

Saved image:

```bash
python scripts/run_air2_segmentation_inference.py \
  --checkpoint checkpoints/air2_segmentation_unet.pth \
  --image datasets/air2_segmentation/images/000000_board_camera_rgb.png
```

Live Isaac camera:

```bash
./IsaacLab/isaaclab.sh -p scripts/run_air2_segmentation_inference.py \
  --checkpoint checkpoints/air2_segmentation_unet.pth \
  --task Isaac-Lift-AIR2-UR3e-RG2-Segmentation-v0 \
  --camera board_camera \
  --enable_cameras
```
