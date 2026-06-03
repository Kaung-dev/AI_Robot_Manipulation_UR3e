# Demo Collection and Training Guide

Full pipeline: collect segmentation data → train the CNN → record manual demos → train the BC policy → evaluate.

---

## Linux vs Windows Differences

| | Linux | Windows |
|---|---|---|
| Launcher | `isaaclab.sh` | `isaaclab.bat` (run with `& "D:\IsaacLab\isaaclab.bat"`) |
| Path separator | `/` | `\` |
| Line continuation | `\` | `` ` `` (backtick) |
| Isaac Lab path | `/home/declan/IsaacLab` | `D:\IsaacLab` (adjust to your install) |
| Repo path | `/home/declan/ur_pick` | `D:\AI_Robot_Manipulation_UR3e` (adjust to your install) |

---

## Step 1 — Collect Segmentation Data

Generates RGB images and segmentation masks used to train the CNN.

**Linux**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/collect_air2_segmentation_data.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --enable_cameras \
  --frames 500 \
  --output /home/declan/ur_pick/datasets/air2_segmentation
```

**Windows**
```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\collect_air2_segmentation_data.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --enable_cameras `
  --frames 500 `
  --output datasets\air2_segmentation
```

**Or via launch script (Linux only):**
```bash
./launch_air2.sh collect-seg 500
```

Output structure:
```
datasets/air2_segmentation/
  images/
  masks/
  overlays/
```

Check `overlays/` to confirm masks align with the tools, basket, robot, and table.

---

## Step 2 — Train the CNN (Segmentation Model)

Trains a ResNet-18 or U-Net encoder to segment objects in camera frames. ResNet-18 is recommended.

**Linux — ResNet-18 (recommended)**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/train_air2_segmentation.py \
  --backbone resnet18 \
  --data /home/declan/ur_pick/datasets/air2_segmentation \
  --epochs 60 \
  --output /home/declan/ur_pick/checkpoints/air2_segmentation_resnet18.pth
```

**Linux — U-Net (from scratch)**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/train_air2_segmentation.py \
  --backbone unet \
  --data /home/declan/ur_pick/datasets/air2_segmentation \
  --epochs 60 \
  --lr 1e-3 \
  --output /home/declan/ur_pick/checkpoints/air2_segmentation_unet.pth
```

**Windows — ResNet-18**
```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\train_air2_segmentation.py `
  --backbone resnet18 `
  --data datasets\air2_segmentation `
  --epochs 60 `
  --output checkpoints\air2_segmentation_resnet18.pth
```

**Or via launch script (Linux only):**
```bash
./launch_air2.sh train-seg 60 resnet18
# or for U-Net:
./launch_air2.sh train-seg 60 unet
```

Output:
```
checkpoints/air2_segmentation_resnet18.pth
```

---

## Step 3 — Record Manual Demos

Teleop the robot and press `Enter` to save each episode. Use `--default_target` to pre-select the object and `--output` / `--hdf5_output` to name files per object.

### Controls

| Key | Action |
|---|---|
| `W` / `S` | Move along X |
| `A` / `D` | Move along Y |
| `Q` / `E` | Move up / down (Z) |
| `Z` / `X` | Rotate X |
| `T` / `G` | Rotate Y |
| `C` / `V` | Rotate Z (yaw) |
| `K` | Open / close gripper |
| `1` | Set target: brush |
| `2` | Set target: pliers |
| `3` | Set target: scissors |
| `4` | Set target: screwdriver |
| `L` | Pause / resume recording |
| `Enter` | Save episode |
| `Backspace` | Discard episode |
| `R` | Reset without saving |

Press the number key for your target object **before** approaching it.

### Linux

```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/collect_air2_manual_demos.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras \
  --num_demos 20 \
  --output /home/declan/ur_pick/datasets/air2_manual_demos_scissors \
  --save_every_n_steps 4 \
  --sensitivity 2 \
  --hdf5_output /home/declan/ur_pick/datasets/air2_mimic_scissors.hdf5 \
  --default_target scissors
```

Change `scissors` to `brush`, `pliers`, or `screwdriver` for other objects.
Change `air2_manual_demos_scissors` and `air2_mimic_scissors.hdf5` to match.

### Windows

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\collect_air2_manual_demos.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --num_envs 1 `
  --teleop_device keyboard `
  --enable_cameras `
  --num_demos 20 `
  --output datasets\air2_manual_demos_scissors `
  --save_every_n_steps 4 `
  --sensitivity 2 `
  --hdf5_output datasets\air2_mimic_scissors.hdf5 `
  --default_target scissors
```

### Or via launch script (Linux only, uses shared output folder):
```bash
./launch_air2.sh collect-demos 20 keyboard scissors
```

> The launch script saves all objects to the same `datasets/air2_manual_demos` folder. Use the full command above if you want per-object output folders.

Output structure per episode:
```
datasets/air2_manual_demos_scissors/
  ep_000/
    meta.json
    states.npz
    wrist_rgb/  (t_0000.png … t_NNNN.png)
    board_rgb/  (t_0000.png … t_NNNN.png)
  ep_001/
    ...
```

Aim for at least 20 successful demos per object, balanced across all 4 targets.

---

## Step 4 — Precompute Centroids (optional, improves board-camera accuracy)

Runs the trained segmentation model over the recorded board images to precompute object centroid locations. Run this before BC training if you want the board-camera centroid feature.

**Linux**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/precompute_centroids.py \
  --demos /home/declan/ur_pick/datasets/air2_manual_demos \
  --unet_ckpt /home/declan/ur_pick/checkpoints/air2_segmentation_resnet18.pth
```

**Or via launch script:**
```bash
./launch_air2.sh precompute
```

---

## Step 5 — Train Behavior Cloning Policy

Trains the BC policy using the recorded demos. The CNN encoder is frozen; only the fusion MLP and action head are trained.

The `--backbone` and `--unet_ckpt` must match the segmentation model trained in Step 2.

**Linux — ResNet-18 backbone (recommended)**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/train_bc.py \
  --demos /home/declan/ur_pick/datasets/air2_manual_demos \
  --backbone resnet18 \
  --unet_ckpt /home/declan/ur_pick/checkpoints/air2_segmentation_resnet18.pth \
  --epochs 50 \
  --batch_size 32 \
  --out /home/declan/ur_pick/checkpoints/policy_bc.pth
```

**Linux — U-Net backbone**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/train_bc.py \
  --demos /home/declan/ur_pick/datasets/air2_manual_demos \
  --backbone unet \
  --unet_ckpt /home/declan/ur_pick/checkpoints/air2_segmentation_unet.pth \
  --epochs 50 \
  --batch_size 32 \
  --out /home/declan/ur_pick/checkpoints/policy_bc.pth
```

**Windows — ResNet-18 backbone**
```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\train_bc.py `
  --demos datasets\air2_manual_demos `
  --backbone resnet18 `
  --unet_ckpt checkpoints\air2_segmentation_resnet18.pth `
  --epochs 50 `
  --batch_size 32 `
  --out checkpoints\policy_bc.pth
```

**Or via launch script (Linux only, U-Net hardcoded):**
```bash
./launch_air2.sh train-bc 50
```

> The launch script hardcodes `--backbone unet`. Use the full command above for ResNet-18.

Output:
```
checkpoints/policy_bc.pth
checkpoints/policy_bc.log.json
```

---

## Step 6 — Evaluate the Policy

Run the trained policy in simulation and report success rate.

**Linux**
```bash
/home/declan/IsaacLab/isaaclab.sh -p /home/declan/ur_pick/scripts/eval_bc.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --bc_ckpt /home/declan/ur_pick/checkpoints/policy_bc.pth \
  --backbone resnet18 \
  --unet_ckpt /home/declan/ur_pick/checkpoints/air2_segmentation_resnet18.pth \
  --target_object scissors \
  --enable_cameras \
  --num_envs 1 \
  --num_episodes 10
```

**Windows**
```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\eval_bc.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --bc_ckpt checkpoints\policy_bc.pth `
  --backbone resnet18 `
  --unet_ckpt checkpoints\air2_segmentation_resnet18.pth `
  --target_object scissors `
  --enable_cameras `
  --num_envs 1 `
  --num_episodes 10
```

**Or via launch script (Linux only):**
```bash
./launch_air2.sh eval scissors
```

Valid values for `--target_object`: `brush`, `pliers`, `scissors`, `screwdriver`.

---

## Known Issues

### Linux: robot freezes after pressing Enter (fixed)

`ep.pre_export()` was called on `EpisodeData` but the method does not exist in Isaac Lab v2.2.1. This caused an `AttributeError` that triggered Isaac Sim shutdown, which appeared as a freeze. The line has been removed from `scripts/collect_air2_manual_demos.py`.

### Corrupt HDF5 file after a crashed run

If a previous run crashed, `datasets/air2_mimic_source.hdf5` (or any `--hdf5_output` file) may be left as a 96-byte truncated file. The next run will fail trying to open it. Delete it before re-running:

```bash
rm datasets/air2_mimic_source.hdf5
```

### Windows: HDF5 DLL errors on startup

Start from a clean shell with no active conda environment:
```powershell
conda deactivate
```
Then rerun the Isaac command. The script pre-imports `h5py` at the top to load HDF5 DLLs before Isaac Sim extensions override them.
