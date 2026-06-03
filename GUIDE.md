# AIR2 Franka pick-and-place — data + full pipeline guide

This zip contains everything we made: the trained **models** (BC policies + CNN
segmentation), the **datasets** (human demos, Mimic-generated demos, segmentation
images), and this guide. It is meant for someone who downloaded the zip and wants
to either **(A) just run our stuff**, or **(B) reproduce the whole pipeline from
scratch** (record human demos → annotate → Mimic-generate → train BC → run).

---

## 0. What's in this folder

```
SAVED_ARTIFACTS/
├── GUIDE.md            ← you are here
├── README.md           ← name-mapping table (saved name → original name → what it is)
├── checkpoints/        ← trained models (.pth)
│   ├── bc_brush_v2.pth                 brush state-BC policy
│   ├── bc_pliers_v2.pth                pliers state-BC policy
│   ├── bc_screwdriver_v2.pth           screwdriver state-BC policy
│   ├── cnn_unet_oldcam.pth             U-Net seg, OLD camera (the demo uses this)
│   ├── cnn_unet_newcam.pth             U-Net seg, NEW camera
│   └── cnn_resnet18_v3_friend.pth      ResNet18 seg, NEW camera (heavier, best mIoU)
└── datasets/           ← all data (.hdf5 + image folders)
    ├── *_annotated_demos.hdf5          human demos, annotated (Mimic INPUT)
    ├── *_mimic_generated_*.hdf5        synthetic demos (Mimic OUTPUT → BC training)
    ├── seg_images_oldcam/              RGB+mask frames, old cam (trained cnn_unet_oldcam)
    └── seg_images_newcam/              RGB+mask frames, new cam (trained cnn_unet_newcam)
```

See `README.md` for the exact original filename of each renamed file.

---

## 1. Prerequisites (one-time machine setup)

You need **Isaac Sim 5.1 + Isaac Lab 2.3.2** and our task extension repo
`AI_Robot_Manipulation_UR3e`. This zip is **data only** — it does not contain the
code. Get the code repo first, then:

1. Install Isaac Lab (follow its docs). Note where it lives, e.g. `~/IsaacLab`.
2. In the repo root, copy `.env.example → .env` and set:
   ```
   ISAACLAB_PATH=/path/to/your/IsaacLab
   export ISAACLAB_PATH
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json   # avoids dup Vulkan ICD
   ```
3. Everything below runs through Isaac Lab's python: `"$ISAACLAB_PATH/isaaclab.sh" -p <script>`.
   The repo's `./launch_air2.sh` wraps the common steps — that's the easy path.

> **GUI vs headless:** add nothing for a GUI window; add `--headless` to run without
> rendering (faster, for training-data generation on a server).

---

## 2. Where to put the data

Copy the contents of this zip into the **repo** (paths are relative to the repo root,
which we'll call `$REPO`). The script defaults expect files in `datasets/` and
`checkpoints/`. The files here are **renamed** for clarity, so either rename them back
(see `README.md`) **or** just pass explicit `--paths` (recommended — shown below).

Quick placement:
```bash
cp checkpoints/*.pth   $REPO/checkpoints/
cp datasets/*.hdf5     $REPO/datasets/
cp -r datasets/seg_images_oldcam $REPO/datasets/air2_segmentation_newscene   # old-cam frames
cp -r datasets/seg_images_newcam $REPO/datasets/air2_segmentation_newcam     # new-cam frames
```

---

## 3. OPTION A — "I just want to use your stuff" (no training)

You already have the policies and the CNN. Just run the demo.

### A1. Sequential multi-tool demo (the headline result)
Franka detects the tools, picks the one(s) closest to the basket, drops each in the
basket. Two run modes:

**Camera-free (most reliable, what we record video with):**
```bash
cd $REPO
PYTHONPATH="$REPO:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_sequential.py \
  --brush_ckpt      checkpoints/bc_brush_v2.pth \
  --pliers_ckpt     checkpoints/bc_pliers_v2.pth \
  --screwdriver_ckpt checkpoints/bc_screwdriver_v2.pth \
  --num_episodes 10 \
  --no_cnn                       # uses ground-truth positions, no camera loaded
```

**With the CNN doing detection (perception in the loop):**
```bash
PYTHONPATH="$REPO:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_sequential.py \
  --brush_ckpt      checkpoints/bc_brush_v2.pth \
  --pliers_ckpt     checkpoints/bc_pliers_v2.pth \
  --screwdriver_ckpt checkpoints/bc_screwdriver_v2.pth \
  --seg_ckpt        checkpoints/cnn_unet_oldcam.pth \
  --num_episodes 10 --enable_cameras
```

> ⚠️ **The seg checkpoint must match the camera the env renders.** `cnn_unet_oldcam`
> matches the OLD camera config; `cnn_unet_newcam` / `cnn_resnet18_v3_friend` match the
> NEW camera (`main_camera` at -4.8,-5.2,2.2). If you change one, change the other or
> detections will be garbage.
>
> 💡 The policies are single-pick specialists trained always-from-reset, so they drift
> after ~2 sequential picks. `--max_picks_per_episode 2` (default) picks two then resets
> for a clean start. That's expected behaviour, not a bug.

### A2. Single-tool eval (sanity-check one policy)
```bash
./launch_air2.sh eval-state-bc brush      checkpoints/bc_brush_v2.pth 20
./launch_air2.sh eval-state-bc pliers     checkpoints/bc_pliers_v2.pth 20
./launch_air2.sh eval-state-bc screwdriver checkpoints/bc_screwdriver_v2.pth 20
```

---

## 4. OPTION B — full pipeline from scratch

This is the data-generation funnel. You **only need ~10–20 human demos per tool**;
Mimic multiplies them into ~1000 synthetic demos, which is what the BC policy trains on.

```
   human teleop          annotate          Mimic generate         train BC          run
 (10-20 demos)   →   (mark subtasks)  →  (~1000 synthetic)  →   (MLP policy)  →   eval
 collect-mimic        annotate-mimic       generate-mimic        train-state-bc    eval-sequential
      │                    │                     │                    │               │
  *_demos.hdf5    *_annotated.hdf5      *_generated.hdf5        *.pth checkpoint   robot moves
```

All steps use the Mimic task ids `Isaac-AIR2-Robotis-Franka-<Object>-Mimic-v0`
where `<Object>` ∈ {Brush, Pliers, Scissors, Screwdriver}.

### Step 1 — record human demos (teleoperation)
Drive the robot by keyboard and save successful grasps.

```bash
./launch_air2.sh collect-mimic brush 20 datasets/brush_demos.hdf5
#                              ^obj  ^n  ^output
```
**Keyboard controls** (printed at launch):
- `W/A/S/D/Q/E` = move end-effector · `Z/X/T/G/C/V` = rotate · `K` = toggle gripper
- `L` = pause/resume · `Enter` = save the demo · `Backspace` = discard & retry

Do ~20 clean pick-and-drop-in-basket demos. Output: `datasets/brush_demos.hdf5`.

> Tip: record from a few different start slots so Mimic has pose variety to interpolate.

### Step 2 — annotate the demos (mark Mimic subtasks)
Mimic needs to know where each subtask (reach / grasp / lift / place) begins. `--auto`
infers them from the gripper signal.

```bash
./launch_air2.sh annotate-mimic brush datasets/brush_demos.hdf5 datasets/brush_annotated.hdf5
```
Output: `datasets/brush_annotated.hdf5` (this is what we ship as
`brush_annotated_demos.hdf5`).

### Step 3 — generate synthetic demos with Mimic
Replays the grasp across hundreds of randomized object poses.

```bash
./launch_air2.sh generate-mimic brush \
  datasets/brush_annotated.hdf5 \
  datasets/brush_generated.hdf5 \
  1000                                # number of trials
```
- Runs `--num_envs 4 --headless` by default. **For a video of generation in action**,
  edit the command to drop `--headless` and bump envs (e.g. `--num_envs 16`).
- Not every trial succeeds. Optionally filter to only the clean/successful episodes
  (we shipped both `*_raw` and `*_clean`). Merge multiple runs with
  `scripts/merge_mimic_hdf5.py` if you generate in batches.

Output: `datasets/brush_generated.hdf5` (≈ what we ship as `brush_mimic_generated.hdf5`).

### Step 4 — train the BC policy
Trains the 43-D-input MLP to imitate the generated demos.

```bash
./launch_air2.sh train-state-bc brush datasets/brush_generated.hdf5 \
  checkpoints/bc_brush.pth 300
#                          ^out         ^epochs
```
Or call the script directly for full control:
```bash
PYTHONPATH="$REPO:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
  scripts/train_state_bc_from_hdf5.py \
  --hdf5 datasets/brush_generated.hdf5 --object brush \
  --out checkpoints/bc_brush.pth --epochs 300
```
Output: `checkpoints/bc_brush.pth` + a `.log.json` loss trace. **Repeat Steps 1–4 for
each tool** (`pliers`, `screwdriver`, …), changing the `<object>` each time.

### Step 5 — use it
Drop your new `.pth` files into the Option A commands (§3). Done.

---

## 5. (Optional) retrain the CNN segmentation model

Only needed if you change the camera or scene. Two stages:

### 5a. Collect segmentation frames
```bash
./launch_air2.sh collect-seg 600        # ~600 RGB+mask frames → datasets/air2_segmentation
```
(We shipped pre-collected frames in `seg_images_oldcam/` and `seg_images_newcam/`.)

### 5b. Train
```bash
./launch_air2.sh train-seg 60 unet      # U-Net, 60 epochs (lighter, what the demo uses)
./launch_air2.sh train-seg 60 resnet18  # ResNet18 (best mIoU, heavier)
```
Output: `checkpoints/air2_segmentation_<backbone>.pth` + `air2_segmentation_metrics.json`
(per-epoch metrics). Our U-Net plateaus around **tool mIoU 0.72** (new cam) / 0.62 (old
cam); the ResNet18 reaches ~0.84.

> ⚠️ Whatever camera you collect frames from is the camera the seg model expects at
> inference. Keep the eval camera (`board_camera`) and the collection camera
> (`main_camera`) identical, and pass the matching `--seg_ckpt` in §3.

---

## 6. Architecture in one paragraph (so the numbers make sense)

Three decoupled pieces, **no PPO** — perception is supervised, control is imitation:
1. **CNN segmentation** (U-Net or ResNet18) labels camera pixels per tool; depth +
   intrinsics back-project each tool's mask into a 3D world position. This is the
   *eyes* — it answers "which tools are here and where," it does **not** move the robot.
2. **Per-tool state-BC policies** (one small MLP each, 43-D obs → joint action). Each is
   a single-pick specialist trained purely by behaviour cloning on Mimic-generated demos.
3. **Orchestrator** (`eval_sequential.py`): runs the CNN once, sorts detected tools
   **closest-to-basket first**, then for each tool loads the matching specialist policy
   and runs a phase machine (APPROACH→GRIP→CARRY→RELEASE). The grasp servos to the tool's
   **true pose** (CNN picks *which/what order*, ground-truth gives grasp *precision*).
   This is how three independent single-tool models behave like one multi-tool system.
   You can swap the CNN without retraining any policy, and vice-versa.

---

## 7. Quick reference — `launch_air2.sh` subcommands

| Command | Does |
|---|---|
| `collect-seg [frames]` | record segmentation training frames |
| `train-seg [epochs] [unet\|resnet18]` | train the CNN |
| `collect-mimic <obj> [n] [out]` | record human teleop demos |
| `annotate-mimic <obj> [in] [out]` | mark Mimic subtasks (`--auto`) |
| `generate-mimic <obj> [in] [out] [n]` | Mimic → synthetic demos |
| `train-state-bc <obj> [hdf5] [out] [epochs]` | train a BC policy |
| `eval-state-bc <obj> [ckpt] [episodes]` | eval one policy |
| `eval-sequential [episodes] [extra args]` | the full multi-tool demo |

Objects: `brush` · `pliers` · `scissors` · `screwdriver`.

---

## 8. Generate CNN evaluation plots

Produces a confusion matrix and training loss/accuracy/IoU curves from the metrics JSON.

```bash
python3 scripts/plot_cnn_training.py \
    --metrics checkpoints/air2_segmentation_unet_newcam.metrics.json \
    --seg_ckpt checkpoints/air2_segmentation_unet_newcam.pth \
    --data datasets/air2_segmentation_newcam \
    --out_dir eval_results/cnn_plots_unet_newcam
```

The training script saves `<checkpoint>.metrics.json` automatically alongside the `.pth` file.

---

## 9. Common issues

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: checkpoints/...` | Ensure all `.pth` files are present. Check `ls checkpoints/` |
| `KeyError: 'main_camera'` | Old scene USD still references `board_camera`. Pull latest `scene/scene_isaaclab.usd` |
| `cv2.error: The function is not implemented` | Isaac Sim's Python lacks GTK GUI. Scripts save PNGs to disk instead of `cv2.imshow()` |
| Isaac Sim freezes on `env.reset()` | Known issue when cameras are active between rounds. Scripts use `go_home()` between picks instead |
| Isaac Sim freezes on launch | Ensure `--enable_cameras` is passed when running CNN-based eval scripts |
| Controller not detected | Plug in gamepad **before** launching. Verify with `ls /dev/input/js0` |
| 3rd pick drifts / fails | IK controller accumulates joint drift after 2 go_home cycles. Use `--max_picks 2` (default) to reset between rounds |
| Checkpoint name not found | The multi-object script searches multiple naming patterns (`policy_state_bc_<tool>.pth`, `policy_state_bc_mimic_<tool>_v2.pth`, etc.). Ensure your checkpoint matches one of these conventions |
