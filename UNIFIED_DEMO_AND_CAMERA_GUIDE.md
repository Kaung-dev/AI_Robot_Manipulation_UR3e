# Unified CNN Pick-and-Place — pipeline map + camera-reposition guide

_For the team. Covers every file in the unified demo sequence, the exact commands,
and — most importantly — **how to reposition the camera and retrain the CNN**
without breaking detection. Last updated 2026-06-02._

---

## 0. TL;DR for the camera reposition

The camera lives in **two places that MUST be identical**, or the CNN trains on one
view and runs on another and detection breaks:

| Role | File | What to edit |
|---|---|---|
| **Collection** (CNN training images) | [isaaclab_ext/tasks/air2_franka/segmentation_env_cfg.py](isaaclab_ext/tasks/air2_franka/segmentation_env_cfg.py) → `_apply_segmentation_cameras()` | the `cfg.scene.main_camera` `OffsetCfg(pos=…, rot=…)` + `focal_length` |
| **Inference** (the live demo) | [scripts/eval_sequential.py](scripts/eval_sequential.py) (~line 471) | the `env_cfg.scene.board_camera` `OffsetCfg(pos=…, rot=…)` + `focal_length` |

**They are currently OUT OF SYNC** (see §4). Repositioning = pick one new
`pos`/`rot`/`focal_length`, paste the **same numbers into both**, then re-collect →
retrain → run. Resolution is already `640×360` in both — keep it.

Procedure:
1. Edit the camera `pos`/`rot`/`focal_length` in **both** files above (identical values).
2. **Re-collect** segmentation data (§5).
3. **Retrain** the U-Net (§6).
4. **Run** the demo with the new checkpoint (§2). The CNN-detected positions only drive
   *detection + sequencing*; the grasp uses the true pose, so cm-level camera error is fine.

---

## 1. What the demo does (the sequence)

One episode = pick-and-place **every tool** into the basket, then reset & re-randomize:

```
env.reset()  →  place tools on constrained-random slots  →  hold robot at home
  └─ loop until all tools handled:
       CNN scan (board_camera)        → which tools are on the pegs + sequence
       pick next detected tool        → grasp servos to the tool's TRUE pose
       APPROACH → GRIP → CARRY → RELEASE → LANDED in basket
       go_home  (lift clear of pegs, drive to home, push targets to sim)
  └─ episode ends when every tool attempted → env.reset() re-randomizes
```

- **CNN role:** detect tools + decide order (cm-level is enough).
- **Grasp role:** servo to the *true* object pose (mm-level) — CNN position (~10 cm off) is
  too coarse to grasp with, so we only use it to know *which* tool and *that* it's present.
- Constrained-random spawn: each tool only spawns on a slot its policy can grasp
  (brush R0/R1/R2, pliers R0/R2, screwdriver R1/R2; scissors has no policy → parked on R3, skipped).

---

## 2. Run the demo

```bash
source .env
# GUI (watch it), hands-free auto:
DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_sequential.py \
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
  --brush_ckpt       checkpoints/policy_state_bc_mimic_v2.pth \
  --pliers_ckpt      checkpoints/policy_state_bc_mimic_pliers_v2.pth \
  --screwdriver_ckpt checkpoints/policy_state_bc_mimic_screwdriver_v2.pth \
  --seg_ckpt         checkpoints/air2_segmentation_unet_newscene.pth \
  --num_episodes 1 --max_steps_per_tool 2000 \
  --out eval_results/seq.json < /dev/null
```
- Add `--headless` for no window. Drop `< /dev/null` to drive tool choice manually
  (prompts `Pick tool >`: `auto` / `<tool>` / `skip <tool>` / `done`).
- Launcher shortcut: `./launch_air2.sh eval-sequential 1 --seg_ckpt … --brush_ckpt …`
- `--no_cnn` uses ground-truth poses and **loads no camera** (good for testing without the CNN).

**Main demo file:** [scripts/eval_sequential.py](scripts/eval_sequential.py). Key pieces:
- `place_tools()` — constrained spawn + settle (uses `env.step`, holds robot at home).
- `run_cnn()` (~line 250) — resize RGB→224 for the U-Net, upsample logits→640×360 for the
  depth lookup, `extract_detections()` → world positions; filters by `PEG_Z_MIN` (= 1.10).
- `run_tool()` — the phase machine + servo (mirrors the per-object `eval_state_bc_v2.py`).
- `go_home()` — lift clear of pegs → drive to `default_joint_pos` → `write_data_to_sim()`
  (the missing `write_data_to_sim` was why tools 2/3 failed before).

---

## 3. The environments (where the scene/robot/tools/cameras come from)

```
isaaclab_ext/tasks/
  air2_franka/                         ← base env family
    joint_pos_env_cfg.py               ← robot (FRANKA_PANDA_HIGH_PD_CFG), init pose,
                                          wrist_camera, IK action
    segmentation_env_cfg.py            ← _apply_segmentation_cameras(): main_camera (CNN view)
    cnn/                               ← the CNN: model.py, postprocess.py
  air2_robotis_franka/                 ← the pegboard scene we actually use (inherits air2_franka)
    joint_pos_env_cfg.py               ← robotis_net_table pegboard, _SLOTS (R0–R3 world coords),
                                          reset_objects_on_slots event, basket
    segmentation_env_cfg.py            ← Robotis Segmentation-Play (collection task)
    mimic_env_cfg.py                   ← Isaac Lab Mimic config (demo generation)
```

- **Robot init pose** (the home pose every pick must start from):
  [air2_franka/joint_pos_env_cfg.py:50-57](isaaclab_ext/tasks/air2_franka/joint_pos_env_cfg.py#L50-L57)
  — `joint1=-1.5708`, `joint4=-2.26892803`, the rest at Franka defaults. The demo reads this
  live via `robot.data.default_joint_pos` (don't hardcode it).
- **Slot world coords** R0–R3: `_SLOTS` in
  [air2_robotis_franka/joint_pos_env_cfg.py:63-72](isaaclab_ext/tasks/air2_robotis_franka/joint_pos_env_cfg.py#L63-L72)
  (mirrored as `SLOT_WORLD` in `eval_sequential.py`).
- **Tasks:** demo = `Isaac-AIR2-Robotis-Franka-Brush-Play-v0` (spawns all 4 tools);
  collection = `Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0`.

---

## 4. The camera — current state (READ BEFORE REPOSITIONING)

| | Collection (`main_camera`) | Inference (`board_camera`) |
|---|---|---|
| File | `air2_franka/segmentation_env_cfg.py` `_apply_segmentation_cameras()` | `scripts/eval_sequential.py` ~L471 |
| `focal_length` | **18.0** | **24.0** |
| `pos` | **(-4.8, -5.2, 2.2)** | **(-1.0, -3.5, 1.8)** |
| `rot` (ros quat) | **(0.1598, -0.3477, 0.8395, -0.3857)** | **(-0.2068, 0.2807, 0.7545, -0.5560)** |
| resolution | 640×360 | 640×360 |
| `convention` | ros | ros |

> ⚠️ These do **not** match. The current working CNN
> (`checkpoints/air2_segmentation_unet_newscene.pth`) was trained on the *board_camera*
> view (the inference one). The `main_camera` in the collection cfg is a newer, wider
> proposed angle that has **not** been used to collect/retrain yet. When you reposition,
> **unify both to the same values** and retrain — don't leave them different.

To reposition: choose new `pos` / `rot` (ros convention; `rot` is a quaternion `(w,x,y,z)`)
and `focal_length`, and set the **identical** values in **both** rows above.

---

## 5. Re-collect segmentation data (after moving the camera)

```bash
source .env
DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/collect_air2_segmentation_data.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --cameras board_camera wrist_camera \
  --enable_cameras --frames 750 \
  --output datasets/air2_segmentation_newscene2
```
- Collect script: [scripts/collect_air2_segmentation_data.py](scripts/collect_air2_segmentation_data.py).
  It saves `images/<id>_<camera>_rgb.png` + masks + per-frame metadata, then a train/val split.
- **Camera-name note:** make sure the `--cameras` name matches the camera you defined in the cfg.
  The current dataset uses `board_camera`; the cfg helper defines `main_camera`. Pick ONE name and
  use it in `_apply_segmentation_cameras` (the `prim_path`/scene key) **and** in `--cameras`
  **and** as the scene key read in `eval_sequential.run_cnn` (`scene["board_camera"]`).
- 9 classes (see `datasets/.../class_map.json`): 0 bg, 1 brush, 2 pliers, 3 scissors,
  4 screwdriver, 5 robot, 6 basket, 7 table, 8 environment.
- Render must run on an **unrestricted display GPU** (GPU 0 here). Do **not** wrap with
  `CUDA_VISIBLE_DEVICES=1` — it breaks camera rendering ("CUDA in bad state").

---

## 6. Retrain the U-Net

```bash
CUDA_VISIBLE_DEVICES=1 ~/isaacsim/python.sh -u scripts/train_air2_segmentation.py \
  --backbone unet \
  --data datasets/air2_segmentation_newscene2 \
  --epochs 60 --lr 1e-3 --image-size 224 \
  --output checkpoints/air2_segmentation_unet_newscene2.pth
```
- Trainer: [scripts/train_air2_segmentation.py](scripts/train_air2_segmentation.py). U-Net,
  `base_channels=32`, 9 classes, trains at **224×224** (the demo resizes the live frame to 224
  to match — keep this consistent if you change `--image-size`).
- Training is **CPU-bound** (data aug). Run **one** at a time — parallel runs on one box
  oversubscribe cores and slow everything down.
- After training, point the demo at the new checkpoint via `--seg_ckpt`.
- **Validate detection without launching the sim** with the viz tool (draws boxes on saved frames):
  ```bash
  ~/isaacsim/python.sh scripts/state_bc_v2/viz_detections.py \
    --checkpoint checkpoints/air2_segmentation_unet_newscene2.pth \
    --data datasets/air2_segmentation_newscene2 --camera board --num 8 --min-area 6 \
    --output viz_detections_new
  ```
  Tiny tools (brush/screwdriver) are ~10–30 px at 224 — keep `--min-area` low (~5–8) or they get
  filtered out (this once looked like a model failure but was just the filter).

---

## 7. The per-tool BC policies (already done — don't retrain for the camera change)

The grasp uses the true pose, so **moving the camera does NOT require retraining the BC policies.**
Each tool has its own state-BC checkpoint + a packaged folder with its own `eval_state_bc_v2.py`:

| Tool | Checkpoint | Package | Slots |
|---|---|---|---|
| brush | `checkpoints/policy_state_bc_mimic_v2.pth` | `R0_R1_R2_ALL_WORKS_FOR_BRUSH/` | R0,R1,R2 |
| pliers | `checkpoints/policy_state_bc_mimic_pliers_v2.pth` | `PLIERS_R0_R2_WORKS/` | R0,R2 |
| screwdriver | `checkpoints/policy_state_bc_mimic_screwdriver_v2.pth` | `SCREWDRIVER_R1_R2_WORKS/` | R1,R2 |
| scissors | — (none) | — | parked R3, skipped |

`eval_sequential.run_tool()` reuses the same grip/carry/release helper logic those three
packages use. Obs is 43-D (joint_pos/vel relative + object pos + target + last_action + eef + phase);
**`joint_pos` is relative to home**, which is why every pick must start at the home pose.

---

## 8. Gotchas (hard-won)

- **Never call bare `env.unwrapped.sim.step()`** to advance while controlling the robot — it
  bypasses the action pipeline and the arm lurches. Use `env.step(neutral)`.
- **`set_joint_position_target()` needs `robot.write_data_to_sim()`** before stepping, or the
  target is never applied (robot doesn't move). This is in `go_home`.
- **No `env.reset()` mid-episode while cameras are active** — it deadlocks the
  `omni.syntheticdata` render pipeline. Only reset at episode start.
- **Checkpoints & datasets are gitignored** — share them out-of-band (not via git).
- **`PEG_Z_MIN = 1.10`** in `eval_sequential.py` — tools sit at z≈1.18; keep this below 1.18
  and above the basket (~1.14) so tools are detected but the basket isn't counted as a peg tool.
