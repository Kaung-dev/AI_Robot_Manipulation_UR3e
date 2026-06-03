# Multi-Object Pick-and-Place Evaluation Guide

Whole-system integrated demo: the robot picks and places all 4 objects in a single episode using per-object state-BC policies. Objects are picked in order of distance to the basket (closest first), adapting to randomized peg placements each round.

---

## Overview

```
                    Environment reset (randomized object positions)
                                     |
                                     v
                    Read GT positions of all 4 objects + basket
                                     |
                                     v
                    Sort objects by distance to basket (closest first)
                                     |
                          +----------+----------+
                          |          |          |
                          v          v          v
                     Object 1   Object 2   Object 3   Object 4
                     (closest)                         (farthest)
                          |
                          v
               Load policy_state_bc_<tool>.pth
                          |
                          v
              Phase machine: APPROACH -> GRIP -> CARRY -> RELEASE
                          |
                  success / timeout
                          |
                          v
                    Next object ...
                          |
                          v
                   Round complete -> reset env -> next round
```

Each per-object policy is a small MLP (43-D obs -> [256, 128, 64] -> 7-D action) trained from Mimic-generated demonstrations. The phase machine handles gripper timing so the policy only needs to output arm motion.

---

## Prerequisites

### 1. Per-object checkpoints

You need one checkpoint per tool in `checkpoints/`:

```
checkpoints/
  policy_state_bc_brush.pth
  policy_state_bc_pliers.pth
  policy_state_bc_scissors.pth
  policy_state_bc_screwdriver.pth
```

The script auto-discovers these files. It also checks for the `_mimic` suffix variant (e.g. `policy_state_bc_brush_mimic.pth`). If a tool's checkpoint is missing, that tool is skipped.

### 2. Isaac Lab environment

The eval uses `Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0` which spawns all 4 objects on randomized peg slots with cameras enabled.

### 3. Repo on `Main+Experimental_merge` branch (or later)

The multi-object eval script and per-object `--object` training flag were added on this branch.

---

## Quick Start

### Linux (launch script)

```bash
# Run 1 round (pick all 4 objects once), GUI mode:
./launch_air2.sh eval-multi

# Run 3 rounds with longer timeout per object:
./launch_air2.sh eval-multi 3 2500
```

Arguments: `eval-multi [num_rounds] [max_steps_per_object]`

### Linux (full command)

```bash
cd /mnt/extra/IsaacLab && ./isaaclab.sh -p \
  /mnt/extra/ai_ws/AI_Robot_Manipulation_UR3e/scripts/eval_multi_object_bc.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --ckpt_dir /mnt/extra/ai_ws/AI_Robot_Manipulation_UR3e/checkpoints \
  --num_envs 1 \
  --num_rounds 3 \
  --max_steps_per_object 2000 \
  --enable_cameras
```

### Windows

```powershell
& "D:\IsaacLab\isaaclab.bat" -p `
  D:\AI_Robot_Manipulation_UR3e\scripts\eval_multi_object_bc.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --ckpt_dir D:\AI_Robot_Manipulation_UR3e\checkpoints `
  --num_envs 1 `
  --num_rounds 3 `
  --max_steps_per_object 2000 `
  --enable_cameras
```

### Headless (for metrics only)

Add `--headless` to any of the commands above. Results are written to JSON.

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0` | Env with all 4 objects + cameras |
| `--ckpt_dir` | `checkpoints/` | Directory containing `policy_state_bc_<tool>.pth` |
| `--num_envs` | `1` | Number of parallel envs (use 1 for GUI demo) |
| `--num_rounds` | `1` | Full pick-all-4 rounds. Env resets between rounds. |
| `--max_steps_per_object` | `2000` | Timeout per object (~40s at 50Hz) |
| `--reset_delay_steps` | `150` | Pause after placing, before next object (~3s) |
| `--episode_length_s` | `120.0` | Env episode timeout (should be > 4 x max_steps) |
| `--out` | `eval_results/multi_object_bc.json` | Output JSON path |

---

## What Happens During a Run

### Console output example

```
[orchestrator] found 4 policies: ['brush', 'pliers', 'scissors', 'screwdriver']
  loaded brush: checkpoints/policy_state_bc_brush.pth
  loaded pliers: checkpoints/policy_state_bc_pliers.pth
  loaded scissors: checkpoints/policy_state_bc_scissors.pth
  loaded screwdriver: checkpoints/policy_state_bc_screwdriver.pth

############################################################
[orchestrator] ROUND 1/3
[orchestrator] pick order (closest to basket first):
  1. screwdriver (dist=0.412m)
  2. brush (dist=0.573m)
  3. pliers (dist=0.689m)
  4. scissors (dist=0.831m)
############################################################

[orchestrator] object 1/4: screwdriver
  [screwdriver] step=   0  phase=0  ee_dist=0.432m  carry=0  xy_basket=0.891m
  [screwdriver] step= 100  phase=0  ee_dist=0.087m  carry=0  xy_basket=0.891m
  [screwdriver] step= 300  phase=2  ee_dist=0.031m  carry=42  xy_basket=0.654m
  [screwdriver] PLACED at step 487! Holding 150 steps...
[orchestrator] screwdriver: SUCCESS (steps=487, min_dist=0.215m)

[orchestrator] object 2/4: brush
  [brush] step=   0  phase=0  ee_dist=0.573m  carry=0  xy_basket=1.102m
  ...
```

### Phase machine

Each object goes through 4 phases:

| Phase | Gripper | Arm | Transition condition |
|---|---|---|---|
| 0 APPROACH | Open | BC policy | EE within 8cm of object for 250 steps |
| 1 GRIP | Closed | Frozen (no motion) | Hold for 50 steps |
| 2 CARRY | Closed | BC policy | Object within 35cm XY of basket AND Z <= 1.4m, after 200+ steps |
| 3 RELEASE | Open | -- | Done, move to next object |

If an object times out (reaches `max_steps_per_object` without placing), it's marked as failed and the robot moves to the next object.

---

## Output JSON

Results are saved to `eval_results/multi_object_bc.json`:

```json
{
  "num_rounds": 3,
  "total_objects_attempted": 12,
  "total_successful_placements": 9,
  "overall_success_rate": 0.75,
  "rounds": [
    {
      "round": 1,
      "pick_order": ["screwdriver", "brush", "pliers", "scissors"],
      "successful": 3,
      "total": 4,
      "objects": [
        {
          "object": "screwdriver",
          "scene_key": "tool_screwdriver",
          "steps": 487,
          "reward": 12.5,
          "min_basket_dist": 0.215,
          "success": true
        },
        ...
      ]
    },
    ...
  ]
}
```

---

## Training Per-Object Policies (if you don't have them yet)

Each tool needs its own Mimic pipeline: source demos -> annotate -> generate -> train.

### Full pipeline for one tool

Replace `TOOL` with `brush`, `pliers`, `scissors`, or `screwdriver`:

```bash
TOOL=screwdriver

# 1. Collect ~10 source demos via keyboard teleop
./launch_air2.sh collect-mimic $TOOL 10 \
  datasets/air2_mimic_${TOOL}_source.hdf5

# 2. Annotate (replay + add subtask signals)
./launch_air2.sh annotate-mimic $TOOL \
  datasets/air2_mimic_${TOOL}_source.hdf5 \
  datasets/air2_mimic_${TOOL}_annotated.hdf5

# 3. Generate ~500 synthetic demos (headless, 4 parallel envs)
./launch_air2.sh generate-mimic $TOOL \
  datasets/air2_mimic_${TOOL}_annotated.hdf5 \
  datasets/air2_mimic_${TOOL}_generated.hdf5 \
  500

# 4. Train state-BC policy
./launch_air2.sh train-state-bc $TOOL \
  datasets/air2_mimic_${TOOL}_generated.hdf5 \
  checkpoints/policy_state_bc_${TOOL}.pth \
  300
```

### Windows equivalent (Step 4 only -- steps 1-3 are similar, swap isaaclab.sh for isaaclab.bat)

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\train_state_bc_from_hdf5.py `
  --hdf5 datasets\air2_mimic_screwdriver_generated.hdf5 `
  --object screwdriver `
  --out checkpoints\policy_state_bc_screwdriver.pth `
  --epochs 300
```

### Train all 4 tools in sequence (Linux)

```bash
for TOOL in brush pliers scissors screwdriver; do
  echo "=== Training $TOOL ==="
  ./launch_air2.sh train-state-bc $TOOL \
    datasets/air2_mimic_${TOOL}_generated.hdf5 \
    checkpoints/policy_state_bc_${TOOL}.pth \
    300
done
```

Training takes ~30-60 seconds per tool on GPU (300 epochs, ~500 demos, pure PyTorch, no Isaac Sim needed).

### Verify checkpoints exist

```bash
ls -la checkpoints/policy_state_bc_*.pth
```

Expected:
```
policy_state_bc_brush.pth        (~200 KB)
policy_state_bc_pliers.pth       (~200 KB)
policy_state_bc_scissors.pth     (~200 KB)
policy_state_bc_screwdriver.pth  (~200 KB)
```

---

## Evaluating Individual Objects

Before running the full multi-object demo, you can test each policy in isolation:

```bash
# Test screwdriver policy alone (20 episodes, single-object env)
./launch_air2.sh eval-state-bc screwdriver \
  checkpoints/policy_state_bc_screwdriver.pth 20 2000
```

This runs `eval_state_bc.py` in the per-object env (`Isaac-AIR2-Robotis-Franka-Screwdriver-Play-v0`) with only that one object spawned.

---

## Troubleshooting

### "No per-object checkpoints found"

The script looks for files matching `policy_state_bc_<tool>.pth` or `policy_state_bc_<tool>_mimic.pth` in `--ckpt_dir`. Check:

```bash
ls checkpoints/policy_state_bc_*.pth
```

If your files are named differently (e.g. `policy_state_bc_mimic_v2.pth`), either rename them or create symlinks:

```bash
cd checkpoints
ln -s policy_state_bc_mimic_v2.pth policy_state_bc_brush.pth
```

### Robot barely moves / raw_arm ~0.003

The policy is outputting near-zero actions, likely due to observation mismatch. Check:
- Was the policy trained with the 43-D observation (42-D obs + 1 phase bit)?
- Does the checkpoint contain `input_dim: 43` and `action_dim: 7`?

```python
import torch
ckpt = torch.load("checkpoints/policy_state_bc_brush.pth", map_location="cpu")
print(f"input_dim={ckpt['input_dim']}, action_dim={ckpt['action_dim']}, hidden={ckpt['hidden_dims']}")
```

### Object falls during carry / gripper doesn't close

The phase machine transitions to GRIP after 250 steps near the object (5 seconds). If the policy doesn't stay near the object long enough, adjust `NEAR_THRESH` in the script.

### Partial runs (only some policies available)

The script automatically skips tools without a checkpoint. You can run with 1, 2, or 3 policies and the others are simply skipped. The JSON output only includes attempted objects.

### Episode times out before all 4 objects are placed

Increase `--episode_length_s` (default 120s). With 2000 max steps per object at 50Hz, worst case is 4 x 40s = 160s. Use `--episode_length_s 200`.

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/eval_multi_object_bc.py` | Multi-object orchestrator (this guide) |
| `scripts/eval_state_bc.py` | Single-object eval (for debugging one policy) |
| `scripts/train_state_bc_from_hdf5.py` | Train state-BC from Mimic HDF5 (supports `--object` param) |
| `scripts/collect_mimic_demos.py` | Collect source demos via teleop |
| `scripts/run_mimic_annotate.py` | Annotate demos with subtask signals |
| `scripts/run_mimic_generate.py` | Generate synthetic demos from annotated source |
| `isaaclab_ext/tasks/air2_franka/objects.py` | Object catalog (scene keys, class IDs, labels) |
| `launch_air2.sh` | Pipeline launcher (`eval-multi`, `train-state-bc`, etc.) |
