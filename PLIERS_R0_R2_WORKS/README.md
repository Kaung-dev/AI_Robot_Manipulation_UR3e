# Pliers pick-and-place — R0 & R2 work; R1 excluded (2026-06-02)

State-BC policy for the **pliers**. Grasps → carries → drops → **LANDS** in the
basket at slots **R0 and R2**. **R1 is excluded from spawning** (the grasp at R1
was unreliable — the pliers sits at an awkward angle there and slides out of the
gripper during extraction). So the demo runs pliers only at the slots it can do.

## Files
- `policy_state_bc_mimic_pliers_v2.pth` — trained policy (43-D obs, normalized + obs-aligned, BC-DW1).
- `policy_state_bc_mimic_pliers_v2.log.json` — training log (best val ≈ 0.00081).
- `train_state_bc_from_hdf5_v2.py` — trainer.
- `eval_state_bc_v2.py` — evaluator (servo + release + landed scoring + `--exclude_slots`).
- `filter_clean_demos.py` — keeps only demos that delivered to the basket (data cleaning).

## What it took (vs brush)
1. **Multi-slot data** via `MIMIC_KEEP_RANDOMIZATION=1` at generation (same as brush).
2. **Data cleaning** — the raw pliers gen was only ~72% clean (vs brush 97%); training on
   the noise made the policy freeze/fly-away. Filtered to 713 clean demos
   (`filter_clean_demos.py`, object ended in basket) → retrain. THIS was the key fix.
3. **Eval** — same servo + release + landed-scoring as brush, plus markers stripped
   (the brush-specific debug spheres crash non-brush scenes), plus `--exclude_slots R1`.

## Run (GUI; R1 excluded)
```
source .env
DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/state_bc_v2/eval_state_bc_v2.py \
    --state_bc_ckpt checkpoints/policy_state_bc_mimic_pliers_v2.pth \
    --task Isaac-AIR2-Robotis-Franka-Pliers-Play-v0 \
    --exclude_slots R1 \
    --num_envs 1 --num_episodes 20 --max_steps 2000 --reset_delay 100
```

## Reproduce training
```
# (dataset air2_mimic_generated_pliers_v2.hdf5 generated with MIMIC_KEEP_RANDOMIZATION=1)
~/isaacsim/python.sh scripts/state_bc_v2/filter_clean_demos.py \
    --in datasets/air2_mimic_generated_pliers_v2.hdf5 \
    --out datasets/air2_mimic_generated_pliers_v2_clean.hdf5 --radius 0.18
CUDA_VISIBLE_DEVICES=1 ~/isaacsim/python.sh -u scripts/state_bc_v2/train_state_bc_from_hdf5_v2.py \
    --hdf5 datasets/air2_mimic_generated_pliers_v2_clean.hdf5 --epochs 300 \
    --out checkpoints/policy_state_bc_mimic_pliers_v2.pth
```

## Status
- R0 ✅ lands · R2 ✅ lands · R1 ⛔ excluded from spawn (unreliable grasp)

## Known limitation (honest note)
The eval is a **hybrid**: BC does the coarse reach + carry; a scripted servo does the
final approach, and a state machine does grip/release. With `--servo_dist 0` (servo off)
the *pure* BC is substantially weaker — the helpers carry a lot of the precision.
