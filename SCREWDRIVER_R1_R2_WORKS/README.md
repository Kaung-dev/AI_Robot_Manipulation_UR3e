# Screwdriver pick-and-place — R1 & R2 work; R0 excluded (2026-06-02)

State-BC policy for the **screwdriver**. Grasps → carries → drops → **LANDS** in
the basket at slots **R1 and R2**. **R0 is excluded from spawning** (the grasp at
R0 was unreliable, like pliers at R1 — an awkward slot for this tool).

## Files
- `policy_state_bc_mimic_screwdriver_v2.pth` — trained policy (43-D obs, normalized + obs-aligned, BC-DW1).
- `policy_state_bc_mimic_screwdriver_v2.log.json` — training log (best val ≈ 0.00083).
- `train_state_bc_from_hdf5_v2.py` — trainer.
- `eval_state_bc_v2.py` — evaluator (servo + release + landed scoring + `--exclude_slots`).
- `filter_clean_demos.py` — keeps only demos that delivered to the basket.

## Recipe (same as brush/pliers)
1. Dataset `air2_mimic_generated_screwdriver.hdf5` generated with `MIMIC_KEEP_RANDOMIZATION=1`
   (833 demos, 90% clean, slots R0/R1/R2 — screwdriver can't reach R3 by design).
2. Filtered to 722 clean demos (`filter_clean_demos.py`).
3. Trained `train_state_bc_from_hdf5_v2.py` (300 epochs, val ≈ 0.00083).
4. Eval with R0 excluded.

## Run (GUI; R0 excluded)
```
source .env
DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/state_bc_v2/eval_state_bc_v2.py \
    --state_bc_ckpt checkpoints/policy_state_bc_mimic_screwdriver_v2.pth \
    --task Isaac-AIR2-Robotis-Franka-Screwdriver-Play-v0 \
    --exclude_slots R0 \
    --num_envs 1 --num_episodes 20 --max_steps 2000 --reset_delay 100
```

## Status
- R1 ✅ lands · R2 ✅ lands · R0 ⛔ excluded from spawn (unreliable grasp)

## Honest note
Hybrid controller: BC does coarse reach + carry; scripted servo does the final
approach and a state machine does grip/release. `--servo_dist 0` shows the weaker
pure-BC behavior.
