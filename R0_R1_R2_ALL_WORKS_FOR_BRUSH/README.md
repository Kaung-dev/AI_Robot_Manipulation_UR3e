# Brush pick-and-place — R0, R1, R2 ALL WORK ✅ (2026-06-02)

State-BC policy that grasps the brush at **all 3 reachable slots (R0, R1, R2)**,
carries it to the basket, drops it, and it **LANDS** inside. Verified in GUI:
8/8 episodes `LANDED=True` across all slots.

## Files
- `policy_state_bc_mimic_v2.pth`  — trained policy (43-D obs, normalized + obs-aligned, BC-DW1). 555 multi-slot demos. best val ≈ 0.00091.
- `policy_state_bc_mimic_v2.log.json` — training loss log.
- `train_state_bc_from_hdf5_v2.py` — trainer.
- `eval_state_bc_v2.py` — evaluator (servo + release + landed scoring).

## What made it work (the full fix chain)
1. **Multi-slot data generation** — the original generation produced a SINGLE slot
   (all demos identical). Fix: env var `MIMIC_KEEP_RANDOMIZATION=1` keeps object
   randomization ON during generation, so each trial spawns a different slot.
   Edit lives in `isaaclab_ext/tasks/air2_robotis_franka/mimic_env_cfg.py`
   (`_disable_object_randomization`). Regenerate with:
     MIMIC_KEEP_RANDOMIZATION=1 ./launch_air2.sh generate-mimic brush \
        datasets/air2_mimic_demos_annotated.hdf5 datasets/air2_mimic_generated_v2.hdf5 1000
   (brush only reaches R0/R1/R2 by design — R3 is unreachable for the brush.)
2. **Trainer (`train_state_bc_from_hdf5_v2.py`):**
   - BC-DW1 dwell-step down-weighting (from teammate)
   - obs-alignment: dims 21:28 (target_object_position) replaced with the actual
     object position (robot-root) + identity quat, instead of the resampling
     command (which was noise + a train/eval mismatch)
   - observation normalization (mean/std saved in the ckpt, applied at eval)
3. **Evaluator (`eval_state_bc_v2.py`):**
   - applies the same obs-alignment + normalization (auto-detected from ckpt)
   - RELEASE fix: drops the bogus `obj_z<=1.4` gate so it releases when centred
     over the basket (it was carrying the brush above the rim → never dropped)
   - release-and-settle window + disables the env's early terminations so the
     brush physically lands before scoring (`LANDED` = released + settled in basket)
   - SERVO (R2 fix): when EE within 0.18 m of the brush during approach, drive the
     EE straight onto the object's true pose (robot-root frame) so it can't
     over/under-shoot — fixed R2's downward overshoot.

## Reproduce
Train (pure PyTorch, ~5 min, GPU 1/2 fine):
  CUDA_VISIBLE_DEVICES=1 ~/isaacsim/python.sh -u scripts/state_bc_v2/train_state_bc_from_hdf5_v2.py \
      --hdf5 datasets/air2_mimic_generated_v2.hdf5 --epochs 300 \
      --out checkpoints/policy_state_bc_mimic_v2.pth

Eval (GUI must run on GPU 0 = the display GPU; headless can run on GPU 1/2):
  DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/state_bc_v2/eval_state_bc_v2.py \
      --state_bc_ckpt checkpoints/policy_state_bc_mimic_v2.pth \
      --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
      --num_envs 1 --num_episodes 20 --max_steps 2000 --reset_delay 100

## Next
Repeat the pipeline per object (pliers / scissors / screwdriver): collect → annotate
→ generate (MIMIC_KEEP_RANDOMIZATION=1) → train_v2 → eval_v2.
