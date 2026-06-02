# state_bc_v2 — obs-aligned state-BC train + eval

Experimental copies of the state-BC scripts with the **train↔eval observation
mismatch fixed**. The originals in `scripts/` are untouched.

## The bug these fix
The 43-D obs vector's `target_object_position` columns (dims **21:28**) come from
`generated_commands("object_pose")` — a `UniformPoseCommand` GOAL that resamples
over time, i.e. noise w.r.t. the action. The original `eval_state_bc.py`
overwrites those dims with the object's **world** pose, but
`train_state_bc_from_hdf5.py` trains on the raw command. So the policy learns one
thing for dims 21:27 and is fed a different thing at test time → unreliable,
slot-dependent grasping.

## The fix (both files, identical obs now)
Dims **21:24** = `object_position` (robot-root frame, already at dims 18:21),
dims **24:28** = identity quat `[1,0,0,0]`.
- `train_state_bc_from_hdf5_v2.py` rewrites these columns when loading the HDF5.
- `eval_state_bc_v2.py` fills these dims from the same **robot-root** object pos
  (the original used **world** pos — frame mismatch).

Everything else is byte-for-byte the original.

## Run
```bash
source .env
# Train (after brush generation completes):
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/state_bc_v2/train_state_bc_from_hdf5_v2.py \
    --hdf5 datasets/air2_mimic_generated_v2.hdf5 \
    --epochs 300 --out checkpoints/policy_state_bc_mimic_v2.pth

# Eval (GUI — drop --headless to watch):
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/state_bc_v2/eval_state_bc_v2.py \
    --state_bc_ckpt checkpoints/policy_state_bc_mimic_v2.pth \
    --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
    --num_envs 1 --num_episodes 10
```

## Not done here (optional next step)
Observation **normalization** (dataset mean/std) — the biggest remaining
generalization win for the un-normalized MLP. Left out so this change is purely
the train/eval alignment.
