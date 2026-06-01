# BC Policy Log

Component: Behaviour Cloning / imitation learning
Pipeline: teleop demo collection → BC training → eval_bc.py rollout

---

## Entry format
**Date:** YYYY-MM-DD
**Who:**
**Changed:**
**Tried:**
**Result:**
**Status:** working / broken / investigating
**Fix:** (if applicable)

---

## Notes
- Train: `./launch_air2.sh train-bc <epochs>` — defaults to 50 epochs
- Eval: `scripts/eval_bc.py --bc_ckpt checkpoints/policy_bc.pth --backbone unet --unet_ckpt checkpoints/air2_segmentation_unet.pth`
- Always use `--backbone unet` with the segmentation checkpoint — ImageNet ResNet-18 features are useless for this env
- Success metric: EE within 0.70 m XY of basket AND above basket rim (Z ≥ basket_z - 0.05)

---

## 2026-05-31 — Switched to Mimic → state-BC pipeline
**Who:** Steph
**Decision:** Abandoned visual BC. Adopted Isaac Lab Mimic for data augmentation → state-BC policy (no camera, GT object_position from physics).
**Reason:** Visual BC memorised joint trajectory and ignored visual features. Mimic + state obs removes the visual dependency entirely and is simpler to get working.
**Target:** sim-only policy (no real-robot transfer needed for now).

### Mimic pipeline results
- 40 source demos collected (air2_mimic_demos.hdf5, ~17 MB) — keyboard teleop, brush only
- 33/40 annotated (air2_mimic_demos_annotated.hdf5) — 7 failed on replay (brush fell off slot)
- 556 synthetic demos generated (air2_mimic_generated.hdf5, ~140 MB) from 1157 attempts (~48% success, stopped early)
- 328k steps, 35-D obs, 7-D actions, no NaNs

### Linux fixes applied to Mimic collection
1. `collect_mimic_demos.py` — `env.device` → `env.unwrapped.device` (gym `OrderEnforcing` wrapper on Linux)
2. `Se3Keyboard.advance()` returns single tensor in IsaacLab 2.3.2, not `(delta_pose, gripper)` tuple
3. `teleop_episode.pre_export()` must be called before `add_episode()` (lists → tensors)
4. `run_mimic_annotate.py` / `run_mimic_generate.py` — inject task imports post-AppLauncher via exec+source-patch (`pxr` unavailable before boot)

### State-BC training script
`scripts/train_state_bc_from_hdf5.py` — pure PyTorch, no Isaac Sim needed.
- Obs: joint_pos(9) + joint_vel(9) + object_position(3) + target_object_position(7) + last_action(7) = 35-D
- Architecture: MLP 35→256→128→64→7 (ELU) — same as PPO actor
- Loss: smooth-L1, cosine LR decay, saves best-val checkpoint
- Run: `./launch_air2.sh train-bc` → `checkpoints/policy_state_bc_mimic.pth`

**Status:** training not yet run — next step

---

## 2026-05-31 — State-BC eval results + gripper debugging
**Who:** Steph
**Checkpoint:** checkpoints/policy_state_bc_mimic.pth (trained on 556 Mimic-generated demos, 328k steps, smooth-L1, val=0.00115)
**Eval setup:** 3-phase state machine in eval_state_bc.py:
- Phase 0 (APPROACH): BC arm output, gripper open
- Phase 1 (GRIP): hold arm still 50 steps, gripper closed
- Phase 2 (CARRY): BC arm output, gripper locked closed
- Gripper latches closed after accumulating 250 steps within 0.08m of object (~5s)

**Results:**
- Arm navigation: good — reliably approaches object in ~500 steps
- Gripper: smooth-L1 loss converges to ~0 for binary ±1 signal (mean of open/close), always reads "open". Tried BCE fix but it dominated the arm loss (BCE ~0.7 vs arm ~0.001). Settled on proximity heuristic in eval instead of policy output.
- Phase 2 (carry): arm stays near object (ee_dist ~0.036m constant = brush gripped and moving), but doesn't reliably reach basket
- Covariate shift: ~50% of episodes the arm drifts backward from the object (unfamiliar obs at episode start for some slot positions)
- Env reset bug: fixed — episodes ending at max_steps now explicitly reset env

**Gripper lesson:** Smooth-L1 regression on binary ±1 signal is wrong — minimizer is at 0 (= open). BCE works but BCE loss magnitude (~0.7) completely dominates arm loss (~0.001) — use weight 0.005x max. Proximity heuristic (accumulate N steps near object) is the practical workaround.

**Status:** partial — arm navigation works, carry unreliable, moving to PPO warm-start

---

## 2026-05-30 — Teammate checkpoint investigation
**Who:** Steph
**Checkpoint:** checkpoints/policy_bc.pth (pushed by teammate, trained on his machine)
**Config:** ResNet-18 + ImageNet weights (no segmentation encoder), trained on datasets\air2_manual_demos (Windows path, his camera config)
**Training:** 50 epochs, val_loss 0.015 — converged but on wrong data
**Tried:** eval_bc.py with num_envs=1, num_episodes=20
**Result:**
- Robot moves to basket area correctly (learned from joint positions alone)
- Never attempts pick — visual features garbage due to wrong camera position + no U-Net
- Gripper stays closed throughout — grip timing never learned
- ep rewards ≈ 0, min_basket_dist ~0.6 m, reached_basket=False every episode
**Status:** not usable — wrong camera config and encoder
**Root cause:** BC only learns what it sees in demos. Wrong viewpoint → visual encoder outputs noise → policy skips pick phase and falls back to the one thing it learned from proprioception (arm trajectory to basket).
**Fix:** Collect own VR demos with current camera config, retrain with `--backbone unet --unet_ckpt checkpoints/air2_segmentation_unet.pth`
