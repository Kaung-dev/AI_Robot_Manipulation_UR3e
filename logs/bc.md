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

## 2026-06-01 — Linux freeze fix for collect_air2_manual_demos.py
**Who:** Declan
**Changed:** `scripts/collect_air2_manual_demos.py` — removed `ep.pre_export()` call before `hdf5_handler.write_episode(ep)`
**Tried:** `./launch_air2.sh collect-demos 2 keyboard scissors` on Linux
**Result:** Every time Enter was pressed to save an episode, the robot froze and could not be moved. HDF5 output file was left as a corrupt 96-byte truncated file. PNG/NPZ/JSON saved correctly (ep_000 with 264 frames was intact).
**Root cause:** `ep.pre_export()` called on `EpisodeData` but the method does not exist in Isaac Lab v2.2.1. Raised `AttributeError` which propagated uncaught and triggered `simulation_app.close()`. Isaac Sim shutdown on Linux takes ~30–60s, appearing as a frozen robot.
**Status:** working after fix
**Fix:**
1. Removed `ep.pre_export()` from line 256 of `collect_air2_manual_demos.py` — method does not exist in this Isaac Lab version and is not needed (tensors are already stacked by `EpisodeData.add()`).
2. Deleted corrupt `datasets/air2_mimic_source.hdf5` (96-byte truncated file left by the crash — next run would fail trying to open it in append mode).
**Note:** Teammate on Windows was unaffected — likely running a newer Isaac Lab version where `pre_export()` exists, or the faster Windows shutdown masked the crash. The warning `"no grasp detected — forcing signal at last frame"` is harmless — it only pads the Mimic grasp signal and does not affect the save.
**Added:** `COLLECT_AND_TRAIN.md` — full pipeline guide with Linux/Windows commands for demo collection and training.

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

---

## Mimic pipeline — how to train from an existing HDF5

If you already have a raw source HDF5 (e.g. from teleop collection), follow these steps:

**Step 1 — Annotate**
```bash
./launch_air2.sh annotate-mimic datasets/your_raw.hdf5 datasets/your_annotated.hdf5
```
Replays each demo and auto-detects subtask boundaries. Expect ~80-85% survival. If <70% survive, the raw demos have quality issues.

**Step 2 — Generate synthetic demos**
```bash
./launch_air2.sh generate-mimic datasets/your_annotated.hdf5 datasets/your_generated.hdf5 1000
```
Tries 1000 generations, keeps successes (~40-50% → ~400-500 demos). Headless, 4 envs, ~1-2 hours on RTX 3050.

**Step 3 — Train state-BC**
```bash
./launch_air2.sh train-state-bc datasets/your_generated.hdf5 checkpoints/policy_state_bc_yourobj.pth 300
```
Pure PyTorch, no Isaac Sim needed. ~10-20 min.

**Step 4 — Eval**
```bash
./launch_air2.sh eval-state-bc checkpoints/policy_state_bc_yourobj.pth 20
```

---

## 2026-06-02 — Gripper learning problem + eval phase logic flaw

**Problem: BC never learns gripper timing**
The gripper action is binary (+1/-1) in demos but BC regresses it continuously with MSE.
The output is some float like 0.1 or -0.3 — ambiguous for `BinaryJointPositionActionCfg`.
More fundamentally, gripper open/close is a mode switch that depends on task history
(have I grasped yet?), not just current obs. A feedforward MLP with no memory can't
reliably learn this from obs alone.

**The eval_state_bc.py workaround masks the problem entirely:**
The eval script ignores the BC policy's gripper output completely and replaces it with
hardcoded phase logic:
- Phase 0 (approach): arm=BC, gripper forced open, waits 250 steps within 0.08m of object
- Phase 1 (grip): arm frozen, gripper forced closed, holds 50 steps
- Phase 2 (carry): arm=BC, gripper forced closed

Phase 2 arm control is also broken: BC has no idea it just grasped. It gets the current
obs and outputs an arm action. The 50-step freeze in phase 1 creates an out-of-distribution
obs at the start of phase 2 (arm frozen, gripper closed — never seen in training demos which
are smooth and continuous). Policy is essentially guessing where to go in phase 2.

**Why Mimic should fix this:**
Mimic-generated demos preserve full trajectory continuity — no artificial phases, no freezes.
BC trains on data matching what it sees at eval time. Gripper timing is still in the action
labels but the obs context around transitions is richer and consistent across 1000 demos.

**Decided fix: phase-conditioned BC**
Add a 1-bit phase label (0=approach, 1=carry) as an extra input to the BC policy (42-D → 43-D obs).

**Training side:**
- Mimic annotation already finds the exact step where `grasp_brush` fires (subtask boundary)
- Use that step index to label every obs in the generated HDF5: steps before boundary = phase 0, steps after = phase 1
- Add phase as a 1-D input to the MLP in `train_state_bc_from_hdf5.py`
- BC now learns two distinct arm behaviors from one policy: phase 0 → move toward object, phase 1 → move toward basket

**Eval side (NO CHANGES):**
- `eval_state_bc.py` phase logic stays identical — it already detects phase via proximity + step count and forces the gripper
- The only difference is the policy now RECEIVES the phase bit as input and was trained with it
- eval owns phase detection + gripper forcing; BC just learns the arm trajectory conditioned on phase
- Result: phase 2 arm no longer guesses where to go — it was trained knowing it's in carry phase

**Why this is better than current approach:**
- Current eval forces phase but policy was trained without it → arm in phase 2 is OOD
- New approach: eval passes phase=0 → policy moves to object; detects grasp, flips to phase=1 → policy moves to basket
- No frozen-arm hack needed, no OOD obs at carry start
- Same `grasp_brush` signal used in annotation AND eval — consistent

**Implementation steps (next session):**
1. After Mimic annotation, the annotated HDF5 has subtask boundary info per demo — extract step index where subtask 0 ends
2. When loading HDF5 in `train_state_bc_from_hdf5.py`, compute phase label per step (0 before boundary, 1 after)
3. Append phase to obs tensor: `obs = torch.cat([obs_42d, phase_1d], dim=-1)` → 43-D
4. Update `OBS_DIM` from 42 to 43 in training script
5. In `eval_state_bc.py`: pass `phase[i].float()` as extra dim when building the obs tensor fed to policy
6. Save new checkpoint as `policy_state_bc_mimic_v2.pth` to distinguish from old 42-D checkpoint

**Gotchas:**
- Each HDF5 must be single-object only — do not mix objects
- `grasp_radius` / `finger_threshold` in `mimic_env_cfg.py` may need per-object tuning (brush uses `0.15` / `0.07` due to the ring)
- Always use unique filenames — never overwrite existing HDF5s
- The annotate/generate task must match the object the demos were collected for
- `BASKET_POS_LOCAL` in `constants.py` must match the actual USD scene — verify before collecting

---

## 2026-06-02 — Phase-conditioned BC implemented + Mimic pipeline completed to annotation

**Who:** Steph

**Changed:**
- `subtask.py` — added `gripper_closed()`: phase boundary now fires on gripper action command (`action[:, -1] < 0`), not physics proximity+finger check. Physics check was unreliable for the brush (ring prevents full closure). `gripper_closed` includes transition print: `[grasp_brush] env N phase change at step X`.
- `mimic_env_cfg.py` — `grasp_brush` ObsTerm now uses `gripper_closed` instead of `grasped`.
- `train_state_bc_from_hdf5.py` — 43-D obs implemented: `OBS_KEYS` extended with `eef_pos`+`eef_quat`, phase label (0=approach, 1=carry) derived per step from `grasp_brush` 0→1 transition in generated HDF5, appended as 1-D.
- `eval_state_bc.py` — appends `ee_pos(3) + ee_quat(4) + phase_bit(1)` to obs before policy call. `phase_bit = (phase >= 2).float()`.

**Mimic pipeline status:**
- ✅ 80 source demos collected → `datasets/air2_mimic_demos_v2.hdf5`
- ✅ 80/80 annotated → `datasets/air2_mimic_demos_annotated.hdf5` (31MB, on Google Drive)
- ⬜ Generation → teammate runs on their device: `./launch_air2.sh generate-mimic datasets/air2_mimic_demos_annotated.hdf5 datasets/air2_mimic_generated_v2.hdf5 1000`
- ⬜ Train: `./launch_air2.sh train-state-bc datasets/air2_mimic_generated_v2.hdf5 checkpoints/policy_state_bc_mimic_v2.pth 300`
- ⬜ Eval: `./launch_air2.sh eval-state-bc checkpoints/policy_state_bc_mimic_v2.pth 20 1600`

**Status:** blocked on generation — waiting for teammate
