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

---

## 2026-06-02 — First full clean run achieved (rightmost brush slot)

**Who:** Steph

**Result:** Phase-conditioned BC (`policy_state_bc_mimic_v2.pth`, 43-D) completes a full APPROACH→GRIP→CARRY→RELEASE run when the brush starts at the rightmost peg slot. Other 2 slot locations do not generalise — arm barely moves (raw_arm~0.003, covariate shift from stale `target_object_position`).

**Commit:** `09a0e40` on `Main+Experimental_merge`

---

### Training script — `scripts/train_state_bc_from_hdf5.py`

**Obs layout (43-D, built from HDF5 keys in this exact order):**
```
joint_pos(9) + joint_vel(9) + object_position(3) + target_object_position(7)
+ actions/last_action(7) + eef_pos(3) + eef_quat(4) + phase(1) = 43-D
```
```python
OBS_KEYS = ["joint_pos", "joint_vel", "object_position", "target_object_position",
            "actions", "eef_pos", "eef_quat"]
```
`object_position` = robot-root-frame (3-D, small values ~[-0.03, -0.67, 0.57]).
`target_object_position` = brush world pose from `generated_commands(object_pose)` (7-D: xyz + identity quat [1,0,0,0]).
`eef_pos` = EE position in env-local frame (world pos minus env_origins, via `ee_frame_pos`).
`eef_quat` = EE quaternion world frame.

**Phase label derivation:**
Reads `obs/datagen_info/subtask_term_signals/grasp_brush` from HDF5 if present (annotated). Falls back to `actions[:, -1] < 0` (gripper close) for generated HDF5 which omits datagen_info. First 0→1 transition = boundary step. Steps before boundary: phase=0.0 (approach). Steps from boundary onward: phase=1.0 (carry).
```python
try:
    signals = d["obs"]["datagen_info"]["subtask_term_signals"]["grasp_brush"][:].flatten()
    transitions = np.where(np.diff(signals.astype(np.int32)) > 0)[0]
    boundary = int(transitions[0]) + 1 if len(transitions) > 0 else T
except (KeyError, ValueError):
    grip = d["actions"][:, -1]
    trans = np.where(np.diff((grip < 0).astype(np.int32)) > 0)[0]
    boundary = int(trans[0]) + 1 if len(trans) > 0 else T
```

**Architecture:** MLP `43 → 256 → 128 → 64 → 7` (ELU activations, no BatchNorm — matches rsl_rl actor layout for PPO warm-start compatibility).

**Training hyperparameters:**
- Loss: smooth-L1
- Optimizer: Adam, lr=3e-4
- LR schedule: CosineAnnealingLR, eta_min=1e-5, T_max=epochs
- Batch size: 512 (full dataset pinned to GPU)
- Val split: 10% random shuffle
- Epochs: 300
- Dataset: `datasets/air2_mimic_generated.hdf5` — 1000 Mimic-generated demos, 458,641 steps, all 1000 demos have clean gripper-close transition (boundary min=132, max=284, mean=199 steps)
- Best val loss: 0.00084

**Checkpoint format** (`checkpoints/policy_state_bc_mimic_v2.pth`):
```python
{"state_dict": ..., "input_dim": 43, "action_dim": 7,
 "hidden_dims": [256, 128, 64], "activation": "elu", "obs_keys": OBS_KEYS, "num_steps": 458641}
```

**Run command:**
```bash
./launch_air2.sh train-state-bc datasets/air2_mimic_generated.hdf5 checkpoints/policy_state_bc_mimic_v2.pth 300
```

---

### Eval script — `scripts/eval_state_bc.py`

**Task:** `Isaac-AIR2-Robotis-Franka-Brush-Play-v0`
**episode_length_s:** 80.0 (set in launch_air2.sh — 40s = ~1500 steps at env control rate, not enough; 80s gives ~3000)
**max_steps:** 2000 (per episode hard cap)

**Obs construction (43-D):**
`obs_policy` from the env is 42-D at runtime:
```
joint_pos(9) + joint_vel(9) + object_position(3) + target_object_position(7)
+ last_action(7) + eef_pos(3) + eef_quat(4) = 42-D
```
`target_object_position` (dims 21-27) is **patched every step** with live brush world pos + identity quat, because `generated_commands(object_pose)` returns a stale default at eval time (reset event is disabled in Play env). Without this patch, the policy is OOD on those 7 dims for all brush slots except the one that happens to match the default — causing near-zero arm output.
```python
obs_policy[:, 21:24] = brush.data.root_pos_w        # live world pos
obs_policy[:, 24:28] = id_quat                       # [1,0,0,0]
obs_input = torch.cat([obs_policy, phase_bit], dim=-1)  # (N, 43)
```

**Phase state machine:**

| Phase | Name | Arm | Gripper | Transition |
|-------|------|-----|---------|------------|
| 0 | APPROACH | BC output, clamp ±0.15 | Open (+1) | `near_counter >= 250` steps within 0.08m of brush |
| 1 | GRIP | Frozen (zeros) | Closed (-1) | `grip_steps >= 50` |
| 2 | CARRY | BC output, clamp ±0.15 | Closed (-1) | `carry_steps >= 200` AND XY dist to basket < 0.35m AND obj Z ≤ 1.4 |
| 3 | RELEASE | BC output, clamp ±0.15 | Open (+1) | episode ends immediately |

`phase_bit = (phase >= 2).float()` — fed as the 43rd dim. Phase 0 and 1 both pass phase_bit=0 (approach mode). Phase 2 and 3 pass phase_bit=1 (carry mode).

**Key constants:**
```python
NEAR_THRESH      = 250    # steps within 0.08m before grip (5s @ ~50Hz)
GRIP_HOLD        = 50     # steps arm frozen while closing (1s)
MIN_CARRY_STEPS  = 200    # cooldown before release check (~5s)
BASKET_XY_RADIUS = 0.35   # m — XY cylinder around basket [-3.941, -5.785]
# Z release threshold: obj_local[:, 2] <= 1.4  (basket at Z=1.140)
BASKET_POS_LOCAL = [-3.941, -5.785, 1.140]
BASKET_REACH_RADIUS = 0.40  # success metric threshold
```

**Episode end conditions:** `terminated | truncated | (ep_step >= max_steps) | (phase == 3)`

**Run command:**
```bash
./launch_air2.sh eval-state-bc checkpoints/policy_state_bc_mimic_v2.pth 20 2000
```

---

---

## 2026-06-02 — BC-TRIM1: trim dwell steps from training data

**Code:** BC-TRIM1
**Who:** Steph
**Changed:** `scripts/train_state_bc_from_hdf5.py`

**Problem:** BC-DW1 (weight=0.1 on 1.2% dwell steps) had no visible effect — too gentle. The dwell phase (first ~5 steps of each demo where arm action < 0.01) trains the policy to map default-reset-pose → near-zero action. At eval, the robot starts at default pose, so the policy outputs near-zero for all slots. R1 happens to escape because the tiny drift aligns with its approach direction by geometry. R0 and R2 drift away.

**Fix:** Trim dwell steps entirely from each demo before training. Find first step where `max|arm_action[:6]| > 0.01` and discard everything before it. Policy never sees "default pose → near-zero" and instead learns "first obs = early approach → output approach direction." At eval step 0, the default-pose obs now maps to the early-approach training distribution → policy immediately outputs correct direction for all slots.

**Pre-patch state (revert to):**
- `DWELL_ARM_THRESH = 0.01`, `DWELL_LOSS_WEIGHT = 0.1` (BC-DW1 weighting, no trimming)
- `load_dataset` computes per-sample weights and returns `(obs_all, act_all, weight_all)`

**Post-patch state:**
- `DWELL_TRIM = True` added — when True, slices each demo from first active step; weights become all 1.0
- Same `(obs_all, act_all, weight_all)` return signature preserved

**Checkpoint produced:** `checkpoints/policy_state_bc_mimic.pth`

**Run command:**
```bash
./launch_air2.sh train-state-bc datasets/air2_mimic_generated.hdf5 checkpoints/policy_state_bc_mimic.pth 300
```

**Status:** training pending

---

### Known issue — other 2 brush slot locations

The arm barely moves (raw_arm~0.001-0.005) for non-rightmost slots even with the `target_object_position` patch. Root cause suspected to be covariate shift: training data may be skewed toward the rightmost slot, so the policy hasn't learned to approach from the other starting configurations. Next step: verify training data slot distribution and either collect more diverse demos or add per-slot conditioning.

---

## 2026-06-02 — BC-DW1: dwell-step loss down-weighting

**Code:** BC-DW1
**Who:** Steph
**Changed:** `scripts/train_state_bc_from_hdf5.py`

**Problem:** Training data is balanced ~33% per slot (confirmed), but 2 of 3 brush slots fail at eval. Root cause: every Mimic demo begins with ~5 dwell steps where arm actions are near-zero. The policy learns "at step 0, output near-zero" for all slots. At eval, the tiny initial output happens to point toward the rightmost slot but away from the other two — a geometry accident of the robot's default joint pose. Non-working slots never bootstrap into the self-reinforcing approach loop.

**Fix:** Down-weight dwell steps in the training loss instead of removing them. Steps where `max|arm_action[:6]| < 0.01` (near-zero arm) get loss weight **0.1**. Active approach/carry steps get weight **1.0**. The policy still sees dwell context but the approach-direction signal dominates by 10x.

**Why this weight:** Gentle enough that the policy still learns the dwell (preserving the "build-up" behaviour seen in the working slot), but strong enough that BC learns correct approach direction from step 5+ data even when step 0 obs looks like dwell.

**Pre-patch state (revert to):**
- `load_dataset` returned `(obs_all, act_all)` — no weights
- `train(obs_np, act_np)` — uniform smooth-L1 loss, no weighting
- `if __name__ == "__main__"`: `obs_np, act_np = load_dataset(...)`

**Post-patch state:**
- `load_dataset` returns `(obs_all, act_all, weight_all)` — weight_all shape (N,), values 0.1 or 1.0
- `train(obs_np, act_np, weight_np)` — weighted smooth-L1 on training steps; val loss remains unweighted
- `if __name__ == "__main__"`: `obs_np, act_np, weight_np = load_dataset(...)`

**Checkpoint produced:** `checkpoints/policy_state_bc_mimic_v2_dw1.pth`

**Run command:**
```bash
./launch_air2.sh train-state-bc datasets/air2_mimic_generated.hdf5 checkpoints/policy_state_bc_mimic_v2_dw1.pth 300
```

**Eval command:**
```bash
./launch_air2.sh eval-state-bc brush checkpoints/policy_state_bc_mimic_v2_dw1.pth 20 2000
```

**Status:** no visible effect — raw_arm still 0.001-0.003 after retrain. Weight 0.1 on 1.2% of steps too gentle; dwell signal barely changed. Superseded by BC-TRIM1.
