# PPO Log

Component: Reinforcement learning / reward design
Env: Isaac-AIR2-Robotis-Franka-*
Pipeline: (optional diffusion warm start) → PPO fine-tuning

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

## 2026-05-29 — Reward design review
**Who:** Steph
**Changed:** nothing yet — design phase
**Tried:** Reviewed existing rewards.py against task requirements
**Result:** Current rewards are barebones. Known issues:

- `reaching_object` still active — targets only 1 of 4 objects, adds noise. Should be disabled.
- `HOOK_LINE_Y = -5.9` hardcoded — may be wrong after AIR2.usd updated (42KB → 65KB). Needs verification against current scene.
- No success termination — episode runs to time limit even after all objects in basket.
- `BASKET_POS_LOCAL` duplicated in 5 files — maintenance risk.

**Status:** not implemented, pending architecture decisions

---

## 2026-05-29 — Proposed reward design
**Who:** Steph

Full term list agreed on. Not yet implemented.
Pending decisions: per-object tasks vs single task with target command, PPO from scratch vs diffusion warm start.

| Term | Type | CNN role |
|---|---|---|
| `ee_to_target` | dense positive | CNN confirms which object is target |
| `target_off_hook` | binary positive | CNN confirms target no longer at slot |
| `target_in_hand` | dense positive | CNN confirms target in gripper frame |
| `target_to_basket` | dense positive | — |
| `target_in_basket` | binary positive (high) | CNN confirms target visible in basket |
| `wrong_object_moved` | penalty | CNN confirms non-target moved |
| `object_slipped` | penalty | CNN detects target reappears at hook |
| `grasp_lost` | penalty | CNN detects target leaves gripper frame |
| `progress_stall` | penalty | CNN confirms scene state unchanged |

Weight order: target_in_basket (highest) → target_off_hook / target_in_hand → ee_to_target / target_to_basket → penalties last.

Note: CNN confirmation on rewards requires position_world fix in cnn.md first.

---

## 2026-05-29 — Reward design finalised (not yet implemented)
**Who:** Steph

Full term list agreed on after discussion. Replacing current barebones rewards entirely.

### Current rewards — problems
- `reaching_object` still active, targets only 1 of 4 objects, adds noise — should be disabled
- `HOOK_LINE_Y = -5.9` may be wrong after AIR2.usd update — off_hook reward could misfire
- No success termination — episode wastes steps after all objects in basket
- `BASKET_POS_LOCAL` duplicated in 5 files
- No grasp confirmation — arm can knock objects off hook and still get off_hook reward
- No carry signal when EE has object — ee_to_object goes to 0 after pick, nothing guides carry

### New reward design

| Term | Type | CNN role |
|---|---|---|
| `ee_to_target` | dense positive | CNN confirms which object is target |
| `target_off_hook` | binary positive | CNN confirms target no longer at slot |
| `target_in_hand` | dense positive | CNN confirms target in gripper frame |
| `target_to_basket` | dense positive | — |
| `target_in_basket` | binary positive (high) | CNN confirms target visible in basket |
| `wrong_object_moved` | penalty | CNN confirms non-target moved |
| `object_slipped` | penalty | CNN detects target reappears at hook |
| `grasp_lost` | penalty | CNN detects target leaves gripper frame |
| `progress_stall` | penalty | scene state unchanged — no task progress |

Weight order: target_in_basket (highest) → target_off_hook / target_in_hand → ee_to_target / target_to_basket → penalties (tune last).

### Design decisions
- **Per-object tasks** — 4 separate envs (one per object), each episode spawns all 4 objects on slots. No target command input needed.
- **Sequence plan** — later, object gathering will run tasks in sequence (e.g. 4→2→3→1 order), but each individual task is still single-object.
- PPO from scratch vs diffusion warm start — still open.
- Time penalty: decided against raw time penalty. Progress stall penalty preferred.

### Dependencies
- CNN `position_world` fix required before CNN confirmation terms can be implemented (see logs/cnn.md)
- Reward redesign blocked until env architecture decisions are made

**Status:** design complete, not implemented

---

## 2026-05-30 — Rewards implemented
**Who:** Steph
**Changed:** rewards.py — added `object_slipped` and `grasp_lost` penalty terms. Both edge-triggered (fire once per event). Wired up all terms in `_apply_target_rewards()` in air2_robotis_franka/joint_pos_env_cfg.py.

### Final implemented weights
| Term | Weight | Notes |
|---|---|---|
| `ee_to_target` | +2.0 | Gaussian, std=0.5 |
| `target_off_slot` | +5.0 | Binary — Y > SLOT_LINE_Y + clearance |
| `target_in_hand` | +3.0 | Physics: dist < 0.08m AND finger_sum < 0.04 |
| `target_to_basket` | +1.0 | Gaussian, std=0.5 |
| `target_in_basket` | +20.0 | Binary — dist < 0.30m from basket |
| `wrong_object_moved` | -5.0 | Count of displaced non-target objects |
| `object_slipped` | -3.0 | Edge: target returned to slot after being off |
| `grasp_lost` | -3.0 | Edge: dropped while carrying (not in basket) |
| `progress_stall` | -0.5 | Per step where min(ee→target, target→basket) didn't improve |
| `task_success` | termination | dist < 0.30m from basket |

CNN confirmation skipped — physics-based approximations used throughout. Weights are initial estimates, expect tuning after first PPO runs.

**Status:** implemented, not yet tested

---

## 2026-05-31 — PPO deprioritised in favour of Mimic → state-BC
**Who:** Steph
**Decision:** Not running PPO for now. Switched to Mimic data augmentation → state-BC policy.
**Root cause of PPO failure:** After 3000 iterations, `ee_to_target ≈ 1.0` (reaches object) but never carries — no reward gradient between grasp and basket. Missing `lift_progress` reward.
**Fix available but not applied:** experimental branch has `grasp_shaping + lift_progress + progress_stall=-0.02` which should fix the carry problem.
**Plan:** If state-BC also fails, apply the PPO reward fix from experimental and run PPO.
**Status:** on hold
