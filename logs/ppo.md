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

### Design decisions still open
- Per-object tasks (4 separate envs) vs single env with target command?
- Does each episode spawn only the target object or all 4?
- PPO from scratch vs diffusion warm start vs diffusion → PPO fine-tune?
- Time penalty: decided against raw time penalty. Progress stall penalty preferred — penalises steps where scene state doesn't advance, not just elapsed time.

### Dependencies
- CNN `position_world` fix required before CNN confirmation terms can be implemented (see logs/cnn.md)
- Reward redesign blocked until env architecture decisions are made

**Status:** design complete, not implemented
