# Environment Log

Component: Isaac Sim scene + Isaac Lab task configs
Active env: Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0
Scene: AIR2.usd + robotis_net_table (slot-based object placement)
Robot: Franka Panda

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

## 2026-05-29 — Task rename
**Who:** Steph
**Changed:** Renamed all 3 task directories, class names, task IDs, all script references, setup_isaaclab.sh
- lift_air2_ur3e_rg2 → air2_franka (Isaac-Lift-AIR2-UR3e-RG2-* → Isaac-AIR2-Franka-*)
- lift_air2_robotis → air2_robotis_franka (Isaac-Lift-AIR2-Robotis-* → Isaac-AIR2-Robotis-Franka-*)
- lift_pegboard_franka → pegboard_franka (Isaac-Lift-Pegboard-Franka-* → Isaac-Pegboard-Franka-*)
**Result:** All tasks register and launch correctly
**Status:** working

---

## 2026-05-29 — Object placement (AIR2 scene)
**Who:** Steph
**Tried:** Launching Isaac-AIR2-Franka-Play-v0 — objects fall off hooks immediately
**Result:** HOOK_POSITIONS in air2_franka/mdp/events.py are calibrated for old AIR2.usd. Scene was updated (42KB → 65KB) in latest pull but positions were not updated.
**Status:** broken — use Robotis env instead (slot-based, stable)
**Fix:** Pending — teammate needs to re-run click_peg_coords.py against current AIR2.usd and update HOOK_POSITIONS

---

## 2026-05-29 — Robotis env verified
**Who:** Steph
**Tried:** ./launch_air2.sh collect-seg 50
**Result:** 50 segmentation frames collected cleanly. Scene looks correct — robot, objects on slots, basket visible.
**Status:** working

---

## 2026-05-29 — PYTHONPATH fix
**Who:** Steph
**Changed:** .env — added PYTHONPATH=/home/steph/ai_in_robotics_project/ur_pick
**Tried:** ./launch_air2.sh teleop — failed with ModuleNotFoundError: No module named 'isaaclab_ext'
**Result:** Fixed. teleop, collect-demos, all modes now work.
**Status:** working
**Fix:** teleop_se3_agent.py is upstream Isaac Lab and doesn't add repo root to sys.path. PYTHONPATH in .env resolves it. Must `source .env` before any launch.

---

## 2026-05-29 — teleop --enable_cameras fix
**Who:** Steph
**Changed:** launch_air2.sh teleop mode — added --enable_cameras flag
**Tried:** ./launch_air2.sh teleop — crashed: "A camera was spawned without the --enable_cameras flag"
**Result:** Fixed. Robotis Play task has cameras in config.
**Status:** working

---

## 2026-05-29 — Full teleop verified
**Who:** Steph
**Tried:** ./launch_air2.sh teleop
**Result:** Scene correct — objects on slots, robot in position, arm moves with keyboard, gripper opens/closes. Ready for demo collection.
**Status:** working

---

## 2026-06-01 — SM_BoxPortableD adjusted
**Who:** Declan
**Changed:** scene/AIR2.usd — raised SM_BoxPortableD height and moved it in front of the robot
**Result:** Basket now positioned correctly for drop-in task
**Status:** working

---

## 2026-05-31 — Right-side slot layout locked (permanent)
**Who:** Steph
**Changed:** `isaaclab_ext/tasks/air2_robotis_franka/mdp/events.py` — `reset_objects_on_slots()`
**Decision:** Left-side slots (L0–L3) abandoned entirely. All 4 objects now placed on right-side slots only (R0–R3).
**Reason:** Left slots add unnecessary diversity during Mimic data collection and reduce teleop comfort. Right side is always within comfortable reach.
**Forbidden placements (from reachability testing):**
- Brush (col 0): cannot reach R3 → swapped with pliers if drawn
- Screwdriver (col 3): cannot reach R3 → swapped with scissors if drawn
- R3 always ends up as pliers or scissors
**Status:** working — permanent, no TEMPORARY marker

---

## 2026-05-31 — Mimic env registered
**Who:** Steph
**Changed:** `mimic_env.py`, `mimic_env_cfg.py`, `__init__.py` (air2_robotis_franka)
**Added:** `Isaac-AIR2-Robotis-Franka-Brush-Mimic-v0`
- Extra obs: `eef_pos` (3-D), `eef_quat` (4-D) for Mimic IK stitching
- `subtask_terms` obs group: `grasp_brush` (proximity + finger closure)
- Two SubTaskConfigs under EEF name `"franka"`: approach+grasp → carry to basket
- `grasp_radius=0.15, finger_threshold=0.07` — brush ring prevents full finger closure
- Cameras stripped (syntheticdata plugin crash on this GPU setup)
- `generation_guarantee=True`
**Status:** working

---

## 2026-06-01 — Switched to Main+Experimental_merge branch; debug prints added
**Who:** Steph
**Reason:** New env has updated basket position (SM_BoxPortableD moved — to the left and closer to robot by Declan) and `panda_joint4 = -2.26892803` restored. Collecting 100 new source demos for Mimic on this env.
**Warning:** `BASKET_POS_LOCAL` in `constants.py`, `eval_state_bc.py`, `eval_ppo.py`, `scripted_controller.py` still hardcoded to old value `[-3.560, -5.370, 1.040]` — needs updating once real position is confirmed from sim.
**Added to `collect_mimic_demos.py` and `collect_air2_manual_demos.py`:**
- On first step: prints basket prim world + env-local coords → use to update `BASKET_POS_LOCAL`
- On every gripper open/close: prints EE + object position in env-local frame → confirms grasp and drop positions
**Dataset cleanup:** old Mimic files (raw 40, annotated, generated) moved to `datasets/old_env/` — wrong env, do not use. New collection goes to `datasets/air2_mimic_demos_v2.hdf5`.
**collect-mimic command:** now accepts output path as 3rd arg: `./launch_air2.sh collect-mimic 100 datasets/air2_mimic_demos_v2.hdf5`
**Status:** ready to collect — run command above, note `[basket]` line on first step, update constants

---

## 2026-06-02 — 3-basket bug identified; collect_mimic_demos.py reworked
**Who:** Steph
**Found:** 3 SM_BoxPortableD prims spawning simultaneously in every env instance.
**Root cause:** `AIR2.usd` references `./AIR.usd` as a payload — `AIR.usd` has the basket at the
original position. `AIR2.usd` had basket position adjusted twice (commits `e6753c2`, `c71cb93`),
each adding a new prim rather than overriding. Results in 3 baskets: inside robot, above correct
spot, correct spot.
**Fix needed:** Open `scene/AIR2.usd` in Isaac Sim Stage panel → find 3 `SM_BoxPortableD` prims
→ delete 2 wrong ones → save. NOT YET DONE.
**Workaround:** Human operator aims for visually correct basket during teleop — collection
proceeds despite bug.
**Changed:** `collect_mimic_demos.py`:
- Manual ENTER to accept demo / R to discard (replaced auto-save on success_term)
- `add_episode()` removed (not needed for 1-env source collection)
- Gripper open/close EE+obj positions logged to `_gripper_log.txt` on accept only
- `debug_vis=True` on `ee_frame` — revert to False before training
**Status:** collecting — basket USD fix pending
**⚠️ IMPORTANT — EE FRAME DEBUG VIS:**
`debug_vis=True` is currently set in `isaaclab_ext/tasks/air2_franka/joint_pos_env_cfg.py:192`.
The wrist camera WILL capture these arrows — any CNN or segmentation data collected with this
enabled will have RGB arrows visible in every frame, corrupting the training data.
REVERT TO `debug_vis=False` BEFORE: collecting segmentation data, collecting visual BC demos, or committing/pushing.
Leave enabled only for teleop inspection sessions.

---

## 2026-06-02 — Basket position fixed + success termination updated
**Who:** Steph
**Changed:**
- `constants.py` + 4 other locations — `BASKET_POS_LOCAL` updated from stale `[-3.560, -5.370, 1.040]` to `[-3.941, -5.785, 1.140]`. Derived from mean obj position at gripper OPEN across 80 source demos (gripper log). XY empirical, Z estimated (floor ~1.04, release ~1.30).
- `terminations.py` — `target_reached_basket` replaced 3D sphere (radius=0.30) with XY box check (±0.3 in X, ±0.5 in Y) + Z≤1.2. Fires when brush hits the basket footprint and has descended to basket height.
**Status:** working — annotation confirmed 80/80 demos accepted

---

## 2026-05-31 — Per-object PPO runner config
**Who:** Steph
**Changed:** `agents/rsl_rl_ppo_cfg.py` added `AIR2RobotisPPORunnerCfg`; all per-object task registrations updated to use it
**Reason:** Per-object tasks previously fell back to parent air2_franka runner — PPO logs went to wrong directory
**Status:** working
