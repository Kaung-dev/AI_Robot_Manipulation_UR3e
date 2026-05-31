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

## 2026-05-31 — Per-object PPO runner config
**Who:** Steph
**Changed:** `agents/rsl_rl_ppo_cfg.py` added `AIR2RobotisPPORunnerCfg`; all per-object task registrations updated to use it
**Reason:** Per-object tasks previously fell back to parent air2_franka runner — PPO logs went to wrong directory
**Status:** working
