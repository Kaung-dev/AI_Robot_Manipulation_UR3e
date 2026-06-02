# Sequential Multi-Tool Eval Log

Component: Sequential 4-policy eval with CNN object detection
Goal: One Isaac Sim episode clears all tools off the pegboard into the basket in order,
using per-tool BC policies and CNN-based object localisation instead of GT physics positions.

---

## Status (2026-06-03)

### What's built and working
- `eval_sequential.py` — full sequential eval script, runs 3 tools (brush, pliers, screwdriver; scissors dropped)
- `eval_state_bc_cnn.py` — single-tool eval with CNN detections printed (test bed, not integrated into sequential yet)
- `launch_air2.sh eval-sequential` — wired up
- `go_home()` — implemented in `eval_state_bc.py` using `set_joint_position_target`, works reliably
- Graceful checkpoint skip — missing checkpoints print a warning and are skipped, no crash
- GT physics fallback — when CNN doesn't detect a tool, falls back to `tool.data.root_pos_w` automatically
- `--no_cnn` flag — skips board_camera + seg model entirely, uses pure GT physics; avoids VRAM OOM on RTX 3050 4GB

### What's NOT working yet
- CNN detection in the new scene — model was trained on old env (no Robotis pegboard). Needs recollection of segmentation data in new scene + retrain. Until then, always use `--no_cnn`.
- Scissors — no dataset, no checkpoint, dropped from pipeline entirely for now
- Screwdriver R0 — carry drops object (peg detachment issue). R0 merged dataset generated, needs retraining.
- Screwdriver R2 — carry reaches basket but MIN_CARRY_STEPS=200 blocks release (38 steps short at step 600)

### Working policies
| Tool        | R0  | R1  | R2  | Checkpoint |
|-------------|-----|-----|-----|------------|
| Brush       | ✅  | ✅  | ✅  | `checkpoints/working/R0_R1_R2_ALL_WORKS_FOR_BRUSH/policy_state_bc_mimic_v2.pth` |
| Pliers      | ✅  | ⛔  | ✅  | `checkpoints/working/PLIERS_R0_R2_WORKS/PLIERS_R0_R2_WORKS/policy_state_bc_mimic_pliers_v2.pth` |
| Screwdriver | ⚠️  | ✅  | ⚠️  | `checkpoints/policy_state_bc_screwdriver.pth` (needs retrain on merged dataset) |
| Scissors    | —   | —   | —   | no checkpoint, not in pipeline |

⛔ = excluded (unreliable grasp)  ⚠️ = policy runs but drops object

---

## Pipeline (implemented)

### Episode start
`env.reset()` — all tools spawn on pegs at random slots, robot at home.

### Per-tool loop (interactive)
After each env.reset(), the script enters a while loop over `TOOL_ORDER = ["brush", "pliers", "scissors", "screwdriver"]`.

Each iteration:
1. CNN scan (or empty dict if `--no_cnn`)
2. Print detected tools + remaining tools
3. Prompt user: type tool name, `auto`, `skip <tool>`, or `done`
4. If tool not in CNN detections → auto fallback to GT physics position
5. Run BC policy loop (phase state machine, same as eval_state_bc.py)
6. `go_home()` after tool is done (LANDED or timeout)
7. Back to CNN scan

### go_home()
Calls `robot.set_joint_position_target(HOME_JOINTS)` + `sim.step()` × up to 200 steps.
Stops early when `max|joint_pos - home| < 0.05 rad`. ~50-100 steps in practice.
No `env.reset()` — tools stay exactly where they are.

### Episode end
After all tools attempted or user types `done`, log per-tool results and `env.reset()`.

---

## Key files

| File | Purpose |
|------|---------|
| `scripts/eval_sequential.py` | Full sequential eval (active) |
| `scripts/eval_state_bc.py` | Single-tool eval — DO NOT MODIFY (working reference) |
| `scripts/eval_state_bc_cnn.py` | Single-tool eval with CNN prints (test bed) |
| `scripts/merge_generated_r0.py` | Strip R0 demos from generated HDF5, stitch in new ones |
| `scripts/collect_mimic_demos.py` | Patched Mimic collection — randomization re-enable block CURRENTLY COMMENTED OUT for R0 collection. REVERT before collecting multi-slot demos. |
| `isaaclab_ext/tasks/air2_robotis_franka/joint_pos_env_cfg.py` | `_strip_debug_markers()` added + called in all PLAY cfgs — fixes FrameTransformer debug_vis crash at eval startup |

---

## CNN position integration detail

CNN gives `position_world` (3D, world frame). Converted to robot-root frame for obs:
```python
from isaaclab.utils.math import subtract_frame_transforms
obj_root, _ = subtract_frame_transforms(
    robot.data.root_pos_w, robot.data.root_quat_w,
    torch.tensor(cnn_position_world, device=device).unsqueeze(0)
)
```
This overrides `obs[18:21]` (object_position) so the policy gets the correct tool position regardless of which tool is the "object" in the env.

GT physics used for: release trigger (`xy_dist_to_basket`), success check (`landed`).
CNN used for: `obs[18:21]`, SERVO approach, `ee_obj_dist`.

---

## Screwdriver dataset history (2026-06-03)

| Dataset | Demos | R0 | R1 | R2 | Notes |
|---------|-------|----|----|----|-------|
| `air2_mimic_screwdriver_annotated.hdf5` | 67 source | 21 | 23 | 23 | good annotated source |
| `air2_mimic_generated_screwdriver.hdf5` | 833 | 292 | 270 | 271 | original multi-slot gen (R0 carry broken) |
| `air2_mimic_screwdriver_r0_raw.hdf5` | 30 | 30 | 0 | 0 | new R0 source demos with upward lift |
| `air2_mimic_screwdriver_r0_annotated.hdf5` | ~28 | 28 | 0 | 0 | annotated new R0 |
| `air2_mimic_generated_screwdriver_r0.hdf5` | 303 | 303 | 0 | 0 | generated from new R0 source |
| `air2_mimic_generated_screwdriver_merged.hdf5` | 844 | 303 | 270 | 271 | **use this for retraining** — replaces bad R0 demos |

Root cause of R0 carry failure: Mimic nearest-neighbor carry selection produced trajectories without the upward peg-detachment lift that R1 learned naturally. New R0 source demos emphasise the upward lift.

R2 carry issue: `MIN_CARRY_STEPS=200` prevented release when policy was already over the basket at carry=162. Either lower threshold or collect more R2 carry data.

---

## Pliers notes (from PLIERS_R0_R2_WORKS README)

- R1 excluded — unreliable grasp (pliers sits at awkward angle, slides out during extraction)
- Data cleaning was critical: raw gen was only 72% clean (vs brush 97%). Filtered to 713 clean demos with `filter_clean_demos.py` before retraining.
- Same `filter_clean_demos.py` should be applied to screwdriver merged dataset before retraining.

---

## Pending

- [ ] Revert `collect_mimic_demos.py` randomization re-enable (currently commented out)
- [ ] Apply `filter_clean_demos.py` to `air2_mimic_generated_screwdriver_merged.hdf5`
- [ ] Retrain screwdriver on filtered merged dataset
- [ ] Recollect segmentation data in new scene (with Robotis pegboard) — ~750 frames
- [ ] Retrain U-Net on new scene data
- [ ] Test CNN detection in new scene, switch from `--no_cnn` to live CNN
- [ ] Investigate scissors viability (no dataset collected yet)
- [ ] Fix screwdriver R2: lower MIN_CARRY_STEPS or collect better R2 carry data

## Launch

```bash
./launch_air2.sh eval-sequential 5 --no_cnn
```
Extra args after episode count are passed through (e.g. `--screwdriver_ckpt <path>`).
