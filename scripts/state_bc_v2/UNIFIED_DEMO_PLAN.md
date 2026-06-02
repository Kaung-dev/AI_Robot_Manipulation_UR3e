# State-BC pick-and-place — context, current state, and the unified-demo plan

_Last updated: 2026-06-02. This chat = **BC eval + unified demo**. A separate chat is
doing **PPO** (GPUs 1/2/3). Keep the two efforts distinct — this doc is the BC side._

---

## 1. Goal
A single **unified pick-and-place demo**: all tools are on the pegboard, and the
robot picks each one and drops it in the basket, in sequence ("pick-place,
pick-place, …"), ideally **perception-driven** (CNN finds the tools), looping.

## 2. What works today (per-object BC, packaged)
Each object has its own state-BC policy (`checkpoints/policy_state_bc_mimic_<obj>_v2.pth`),
trained + evaluated with the v2 scripts in `scripts/state_bc_v2/`. Slots are R0/R1/R2
(right panel; R3 is unreachable for brush & screwdriver by design).

| Object | Working slots | Deliverable (folder + .zip in repo root) |
|---|---|---|
| **Brush** | R0, R1, R2 (all) | `R0_R1_R2_ALL_WORKS_FOR_BRUSH` |
| **Pliers** | R0, R2 (R1 excluded) | `PLIERS_R0_R2_WORKS` |
| **Screwdriver** | R1, R2 (R0 excluded) | `SCREWDRIVER_R1_R2_WORKS` |
| **Scissors** | not done | — |

Each tool reliably does ~2–3 slots; the "bad" slot differs per tool (an awkward
grasp angle). Brush (ring) is the easy one and does all 3.

## 3. The v2 recipe (how each object was built)
1. **Generate** mimic data with multi-slot diversity — REQUIRES env var:
   `MIMIC_KEEP_RANDOMIZATION=1 ./launch_air2.sh generate-mimic <obj> <annotated_in> <out> <N>`
   (without it, every demo spawns on ONE slot — the original single-slot bug).
2. **Filter to clean demos** (object actually reached basket): `filter_clean_demos.py`.
   Raw gen quality varied (brush 97%, pliers 72%, screwdriver 90%); training on the
   junk made the policy freeze / fly-away. Cleaning was the key fix.
3. **Train**: `train_state_bc_from_hdf5_v2.py` — BC-DW1 (dwell down-weight) +
   **obs-alignment** (dims 21:28 = object pos + identity quat, not the noisy command)
   + **obs normalization** (mean/std saved in ckpt). 300 epochs, val ≈ 0.0008.
4. **Eval**: `eval_state_bc_v2.py` (GUI on GPU 0 = display; headless on a spare GPU).
   Key eval mechanics:
   - **servo** final approach (drive EE to true object pose; fixes under/over-shoot),
   - **release-and-settle** + **disable env early-terminations** (let the object physically land),
   - **LANDED** = released AND settled in basket (not min-dist while gripped),
   - **strip cosmetic markers** (brush-specific debug spheres crash non-brush scenes),
   - **`--exclude_slots R1`** to skip a tool's bad slot.

## 4. HONEST framing (important)
The eval is a **hybrid**, not pure BC: the BC does the **coarse reach + carry**;
**scripted helpers** do the precise grasp positioning (servo, using the true object
pose), the gripper open/close, and the release timing. With `--servo_dist 0` (servo
off) the pure-BC behavior is much weaker. State this in any writeup. The unified
demo below leans even more on scripting unless the CNN provides perception.

## 5. The unified demo — design (CNN-driven, the target)
Per-episode loop:
1. **reset** — all 4 tools spawn on pegs (`Isaac-AIR2-Robotis-Franka-Play-v0` has all 4
   real tools), robot home, scores zeroed.
2. **CNN scan** — run U-Net inference on `board_camera`; for each tool still on the
   pegs, store its world position. Nothing detected ⇒ episode done.
3. **Sequence** — pick next tool (fixed priority brush→pliers→scissors→screwdriver,
   skip undetected); load that tool's BC ckpt; reset the phase state machine.
4. **Policy loop** — step sim; CNN re-runs every N steps and caches the current tool's
   position; that cached pos **replaces `obj.data.root_pos_w`** in `ee_obj_dist`, the
   obs patch, the servo, and `xy_dist_to_basket`. Phase logic + gripper forcing as in
   the current eval. Exit on LANDED or timeout.
   - NOTE: once grasped, the tool leaves the pegboard view → during CARRY/RELEASE fall
     back to **EE-based** logic (CNN can't track the held/occluded tool).
5. **go_home** — drive robot to home joints (no env reset; placed tools stay in basket,
   remaining stay on pegs). Important: each per-object model expects a home-ish start.
6. **re-scan** → pick next → repeat until pegs empty / all attempted.
7. **episode end** — log per-tool results, then `env.reset()` for a new episode.

### Fallback design (if cameras/CNN don't pan out)
Same loop but **ground-truth object poses** instead of CNN (works today, no cameras).
Less "real" (privileged poses) but a working multi-tool demo. Place each tool on a
slot it can do (brush R0 / screwdriver R1 / pliers R2) so no bad-slot grasps.

## 6. Prerequisites status (for the CNN path)
- ✅ **CNN checkpoint**: have it — `checkpoints/air2_segmentation_unet.pth` (U-Net,
  base_channels=32, 9 classes incl. brush/pliers/scissors/screwdriver/basket, 0.919 mIoU).
  `cnn/postprocess.py` already computes `position_world` from camera extrinsics.
- ⚠️ **Cameras on the RTX 5090**: the T4 `omni.syntheticdata` crash did **NOT** happen
  here (good). But the one test so far was **inconclusive** — run under
  `CUDA_VISIBLE_DEVICES=1`, which broke camera *rendering* ("CUDA in bad state",
  no detections printed). **TODO: re-run the seg inference on a FREE, unrestricted
  render GPU** to confirm cameras render + CNN detects the tools + check position
  accuracy (cm-level CNN vs the mm-level the grasp needs — likely need CNN-for-find +
  a tighter final approach).
  Test command (when a GPU is free):
  ```
  "$ISAACLAB_PATH/isaaclab.sh" -p scripts/run_air2_segmentation_inference.py \
      --checkpoint checkpoints/air2_segmentation_unet.pth \
      --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
      --camera board_camera --enable_cameras --headless
  ```

## 7. Open issues / risks
- **CNN position precision**: grasp needs ~mm; CNN gives ~cm. CNN great for
  detect/sequence/coarse-approach; final grasp may need refinement.
- **Carry tracking**: CNN loses the held tool → EE-based carry/release.
- **GPU contention**: GPUs 1/2/3 are the other chat's PPO; GPU 0 is the display.
  The camera test needs a free GPU — paused until one frees up. Do NOT `pkill`
  (it hits other sessions); stop only your own tasks.

## 8. Parallel work / incoming
- **Friend is re-collecting pliers source demos (slot R0 per their note).** When the
  new `air2_mimic_generated_pliers*.hdf5` lands → re-run filter→train→eval for pliers
  and re-check whether the previously-excluded slot now works (may drop `--exclude_slots`).
- Other chat: PPO on brush/pliers/screwdriver (GPUs 1/2/3) — separate from this BC work.

## 9. Next steps (this chat)
1. Wait for a free GPU → re-run the camera/CNN test (Section 6).
2. If cameras+CNN good → build the unified CNN loop (Section 5); else build the GT fallback.
3. Retrain pliers when the friend's re-collected data arrives.
4. (Optional) scissors, same recipe.
