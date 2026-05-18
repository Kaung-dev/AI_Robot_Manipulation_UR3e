# CNN Perception + Learning Pipeline

This document describes the planned architecture for the pegboard pick-and-place
project. It is intentionally separated into **perception** (a CNN) and **control**
(an imitation-learned policy, optionally fine-tuned with PPO). The goal is to
have three independently verifiable modules so each can be evaluated and
debugged on its own.

> **Status**: design doc. The Franka pegboard env is built ([lift_pegboard_franka/](isaaclab_ext/tasks/lift_pegboard_franka/)) and the visuomotor variant with cameras is registered. The detector, BC training loop, and PPO fine-tune step are not yet implemented — this file is the agreed plan before we start coding.

## TL;DR

```
   Camera frame
        │
        ▼
   CNN detector   ─── outputs: class + bbox + (x, y, z) target
        │
        ▼
   Imitation-learned policy   ─── trained from human teleop demos
        │
        ▼
   (Optional) PPO fine-tune   ─── refines the BC policy with sim reward
        │
        ▼
   IK-Rel action → Franka in Isaac Sim
```

| Module | Role | Trained how | Evaluated how |
|---|---|---|---|
| **CNN detector** | "What object is visible and where is it?" | Supervised, on synthetic images auto-labeled from Isaac Sim | mAP, precision/recall, confusion matrix |
| **BC policy** | "Given the object position, what action moves the EE toward it?" | Supervised, on human teleop demos | Rollout success rate |
| **PPO fine-tune** | "Improve the BC policy past human performance." | Reinforcement, using existing pegboard reward function | Δ success rate vs BC alone |

---

## Why this split?

**Do NOT** treat the CNN as the whole controller. Train it to do **perception
only** (object detection + localization). The control policy is a separate
network.

Benefits:

1. **Interpretability** — the detector output is human-readable (bounding boxes, class labels). You can visually verify perception works before any policy training.
2. **Cleaner story for the report** — three modules, three sets of metrics. "Perception works at X mAP; given correct perception, the policy reaches Y success rate."
3. **Smaller policy network** — the policy only consumes (x, y, z) coordinates, not raw pixels. Faster to train, fewer demos needed.
4. **Mirrors real robotics stacks** — perception → planning → control is the standard decomposition. Industry-grade pipelines look like this.
5. **Two debuggable failure modes** — if the robot fails, you can ask separately "did perception fail?" vs "did control fail?".

The wrong story to tell: *"a single CNN controls the UR3e/Franka end-to-end."* The right story: *"the CNN provides visual perception by detecting the object and estimating its position; the manipulation policy then uses this information to execute the pick-and-place task."*

---

## Module 1 — CNN object detector

### What it outputs

For each camera frame the detector produces, per detected object:

- **Class** — one of `{brush, toothbrush, silicone, scissors, pliers, screwdriver}` (or "empty slot" if we want that).
- **Bounding box** — `(x1, y1, x2, y2)` in image pixels.
- **Confidence** — `[0, 1]`.

We then convert the bbox center + depth into a 3D target point in robot base frame (see §"2D → 3D" below).

### Where the training data comes from

Synthetic, generated in Isaac Sim. No hand-labeling.

For each tool USD in [exported_assets/object/](exported_assets/object/):

1. Spawn it in a minimal scene with one camera looking at it.
2. Randomize: yaw, small xy translation, dome-light intensity/color, camera angle, optional background texture.
3. Render → save the RGB image.
4. Write the label automatically — we know the class (because we spawned that tool) and the bounding box (from the projected geometry or instance segmentation mask).

Target: ~500-2000 images per class. Realistic budget: minutes of sim time per class.

### Architecture options

Recommended for an assignment-grade implementation (in increasing complexity):

- **"Baby YOLO"** — a CNN with a single regression head outputting `(cx, cy, w, h, class_logits)` per cell of a small grid (e.g. 7×7). ~200 lines of PyTorch, clearly your own work.
- **Heatmap detector** — predict a per-pixel "where is the object" heatmap; take argmax for `(cx, cy)`. Simpler than YOLO; gives a nice visualization.
- **YOLOv8-nano fine-tuned** — strongest accuracy fastest, but uses `ultralytics` and most of the work isn't yours. Weaker story.

Pick the one that lets you say "I built this" in the writeup. **Build your own** (option 1 or 2). Aim for ~3 conv blocks + a small head; you don't need ResNet-scale depth for 84-px synthetic images.

### Training procedure

Standard supervised PyTorch loop:

- Train/val split: 80/20, stratified by class.
- Loss: cross-entropy for classification + smooth-L1 (or MSE) for bbox regression.
- Augmentation: random flip, 90° rotation, brightness/contrast jitter.
- Optimizer: `Adam(lr=1e-3)` or `AdamW(lr=1e-3, weight_decay=1e-4)`.
- Epochs: 20-50; early stop on val mAP.

### Metrics to report

- Per-class precision/recall and confusion matrix.
- Mean average precision (mAP).
- Sample predictions overlaid on test images (good visual for the writeup).

### 2D → 3D unprojection

The detector gives 2D pixel coordinates. The policy needs 3D world coordinates.

Two viable methods:

1. **Depth channel** (recommended). Our `CameraCfg` already supports `data_types=["rgb", "distance_to_image_plane"]`. Read depth at the bbox center, then unproject through the camera intrinsics:
   ```
   (cx, cy, depth) + intrinsics + camera pose → (X, Y, Z) in world
   ```
   Robust to objects at any height (pegboard, table, mid-air).

2. **Known z-plane**. Assume the object is at a fixed height (e.g. on the pegboard surface). Intersect the camera ray with that plane. Breaks once the robot lifts the object — not recommended.

> **Action item**: when implementing the visuomotor env, make sure `"distance_to_image_plane"` is in the camera `data_types`.

---

## Module 2 — Imitation learning (BC policy)

### What it does

Takes the perception output (object position + low-dim robot state) and outputs an IK-Rel action: `(Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)`.

### Where the demos come from

Human teleop, recorded with Isaac Lab's `record_demos.py`:

- **Keyboard teleop** — `teleop_se3_agent.py` with `--teleop_device keyboard`. Slow but works on any machine. Budget: ~50 demos in an evening.
- **VR teleop** — already wired up in our `ik_rel_env_cfg.py` via `OpenXRDevice`. Much faster and more natural. Budget: ~200 demos in an evening.

Each demo = one successful trajectory of "pick the toothbrush from L1, drop it in the basket."

### What's in the demo HDF5

```
obs/
  wrist_cam_rgb     (T, 84, 84, 3)     ← raw image
  wrist_cam_depth   (T, 84, 84)        ← for 2D→3D
  table_cam_rgb     (T, 84, 84, 3)
  table_cam_depth   (T, 84, 84)
  joint_pos         (T, 9)
  joint_vel         (T, 9)
  last_action       (T, 7)
actions             (T, 7)              ← what the human did
```

> **Important**: store the **raw images and depth**, not the CNN's processed output. This lets us iterate on the CNN architecture without re-recording demos. The CNN runs at training time, not at recording time.

### BC algorithm

**BC-RNN** (LSTM policy) — the standard for manipulation. Handles "the gripper has the object, now navigate to the basket" naturally via temporal context.

Alternatives: BC-MLP (simpler but underfits long-horizon tasks), BC-Transformer (heavier), Diffusion Policy (current SOTA, slow to train). Start with BC-RNN.

### Architecture

```
At every step:
   wrist_img + wrist_depth ──► CNN detector ──► (cls, x, y, z)
   table_img + table_depth ──► CNN detector ──► (cls, x, y, z)
   merge detections + low-dim state           ──► observation vector
                                                          │
                                                          ▼
                                                       LSTM
                                                          │
                                                          ▼
                                                   7-D IK-Rel action
```

The CNN is **frozen** during BC training. Only the LSTM and a small input MLP have gradients. If accuracy stalls after a long training run, optionally unfreeze the CNN for a fine-tune pass.

### Framework choice

Two options:

- **Robomimic** — Isaac Lab's standard BC framework, used by upstream stack-cube. Gives you HDF5 loading, batching, checkpointing for free. **But** swapping in our custom CNN requires registering it with robomimic's encoder system.
- **Custom PyTorch loop** — write `train_bc.py` (~150 lines). Loads the HDF5, batches `(images, low_dim, action)`, forward through the policy, MSE loss vs recorded action, backprop. You own every line.

For the assignment, **custom loop** wins. Robomimic hides the interesting parts.

### Metrics to report

- BC training loss curve (should drop smoothly).
- Rollout success rate over N evaluation episodes (e.g. 50).
- Time-to-grasp and drop-rate breakdowns.
- Comparison of "CNN-estimated object_position" vs "ground-truth object_position" — if BC trained on CNN output performs similarly to BC trained on ground truth, the CNN is good enough.

---

## Module 3 — PPO fine-tune (optional but strong)

### Why add PPO on top of BC

BC alone is bounded by the demonstrator's skill — typically 60-80% success rate. PPO can refine the BC policy past the human, using the existing reward signals in the env.

The handoff:

```
demos.hdf5  ──► train_bc.py  ──► policy_bc.pth   (success rate ~60-80%)
                                       │
                                       ▼
                              load as PPO actor init
                                       │
                                       ▼
              rsl_rl PPO training in pegboard env  ──► policy_final.pth   (target ~90-95%)
```

### Reward function

Already in our env via `LiftEnvCfg`:

| Term | Weight | Effect |
|---|---|---|
| `reaching_object` | 1.0 | EE distance to object |
| `lifting_object` | 15.0 | Object height above table |
| `object_goal_tracking` | 16.0 | Object distance to goal (basket) |
| `object_goal_tracking_fine_grained` | 5.0 | Same, with tighter Gaussian |
| `action_rate` | -0.0001 | Smoothness penalty |
| `joint_vel` | -0.0001 | Energy penalty |

These don't need to change for PPO fine-tune.

### What gets copied / reset

- **Actor weights**: load from BC checkpoint.
- **Critic weights**: train from scratch (BC doesn't produce a value function).
- **Optimizer state**: fresh.
- **Empirical observation normalization stats**: warm up for ~100 iters before allowing actor updates, so the critic has time to catch up.

Implementation note: this requires a small adapter script — `rsl_rl` doesn't have a built-in "load BC checkpoint as PPO actor init" mode. We'd write `scripts/bc_to_ppo.py` that loads `policy_bc.pth`, builds an `RslRlOnPolicyRunner` with matching actor architecture, copies weights into `runner.alg.actor_critic.actor`, and starts training.

### When to skip PPO

If the project deadline is tight or if BC alone hits ≥85% success rate, skip PPO. Story 1 (BC only) is a complete, defensible submission. PPO is the "nice-to-have second result."

---

## Why not vision PPO from scratch?

The naive alternative ("just train PPO with cameras directly, no BC, no detector") is **not recommended**:

- Sample-hungry: vision RL needs 100M+ steps, typically weeks of wall-clock.
- Unstable: reward hacking, gradient issues, sensitive to hyperparams.
- Rarely succeeds for manipulation without demonstrations.
- No interpretable perception module — the report has nothing to show for the CNN.

Upstream Isaac Lab stack-cube ships **BC only** (no PPO config) for the same reason.

---

## Pipeline summary

```
┌─────────────────────────┐
│  1. Generate dataset    │   ./isaaclab.sh -p scripts/collect_detector_data.py
│  in Isaac Sim           │
│  (random poses, lights) │   → datasets/tool_detector_raw/   (images + bbox labels)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Train CNN detector  │   python scripts/train_detector.py
│  (PyTorch, custom)      │
│                         │   → checkpoints/detector.pth   (+ confusion matrix png)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Record demos        │   ./isaaclab.sh -p record_demos.py
│  (human teleop)         │       --task Isaac-Lift-Pegboard-Franka-IK-Rel-Visuomotor-Play-v0
│                         │       --teleop_device keyboard  (or handtracking)
│                         │   → datasets/pegboard_demos.hdf5
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. Train BC policy     │   python scripts/train_bc.py
│  (CNN frozen as         │       --demos datasets/pegboard_demos.hdf5
│   feature extractor)    │       --detector checkpoints/detector.pth
│                         │   → checkpoints/policy_bc.pth
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. (Optional) PPO      │   ./isaaclab.sh -p scripts/bc_to_ppo.py
│  fine-tune              │       --bc_ckpt checkpoints/policy_bc.pth
│                         │       --task Isaac-Lift-Pegboard-Franka-Play-v0
│                         │   → checkpoints/policy_final.pth
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  6. Evaluate & rollout  │   ./isaaclab.sh -p scripts/eval_policy.py
│                         │       --checkpoint checkpoints/policy_final.pth
│                         │   → success rate table, video rollouts
└─────────────────────────┘
```

---

## File layout (planned)

```
ur_pick/
├── isaaclab_ext/
│   ├── tasks/lift_pegboard_franka/          # already exists
│   │   └── ik_rel_visuomotor_env_cfg.py     # cameras already wired
│   └── policies/                            # NEW
│       ├── __init__.py
│       ├── detector.py                      # CNN object detector class
│       └── visuomotor_policy.py             # BC policy wrapping the detector
├── scripts/                                 # NEW additions
│   ├── collect_detector_data.py             # synthetic data collector
│   ├── train_detector.py                    # CNN training loop
│   ├── train_bc.py                          # BC training loop
│   ├── bc_to_ppo.py                         # load BC weights into PPO
│   └── eval_policy.py                       # rollout + success rate
├── datasets/                                # NEW
│   ├── tool_detector_raw/                   # images + bbox labels
│   └── pegboard_demos.hdf5                  # human teleop demos
└── checkpoints/                             # NEW
    ├── detector.pth
    ├── policy_bc.pth
    └── policy_final.pth
```

---

## Open questions for the team

1. **Demo source** — keyboard or VR? VR is faster but requires the Quest 2 + WiVRn setup ([REBUILD_GUIDE.md Step 9](REBUILD_GUIDE.md)).
2. **Detector architecture** — baby YOLO vs heatmap? Both are ~200 lines of PyTorch; pick based on team preference.
3. **How many demos** to commit to recording? 50 (one evening) for proof-of-concept; 200 for serious BC.
4. **Are we doing PPO fine-tune?** Decide early — it doesn't add much complexity if planned in, but bolting it on at the end is painful.
5. **What success rate is "good enough"** for the writeup? 70%? 90%? Set the target before training.

---

## Quick glossary

- **BC (Behavioral Cloning)** — supervised learning of `(observation, action)` pairs from demos. Pure imitation.
- **PPO (Proximal Policy Optimization)** — on-policy RL algorithm. Improves a policy from reward.
- **IL (Imitation Learning)** — umbrella term that includes BC and adversarial methods like GAIL.
- **IK-Rel** — relative differential inverse kinematics. The action is a small `(Δposition, Δrotation, gripper)` delta the controller integrates into joint targets.
- **mAP (mean Average Precision)** — standard object-detection metric (area under the precision-recall curve, averaged across classes).
- **Visuomotor** — a policy that takes raw vision (camera images) as input.
