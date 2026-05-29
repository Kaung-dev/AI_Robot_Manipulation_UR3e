# CNN Log

Component: Semantic segmentation + visual encoding
Model: Custom U-Net (base_channels=32, 9 classes)
Pipeline: collect_air2_segmentation_data.py → train_air2_segmentation.py → frozen encoder in policy

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

## 2026-05-29
**Who:** Steph
**Changed:** cnn/__init__.py — made imports lazy so Isaac Lab auto-discovery doesn't try to load torch/PIL at task registration time and crash
**Tried:** Launching air2_franka task — failed with ImportError on AIR2SegmentationDataset
**Result:** Fixed. Lazy imports in __init__.py let the task register without loading training dependencies
**Status:** working

---

## 2026-05-29 — Deep dive findings
**Who:** Steph
**Changed:** nothing yet
**Tried:** Full pipeline review
**Result:** Three issues found:

1. `position_world` always None in postprocess.py — camera extrinsics never extracted or passed to extract_detections(). Blocks using CNN as reward confirmation signal. To fix: pass camera pos_w/rot_w from Isaac Lab camera data into extract_detections() and compute world transform.

2. No data augmentation in train_air2_segmentation.py — with ~500 frames model will overfit. Needs RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90, brightness/contrast jitter.

3. num_classes=7 default in model.py — training correctly uses 9 from class map, but default is wrong and inference fallback `checkpoint.get("num_classes", 7)` would load broken model if checkpoint is missing that key.

**Status:** 1=broken, 2=not implemented, 3=minor risk

---

## 2026-05-29 — Segmentation data collection verified
**Who:** Steph
**Tried:** ./launch_air2.sh collect-seg 50
**Result:** 50 frames collected cleanly to datasets/air2_segmentation/. Images, masks, overlays all present.
**Status:** working
**Note:** 50 frames is smoke test only. Real collection needs 500+ frames for meaningful training.

---

## 2026-05-30 — Full dataset collection + retraining
**Who:** Steph
**Changed:** Collected 757 good frames (758 raw, 1 corrupt at boundary removed). Split: 606 train / 151 val. Regenerated train.txt/val.txt from all images on disk (previous split files were stale from 50-frame run — Ctrl+C killed the write).
**Tried:** Trained both ResNet-18 and U-Net for 60 epochs with: encoder freeze (ResNet-18, first 10 epochs), cosine LR decay, dice loss weight 0.5, data augmentation.
**Result:**
- ResNet-18: best tool_miou = 0.793, smooth training
- U-Net: best tool_miou = 0.919, noisy early training but stabilised from epoch 50
**Status:** working — U-Net checkpoint in use (checkpoints/air2_segmentation_unet.pth)

---

## 2026-05-30 — position_world fix
**Who:** Steph
**Changed:** postprocess.py — added `pos_w` and `rot_w_quat` params to `extract_detections()`. Added `_quat_to_rot()` helper. Computes `position_world = pos_w + R_cam_to_world @ position_camera` when extrinsics provided. run_air2_segmentation_inference.py updated to pass `camera.data.pos_w[0]` and `camera.data.quat_w_ros[0]` in live mode.
**Result:** position_world now populated when camera extrinsics are passed. Backward compatible — callers that omit extrinsics still get None.
**Status:** working (code only — not yet wired into reward functions)
**Note:** Reward functions will need to pass extrinsics when calling extract_detections. That's part of the reward redesign step.
