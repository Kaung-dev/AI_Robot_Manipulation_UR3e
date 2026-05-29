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
