# Diffusion Policy Log

Component: Diffusion policy / imitation learning
Pipeline: teleop demo collection → diffusion policy training → (optional PPO warm start)

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
- Demo collection: keyboard teleop or VR teleop (VR produces more natural, less robotic motion)
- Target: 20-50 demos per object, 4 objects = 80-200 total demos
- Each demo: press 1/2/3/4 to select target → pick → drop in basket

---

## 2026-05-30 — Initial implementation
**Who:** Steph
**Changed:** Created isaaclab_ext/tasks/air2_franka/diffusion_policy.py and scripts/train_diffusion.py. Added train-diffusion to launch_air2.sh.
**Status:** coded, not yet tested (no demos collected yet)

### Architecture
- Noise predictor: 4-layer MLP with FiLM conditioning (obs modulates via scale+shift)
- Timestep embedding: sinusoidal (64-dim)
- Observation conditioning: frozen encoder (wrist + board visual features) + state + target_one_hot
- Action space: (CHUNK_SIZE=16, ACTION_DIM=7) flattened to 112-dim
- Training: cosine DDPM, T=100 steps, MSE noise loss, masked over padded chunk positions
- Inference: DDIM deterministic sampling, 10 steps (fast)
- Encoder: same frozen U-Net or ResNet-18 as BC policy — same checkpoint

### Sizes
| Backbone | obs_dim | noise_pred params (approx) |
|---|---|---|
| U-Net | 541 (512+25+4) | ~1.3M |
| ResNet-18 | 1053 (1024+25+4) | ~2.1M |

### Launch
```bash
./launch_air2.sh train-diffusion 200 unet   # after collecting demos
./launch_air2.sh train-diffusion 200 resnet18
```

### Pending
- Collect 20-50 VR demos per object before training
- Run first training and check loss curve
- Evaluate: compare BC vs diffusion rollout success rate
- Possible improvements to review: observation history stacking (T_o=2-4 steps), larger hidden_dim, action normalisation
