# Checkpoints

## Behaviour-cloning models

### Visual BC (ResNet-18 + state + chunked actions)
- **`policy_bc.pth`** — visual BC trained on `datasets/air2_manual_demos`
  (80 episodes, 4 targets merged: brush, pliers, scissors, screwdriver). Goal-
  conditioned via `target_one_hot` in `states.npz`. Backbone = `resnet18`
  (ImageNet pretrained, frozen encoder). Best val loss = **0.0151** at epoch 49.
- **`policy_bc.log.json`** — per-epoch train/val loss trace.

### State-only BC (MLP matching rsl_rl actor, per-target)
Trained from friend's HDF5 demos (`_archive/v2_friend_hdf5/*.hdf5`, 20 demos
per target, state-only obs). Architecture exactly matches the rsl_rl PPO actor
(`MLP(35 → 256 → 128 → 64 → 7, elu)`) so the state-dict loads directly via
`bc_to_ppo.py --state_bc_ckpt`.

| Checkpoint | Target | Best val loss |
|---|---|---|
| `policy_state_bc_brush.pth` | brush (paintbrush ring) | **0.00160** |
| `policy_state_bc_pliers.pth` | pliers | **0.00151** |
| `policy_state_bc_scissors.pth` | scissors | **0.00225** |
| `policy_state_bc_screwdriver.pth` | screwdriver | **0.00281** |

Each has a sibling `policy_state_bc_<target>.log.json` with per-epoch losses.

## PPO checkpoints
Multiple PPO runs were attempted on the Robotis pegboard task
(`Isaac-AIR2-Robotis-Franka-{Brush,Pliers,Scissors,Screwdriver}-v0`). The full
artefacts live under `logs/rsl_rl/air2_ppo/<timestamp>/model_*.pt` (gitignored
locally, not pushed). None of the trained PPO policies achieved consistent
`task_success` — at best one of 16 envs delivered an object on a single
iteration. The state-BC checkpoints above are the most reliable trained
policies in this repo today.

### Known issue
`scripts/bc_to_ppo.py` derives its log dir from the current second, so
launching 4 parallel runs at the same second causes them to share one dir
and race each other's `model_*.pt` writes. Future runs should add a `--seed`
or `--run_name` suffix to the log dir.

## What's NOT here
- The U-Net segmentation checkpoint (`air2_segmentation_unet.pth`) — Stephen
  trained this to 0.919 tool mIoU but it lives only on his machine; not yet
  shared via git.
