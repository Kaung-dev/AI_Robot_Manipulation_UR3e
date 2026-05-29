#!/usr/bin/env bash
# AIR2 pipeline launcher — Robotis env (slot-based, stable object placement)
#
# Usage:
#   ./launch_air2.sh collect-seg          # collect segmentation data (scripted, ~500 frames)
#   ./launch_air2.sh train-seg            # train U-Net segmentation model
#   ./launch_air2.sh collect-demos        # record manual keyboard demos (~20 episodes)
#   ./launch_air2.sh train-bc             # train behaviour cloning policy
#   ./launch_air2.sh eval [object]        # evaluate policy (object: brush|pliers|scissors|screwdriver)
#   ./launch_air2.sh teleop               # free keyboard teleop (no recording)
#   ./launch_air2.sh ppo [object]         # run PPO training for one object task
#   ./launch_air2.sh ppo-teleop [object]  # launch per-object env in teleop for inspection

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$REPO_ROOT/scripts/find_isaaclab.sh"
_resolve_isaaclab_path "$REPO_ROOT" || exit 1

TASK="Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0"
MODE="${1:-}"

case "$MODE" in

  collect-seg)
    FRAMES="${2:-500}"
    echo "[INFO] Collecting $FRAMES segmentation frames → datasets/air2_segmentation"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/collect_air2_segmentation_data.py" \
      --task "$TASK" --enable_cameras --frames "$FRAMES" \
      --output "$REPO_ROOT/datasets/air2_segmentation"
    ;;

  train-seg)
    EPOCHS="${2:-30}"
    echo "[INFO] Training segmentation model ($EPOCHS epochs)"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/train_air2_segmentation.py" \
      --data "$REPO_ROOT/datasets/air2_segmentation" \
      --epochs "$EPOCHS" \
      --output "$REPO_ROOT/checkpoints/air2_segmentation_unet.pth"
    ;;

  collect-demos)
    NUM="${2:-20}"
    echo "[INFO] Recording $NUM manual demos → datasets/air2_manual_demos"
    echo "[INFO] Controls: W/A/S/D/Q/E move | Z/X/T/G/C/V rotate | K gripper"
    echo "[INFO]           1=brush 2=pliers 3=scissors 4=screwdriver"
    echo "[INFO]           L=pause/resume | Enter=save | Backspace=discard"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/collect_air2_manual_demos.py" \
      --task "$TASK" --num_envs 1 --teleop_device keyboard --enable_cameras \
      --num_demos "$NUM" --output "$REPO_ROOT/datasets/air2_manual_demos"
    ;;

  train-bc)
    EPOCHS="${2:-50}"
    echo "[INFO] Training BC policy ($EPOCHS epochs)"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/train_bc.py" \
      --demos "$REPO_ROOT/datasets/air2_manual_demos" \
      --unet_ckpt "$REPO_ROOT/checkpoints/air2_segmentation_unet.pth" \
      --epochs "$EPOCHS" --batch_size 32 \
      --out "$REPO_ROOT/checkpoints/policy_bc.pth"
    ;;

  eval)
    OBJECT="${2:-brush}"
    echo "[INFO] Evaluating policy — target: $OBJECT"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/eval_bc.py" \
      --task "$TASK" --enable_cameras --num_envs 1 --num_episodes 5 \
      --bc_ckpt "$REPO_ROOT/checkpoints/policy_bc.pth" \
      --unet_ckpt "$REPO_ROOT/checkpoints/air2_segmentation_unet.pth" \
      --target_object "$OBJECT"
    ;;

  teleop)
    echo "[INFO] Free teleop — Isaac-AIR2-Robotis-Franka-Play-v0"
    "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$ISAACLAB_PATH/scripts/environments/teleoperation/teleop_se3_agent.py" \
      --task "Isaac-AIR2-Robotis-Franka-Play-v0" \
      --num_envs 1 --teleop_device keyboard --enable_cameras
    ;;

  ppo)
    OBJECT="${2:-brush}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    TASK="Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-v0"
    echo "[INFO] PPO training — $TASK"
    "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py" \
      --task "$TASK" --headless
    ;;

  ppo-teleop)
    OBJECT="${2:-brush}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    TASK="Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-Play-v0"
    echo "[INFO] Teleop inspection — $TASK"
    "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$ISAACLAB_PATH/scripts/environments/teleoperation/teleop_se3_agent.py" \
      --task "$TASK" --num_envs 1 --teleop_device keyboard --enable_cameras
    ;;

  *)
    echo "Usage: $0 collect-seg | train-seg | collect-demos | train-bc | eval [object] | teleop"
    exit 1
    ;;

esac
