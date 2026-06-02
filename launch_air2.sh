#!/usr/bin/env bash
# AIR2 pipeline launcher — Robotis env (slot-based, stable object placement)
#
# Usage:
#   ./launch_air2.sh collect-seg          # collect segmentation data (scripted, ~500 frames)
#   ./launch_air2.sh train-seg            # train U-Net segmentation model
#   ./launch_air2.sh collect-demos [n] [device] [object]  # record manual demos (device: keyboard|handtracking; object: brush|pliers|scissors|screwdriver)
#   ./launch_air2.sh precompute           # precompute brush centroids from demo board_rgb images
#   ./launch_air2.sh train-bc             # train behaviour cloning policy (U-Net encoder)
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
    FRAMES="${2:-0}"
    FRAMES_ARG=""
    [ "$FRAMES" -gt 0 ] && FRAMES_ARG="--frames $FRAMES"
    echo "[INFO] Collecting segmentation frames → datasets/air2_segmentation (Ctrl+C to stop)"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/collect_air2_segmentation_data.py" \
      --task "$TASK" --enable_cameras $FRAMES_ARG \
      --output "$REPO_ROOT/datasets/air2_segmentation"
    ;;

  train-seg)
    EPOCHS="${2:-60}"
    BACKBONE="${3:-resnet18}"
    LR_ARG=""
    [ "$BACKBONE" = "unet" ] && LR_ARG="--lr 1e-3"
    echo "[INFO] Training segmentation model ($BACKBONE, $EPOCHS epochs)"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/train_air2_segmentation.py" \
      --backbone "$BACKBONE" \
      --data "$REPO_ROOT/datasets/air2_segmentation" \
      --epochs "$EPOCHS" \
      --output "$REPO_ROOT/checkpoints/air2_segmentation_${BACKBONE}.pth" \
      $LR_ARG
    ;;

  collect-demos)
    NUM="${2:-20}"
    DEVICE="${3:-keyboard}"
    OBJECT="${4:-}"
    TARGET_ARG=""
    [ -n "$OBJECT" ] && TARGET_ARG="--default_target $OBJECT"
    HDF5_OUT="$REPO_ROOT/datasets/air2_mimic_source.hdf5"
    echo "[INFO] Recording $NUM manual demos → datasets/air2_manual_demos (device: $DEVICE${OBJECT:+, target: $OBJECT})"
    echo "[INFO] Also writing Mimic-compatible HDF5 → $HDF5_OUT"
    echo "[INFO] Controls: W/A/S/D/Q/E move | Z/X/T/G/C/V rotate | K gripper"
    echo "[INFO]           1=brush 2=pliers 3=scissors 4=screwdriver"
    echo "[INFO]           L=pause/resume | Enter=save | Backspace=discard"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/collect_air2_manual_demos.py" \
      --task "$TASK" --num_envs 1 --teleop_device "$DEVICE" --enable_cameras \
      --num_demos "$NUM" --output "$REPO_ROOT/datasets/air2_manual_demos" \
      --save_every_n_steps 4 --sensitivity 2 \
      --hdf5_output "$HDF5_OUT" \
      $TARGET_ARG
    ;;

  train-diffusion)
    EPOCHS="${2:-200}"
    BACKBONE="${3:-unet}"
    SEG_CKPT="$REPO_ROOT/checkpoints/air2_segmentation_${BACKBONE}.pth"
    echo "[INFO] Training diffusion policy ($BACKBONE encoder, $EPOCHS epochs)"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/train_diffusion.py" \
      --demos "$REPO_ROOT/datasets/air2_manual_demos" \
      --seg_ckpt "$SEG_CKPT" \
      --backbone "$BACKBONE" \
      --epochs "$EPOCHS" \
      --out "$REPO_ROOT/checkpoints/policy_diffusion_${BACKBONE}.pth"
    ;;

  precompute)
    echo "[INFO] Precomputing brush centroids from demo board_rgb images"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/precompute_centroids.py" \
      --demos "$REPO_ROOT/datasets/air2_manual_demos" \
      --unet_ckpt "$REPO_ROOT/checkpoints/air2_segmentation_unet.pth"
    ;;

  train-bc)
    EPOCHS="${2:-50}"
    echo "[INFO] Training BC policy (U-Net encoder, $EPOCHS epochs)"
    "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/train_bc.py" \
      --demos "$REPO_ROOT/datasets/air2_manual_demos" \
      --backbone unet \
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
      --backbone unet \
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
    NUM_ENVS="${3:-4}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    TASK="Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-v0"
    echo "[INFO] PPO training — $TASK (num_envs=$NUM_ENVS)"
    "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py" \
      --task "$TASK" --headless --num_envs "$NUM_ENVS"
    ;;

  collect-mimic)
    OBJECT="${2:-brush}"
    NUM="${3:-40}"
    OUT="${4:-$REPO_ROOT/datasets/air2_mimic_demos.hdf5}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    echo "[INFO] Collecting $NUM Mimic source demos ($OBJECT) → $OUT"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/collect_mimic_demos.py" \
      --task "Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-Mimic-v0" \
      --teleop_device keyboard \
      --num_demos "$NUM" \
      --num_envs 1 \
      --teleop_env_index 0 \
      --output_file "$OUT"
    ;;

  annotate-mimic)
    OBJECT="${2:-brush}"
    IN="${3:-$REPO_ROOT/datasets/air2_mimic_demos.hdf5}"
    OUT="${4:-$REPO_ROOT/datasets/air2_mimic_demos_annotated.hdf5}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    echo "[INFO] Annotating Mimic demos ($OBJECT): $IN → $OUT"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/run_mimic_annotate.py" \
      --task "Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-Mimic-v0" \
      --input_file "$IN" \
      --output_file "$OUT" \
      --auto
    ;;

  generate-mimic)
    OBJECT="${2:-brush}"
    IN="${3:-$REPO_ROOT/datasets/air2_mimic_demos_annotated.hdf5}"
    OUT="${4:-$REPO_ROOT/datasets/air2_mimic_generated.hdf5}"
    NUM="${5:-1000}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    echo "[INFO] Generating $NUM synthetic demos ($OBJECT) from $IN → $OUT"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/run_mimic_generate.py" \
      --task "Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-Mimic-v0" \
      --input_file "$IN" \
      --output_file "$OUT" \
      --generation_num_trials "$NUM" \
      --num_envs 4 --headless
    ;;

  train-state-bc)
    OBJECT="${2:-brush}"
    HDF5="${3:-$REPO_ROOT/datasets/air2_mimic_${OBJECT}_generated.hdf5}"
    OUT="${4:-$REPO_ROOT/checkpoints/policy_state_bc_${OBJECT}.pth}"
    EPOCHS="${5:-300}"
    echo "[INFO] Training state-BC ($OBJECT) from $HDF5 → $OUT  (${EPOCHS} epochs)"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/train_state_bc_from_hdf5.py" \
      --hdf5 "$HDF5" --object "$OBJECT" --out "$OUT" --epochs "$EPOCHS"
    ;;

  ppo-warm-start)
    BC_CKPT="${2:-$REPO_ROOT/checkpoints/policy_state_bc_mimic.pth}"
    ITERS="${3:-2000}"
    echo "[INFO] PPO warm-start from $BC_CKPT  ($ITERS iterations)"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/bc_to_ppo.py" \
      --task "Isaac-AIR2-Robotis-Franka-Brush-v0" \
      --state_bc_ckpt "$BC_CKPT" \
      --num_envs 4 --max_iterations "$ITERS" --headless
    ;;

  eval-state-bc)
    OBJECT="${2:-brush}"
    CKPT="${3:-$REPO_ROOT/checkpoints/policy_state_bc_mimic.pth}"
    EPISODES="${4:-20}"
    MAX_STEPS="${5:-1600}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    echo "[INFO] Evaluating state-BC ($OBJECT) — $CKPT  ($EPISODES episodes, max $MAX_STEPS steps)"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/eval_state_bc.py" \
      --state_bc_ckpt "$CKPT" \
      --task "Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-Play-v0" \
      --num_envs 1 --num_episodes "$EPISODES" --max_steps "$MAX_STEPS" --episode_length_s 80.0
    ;;

  ppo-play)
    OBJECT="${2:-brush}"
    RUN="${3:-}"
    CHECKPOINT="${4:-model_2999.pt}"
    OBJECT_CAP="$(echo "$OBJECT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
    TASK="Isaac-AIR2-Robotis-Franka-${OBJECT_CAP}-Play-v0"
    RUN_ARG=""
    [ -n "$RUN" ] && RUN_ARG="--load_run $RUN"
    echo "[INFO] PPO play — $TASK  run=$RUN  checkpoint=$CHECKPOINT"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/play.py" \
      --task "$TASK" --num_envs 1 \
      $RUN_ARG --checkpoint "$CHECKPOINT"
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

  eval-multi)
    ROUNDS="${2:-1}"
    MAX_STEPS="${3:-2000}"
    echo "[INFO] Multi-object BC eval — $ROUNDS rounds, max $MAX_STEPS steps/object"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p \
      "$REPO_ROOT/scripts/eval_multi_object_bc.py" \
      --task "Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0" \
      --ckpt_dir "$REPO_ROOT/checkpoints" \
      --num_envs 1 --num_rounds "$ROUNDS" --max_steps_per_object "$MAX_STEPS" \
      --enable_cameras
    ;;

  *)
    echo "Usage: $0 collect-seg | train-seg | collect-demos | precompute | train-diffusion | train-bc | eval [object] | eval-multi [rounds] | teleop"
    exit 1
    ;;

esac
