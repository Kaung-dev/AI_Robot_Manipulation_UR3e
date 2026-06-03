# Install and Run Guide

This guide is for the `main` branch of this repository. It covers:

- installing Isaac Sim and Isaac Lab;
- where to put this repository;
- bootstrapping the custom Isaac Lab tasks;
- the commands used for data collection, training, evaluation, and showing trained results.

The repo is an Isaac Lab project for the AIR2 / Robotis pegboard Franka manipulation pipeline. The central runner is `launch_air2.sh`; direct `isaaclab.sh` and `isaaclab.bat` commands are included where useful.

For the end-to-end data and model workflow, refer to [`GUIDE.md`](GUIDE.md). It covers collecting data, training, Mimic generation, and executing/evaluating the trained policies.

Unless a command explicitly says otherwise, run it from the repository root:

```text
D:\AI_Robot_Manipulation_UR3e
```

## 1. Install Isaac Sim

Use these versions for this branch:

| Component | Expected version |
| --- | --- |
| Isaac Sim standalone | 5.1.0 |
| Isaac Lab | 2.3.2 |
| Python | 3.11, matching Isaac Sim 5.x |

Isaac Lab 2.3.2 expects Isaac Sim 5.1.0. Basic machine requirements are Ubuntu 22.04 or Windows 11, 32 GB RAM, and 16 GB GPU VRAM. Use the latest NVIDIA production driver; 580-series drivers are the recommended line for this setup.

Reference websites:

- Isaac Sim download page: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html
- Isaac Sim workstation installation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html
- Isaac Lab 2.3.2 local installation: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html
- Isaac Lab 2.3.2 binary installation path: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html

### Downloads

Download the Isaac Sim 5.1.0 standalone zip from the Isaac Sim download page above. Save it to your `Downloads` folder.

Expected zip names:

```text
Linux:   isaac-sim-standalone-5.1.0-linux-x86_64.zip
Windows: isaac-sim-standalone-5.1.0-windows-x86_64.zip
```

Download Isaac Lab from GitHub with `git`:

Linux:

```bash
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
```

Windows PowerShell:

```powershell
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git D:\IsaacLab
```

If `git` is not installed, download the Isaac Lab source zip instead:

Linux:

```bash
cd ~/Downloads
curl -L https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.3.2.zip -o IsaacLab-v2.3.2.zip
unzip IsaacLab-v2.3.2.zip -d ~
mv ~/IsaacLab-2.3.2 ~/IsaacLab
```

Windows PowerShell:

```powershell
cd $env:USERPROFILE\Downloads
Invoke-WebRequest -Uri https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.3.2.zip -OutFile IsaacLab-v2.3.2.zip
Expand-Archive IsaacLab-v2.3.2.zip -DestinationPath D:\
Move-Item D:\IsaacLab-2.3.2 D:\IsaacLab
```

### Linux standalone install

After downloading the Isaac Sim 5.1.0 standalone zip, install it into a stable folder. NVIDIA's example uses `~/isaacsim`; this repo can use any path as long as `.env` points to the Isaac Lab root.

```bash
mkdir -p ~/isaacsim
cd ~/Downloads
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
./isaac-sim.selector.sh
```

In the selector, choose Isaac Sim Full and start it once. The first launch can take several minutes while shaders and caches are built.

Optional compatibility check:

```bash
cd ~/isaacsim
./isaac-sim.compatibility_check.sh
```

### Windows standalone install

After downloading the Isaac Sim 5.1.0 standalone Windows zip, install it into `C:\isaacsim`:

```powershell
mkdir C:\isaacsim
cd $env:USERPROFILE\Downloads
tar -xvzf "isaac-sim-standalone-5.1.0-windows-x86_64.zip" -C C:\isaacsim
cd C:\isaacsim
.\post_install.bat
.\isaac-sim.selector.bat
```

Optional compatibility check:

```powershell
cd C:\isaacsim
.\isaac-sim.compatibility_check.bat
```

## 2. Install Isaac Lab

This repo expects an Isaac Lab source tree that contains `isaaclab.sh` on Linux or `isaaclab.bat` on Windows. If your Isaac Sim standalone package already contains an `IsaacLab` folder, use that. If you already cloned or zip-downloaded Isaac Lab in the previous section, do not clone it again.

### Linux source install

```bash
if [ ! -d "$HOME/IsaacLab" ]; then
  git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
fi
cd ~/IsaacLab

# Link Isaac Sim into Isaac Lab.
ln -sfn ~/isaacsim _isaac_sim

# Optional but recommended isolated environment.
./isaaclab.sh --conda
conda activate env_isaaclab

# Linux dependency needed by robomimic and some learning-framework installs.
sudo apt install -y cmake build-essential

# Install Isaac Lab and learning frameworks.
./isaaclab.sh --install

# Verify Isaac Lab can launch Isaac Sim.
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

### Windows source install

Open Command Prompt or PowerShell as Administrator for the symlink step:

```powershell
if (-not (Test-Path D:\IsaacLab)) {
  git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git D:\IsaacLab
}
cd D:\IsaacLab

# In an elevated shell:
if (-not (Test-Path D:\IsaacLab\_isaac_sim)) {
  cmd /c mklink /D _isaac_sim C:\isaacsim
}

.\isaaclab.bat --conda
conda activate env_isaaclab
.\isaaclab.bat --install
.\isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

If you install Isaac Lab somewhere else, keep that path handy. You will put it in this repo's `.env`.

## 3. Put This Repository Somewhere Stable

Do not put this repository inside Isaac Lab. Keep it as a separate project folder, then let `setup_isaaclab.sh` symlink the custom task files into Isaac Lab.

Recommended Linux layout:

```bash
mkdir -p ~/ai_ws
cd ~/ai_ws
git clone https://github.com/Kaung-dev/AI_Robot_Manipulation_UR3e.git
cd AI_Robot_Manipulation_UR3e
```

Recommended Windows layout:

```powershell
cd D:\
git clone https://github.com/Kaung-dev/AI_Robot_Manipulation_UR3e.git
cd D:\AI_Robot_Manipulation_UR3e
```

The current checked-out workspace is:

```text
D:\AI_Robot_Manipulation_UR3e
```

## 4. Configure the Repo

Create `.env` from the example and point it at your Isaac Lab root:

```bash
cp .env.example .env
```

Linux example:

```bash
export ISAACLAB_PATH=$HOME/IsaacLab
export PYTHONPATH=/home/YOUR_USER/ai_ws/AI_Robot_Manipulation_UR3e:${PYTHONPATH:-}
```

If you are using a bundled Isaac Lab under Isaac Sim, use that instead:

```bash
export ISAACLAB_PATH=$HOME/isaacsim/IsaacLab
export PYTHONPATH=/home/YOUR_USER/ai_ws/AI_Robot_Manipulation_UR3e:${PYTHONPATH:-}
```

Windows example:

```powershell
$env:ISAACLAB_PATH = "D:\IsaacLab"
$env:PYTHONPATH = "D:\AI_Robot_Manipulation_UR3e;$env:PYTHONPATH"
```

The shell launchers read `.env` on Linux. On Windows, use the direct `isaaclab.bat` commands unless you are running the bash launchers through Git Bash or WSL.

## 5. Link the Repo Into Isaac Lab

Run this once after cloning, and again after pulling changes that modify `isaaclab_patches/`.

```bash
source .env
./setup_isaaclab.sh
```

What it does:

- symlinks `isaaclab_ext/robots/ur3e_rg2.py` into Isaac Lab assets;
- symlinks the AIR2, AIR2 Robotis, and Pegboard task folders into Isaac Lab tasks;
- copies VR and recording patches from `isaaclab_patches/` into Isaac Lab;
- generates `scene/scene_isaaclab.usd` if it is missing.

On `main`, `scene/scene_isaaclab.usd` is already present. Do not delete it unless you also restore `scripts/fix_scene_for_isaaclab.py`; `setup_isaaclab.sh` references that generator script, but the script is not present on this branch.

After this, Isaac Lab can discover the custom Gym task IDs when scripts import `isaaclab_ext.tasks.*`.

## 6. Task IDs

Main stable Robotis AIR2 task family:

| Task ID | Use |
| --- | --- |
| `Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0` | Segmentation data, manual demos, visual BC eval |
| `Isaac-AIR2-Robotis-Franka-Play-v0` | Free teleop / inspection |
| `Isaac-AIR2-Robotis-Franka-Brush-v0` | PPO training for brush |
| `Isaac-AIR2-Robotis-Franka-Pliers-v0` | PPO training for pliers |
| `Isaac-AIR2-Robotis-Franka-Scissors-v0` | PPO training for scissors |
| `Isaac-AIR2-Robotis-Franka-Screwdriver-v0` | PPO training for screwdriver |
| `Isaac-AIR2-Robotis-Franka-Brush-Play-v0` | Brush eval / PPO play |
| `Isaac-AIR2-Robotis-Franka-Pliers-Play-v0` | Pliers eval / PPO play |
| `Isaac-AIR2-Robotis-Franka-Scissors-Play-v0` | Scissors eval / PPO play |
| `Isaac-AIR2-Robotis-Franka-Screwdriver-Play-v0` | Screwdriver eval / PPO play |
| `Isaac-AIR2-Robotis-Franka-Brush-Mimic-v0` | Mimic demos / generation for brush |
| `Isaac-AIR2-Robotis-Franka-Pliers-Mimic-v0` | Mimic demos / generation for pliers |
| `Isaac-AIR2-Robotis-Franka-Scissors-Mimic-v0` | Mimic demos / generation for scissors |
| `Isaac-AIR2-Robotis-Franka-Screwdriver-Mimic-v0` | Mimic demos / generation for screwdriver |

Other registered task families:

| Task ID | Use |
| --- | --- |
| `Isaac-AIR2-Franka-Play-v0` | Hook-based AIR2 inspection, currently WIP |
| `Isaac-AIR2-Franka-Segmentation-Play-v0` | Hook-based segmentation, currently WIP |
| `Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Toothbrush-v0` | Original pegboard VR / keyboard recording |
| `Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Scissors-v0` | Original pegboard VR / keyboard recording |
| `Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Silicone-v0` | Original pegboard VR / keyboard recording |
| `Isaac-Pegboard-Franka-IK-Rel-Visuomotor-Pliers-v0` | Original pegboard VR / keyboard recording |

## 7. Quick Smoke Test

Use this before a long collection or training run:

```bash
source .env
./launch_air2.sh collect-seg 50
./launch_air2.sh train-seg 2 resnet18
./launch_air2.sh collect-demos 2 keyboard brush
./launch_air2.sh train-bc 2
./launch_air2.sh eval brush
```

On Windows, use the direct commands in the next sections with small values such as `--frames 50`, `--epochs 2`, and `--num_demos 2`.

## 8. Main Pipeline With `launch_air2.sh`

This is the shortest Linux workflow.

```bash
source .env

# 1. Collect segmentation frames.
./launch_air2.sh collect-seg 500

# 2. Train segmentation model.
./launch_air2.sh train-seg 60 resnet18
# or:
./launch_air2.sh train-seg 60 unet

# 3. Record manual demos.
./launch_air2.sh collect-demos 20 keyboard brush
./launch_air2.sh collect-demos 20 keyboard pliers
./launch_air2.sh collect-demos 20 keyboard scissors
./launch_air2.sh collect-demos 20 keyboard screwdriver

# 4. Train visual behavior cloning.
./launch_air2.sh train-bc 50

# 5. Show / evaluate a trained BC result.
./launch_air2.sh eval brush
./launch_air2.sh eval pliers
./launch_air2.sh eval scissors
./launch_air2.sh eval screwdriver

# 6. Free teleop inspection.
./launch_air2.sh teleop
```

Outputs:

| Step | Output |
| --- | --- |
| `collect-seg` | `datasets/air2_segmentation/` |
| `train-seg` | `checkpoints/air2_segmentation_resnet18.pth` or `checkpoints/air2_segmentation_unet.pth` |
| `collect-demos` | `datasets/air2_manual_demos/` and `datasets/air2_mimic_source.hdf5` |
| `train-bc` | `checkpoints/policy_bc.pth` and `checkpoints/policy_bc.log.json` |
| `eval` | console success metrics and `eval_results/bc_rollouts.json` if using direct defaults |

### `launch_air2.sh` command reference

| Command | What it runs |
| --- | --- |
| `./launch_air2.sh collect-seg [frames]` | collect segmentation frames into `datasets/air2_segmentation` |
| `./launch_air2.sh train-seg [epochs] [resnet18|unet]` | train segmentation checkpoint |
| `./launch_air2.sh collect-demos [num] [keyboard|handtracking] [object]` | collect manual demos and optional Mimic source HDF5 |
| `./launch_air2.sh precompute` | legacy/unavailable on `main`; it references missing `scripts/precompute_centroids.py` |
| `./launch_air2.sh train-bc [epochs]` | train visual BC policy |
| `./launch_air2.sh eval [object]` | show / evaluate visual BC policy |
| `./launch_air2.sh teleop` | free keyboard teleop |
| `./launch_air2.sh train-diffusion [epochs] [resnet18|unet]` | train diffusion policy |
| `./launch_air2.sh ppo [object] [num_envs]` | train PPO from scratch |
| `./launch_air2.sh ppo-play [object] [run] [checkpoint]` | show a trained PPO run |
| `./launch_air2.sh ppo-teleop [object]` | inspect a per-object PPO task with teleop |
| `./launch_air2.sh collect-mimic [object] [num] [out.hdf5]` | collect Mimic source demos |
| `./launch_air2.sh annotate-mimic [object] [in.hdf5] [out.hdf5]` | annotate Mimic source demos |
| `./launch_air2.sh generate-mimic [object] [annotated.hdf5] [out.hdf5] [num_trials]` | generate synthetic Mimic demos |
| `./launch_air2.sh train-state-bc [object] [generated.hdf5] [out.pth] [epochs]` | train per-object state-BC |
| `./launch_air2.sh eval-state-bc [object] [ckpt.pth] [episodes] [max_steps]` | show / evaluate state-BC |
| `./launch_air2.sh ppo-warm-start [state_bc_ckpt] [iterations]` | warm-start PPO from state-BC |
| `./launch_air2.sh eval-sequential [episodes] [extra eval_sequential.py args]` | run the unified sequential demo |
| `./launch_air2.sh eval-multi [rounds] [max_steps_per_object]` | run the multi-object BC orchestrator |

## 9. Segmentation Commands

### Linux direct command

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/collect_air2_segmentation_data.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --enable_cameras \
  --frames 500 \
  --output datasets/air2_segmentation
```

Train ResNet-18:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train_air2_segmentation.py \
  --backbone resnet18 \
  --data datasets/air2_segmentation \
  --epochs 60 \
  --output checkpoints/air2_segmentation_resnet18.pth
```

Train U-Net:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train_air2_segmentation.py \
  --backbone unet \
  --data datasets/air2_segmentation \
  --epochs 60 \
  --lr 1e-3 \
  --output checkpoints/air2_segmentation_unet.pth
```

### Windows direct command

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\collect_air2_segmentation_data.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --enable_cameras `
  --frames 500 `
  --output datasets\air2_segmentation
```

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\train_air2_segmentation.py `
  --backbone resnet18 `
  --data datasets\air2_segmentation `
  --epochs 60 `
  --output checkpoints\air2_segmentation_resnet18.pth
```

To inspect training quality, check:

```text
checkpoints/air2_segmentation_metrics.json
checkpoints/air2_segmentation_overlays/
datasets/air2_segmentation/overlays/
```

## 10. Manual Demo Collection Commands

Keyboard controls:

| Key | Action |
| --- | --- |
| `1` | target brush |
| `2` | target pliers |
| `3` | target scissors |
| `4` | target screwdriver |
| `W/A/S/D/Q/E` | translate end effector |
| `Z/X/T/G/C/V` | rotate end effector |
| `K` | toggle gripper |
| `L` | pause / resume recording |
| `Enter` | save episode |
| `Backspace` | discard episode |
| `R` | reset |

Press the object number before moving toward it.

Linux:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/collect_air2_manual_demos.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras \
  --num_demos 20 \
  --output datasets/air2_manual_demos_brush \
  --save_every_n_steps 4 \
  --sensitivity 2 \
  --hdf5_output datasets/air2_mimic_brush.hdf5 \
  --default_target brush
```

Windows:

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\collect_air2_manual_demos.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --num_envs 1 `
  --teleop_device keyboard `
  --enable_cameras `
  --num_demos 20 `
  --output datasets\air2_manual_demos_brush `
  --save_every_n_steps 4 `
  --sensitivity 2 `
  --hdf5_output datasets\air2_mimic_brush.hdf5 `
  --default_target brush
```

Valid targets: `brush`, `pliers`, `scissors`, `screwdriver`.

## 11. Visual Behavior Cloning Commands

Train visual BC with a frozen segmentation encoder:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train_bc.py \
  --demos datasets/air2_manual_demos \
  --backbone resnet18 \
  --unet_ckpt checkpoints/air2_segmentation_resnet18.pth \
  --epochs 50 \
  --batch_size 32 \
  --out checkpoints/policy_bc.pth
```

If using the U-Net checkpoint:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train_bc.py \
  --demos datasets/air2_manual_demos \
  --backbone unet \
  --unet_ckpt checkpoints/air2_segmentation_unet.pth \
  --epochs 50 \
  --batch_size 32 \
  --out checkpoints/policy_bc.pth
```

Show / evaluate the visual BC policy:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_bc.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --bc_ckpt checkpoints/policy_bc.pth \
  --backbone resnet18 \
  --unet_ckpt checkpoints/air2_segmentation_resnet18.pth \
  --target_object brush \
  --enable_cameras \
  --num_envs 1 \
  --num_episodes 10
```

For U-Net, set `--backbone unet --unet_ckpt checkpoints/air2_segmentation_unet.pth`.

## 12. Mimic and State-BC Commands

For this branch, the more reliable imitation-learning path is Isaac Lab Mimic plus state-BC. Visual BC is still documented above, but state-BC is preferred because it uses state observations and avoids the visual-policy overfitting seen in earlier runs. This pipeline is:

```text
collect source demos -> annotate demos -> generate synthetic demos -> train state-BC -> evaluate
```

For one tool:

```bash
TOOL=brush

./launch_air2.sh collect-mimic $TOOL 40 datasets/air2_mimic_${TOOL}_source.hdf5

./launch_air2.sh annotate-mimic $TOOL \
  datasets/air2_mimic_${TOOL}_source.hdf5 \
  datasets/air2_mimic_${TOOL}_annotated.hdf5

./launch_air2.sh generate-mimic $TOOL \
  datasets/air2_mimic_${TOOL}_annotated.hdf5 \
  datasets/air2_mimic_${TOOL}_generated.hdf5 \
  1000

./launch_air2.sh train-state-bc $TOOL \
  datasets/air2_mimic_${TOOL}_generated.hdf5 \
  checkpoints/policy_state_bc_${TOOL}.pth \
  300

./launch_air2.sh eval-state-bc $TOOL \
  checkpoints/policy_state_bc_${TOOL}.pth \
  20 \
  2000
```

Train all four state-BC policies once generated HDF5s exist:

```bash
for TOOL in brush pliers scissors screwdriver; do
  ./launch_air2.sh train-state-bc $TOOL \
    datasets/air2_mimic_${TOOL}_generated.hdf5 \
    checkpoints/policy_state_bc_${TOOL}.pth \
    300
done
```

Direct state-BC training command:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train_state_bc_from_hdf5.py \
  --hdf5 datasets/air2_mimic_brush_generated.hdf5 \
  --object brush \
  --out checkpoints/policy_state_bc_brush.pth \
  --epochs 300
```

Direct state-BC eval command:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_state_bc.py \
  --state_bc_ckpt checkpoints/policy_state_bc_brush.pth \
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
  --num_envs 1 \
  --num_episodes 20 \
  --max_steps 2000 \
  --episode_length_s 80.0
```

## 13. Multi-Object and Sequential Evaluation

`scripts/eval_multi_object_bc.py` looks for checkpoint files named `policy_state_bc_<tool>.pth` or `policy_state_bc_<tool>_mimic.pth` inside `--ckpt_dir`. If your trained checkpoints have different names, either pass a directory with matching filenames or create copies/symlinks with these names:

```text
checkpoints/policy_state_bc_brush.pth
checkpoints/policy_state_bc_pliers.pth
checkpoints/policy_state_bc_scissors.pth
checkpoints/policy_state_bc_screwdriver.pth
```

Multi-object state-BC orchestrator:

```bash
./launch_air2.sh eval-multi 1 2000
```

Direct command:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_multi_object_bc.py \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \
  --ckpt_dir checkpoints \
  --num_envs 1 \
  --num_rounds 3 \
  --max_steps_per_object 2000 \
  --enable_cameras
```

Sequential unified demo with the checkpoints currently present in this checkout:

```bash
./launch_air2.sh eval-sequential 1 --no_cnn \
  --brush_ckpt checkpoints/policy_state_bc_mimic_v2.pth \
  --pliers_ckpt checkpoints/policy_state_bc_mimic_pliers_v2.pth \
  --screwdriver_ckpt checkpoints/policy_state_bc_mimic_screwdriver_v2.pth
```

Sequential demo with CNN:

```bash
DISPLAY=:0 "$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_sequential.py \
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
  --brush_ckpt checkpoints/policy_state_bc_mimic_v2.pth \
  --pliers_ckpt checkpoints/policy_state_bc_mimic_pliers_v2.pth \
  --screwdriver_ckpt checkpoints/policy_state_bc_mimic_screwdriver_v2.pth \
  --seg_ckpt checkpoints/air2_segmentation_unet_newscene.pth \
  --num_episodes 1 \
  --max_steps_per_tool 2000 \
  --out eval_results/sequential.json
```

Add `--headless` for metrics-only runs. Add `--no_cnn` to use ground-truth object positions and skip camera loading.

## 14. Diffusion Policy Commands

Train diffusion policy from manual demo folders:

```bash
./launch_air2.sh train-diffusion 200 unet
./launch_air2.sh train-diffusion 200 resnet18
```

Direct command:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train_diffusion.py \
  --demos datasets/air2_manual_demos \
  --seg_ckpt checkpoints/air2_segmentation_unet.pth \
  --backbone unet \
  --epochs 200 \
  --out checkpoints/policy_diffusion_unet.pth
```

Evaluate a diffusion state-BC checkpoint:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_diffusion_bc.py \
  --diffusion_ckpt checkpoints/policy_diffusion_unet.pth \
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
  --num_envs 1 \
  --num_episodes 5 \
  --episode_length_s 40.0
```

## 15. PPO Commands

Train PPO from scratch for one object:

```bash
./launch_air2.sh ppo brush 4
./launch_air2.sh ppo pliers 4
./launch_air2.sh ppo scissors 4
./launch_air2.sh ppo screwdriver 4
```

Direct PPO training command:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-AIR2-Robotis-Franka-Brush-v0 \
  --headless \
  --num_envs 4
```

Show a trained PPO result:

```bash
./launch_air2.sh ppo-play brush RUN_NAME model_2999.pt
```

Direct PPO play command:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
  --num_envs 1 \
  --load_run RUN_NAME \
  --checkpoint model_2999.pt
```

PPO warm-start from a state-BC checkpoint:

```bash
./launch_air2.sh ppo-warm-start checkpoints/policy_state_bc_mimic.pth 2000
```

Direct warm-start command:

```bash
PYTHONPATH="$PWD:${PYTHONPATH:-}" "$ISAACLAB_PATH/isaaclab.sh" -p scripts/bc_to_ppo.py \
  --task Isaac-AIR2-Robotis-Franka-Brush-v0 \
  --state_bc_ckpt checkpoints/policy_state_bc_mimic.pth \
  --num_envs 4 \
  --max_iterations 2000 \
  --headless
```

Evaluate PPO with the local evaluator:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/eval_ppo.py \
  --ppo_ckpt logs/rsl_rl/air2_ppo/RUN_NAME/model_final.pt \
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 \
  --num_envs 1 \
  --num_episodes 20 \
  --max_steps 800
```

## 16. Teleop and VR Commands

Free keyboard teleop in the Robotis AIR2 env:

```bash
./launch_air2.sh teleop
```

Per-object PPO teleop inspection:

```bash
./launch_air2.sh ppo-teleop brush
./launch_air2.sh ppo-teleop pliers
./launch_air2.sh ppo-teleop scissors
./launch_air2.sh ppo-teleop screwdriver
```

Original pegboard VR / keyboard demo recording:

```bash
./launch_teleop.sh handtracking toothbrush
./launch_teleop.sh handtracking scissors
./launch_teleop.sh handtracking silicone
./launch_teleop.sh handtracking pliers
```

Keyboard fallback:

```bash
./launch_teleop.sh keyboard toothbrush ~/datasets/test_toothbrush.hdf5
```

For VR, start WiVRn first, then put on the Quest and connect:

```bash
flatpak run io.github.wivrn.wivrn
```

`launch_preview.sh` exists, but on `main` it references `scripts/preview_cameras.py`, which is not present. Treat that preview launcher as legacy until the missing script is restored.

## 17. Plotting and Inspecting Results

Plot available training and evaluation curves:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/plot_training_curves.py \
  --unet_metrics checkpoints/air2_segmentation_metrics.json \
  --bc_log checkpoints/policy_bc.log.json \
  --bc_eval eval_results/bc_rollouts.json \
  --ppo_eval eval_results/ppo.json \
  --out_dir eval_results/plots
```

Inspect annotated Mimic HDF5 files:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/_inspect_annotated.py \
  datasets/air2_mimic_brush_annotated.hdf5
```

Note: `scripts/_inspect_hdf5.py` and `scripts/_inspect_demos_hdf5.py` are hardcoded one-off diagnostics on `main`; edit their path constants before using them.

Merge Mimic HDF5 files:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/merge_mimic_hdf5.py \
  --inputs datasets/a.hdf5 datasets/b.hdf5 \
  --output datasets/merged.hdf5
```

Run segmentation inference on a saved image or live sim:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/run_air2_segmentation_inference.py \
  --checkpoint checkpoints/air2_segmentation_unet.pth \
  --image path/to/collected_rgb_image.png \
  --mask-output eval_results/mask.png
```

Live inference:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/run_air2_segmentation_inference.py \
  --checkpoint checkpoints/air2_segmentation_unet.pth \
  --task Isaac-AIR2-Robotis-Franka-Segmentation-v0 \
  --camera main_camera \
  --enable_cameras
```

## 18. Windows Command Pattern

Most project scripts work on Windows through Isaac Lab's batch launcher:

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\<script_name>.py `
  --arg value
```

Examples:

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\eval_state_bc.py `
  --state_bc_ckpt checkpoints\policy_state_bc_brush.pth `
  --task Isaac-AIR2-Robotis-Franka-Brush-Play-v0 `
  --num_envs 1 `
  --num_episodes 20 `
  --max_steps 2000 `
  --episode_length_s 80.0
```

```powershell
& "D:\IsaacLab\isaaclab.bat" -p scripts\eval_multi_object_bc.py `
  --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 `
  --ckpt_dir checkpoints `
  --num_envs 1 `
  --num_rounds 3 `
  --max_steps_per_object 2000 `
  --enable_cameras
```

Start from a clean shell if HDF5 DLL errors appear:

```powershell
conda deactivate
```

## 19. Common Gotchas

- Always source `.env` before Linux launcher commands.
- Keep `ISAACLAB_PATH` pointing to the directory containing `isaaclab.sh` or `isaaclab.bat`.
- Keep this repo separate from Isaac Lab. `setup_isaaclab.sh` handles the symlinks.
- Re-run `./setup_isaaclab.sh` after pulling changes to `isaaclab_patches/`.
- Use `--enable_cameras` for camera-based collection and evaluation.
- For camera rendering on Linux, run on the display GPU. Forcing the wrong `CUDA_VISIBLE_DEVICES` can break camera rendering.
- Check `datasets/.../overlays/` after segmentation collection before training.
- Checkpoints and datasets are gitignored; share them separately from Git.
- `launch_air2.sh train-bc` is hardcoded to U-Net. Use the direct command if you want ResNet-18.
- The hook-based `Isaac-AIR2-Franka-*` env is WIP. Use the Robotis task family for stable collection and training.
