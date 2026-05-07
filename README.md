# ur_pick — UR3e + RG2 in Isaac Lab

Custom Isaac Lab task and robot configuration for the UR3e arm + OnRobot RG2 gripper,
plus the project's ROS2 / MoveIt scaffolding.

The Isaac Lab integration registers a custom gym task — `Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0`
and friends — that can be teleoperated, used to record demos, and used to train RL
or imitation-learning policies.

---

## Layout

| Dir | What |
|---|---|
| `scene/scene.usd` | source of truth for the robot+gripper scene (with ROS bridge graphs) |
| `scene/scene_isaaclab.usd` | generated; clean copy with single articulation root and no graphs |
| `scripts/` | one-off tools: USD inspector, scene fixer, gripper utilities |
| `isaaclab_ext/robots/ur3e_rg2.py` | `UR3E_RG2_CFG` and `UR3E_RG2_HIGH_PD_CFG` |
| `isaaclab_ext/tasks/lift_cube_ur3e_rg2/` | gym task variants (joint-pos, IK-Rel) |
| `setup_isaaclab.sh` | bootstrap: links `isaaclab_ext/` into a local IsaacLab clone |
| `inria_onrobot/`, `rg2_*/`, `ur3e_only/`, `ros2_ws/` | URDFs and ROS2 driver workspace |

---

## Prerequisites

These have to be installed **on each machine** before this repo is useful. They
are large and not in the zip.

### 1. Isaac Sim 4.5 (pip install)

Tested with `isaacsim==4.5.0.0` in a Python 3.10 venv:

```bash
python3.10 -m venv ~/isaacsim_env
source ~/isaacsim_env/bin/activate
pip install --upgrade pip "setuptools<81" wheel
pip install "isaacsim[all]==4.5.*" --extra-index-url https://pypi.nvidia.com
```

> The `setuptools<81` pin matters: 81+ removed `pkg_resources`, which Isaac Lab's
> build steps still need.

### 2. Isaac Lab v2.2.1 (matching tag for Isaac Sim 4.5)

```bash
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
source ~/isaacsim_env/bin/activate
./isaaclab.sh --install
pip install --editable source/isaaclab --no-build-isolation
```

Make Isaac Lab's `isaaclab.sh -p` wrapper use the venv's Python:

```bash
mkdir -p ~/IsaacLab/_isaac_sim
cat > ~/IsaacLab/_isaac_sim/python.sh <<'EOF'
#!/bin/bash
exec ~/isaacsim_env/bin/python "$@"
EOF
chmod +x ~/IsaacLab/_isaac_sim/python.sh
```

### 3. This repo

```bash
unzip ur_pick.zip            # or: git clone <url>
cd ur_pick
./setup_isaaclab.sh           # symlinks isaaclab_ext/ into ~/IsaacLab and builds scene_isaaclab.usd
```

If your IsaacLab clone lives somewhere other than `~/IsaacLab`:

```bash
ISAACLAB_PATH=/path/to/IsaacLab ./setup_isaaclab.sh
```

---

## Running

All commands below assume the prereqs above are done. They work from any cwd.

### Teleoperate the robot (keyboard)

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device keyboard
```

Click the viewport, then: `W A S D Q E` translate, `Z X T G V B` rotate, `K L` close/open gripper, `R` reset.

### Record demos for imitation learning

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/record_demos.py \
  --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --teleop_device keyboard \
  --dataset_file ./datasets/lift_cube_demos.hdf5
```

### Train PPO from scratch

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-UR3e-RG2-v0 --headless --num_envs 4096
```

### Train BC on collected demos

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --algo bc \
  --dataset ./datasets/lift_cube_demos.hdf5
```

---

## Editing the Isaac Lab integration

Edit files under `isaaclab_ext/` directly. Because `setup_isaaclab.sh` symlinks
those into `~/IsaacLab/source/...`, your changes are immediately picked up by
`isaaclab.sh -p`. No re-install needed.

Files of interest:

- [`isaaclab_ext/robots/ur3e_rg2.py`](isaaclab_ext/robots/ur3e_rg2.py) — actuator gains, armature, init pose, USD path
- [`isaaclab_ext/tasks/lift_cube_ur3e_rg2/joint_pos_env_cfg.py`](isaaclab_ext/tasks/lift_cube_ur3e_rg2/joint_pos_env_cfg.py) — gripper open/close values, EE-frame TCP offset, cube spawn
- [`isaaclab_ext/tasks/lift_cube_ur3e_rg2/ik_rel_env_cfg.py`](isaaclab_ext/tasks/lift_cube_ur3e_rg2/ik_rel_env_cfg.py) — IK controller scale, body offset
- [`isaaclab_ext/tasks/lift_cube_ur3e_rg2/agents/rsl_rl_ppo_cfg.py`](isaaclab_ext/tasks/lift_cube_ur3e_rg2/agents/rsl_rl_ppo_cfg.py) — PPO hyperparameters

---

## Real-robot deployment (after RL/IL training)

After training you'll have a checkpoint (`.pth` for BC, `.pt` for rsl_rl). To run
on the real UR3e+RG2:

1. Export the policy to TorchScript: `torch.jit.script(policy).save("policy.pt")`.
2. Write a small ROS2 node that:
   - Subscribes to `/joint_states` (and any other observation topics the policy used in training).
   - At ~50 Hz: builds the observation in the **same order** as in training, runs `policy(obs)`,
     publishes the resulting joint commands to your existing `ur_onrobot_control` controllers
     and the RG2 gripper topic.

Use MoveIt **outside** the policy as a "move to ready pose" / safety wrapper, not for the
trained skill itself — the policy is the controller now.
