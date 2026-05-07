# `AI_Robot_Manipulation_UR3e` — UR3e + RG2 in Isaac Sim / Isaac Lab

UR3e arm + OnRobot RG2 gripper, set up for **two** pipelines that share the same scene:

- **A. ROS2 / MoveIt → Isaac Sim.** Drag the marker in RViz, click *Plan & Execute*, the arm tracks in Isaac Sim. Use MoveIt's `open` / `closed` named gripper states to drive the gripper. Bridge node ([scripts/moveit_to_isaac_bridge.py](scripts/moveit_to_isaac_bridge.py)) translates `/joint_states` ↔ `/isaac_joint_commands`.
- **B. Isaac Lab gym task.** Custom registrations: `Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0`, `Isaac-Lift-Cube-UR3e-RG2-v0`. Teleoperate, record demos, train RL (PPO) or imitation learning (BC).

> **First-time setup or rebuilding from scratch?** See [REBUILD_GUIDE.md](REBUILD_GUIDE.md) — it walks the entire path from a fresh OS install to a working demo, including how the ROS workspace is built and how the gripper USD is generated. This README is the **operate-it** doc; that one is the **build-it** doc.

---

## Layout

| Dir | What |
|---|---|
| `scene/scene.usd` | source-of-truth (UR3e + gripper + ROS bridge OmniGraph) |
| `scene/scene_isaaclab.usd` | generated; single articulation, no graphs (Isaac Lab uses this) |
| `scripts/` | scene-fixing + bridge + diagnostic Python scripts |
| `scripts/moveit_to_isaac_bridge.py` | the runtime bridge node for path A |
| `scripts/fix_scene_for_isaaclab.py` | converts `scene.usd` → `scene_isaaclab.usd` |
| `isaaclab_ext/robots/ur3e_rg2.py` | `UR3E_RG2_CFG` and `UR3E_RG2_HIGH_PD_CFG` |
| `isaaclab_ext/tasks/lift_cube_ur3e_rg2/` | gym task variants (joint-pos, IK-Rel) |
| `setup_isaaclab.sh` | bootstrap: links `isaaclab_ext/` into a local IsaacLab clone + builds `scene_isaaclab.usd` |
| `ros2_ws/` | ROS2 workspace (`tonydle/UR_OnRobot_ROS2` + drivers) — `colcon build`-able |
| `inria_onrobot/`, `rg2_inria/`, `rg2_inria_usd/` | Inria RG2 URDF + flattened URDF + imported USD |
| `ur3e_only/` | UR3e URDF + imported USD |
| `commands.txt` | quick-reference cheat-sheet of run commands |

---

## Prerequisites

These are **not** in the repo (large + machine-specific). Install once per machine.

### 1. ROS2 Humble + MoveIt

```bash
sudo apt install ros-humble-desktop ros-humble-moveit ros-humble-moveit-task-constructor \
                 ros-humble-vision-msgs python3-colcon-common-extensions ros-humble-topic-based-ros2-control
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 2. Isaac Sim 4.5 (pip)

```bash
python3.10 -m venv ~/isaacsim_env
source ~/isaacsim_env/bin/activate
pip install --upgrade pip "setuptools<81" wheel
pip install "isaacsim[all]==4.5.*" --extra-index-url https://pypi.nvidia.com
```

> The `setuptools<81` pin matters: 81+ removed `pkg_resources`, which Isaac Lab still needs.

### 3. Isaac Lab v2.2.1 (matching tag for Isaac Sim 4.5) — only needed for path B

```bash
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
source ~/isaacsim_env/bin/activate
./isaaclab.sh --install
pip install --editable source/isaaclab --no-build-isolation

# Make Isaac Lab's `isaaclab.sh -p` wrapper use the venv's Python:
mkdir -p ~/IsaacLab/_isaac_sim
cat > ~/IsaacLab/_isaac_sim/python.sh <<'EOF'
#!/bin/bash
exec ~/isaacsim_env/bin/python "$@"
EOF
chmod +x ~/IsaacLab/_isaac_sim/python.sh
```

### 4. This repo

```bash
git clone git@github.com:Kaung-dev/AI_Robot_Manipulation_UR3e.git
cd AI_Robot_Manipulation_UR3e
```

Then build the ROS2 workspace (one-time):

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
vcs import src --input src/UR_OnRobot_ROS2/required.repos --recursive   # if a fresh clone
sudo apt install -y libnet1-dev    # for the OnRobot Modbus driver
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
cd ..
```

For Isaac Lab integration (only needed for path B):

```bash
./setup_isaaclab.sh                              # uses ~/IsaacLab by default
# or:
ISAACLAB_PATH=/path/to/IsaacLab ./setup_isaaclab.sh
```

That symlinks `isaaclab_ext/robots/ur3e_rg2.py` and `isaaclab_ext/tasks/lift_cube_ur3e_rg2/` into your IsaacLab clone, and produces `scene/scene_isaaclab.usd` from `scene/scene.usd` if missing.

---

## Path A — ROS2 / MoveIt → Isaac Sim

### Run

In each new terminal first:

```bash
cd ~/Desktop/AI_Robot_Manipulation_UR3e   # wherever you cloned
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

| | command |
|---|---|
| Isaac Sim | `~/isaacsim_env/bin/isaacsim isaacsim.exp.full` → File → Open `scene/scene.usd` → ▶ Play |
| T1 | `ros2 launch ur_onrobot_control start_robot.launch.py ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false` |
| T2 | `ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py ur_type:=ur3e onrobot_type:=rg2` |
| T3 | `python3 scripts/moveit_to_isaac_bridge.py` |

In RViz: drag the orange end-effector marker → Plan & Execute (arm). Switch Planning Group to gripper, Goal State `closed` / `open` → Plan & Execute.

If nothing moves: `ros2 topic echo /isaac_joint_commands` should be printing JointState messages with all 6 arm joints + `rg2_gripper_joint`.

---

## Path B — Isaac Lab tasks

### Teleoperate

The same `teleop_se3_agent.py` script supports keyboard, gamepad, and SpaceMouse — pick one with `--teleop_device`.

**Keyboard:**

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device keyboard
```

| key | action |
|---|---|
| `W A S D Q E` | translate |
| `Z X T G V B` | rotate |
| `K` / `L` | close / open gripper |
| `R` | reset |

**Gamepad** (Xbox / PS-style controller, USB or Bluetooth):

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device gamepad
```

| input | action |
|---|---|
| Left stick | translate X / Y |
| Right stick up/down | translate Z |
| Right stick left/right | rotate Z (yaw) |
| D-pad left/right | rotate X (roll) |
| D-pad up/down | rotate Y (pitch) |
| `X` button | toggle gripper open/close |

Plug the controller in *before* launching, click the Isaac Sim viewport so it has input focus, then move sticks. Verify Linux sees it with `ls /dev/input/js0`. The connection runs through Omniverse's Carb input layer ([`Se3Gamepad`](../IsaacLab/source/isaaclab/isaaclab/devices/gamepad/se3_gamepad.py)) — no extra ROS / `joy_node` needed.

### Record demos for IL

Both teleop devices work for demonstration capture — swap `--teleop_device keyboard` for `gamepad` to use the controller. The gamepad is usually faster and produces smoother trajectories than keyboard for IL data.

```bash
mkdir -p datasets

# keyboard:
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --teleop_device keyboard \
    --dataset_file ./datasets/lift_cube_demos.hdf5

# gamepad:
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --teleop_device gamepad \
    --dataset_file ./datasets/lift_cube_demos.hdf5
```

Each successful episode is appended to the HDF5 file. Press the reset key (`R` on keyboard) between attempts. The resulting dataset feeds straight into the BC training command below.

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

Edit files under `isaaclab_ext/` directly. Because `setup_isaaclab.sh` symlinks those into `~/IsaacLab/source/...`, your changes are picked up immediately by `isaaclab.sh -p` — no re-install.

Files of interest:

- [`isaaclab_ext/robots/ur3e_rg2.py`](isaaclab_ext/robots/ur3e_rg2.py) — actuator gains, armature, init pose, USD path.
- [`isaaclab_ext/tasks/lift_cube_ur3e_rg2/joint_pos_env_cfg.py`](isaaclab_ext/tasks/lift_cube_ur3e_rg2/joint_pos_env_cfg.py) — gripper open/close values, EE-frame TCP offset, cube spawn.
- [`isaaclab_ext/tasks/lift_cube_ur3e_rg2/ik_rel_env_cfg.py`](isaaclab_ext/tasks/lift_cube_ur3e_rg2/ik_rel_env_cfg.py) — IK controller scale, body offset.
- [`isaaclab_ext/tasks/lift_cube_ur3e_rg2/agents/rsl_rl_ppo_cfg.py`](isaaclab_ext/tasks/lift_cube_ur3e_rg2/agents/rsl_rl_ppo_cfg.py) — PPO hyperparameters.

---

## Real-robot deployment (after RL/IL training)

After training you'll have a checkpoint (`.pth` for BC, `.pt` for rsl_rl). To run on the real UR3e + RG2:

1. Export the policy to TorchScript: `torch.jit.script(policy).save("policy.pt")`.
2. Write a small ROS2 node that:
   - Subscribes to `/joint_states` (and any other observation topics the policy used in training).
   - At ~50 Hz: builds the observation in the **same order** as in training, runs `policy(obs)`, publishes the resulting joint commands to your existing `ur_onrobot_control` controllers and the RG2 gripper topic.

Use MoveIt **outside** the policy as a "move to ready pose" / safety wrapper, not for the trained skill itself — the policy is the controller now.

---

## Known limitations

- **Gripper close is cosmetically imperfect.** The Inria URDF doesn't have the real RG2 closed-chain 4-bar; it fakes parallel motion with single-axis mimics. Fingertips don't quite stay parallel at the close extreme. Functionally fine — pads can grip any object of finite width. See REBUILD_GUIDE.md *Known limitations* for details and the alternatives we tried.
- **Two articulations in `scene.usd`, one in `scene_isaaclab.usd`.** Path A needs two so the wrist-mount fixed joint becomes a stable inter-articulation constraint. Path B collapses to one (Isaac Lab's GPU pipeline forbids closed loops). Both are correct for their pipeline; **don't open `scene_isaaclab.usd` in path A** — it lacks the OmniGraph and won't accept ROS commands.
- **Per-joint drive gains in URDF are tiny** (URDF inertials are placeholder ~1 g). [`scripts/bump_drives.py`](scripts/bump_drives.py) overrides them. Retune if you change kinematics or payload.

---

## Quick command reference

See [`commands.txt`](commands.txt) for a copy-pasteable cheat-sheet.
