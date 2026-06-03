# `ur_pick` — full rebuild from scratch

End-states this delivers:

- **A.** ROS2 / MoveIt → Isaac Sim demo: drag a marker in RViz, Plan & Execute, arm tracks in Isaac Sim. Use MoveIt's `open` / `closed` named gripper states to drive the gripper.
- **B.** Isaac Lab gym task: `Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0` (and `joint-pos` variant) registered for teleoperation, demo recording, and RL/IL training.

Path to A only is steps 1–6. Path to B adds steps 7–8 on top.

---

## Repo layout (target end-state)

```
ur_pick/
├── README.md                       # how to run (Isaac Lab side)
├── REBUILD_GUIDE.md                # this file (how to rebuild)
├── commands.txt                    # cheat-sheet of frequently-used commands
├── setup_isaaclab.sh               # bootstrap: links isaaclab_ext/ into IsaacLab + builds scene_isaaclab.usd
├── scene/
│   ├── scene.usd                   # source-of-truth (UR3e + gripper + ROS bridge graphs)
│   └── scene_isaaclab.usd          # generated, single-articulation, no graphs (Isaac Lab uses this)
├── scripts/                        # one-off USD + ROS scripts
│   ├── moveit_to_isaac_bridge.py
│   ├── fix_scene_for_isaaclab.py
│   ├── fix_mimic_limits.py
│   ├── fix_mimic_reference.py
│   ├── integrate_inria_gripper.py
│   ├── position_gripper_v2.py
│   ├── clean_articulations.py
│   ├── split_articulations.py
│   ├── fix_ur_articulation_v2.py
│   ├── add_gripper_artctrl.py
│   ├── bump_drives.py
│   ├── reset_drive_targets.py
│   ├── inspect_scene.py            # diagnostics
│   ├── inspect_graph_nodes.py
│   ├── list_joints.py
│   ├── dump_xforms.py
│   ├── verify_gripper_pipeline.py
│   ├── record_play.py
│   └── ... (debug fossils — safe to delete)
├── isaaclab_ext/
│   ├── README.md
│   ├── robots/ur3e_rg2.py          # UR3E_RG2_CFG, UR3E_RG2_HIGH_PD_CFG
│   └── tasks/lift_cube_ur3e_rg2/
│       ├── __init__.py             # gym registrations
│       ├── joint_pos_env_cfg.py
│       ├── ik_rel_env_cfg.py
│       └── agents/rsl_rl_ppo_cfg.py
├── ros2_ws/
│   ├── src/                        # cloned ROS2 repos here (see step 2)
│   ├── build/  install/  log/      # produced by colcon
│   └── ...
├── ur3e_only/                      # UR3e URDF + imported USD
│   ├── ur3e.urdf
│   └── ur3e/ur3e.usd
├── rg2_inria/                      # generated flat URDF + meshes for RG2
│   └── rg2_inria.urdf
├── rg2_inria_usd/
│   └── rg2_inria.usd               # imported gripper USD (referenced by scene.usd)
└── inria_onrobot/                  # cloned Inria onrobot_ros (just for the URDF)
```

---

## Step 0 — Machine prerequisites

(Already on this box; listed for fresh-machine rebuild.)

```bash
# Ubuntu 22.04 + NVIDIA driver + CUDA-capable GPU
# ROS2 Humble:
sudo apt install ros-humble-desktop ros-humble-moveit ros-humble-moveit-task-constructor \
                 ros-humble-vision-msgs python3-colcon-common-extensions
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

**Isaac Sim 4.5 (pip in venv):**

```bash
python3.10 -m venv ~/isaacsim_env
source ~/isaacsim_env/bin/activate
pip install --upgrade pip "setuptools<81" wheel
pip install "isaacsim[all]==4.5.*" --extra-index-url https://pypi.nvidia.com
```

The `setuptools<81` pin matters (81+ removed `pkg_resources` which Isaac Lab still needs).

**Isaac Lab v2.2.1** (matching tag for Isaac Sim 4.5):

```bash
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
source ~/isaacsim_env/bin/activate
./isaaclab.sh --install
pip install --editable source/isaaclab --no-build-isolation

# Make Isaac Lab's wrapper use the venv:
mkdir -p ~/IsaacLab/_isaac_sim
cat > ~/IsaacLab/_isaac_sim/python.sh <<'EOF'
#!/bin/bash
exec ~/isaacsim_env/bin/python "$@"
EOF
chmod +x ~/IsaacLab/_isaac_sim/python.sh
```

---

## Step 1 — Get the project tree

```bash
cd ~/Desktop
git clone <your-repo-url> ur_pick
# or unzip ur_pick.zip
cd ur_pick
```

---

## Step 2 — Build the ROS 2 workspace

This pulls in `tonydle/UR_OnRobot_ROS2` (UR + RG2 control + MoveIt config) and `tonydle/moveit2_tutorials_ur_onrobot` (RViz tutorials), plus the OnRobot driver + Modbus deps via `vcs`.

```bash
mkdir -p ros2_ws/src && cd ros2_ws/src
git clone https://github.com/tonydle/UR_OnRobot_ROS2.git
git clone https://github.com/tonydle/moveit2_tutorials_ur_onrobot.git
cd ..
vcs import src --input src/UR_OnRobot_ROS2/required.repos --recursive
sudo apt install -y libnet1-dev   # Modbus dep for the OnRobot driver
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

Smoke test (no Isaac yet):

```bash
source install/setup.bash
ros2 launch ur_onrobot_control start_robot.launch.py \
    ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false
```

You should see all controllers `Configured and activated` with no `[ERROR]`.

---

## Step 3 — Build the gripper USD

The original tonydle URDF for the RG2 has a closed-chain 4-bar parallelogram which **PhysX 4.5 articulations don't support**. So we use the **Inria** URDF, which fakes parallel motion with a tree of `<mimic>` joints (no closed loop) and is friendly to the URDF importer.

### 3.1 Flatten the Inria URDF with absolute mesh paths

```bash
cd ~/Desktop/ur_pick
git clone --depth 1 https://github.com/inria-paris-robotics-lab/onrobot_ros.git inria_onrobot

INRIA_SRC=$PWD/inria_onrobot/onrobot_description
WORK=$PWD/rg2_inria
rm -rf "$WORK" && mkdir -p "$WORK"
cp -r "$INRIA_SRC/urdf"   "$WORK/"
cp -r "$INRIA_SRC/config" "$WORK/"
cp -r "$INRIA_SRC/meshes" "$WORK/"

# substitute $(find onrobot_description) with the absolute work-copy path
find "$WORK" -name '*.xacro' -exec sed -i "s|\$(find onrobot_description)|$WORK|g" {} \;

# strip the Gazebo include/invocation (depends on a missing onrobot_control package)
sed -i '/onrobot_rg.gazebo.xacro/d' "$WORK/urdf/onrobot_rg.urdf.xacro"
sed -i '/<!-- si standalone alors/,/<\/xacro:unless>/d' "$WORK/urdf/onrobot_rg.urdf.xacro"

source /opt/ros/humble/setup.bash
xacro "$WORK/urdf/test.urdf.xacro" model:=rg2_v1 > "$WORK/rg2_inria.urdf"
```

### 3.2 Import into Isaac Sim

```bash
source /opt/ros/humble/setup.bash
~/isaacsim_env/bin/isaacsim isaacsim.exp.full
```

In Isaac Sim:

1. **File → Import → URDF** → `~/Desktop/ur_pick/rg2_inria/rg2_inria.urdf`.
2. Settings:
   - Model → **Create in Stage** ✓, **Set as Default Prim** ✓, **Clear Stage on Import** ✓.
   - Links → **Static Base** ✓.
   - Joints & Drives → **Ignore Mimic** ❌ (leave UNCHECKED). Joint Configuration: **Natural Frequency**. Drive Type: **Force**.
   - Colliders → **Convex Decomposition** (fingertips are concave; Hull will give terrible contacts).
3. Import. Press Play briefly and read the console — you'll see two PhysX errors:
   - `revolute joint ... needs a finite limit set to be used by the mimic joint feature`
   - `PhysxMimicJointAPI ... must have exactly 1 "referenceJoint" relationship defined`
4. Stop Play. In Script Editor, run [_archive/debug_scripts/fix_mimic_limits.py](_archive/debug_scripts/fix_mimic_limits.py), then [_archive/debug_scripts/fix_mimic_reference.py](_archive/debug_scripts/fix_mimic_reference.py). Save the stage.
5. Press Play again. Errors should be gone. Verify by selecting `rg2_gripper_joint` in the Stage panel → Property panel → Drive (angular) → set **Target Position** to 50° → gripper closes; back to 0° → opens.
6. **File → Save As** → `~/Desktop/ur_pick/rg2_inria_usd/rg2_inria.usd`.

---

## Step 4 — Build the UR3e USD

Same import flow as the gripper, but with the UR3e URDF you already have at `ur3e_only/ur3e.urdf`.

1. **File → Import → URDF** → `~/Desktop/ur_pick/ur3e_only/ur3e.urdf`.
2. Same settings. **Static Base** ✓, **Convex Decomposition** for colliders.
3. Save the imported USD to `~/Desktop/ur_pick/ur3e_only/ur3e/ur3e.usd`.

---

## Step 5 — Build `scene/scene.usd` (UR3e + gripper + ROS bridge)

Open Isaac Sim, **File → New** to start from an empty stage.

### 5.1 References

In Stage panel, add references:
- `/World/ur3e` ← Xform with reference to `ur3e_only/ur3e/ur3e.usd`.
- `/World/rg2`  ← Xform with reference to `rg2_inria_usd/rg2_inria.usd`.

Save the empty-but-referencing stage to `~/Desktop/ur_pick/scene/scene.usd`.

### 5.2 Run the integration scripts (in this order)

In Script Editor, run each, **Ctrl+S after each** (or just at the end):

| script | what it does |
|---|---|
| [_archive/debug_scripts/integrate_inria_gripper.py](_archive/debug_scripts/integrate_inria_gripper.py) | drops the redundant `ArticulationRootAPI` from inside the gripper, creates `/World/wrist_to_rg2` fixed joint pinning bracket → `/World/ur3e/wrist_3_link` |
| [_archive/debug_scripts/position_gripper_v2.py](_archive/debug_scripts/position_gripper_v2.py) | sets `/World/rg2`'s transform so the bracket lands exactly on the wrist's rest pose (no startup impulse) |
| [_archive/debug_scripts/clean_articulations.py](_archive/debug_scripts/clean_articulations.py) | deactivates the gripper's internal `root_joint` (pinned the bracket to its import-time world position; conflicts with the wrist mount) |
| [_archive/debug_scripts/split_articulations.py](_archive/debug_scripts/split_articulations.py) | re-applies `ArticulationRootAPI` to `/World/rg2` so gripper is its OWN articulation; wrist mount is then an inter-articulation constraint, which PhysX handles cleanly |
| [_archive/debug_scripts/fix_ur_articulation_v2.py](_archive/debug_scripts/fix_ur_articulation_v2.py) | moves UR's `ArticulationRootAPI` from the legacy `root_joint` prim to `/World/ur3e/base_link` (where Isaac actually finds it), updates OmniGraph node target paths |
| [_archive/debug_scripts/bump_drives.py](_archive/debug_scripts/bump_drives.py) | bumps UR3e arm-joint drive stiffness/damping (the URDF importer's defaults are far too weak for a real load) |
| [scripts/reset_drive_targets.py](scripts/reset_drive_targets.py) | copies each UR joint's *current* angle into its drive target, so on Play the drives don't yank the arm to zero |

Save the stage. Press Play. The arm should sit still at rest pose; the gripper should hang quietly on the wrist.

### 5.3 Wire the OmniGraph (ROS2 bridge)

Create an Action Graph at `/World/RosBridgeGraph` with these nodes:

| node | type | key inputs |
|---|---|---|
| `Tick` | `omni.graph.action.OnPlaybackTick` | — |
| `ReadSimTime` | `isaacsim.core.nodes.IsaacReadSimulationTime` | — |
| `ros2_context` | `isaacsim.ros2.bridge.ROS2Context` | — |
| `PubJS` | `isaacsim.ros2.bridge.ROS2PublishJointState` | topicName=`isaac_joint_states`, targetPrim=`/World/ur3e/base_link` |
| `SubJS` | `isaacsim.ros2.bridge.ROS2SubscribeJointState` | topicName=`isaac_joint_commands` |
| `ArtCtrl` | `isaacsim.core.nodes.IsaacArticulationController` | targetPrim=`/World/ur3e/base_link`, robotPath=`/World/ur3e/base_link` |

Connections:
- `Tick.outputs:tick` → `PubJS.inputs:execIn` AND `SubJS.inputs:execIn`
- `ros2_context.outputs:context` → `PubJS.inputs:context` AND `SubJS.inputs:context`
- `ReadSimTime.outputs:simulationTime` → `PubJS.inputs:timeStamp`
- `SubJS.outputs:execOut` → `ArtCtrl.inputs:execIn`
- `SubJS.outputs:positionCommand/velocityCommand/effortCommand/jointNames` → corresponding `ArtCtrl.inputs:*`

For the gripper, run [_archive/debug_scripts/add_gripper_artctrl.py](_archive/debug_scripts/add_gripper_artctrl.py) — it adds a second `IsaacArticulationController` named `ArtCtrlGripper` targeting `/World/rg2`, wired to the same SubJS.

Save. Press Play.

---

## Step 6 — Run the demo (path A)

In each terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/ur_pick/ros2_ws/install/setup.bash
```

Then:

| | command |
|---|---|
| Isaac Sim | `~/isaacsim_env/bin/isaacsim isaacsim.exp.full` → File → Open `scene/scene.usd` → ▶ Play |
| T1 | `ros2 launch ur_onrobot_control start_robot.launch.py ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false` |
| T2 | `ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py ur_type:=ur3e onrobot_type:=rg2` |
| T3 | `python3 ~/Desktop/ur_pick/scripts/moveit_to_isaac_bridge.py` |
| RViz | drag the orange end-effector marker → Plan & Execute (arm). Switch Planning Group to gripper, Goal State `closed`/`open` → Plan & Execute. |

If nothing moves: `ros2 topic echo /isaac_joint_commands` should be printing JointState messages with all 6 arm joints + `rg2_gripper_joint`. If not, the bridge isn't seeing `finger_width` in `/joint_states`.

---

## Step 7 — Isaac Lab integration (path B)

Path A's `scene.usd` has TWO articulations (UR + gripper) and ROS bridge graphs. Isaac Lab needs **one** articulation rooted at `/World/ur3e/base_link` and **no** OmniGraphs (their legacy `setDriveTarget` calls fight Isaac Lab's GPU pipeline).

[scripts/fix_scene_for_isaaclab.py](scripts/fix_scene_for_isaaclab.py) does both transformations and produces `scene/scene_isaaclab.usd`. The wrapper `setup_isaaclab.sh` runs it for you and links `isaaclab_ext/` into your IsaacLab clone:

```bash
cd ~/Desktop/ur_pick
./setup_isaaclab.sh                       # uses ~/IsaacLab
# or: ISAACLAB_PATH=/path/to/IsaacLab ./setup_isaaclab.sh
```

What that script does:
- `ln -sfn isaaclab_ext/robots/ur3e_rg2.py ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/ur3e_rg2.py`
- `ln -sfn isaaclab_ext/tasks/lift_cube_ur3e_rg2 ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur3e_rg2`
- generates `scene/scene_isaaclab.usd` if missing

After that the gym IDs `Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0`, `Isaac-Lift-Cube-UR3e-RG2-v0`, etc. are auto-discovered by IsaacLab on next launch.

---

## Step 8 — Run a demo task

```bash
# Teleoperate (keyboard)
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device keyboard
# Click the viewport, then: WASDQE translate, ZXTGVB rotate, K/L close/open gripper, R reset.

# Record demos for IL
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/record_demos.py \
  --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --teleop_device keyboard \
  --dataset_file ./datasets/lift_cube_demos.hdf5

# Train PPO
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-UR3e-RG2-v0 --headless --num_envs 4096

# Train BC on collected demos
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --algo bc \
  --dataset ./datasets/lift_cube_demos.hdf5
```

---

## Step 9 — VR teleop install (optional, path B extension)

This step adds **Meta Quest 2** teleop on top of path B. The task configs at `isaaclab_ext/tasks/lift_*_ur3e_rg2/ik_rel_env_cfg.py` already declare a `handtracking` teleop device using Isaac Lab's shipped `OpenXRDevice` + `Se3RelRetargeter` + `GripperRetargeter` — Step 9 is just installing the runtime side so the headset can reach Isaac Sim.

> **Verify install commands against the current upstream docs.** Monado and WiVRn move fast; package names and the WiVRn build dep list change between releases. Treat this section as a roadmap, not a 1:1 spec.

### 9.0 Hardware floor

| Component | Minimum | Comfortable |
|---|---|---|
| GPU | RTX 3070 (8 GB) | RTX 3080+ (12 GB+) |
| RAM | 16 GB | 32 GB |
| Wi-Fi | Dual-band 5 GHz AP | Dedicated Wi-Fi 6/6E AP |
| Alternative to Wi-Fi | USB-C tether (10 ft fiber) | — |

**RTX 3050 8 GB is below Isaac Sim's minimum spec before VR adds its load.** Expect 10–20 FPS at the headset on a 3050 — the pipeline works but isn't comfortable. Render-side bottleneck, not a code fix.

### 9.1 Install Monado (OpenXR runtime)

```bash
# Build deps + Monado from apt (Ubuntu 22.04 universe):
sudo apt install -y \
  build-essential cmake ninja-build pkg-config git \
  libsdl2-dev libgl-dev libvulkan-dev libxcb-randr0-dev \
  libavcodec-dev libavutil-dev libswscale-dev libusb-1.0-0-dev \
  libudev-dev libv4l-dev libopencv-dev libeigen3-dev \
  libbluetooth-dev libhidapi-dev libwayland-dev libxkbcommon-dev \
  libuvc-dev libjpeg-dev

# Monado from source (ppa often lags upstream):
git clone https://gitlab.freedesktop.org/monado/monado.git ~/monado
cd ~/monado
cmake -B build -G Ninja
ninja -C build
sudo ninja -C build install

# Point OpenXR loader at Monado:
mkdir -p ~/.config/openxr/1
ln -sfn /usr/local/share/openxr/1/openxr_monado.json \
        ~/.config/openxr/1/active_runtime.json
```

Smoke test the runtime alone:

```bash
monado-service &           # background OpenXR service
xrgears                    # or any OpenXR sample
```

If `xrgears` complains about a missing runtime, the symlink in `~/.config/openxr/1` is wrong.

### 9.2 Build WiVRn server (Linux → Quest streamer)

```bash
sudo apt install -y \
  libavfilter-dev libavdevice-dev libopus-dev libx264-dev \
  libpipewire-0.3-dev libsystemd-dev nlohmann-json3-dev

git clone https://github.com/WiVRn/WiVRn.git ~/WiVRn
cd ~/WiVRn
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -C build
```

The server binary is at `~/WiVRn/build/server/wivrn-server`. Run it (foreground for the first test):

```bash
~/WiVRn/build/server/wivrn-server
```

It will listen on the LAN and advertise itself via mDNS so the Quest client can discover it.

### 9.3 Quest 2 — developer mode + WiVRn client APK

1. **Enable developer mode** on the Quest 2:
   - Create a Meta developer account at `developer.oculus.com` (free).
   - In the Meta Quest mobile app on your phone → Devices → your Quest → Developer Mode → ON.
   - Reboot the headset.

2. **Sideload the WiVRn client APK**:
   - Download the latest `WiVRn-client-*.apk` from the [WiVRn releases page](https://github.com/WiVRn/WiVRn/releases).
   - Install **SideQuest** on Linux (AppImage from `sidequestvr.com`).
   - Plug Quest in via USB-C, accept the "Allow USB debugging" prompt **inside the headset**.
   - In SideQuest: "Install APK file from folder on computer" → pick the downloaded APK.

   Alternative (no SideQuest):
   ```bash
   sudo apt install -y adb
   adb devices                                  # should show your Quest
   adb install ~/Downloads/WiVRn-client-*.apk
   ```

3. **Pair Quest to the WiVRn server**:
   - Put the headset on, find the WiVRn client in `Apps → Unknown Sources`.
   - Launch it. It should auto-discover the Linux server on the LAN. Confirm pairing.

### 9.4 Validation gates (run in this order — don't skip)

| Gate | Command | Pass criterion |
|---|---|---|
| **9.4.a** OpenXR + Quest alone | Launch WiVRn client on Quest, run `xrgears` on Linux | See the gears spinning in the headset |
| **9.4.b** Isaac Sim XR alone | `~/isaacsim_env/bin/isaacsim isaacsim.exp.full` → Window → Extensions → enable `omni.kit.xr.core` → File → Open `scene/scene.usd` → Play | See your scene in the headset, no robot control |
| **9.4.c** Full VR teleop | The two commands below | Right wrist moves the EE; pinch closes RG2 |

**If gate 9.4.a fails**, the issue is Monado / WiVRn / Wi-Fi — not Isaac Lab. Debug there before touching Isaac Sim.
**If gate 9.4.b fails**, the issue is Isaac Sim's XR extension or GPU. Skip 9.4.c until 9.4.b passes.

### 9.5 Run VR teleop

```bash
# Cube task — VR teleop
cd ~/IsaacLab && ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device handtracking

# Pegboard task — VR teleop
cd ~/IsaacLab && ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device handtracking

# Record VR demos to HDF5
cd ~/IsaacLab && ./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Lift-Pegboard-UR3e-RG2-IK-Rel-v0 --teleop_device handtracking \
    --dataset_file ~/Desktop/ur_pick/datasets/lift_pegboard_vr.hdf5
```

Controls (bare-hand tracking, no Quest controllers used):

| input | action |
|---|---|
| right wrist position | EE translation |
| right wrist yaw | EE yaw (roll/pitch zeroed for stability) |
| pinch thumb + index < 3 cm | gripper close |
| open thumb + index > 5 cm | gripper open |
| WiVRn/dashboard "start" event | begin teleop (default state: paused) |
| WiVRn/dashboard "reset" event | reset env |

### 9.6 Tuning — when behavior feels wrong

Edit the `teleop_devices` block in `isaaclab_ext/tasks/lift_*_ur3e_rg2/ik_rel_env_cfg.py`:

| Symptom | Field to change |
|---|---|
| EE moves too fast / overshoots | `delta_pos_scale_factor` down (default 5.0 — upstream Franka uses 10.0) |
| EE rotation too twitchy | `delta_rot_scale_factor` down |
| Gripper toggles rapidly when fingers half-pinched | adjust `GRIPPER_CLOSE_METERS` / `GRIPPER_OPEN_METERS` (need to subclass `GripperRetargeter` — they're class constants) |
| Operator spawns inside the robot / behind the table | edit `self.xr = XrCfg(anchor_pos=...)` |
| Want roll/pitch control | set `zero_out_xy_rotation=False` (less stable, more expressive) |
| Want fingertip pinch pose instead of wrist | set `use_wrist_position=False` and `use_wrist_rotation=False` |

### 9.7 Known issues specific to VR

- **Hand tracking needs light.** Quest 2's hand tracking is optical; in a dim room your hand will dropout and the robot will freeze on the last pose. The device class doesn't crash, it just stops emitting deltas.
- **No controller fallback in shipped device.** Isaac Lab's `OpenXRDevice` reads `/user/hand/left` and `/user/hand/right` (hand-tracking endpoints). To use trigger/grip from physical controllers instead, subclass it and read from `/user/hand/left/input/trigger/value` etc. — out of scope here.
- **Wi-Fi latency above ~30 ms is unusable.** Tether via USB-C if your Wi-Fi adds >50 ms RTT to the Quest.
- **The 3050 will be slow.** Said elsewhere. Not a bug.

---

## Troubleshooting (setup-time crashes)

### A. Isaac Sim segfaults right after `app ready` (NVIDIA driver 595)

Symptom — Isaac Sim window opens for a second, then dies with:

```
app ready
[Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit buffer size ...
Segmentation fault (core dumped)
```

Cause — NVIDIA driver **595.x** is not compatible with Isaac Sim 4.5's RTX scene-DB plugin. Isaac Sim 4.5 is validated against the 535 / 550 / **580** line.

Fix — downgrade to driver 580:

```bash
nvidia-smi                                # check current version
sudo apt -s install nvidia-driver-580     # dry-run first
sudo apt install nvidia-driver-580
sudo reboot
nvidia-smi                                # should now show Driver Version: 580.x
```

Then re-launch Isaac Sim; the segfault should be gone.

### B. `Failed to acquire interface: omni::physx::IPhysxFabric` (Fabric extension version mismatch)

Symptom — environment creation fails with:

```
Pulling extension: omni.physx.fabric-106.3.2
Interface: [omni::physx::IPhysxPrivate v0.2] requested but
default plugin [omni::physx::IPhysxPrivate v1.2] cannot provide requested version
RuntimeError: Failed to acquire interface: omni::physx::IPhysxFabric
Failed to create environment
```

Cause — two versions of `omni.physx.fabric` ended up in the Kit extension cache (typically when both `pip install isaacsim[all]==4.5.*` and `./isaaclab.sh --install` populate caches). The older **106.3.2** gets pulled first and tries to talk to **106.5.x** PhysX core → ABI mismatch.

Fix — delete the stale 106.3.2 extension from both caches so only 106.5.3 remains.

1. Find what's installed:

   ```bash
   find ~/.local/share/ov/data/exts/v2 -maxdepth 1 -type d -name "omni.physx.fabric*" -print
   find ~/isaacsim_env/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/exts/3 \
     -maxdepth 1 -name "omni.physx.fabric*" -ls
   ```

2. If you see **both** `106.3.2` and `106.5.3`, remove the 106.3.2 dirs (paths shown with the actual suffixes used by Kit):

   ```bash
   rm -rf \
     ~/isaacsim_env/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/exts/3/omni.physx.fabric-106.3.2+106.3.0.lx64.r.cp310.ub3f \
     ~/.local/share/ov/data/exts/v2/omni.physx.fabric-106.3.2+106.3.0.lx64.r.cp310.ub3f
   ```

3. Verify only 106.5.3 remains:

   ```bash
   find ~/isaacsim_env/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/exts/3 \
     -maxdepth 1 -name "omni.physx.fabric*" -ls
   # expected:  omni.physx.fabric-106.5.3+106.5.0.lx64.r.cp310.ub3f
   ```

4. Re-run the task:

   ```bash
   cd ~/Desktop/ur_pick
   ~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
     --task Isaac-Lift-Cube-UR3e-RG2-IK-Rel-v0 --num_envs 1 --teleop_device keyboard
   ```

---

## Known limitations / gotchas

- **Gripper close is cosmetically imperfect.** The Inria URDF doesn't have the real RG2 closed-chain 4-bar; it fakes parallel motion with single-axis mimics. Fingertips don't quite stay parallel at the close extreme. Functionally fine — pads can grip any object of finite width.
- **Per-joint drive gains in URDF are tiny.** `bump_drives.py` overrides them; if you change kinematics/payload, retune.
- **Two articulations in `scene.usd`, one in `scene_isaaclab.usd`.** Path A needs two so the wrist mount fixed joint becomes a stable inter-articulation constraint. Path B (Isaac Lab) collapses to one because `fix_scene_for_isaaclab.py` strips the gripper's `ArticulationRootAPI`. Both are correct for their pipeline; **don't open `scene_isaaclab.usd` in path A** — it lacks the OmniGraph and won't accept ROS commands.
- **`PhysxMimicJointAPI` is fragile.** When the URDF importer applies it, finite joint limits and a `referenceJoint` relationship are *required*; older importer versions sometimes leave them out (hence `fix_mimic_limits.py` + `fix_mimic_reference.py`). If you reimport the gripper from a newer Isaac Sim, re-run those after import.

---

## Throwaway debugging scripts

These were dead-ends or one-off diagnostics; safe to delete from `scripts/`:

`add_loop_closure.py`, `cleanup_scene.py`, `configure_graph.py`, `detach_gripper.py`, `disable_mimic_use_drives.py`, `disable_rosbridge_graph.py`, `find_and_zero_offset.py`, `fix_gripper_integration.py`, `fix_ur_articulation.py` (v1; v2 supersedes), `flip_finger_tip_mimic.py`, `inspect_scene_cli.py`, `move_gripper_to_origin.py`, `position_gripper.py` (v1; v2 supersedes), `remove_loop_closures.py`, `revert_to_mimic.py`, `rollback_mimic.py`, `tune_gripper_drives.py`.

Plus all the `*_report.txt`, `*_log.txt`, `*.json` artifacts under `scripts/`.

---

## In one sentence

> Build a ROS2 workspace from `tonydle/UR_OnRobot_ROS2`, import the **Inria** RG2 URDF and your UR3e URDF into Isaac Sim, glue them together as two articulations connected by a fixed wrist-mount joint with the right `ArticulationRootAPI` placement and drive tuning, add a six-node ROS2 OmniGraph plus a per-articulation `IsaacArticulationController` for the gripper, run `moveit_to_isaac_bridge.py` to translate MoveIt's `/joint_states` into `/isaac_joint_commands`, and optionally strip the graph + collapse to a single articulation via `fix_scene_for_isaaclab.py` to feed Isaac Lab's gym task system.
