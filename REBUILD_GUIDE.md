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
4. Stop Play. In Script Editor, run [scripts/fix_mimic_limits.py](scripts/fix_mimic_limits.py), then [scripts/fix_mimic_reference.py](scripts/fix_mimic_reference.py). Save the stage.
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
| [scripts/integrate_inria_gripper.py](scripts/integrate_inria_gripper.py) | drops the redundant `ArticulationRootAPI` from inside the gripper, creates `/World/wrist_to_rg2` fixed joint pinning bracket → `/World/ur3e/wrist_3_link` |
| [scripts/position_gripper_v2.py](scripts/position_gripper_v2.py) | sets `/World/rg2`'s transform so the bracket lands exactly on the wrist's rest pose (no startup impulse) |
| [scripts/clean_articulations.py](scripts/clean_articulations.py) | deactivates the gripper's internal `root_joint` (pinned the bracket to its import-time world position; conflicts with the wrist mount) |
| [scripts/split_articulations.py](scripts/split_articulations.py) | re-applies `ArticulationRootAPI` to `/World/rg2` so gripper is its OWN articulation; wrist mount is then an inter-articulation constraint, which PhysX handles cleanly |
| [scripts/fix_ur_articulation_v2.py](scripts/fix_ur_articulation_v2.py) | moves UR's `ArticulationRootAPI` from the legacy `root_joint` prim to `/World/ur3e/base_link` (where Isaac actually finds it), updates OmniGraph node target paths |
| [scripts/bump_drives.py](scripts/bump_drives.py) | bumps UR3e arm-joint drive stiffness/damping (the URDF importer's defaults are far too weak for a real load) |
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

For the gripper, run [scripts/add_gripper_artctrl.py](scripts/add_gripper_artctrl.py) — it adds a second `IsaacArticulationController` named `ArtCtrlGripper` targeting `/World/rg2`, wired to the same SubJS.

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
