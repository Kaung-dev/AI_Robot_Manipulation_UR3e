# UR3e + OnRobot RG2 in Isaac Sim — Build Log

End-to-end process of producing a self-contained URDF for a UR3e arm with an OnRobot RG2 parallel-jaw gripper, importing it into Isaac Sim 4.5.0, and exposing it on ROS 2 topics for MoveIt 2 to drive.

## Overview

| Layer | What it does | Status |
| --- | --- | --- |
| URDF | Kinematic + visual model of UR3e + RG2 | done |
| Mesh assets | Local copies of all `.dae` / `.stl` files | done |
| USD | Isaac-Sim-native version of the robot | done |
| ROS 2 bridge | `/joint_states` out, `/joint_command` in | done |
| MoveIt 2 | Plans trajectories, drives the bridge topics | not yet wired |
| OnRobot driver | Real-gripper hardware interface | not yet wired |

## Hardware / OS

- Ubuntu 22.04.5 LTS, Linux 6.8
- 2× NVIDIA RTX 4090, driver 580.105.08 (CUDA 13)
- Python 3.10, ROS 2 Humble pre-installed at `/opt/ros/humble`
- ~13 GB of mesh + USD assets, ~30 GB of Isaac Sim install

## Repositories pulled

```
tonydle/moveit2_tutorials_ur_onrobot      tutorials (reference only)
tonydle/UR_OnRobot_ROS2                   tonydle's wrapper macros
tonydle/OnRobot_ROS2_Description          RG2 macros + meshes
UniversalRobots/Universal_Robots_ROS2_Description  (tag 3.5.0)  UR3e macros + meshes
```

Cloned into the working directory as `tonydle_repo/`, `ur_onrobot_repo/`, `onrobot_desc/`, and `ur_desc/`.

## Step-by-step

### 1. xacro tooling

```bash
pip install xacro
```

`xacro` (pip 2.1.1) imports `ament_index_python` to resolve `$(find pkg)` references. That module isn't on PyPI, so a shim was written at:

`~/.local/lib/python3.10/site-packages/ament_index_python/__init__.py` and `packages.py`

The shim maps known package names to their locally cloned paths. Stub directories were also created for `ur_client_library` and `ur_robot_driver` so the resource lookups don't fail.

### 2. xacro wrapper

tonydle's `ur_onrobot_macro.xacro` calls the UR `ur_robot` macro with arguments (`robot_ip`, `use_fake_hardware`, `script_filename`, …) that don't exist in modern `ur_description`. Wrote a minimal replacement at [ur3e_rg2.urdf.xacro](ur3e_rg2.urdf.xacro) that uses only the description-side macro signature, then includes `onrobot_macro.xacro` and joins it at `tool0` with `rpy="0 0 -pi/2"` (matching tonydle's geometry).

```bash
xacro ur3e_rg2.urdf.xacro -o ur3e_rg2.urdf
```

Output: 24 links, 25 joints. Validated with `check_urdf`.

### 3. Self-contained meshes

Mesh refs were `package://ur_description/...` and `package://onrobot_description/...`. Isaac Sim's URDF importer doesn't resolve `package://` reliably, so:

```bash
mkdir -p meshes/{ur3e/{visual,collision},rg2/{visual,collision}}
cp ur_desc/meshes/ur3e/visual/*.dae       meshes/ur3e/visual/
cp ur_desc/meshes/ur3e/collision/*.stl    meshes/ur3e/collision/
cp onrobot_desc/meshes/rg2/visual/*.stl   meshes/rg2/visual/
cp onrobot_desc/meshes/rg2/collision/*.stl meshes/rg2/collision/
```

Then `sed`'d every mesh path in the URDF to absolute paths under `/home/user/Desktop/ur_pick/meshes/`. Result: 22 mesh refs (15 UR3e, 8 RG2) all resolvable without ROS package context.

### 4. Isaac Sim 4.5.0 install

```bash
python3 -m venv ~/isaacsim_env
source ~/isaacsim_env/bin/activate
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

~10 GB download, ~30 GB installed. Includes Kit 106.5, omni.graph, RTX renderer, ROS 2 bridge for Humble.

### 5. Vulkan ICD fixup (one-off system change)

First launch errored with:

```
[Error] [gpu.foundation.plugin] Multiple Installable Client Drivers (ICDs) are found for the same GPU on the system.
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices
```

Two `nvidia_icd.json` files existed:
- `/etc/vulkan/icd.d/nvidia_icd.json` — stale (Vulkan 1.3.204), from an old `.run` install
- `/usr/share/vulkan/icd.d/nvidia_icd.json` — current (1.4.312), matches live driver

Each GPU was therefore enumerated twice; Isaac Sim refused to pick one. Fix:

```bash
sudo mv /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json.bak
```

Reversible (`mv` back to undo).

### 6. URDF → USD import

[import_ur3e_rg2.py](import_ur3e_rg2.py) — runs the URDF importer with `make_default_prim=True`, position-drive joints, no fixed-joint merging. Writes `ur3e_rg2.usd`. Took ~3 minutes for first launch (extension + shader downloads), ~30 s thereafter.

Visual + joint check passed in the viewport — all meshes loaded, no pink missing-mesh squares, all 6 UR joints sweep without explosions, `finger_width` opens/closes 0–110 mm.

### 7. Mimic joint problem (and fix)

The bridge script attached an `IsaacArticulationController` to `/ur3e_rg2`. The console spammed:

```
[Error] [omni.physx.plugin] Usd Physics: the revolute joint at prim /ur3e_rg2/joints/finger_joint
        needs a finite limit set to be used by the mimic joint feature.
[Error] [omni.physx.tensors.plugin] Pattern '/ur3e_rg2' did not match any rigid bodies
[Error] [omni.physx.tensors.plugin] Provided pattern list did not match any articulations
```

Cause: the RG2 URDF uses 6 `<mimic>` tags so all gripper joints follow `finger_width`. Isaac Sim 4.5's USD-physics mimic implementation requires finite limits on the slave joints, but the URDF→USD pipeline lost them. PhysX consequently refused to form an articulation, so every per-frame articulation lookup failed.

Fix: strip the `<mimic>` tags from the URDF and let the higher level (MoveIt's gripper controller) command all 7 gripper joints together based on the linkage equations.

```python
import re
with open('ur3e_rg2.urdf') as f: s = f.read()
s = re.sub(r'\s*<mimic[^/]*/>\s*\n', '\n', s)
with open('ur3e_rg2.urdf', 'w') as f: f.write(s)
```

After this, the USD must be regenerated — the URDF edit doesn't propagate.

### 8. ROS 2 bridge graph

[isaac_ros2_sim.py](isaac_ros2_sim.py) does the whole loop in one script:

1. Boots `SimulationApp`
2. Enables `isaacsim.ros2.bridge`
3. Re-imports URDF → USD (so the latest URDF edits always take effect)
4. Opens the stage
5. Auto-detects the articulation root by walking the stage looking for `UsdPhysics.ArticulationRootAPI` (avoids hardcoding prim paths)
6. Builds an OmniGraph wired:
   ```
   OnPlaybackTick ── PublishJointState   (publishes sensor_msgs/JointState on /joint_states)
                ── SubscribeJointState ──┐
                                         ├─→ ArticulationController  (writes positions to the articulation)
                ── ArticulationController┘
   IsaacReadSimulationTime ── PublishJointState.timeStamp
   ```
7. Plays the world

Topics published: `/joint_states` (out), `/joint_command` (in).

## How to run

```bash
source /opt/ros/humble/setup.bash && \
source ~/isaacsim_env/bin/activate && \
python -u /home/user/Desktop/ur_pick/isaac_ros2_sim.py 2>&1 | tee ~/isaac_run.log
```

`-u` forces unbuffered Python stdout so `[bridge]` prints land in the log in real time. Without it, prints are block-buffered when piped through `tee` and only flush when the process exits. `tee ~/isaac_run.log` keeps a copy on disk for debugging.

In a second terminal (must be bash, not dash):

```bash
bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic echo /joint_states --once
```

Should print `/joint_states` + `/joint_command` and an actual JointState message with all 7 driven joints.

## Files

| File | Purpose |
| --- | --- |
| [ur3e_rg2.urdf.xacro](ur3e_rg2.urdf.xacro) | Source xacro — UR3e macro + RG2 macro joined at tool0 |
| [ur3e_rg2.urdf](ur3e_rg2.urdf) | Flat URDF, mimics stripped, absolute mesh paths |
| [meshes/](meshes/) | Local copies of all UR3e + RG2 visual/collision meshes |
| [ur3e_rg2.usd](ur3e_rg2.usd) | Isaac Sim USD (regenerated each run by the bridge script) |
| [import_ur3e_rg2.py](import_ur3e_rg2.py) | Standalone URDF → USD import script |
| [isaac_ros2_sim.py](isaac_ros2_sim.py) | Re-imports + opens stage + builds ROS 2 bridge + plays sim |
| [display.launch.py](display.launch.py) | RViz-only sanity check (robot_state_publisher + JSP GUI + RViz) |
| [moveit_config/](moveit_config/) | Hand-written MoveIt 2 config (SRDF, kinematics, controllers, OMPL) |
| [trajectory_bridge.py](trajectory_bridge.py) | Translates MoveIt FollowJointTrajectory action goals → /joint_command JointState |
| [moveit.launch.py](moveit.launch.py) | Brings up move_group + RViz + trajectory_bridge |

## Running the full stack (Isaac Sim + MoveIt + RViz)

Two terminals.

**Terminal 1 — Isaac Sim:**
```bash
source /opt/ros/humble/setup.bash && \
source ~/isaacsim_env/bin/activate && \
python -u /home/user/Desktop/ur_pick/isaac_ros2_sim.py 2>&1 | tee ~/isaac_run.log
```

**Terminal 2 — MoveIt + RViz:**
```bash
source /opt/ros/humble/setup.bash
ros2 launch /home/user/Desktop/ur_pick/moveit.launch.py
```

In RViz: **Add → MotionPlanning**. Set "Planning Group" to `arm`. Drag the interactive marker to a goal pose, click **Plan**, then **Execute**. The trajectory_bridge node receives the FollowJointTrajectory goal, walks the points, publishes `/joint_command`, Isaac Sim drives the articulation, and `/joint_states` flows back to RViz.

## Known caveats

- **Mimic joints removed** — the gripper fingers are 7 independent joints in sim. They will not couple automatically; the controller (MoveIt or a custom node) must command all 7 from the master `finger_width` value. For pure simulation this is fine; for the real robot, the OnRobot driver re-introduces the coupling at the hardware level.
- **`source: command not found`** — the default `/bin/sh` is dash, which has no `source`. Always run from `bash`, or use `.` instead of `source`.
- **One Isaac Sim instance at a time.** Close the prior viewport before launching another script.
- **Articulation root prim is `/ur3e_rg2/root_joint`, NOT `/ur3e_rg2`.** The URDF importer with `fix_base=True` applies `UsdPhysics.ArticulationRootAPI` to the auto-generated `root_joint` fixed-joint prim, not the parent xform. Hardcoding `/ur3e_rg2` causes "Pattern did not match any rigid bodies" spam. The current script auto-detects via `prim.HasAPI(UsdPhysics.ArticulationRootAPI)` which finds the right prim.

## Open issues (currently unresolved)

### A. Topics not visible from a separate shell

The bridge logs `[bridge] sim playing — publishing /joint_states, listening on /joint_command` and shows no PhysX errors, but in a fresh shell:

```bash
$ bash -c "source /opt/ros/humble/setup.bash && ros2 topic list"
/parameter_events
/rosout
```

`/joint_states` and `/joint_command` are missing. Restarting the ROS 2 daemon (`ros2 daemon stop; ros2 daemon start`) does not help. Likely cause: RMW or DDS-discovery mismatch between Isaac Sim's bundled bridge and the system Humble shell. To investigate:
- Check `RMW_IMPLEMENTATION` is the same in both shells (system default is `rmw_fastrtps_cpp` for Humble)
- Check `ROS_DOMAIN_ID` — Isaac Sim's bridge defaults to 0, the shell should match
- Try `ros2 topic list --no-daemon`
- Run `ros2 topic list` from the same shell that launched Isaac Sim, before sourcing the venv (so system ROS env is intact)

### B. Robot freefalls in Isaac Sim during MoveIt/RViz testing

Symptom: with MoveIt + RViz running on the side, the robot in the Isaac Sim viewport falls under gravity instead of holding its commanded pose.

Likely causes (in order of likelihood):
1. **Joint drive stiffness lost.** The import sets `default_drive_strength=1e7` (very stiff PD drives), but Isaac Sim sometimes drops drive properties during URDF→USD conversion when no `<dynamics>` block is present in the URDF. Without stiffness, the PD controller has no restoring force and the arm sags. Verify in the viewport: select any revolute joint prim under `/ur3e_rg2/joints/`, look for `drive:angular:physics:stiffness` in the property panel.
2. **No `/joint_command` reaching the bridge** (related to issue A). When `SubscribeJointState` never receives a message, `ArticulationController.positionCommand` stays empty → no PD targets → joints drift.
3. **`fix_base=True` not applied to the saved USD.** If the world→base_link fixed joint isn't actually fixed in physics, the entire arm falls together. Check whether the base stays put or the whole robot falls as one rigid body.

Note: the bridge will not move the robot until MoveIt is actually wired to the topics — see "Next steps" below. So without MoveIt running, the only thing keeping the arm up is joint stiffness, which is what (1) is about.

## Next steps (not done yet)

1. **MoveIt 2 wiring** — build tonydle's `ur_onrobot_moveit_config` workspace, swap its `ros2_control` hardware to `topic_based_ros2_control/TopicBasedSystem` pointing at `/joint_command` and `/joint_states`. MoveIt then plans and Isaac Sim executes.
2. **OnRobot real-gripper path** — when swapping to a real RG2, drop `topic_based_ros2_control` and bring up `ur_onrobot_control` with `use_fake_hardware:=false`. The MoveIt-side topics stay the same, only the controller backend changes.
3. **Save the configured stage** — once MoveIt is wired, save the Isaac Sim stage with the bridge action graph baked in, so we don't rebuild the graph every launch.
