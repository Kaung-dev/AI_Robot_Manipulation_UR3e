#!/usr/bin/env python3
"""
Bridge: /joint_states  ->  /isaac_joint_commands

What it does
------------
Subscribes to the /joint_states topic published by the MoveIt mock hardware
(joint_state_broadcaster reflects the trajectory the controllers are tracking),
filters down to the 6 UR3e arm joints, and republishes them as JointState
messages on /isaac_joint_commands. The Isaac Sim ROS2SubscribeJointState ->
IsaacArticulationController graph nodes consume those and drive the simulated
arm.

How to run
----------
1. In Isaac Sim: open scene.usd and press Play (so the OmniGraph ticks).
2. Open 4 terminals; in each: source /opt/ros/humble/setup.bash AND
   source ros2_ws/install/setup.bash. Then run:
     T1: ros2 launch ur_onrobot_control start_robot.launch.py \
           ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false
     T2: ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \
           ur_type:=ur3e onrobot_type:=rg2 launch_rviz:=false
     T3: ros2 launch ur_onrobot_hello_moveit tutorials_rviz.launch.py
     T4: python3 scripts/moveit_to_isaac_bridge.py
3. In a 5th terminal, run any demo, e.g.:
     ros2 run ur_onrobot_hello_moveit your_first_project
   The arm in RViz mirrors MoveIt's plan; the arm in Isaac Sim physically tracks it.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Gripper: drive BOTH master joints (left + right) with the same target,
# matching the Isaac Lab lift task config (close=1.30 rad, open=0.0 rad).
# This is why scene.usd's mirror_joint had its mimic stripped via
# _archive/debug_scripts/align_gripper_with_isaaclab.py - it's now a free drive.
GRIPPER_WIDTH_JOINT = "finger_width"
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = 1.30
WIDTH_OPEN_M = 0.110
WIDTH_CLOSED_M = 0.0
GRIPPER_MASTER_JOINTS = ["rg2_gripper_joint", "rg2_gripper_mirror_joint"]


def width_to_master(width_m: float) -> float:
    """Linear, clamped: finger_width(m) -> master angle (rad)."""
    w = max(WIDTH_CLOSED_M, min(WIDTH_OPEN_M, width_m))
    frac_open = w / WIDTH_OPEN_M  # 0 closed, 1 open
    return GRIPPER_CLOSED_RAD + frac_open * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


class MoveItToIsaacBridge(Node):
    def __init__(self) -> None:
        super().__init__("moveit_to_isaac_bridge")
        self.sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 50
        )
        self.pub = self.create_publisher(JointState, "/isaac_joint_commands", 50)
        self._missing_logged: set[str] = set()
        self.get_logger().info(
            "Bridging /joint_states -> /isaac_joint_commands. "
            f"Arm joints: {ARM_JOINTS}. "
            f"Gripper: {GRIPPER_WIDTH_JOINT} -> finger_joint + 5 mimicked."
        )

    def _on_joint_states(self, msg: JointState) -> None:
        idx = {name: i for i, name in enumerate(msg.name)}
        out = JointState()
        out.header = msg.header

        # ---- arm: pass through ----
        for joint in ARM_JOINTS:
            i = idx.get(joint)
            if i is None:
                if joint not in self._missing_logged:
                    self.get_logger().warn(
                        f"Joint '{joint}' not in /joint_states (have {list(msg.name)})"
                    )
                    self._missing_logged.add(joint)
                continue
            out.name.append(joint)
            if i < len(msg.position):
                out.position.append(msg.position[i])

        # ---- gripper: drive both master joints with same target ----
        # The 4 finger-side mimic followers (truss + finger_tip x left/right)
        # still track via PhysX mimic. The mirror master gets driven directly
        # like Isaac Lab does.
        gi = idx.get(GRIPPER_WIDTH_JOINT)
        if gi is not None and gi < len(msg.position):
            target = width_to_master(msg.position[gi])
            for jname in GRIPPER_MASTER_JOINTS:
                out.name.append(jname)
                out.position.append(target)
        elif GRIPPER_WIDTH_JOINT not in self._missing_logged:
            self.get_logger().warn(
                f"'{GRIPPER_WIDTH_JOINT}' not in /joint_states yet; "
                "gripper will not be commanded until the gripper controller is up."
            )
            self._missing_logged.add(GRIPPER_WIDTH_JOINT)

        if out.name:
            self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = MoveItToIsaacBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
