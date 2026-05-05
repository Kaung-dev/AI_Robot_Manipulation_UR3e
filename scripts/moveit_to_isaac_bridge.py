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
   source /home/user/Desktop/ur_pick/ros2_ws/install/setup.bash. Then run:
     T1: ros2 launch ur_onrobot_control start_robot.launch.py \
           ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false
     T2: ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \
           ur_type:=ur3e onrobot_type:=rg2 launch_rviz:=false
     T3: ros2 launch ur_onrobot_hello_moveit tutorials_rviz.launch.py
     T4: python3 /home/user/Desktop/ur_pick/scripts/moveit_to_isaac_bridge.py
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

# RG2 gripper: convert prismatic finger_width (m) to the 6 revolute joints in
# Isaac (which has no <mimic> support, so we do it here). Coefficients come
# straight from onrobot_description/urdf/rg2_macro.xacro:
#   finger_joint(rad) = MULT * finger_width(m) + OFFSET
GRIPPER_WIDTH_JOINT = "finger_width"
RG2_FINGER_JOINT_MULT = 0.85 * ((-0.558505 - 0.785398) / 0.110)  # ~ -10.38470
RG2_FINGER_JOINT_OFFSET = 0.785398

# multipliers of the OTHER 5 joints, relative to finger_joint (from URDF mimic tags)
RG2_MIMIC = {
    "right_outer_knuckle_joint": -1.0,
    "left_inner_knuckle_joint": -1.0,
    "right_inner_knuckle_joint": -1.0,
    "left_inner_finger_joint": +1.0,
    "right_inner_finger_joint": +1.0,
}


def width_to_finger_joint(width_m: float) -> float:
    return RG2_FINGER_JOINT_MULT * width_m + RG2_FINGER_JOINT_OFFSET


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

        # ---- gripper: convert finger_width -> 6 revolute joints ----
        gi = idx.get(GRIPPER_WIDTH_JOINT)
        if gi is not None and gi < len(msg.position):
            width = msg.position[gi]
            fj_rad = width_to_finger_joint(width)
            out.name.append("finger_joint")
            out.position.append(fj_rad)
            for jname, mult in RG2_MIMIC.items():
                out.name.append(jname)
                out.position.append(mult * fj_rad)
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
