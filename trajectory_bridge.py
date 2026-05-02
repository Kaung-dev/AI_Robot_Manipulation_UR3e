"""
Bridges MoveIt's FollowJointTrajectory action to Isaac Sim's /joint_command topic.

Two action servers are started — one per controller defined in moveit_controllers.yaml:
  /arm_controller/follow_joint_trajectory
  /gripper_controller/follow_joint_trajectory

Each receives a control_msgs/FollowJointTrajectory goal, walks through trajectory points
in real time, and publishes a sensor_msgs/JointState on /joint_command. Isaac Sim's
SubscribeJointState node consumes that and drives the articulation.

Also republishes the latest /joint_states it sees back out as joint feedback during
trajectory execution (so MoveIt's controller manager sees the goal is being tracked).
"""

import time
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration


class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__("trajectory_bridge")
        cb = ReentrantCallbackGroup()

        self.cmd_pub = self.create_publisher(JointState, "/joint_command", 10)
        self.state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.last_state = {}
        self.create_subscription(JointState, "/isaac_joint_states", self._on_isaac_state, 10, callback_group=cb)
        self.create_subscription(JointState, "/joint_states", self._on_state, 10, callback_group=cb)

        self._mk_server("arm_controller", cb)
        self._mk_server("gripper_controller", cb)
        self.get_logger().info("ready: arm_controller + gripper_controller + /joint_states restamper")

    def _mk_server(self, name, cb):
        ActionServer(
            self,
            FollowJointTrajectory,
            f"/{name}/follow_joint_trajectory",
            execute_callback=self._execute,
            callback_group=cb,
        )

    def _on_isaac_state(self, msg: JointState):
        # Isaac Sim publishes /isaac_joint_states with stamp=0; restamp with ROS time
        # and republish on /joint_states so MoveIt's CurrentStateMonitor accepts it.
        msg.header.stamp = self.get_clock().now().to_msg()
        self.state_pub.publish(msg)

    def _on_state(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.last_state[n] = p

    def _execute(self, goal_handle):
        traj = goal_handle.request.trajectory
        names = list(traj.joint_names)
        points = traj.points
        self.get_logger().info(f"executing trajectory: {len(points)} pts on {names}")

        start = self.get_clock().now()
        for i, pt in enumerate(points):
            target_t = _dur_to_sec(pt.time_from_start)
            while True:
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
                if elapsed >= target_t:
                    break
                # use sleep, not spin_once — the MultiThreadedExecutor in main()
                # is already pumping subscription callbacks (incl. /isaac_joint_states
                # restamper) on a separate thread. spin_once here would deadlock /
                # block restamper while we wait between trajectory points.
                time.sleep(0.005)

            cmd_names = list(names)
            cmd_pos = list(pt.positions)
            # Expand finger_width into the 7 coupled RG2 joints.
            # Originally <mimic> in URDF; stripped for Isaac Sim physics, recreated here in software.
            if "finger_width" in cmd_names:
                w = cmd_pos[cmd_names.index("finger_width")]
                fj = -10.384705 * w + 0.785398
                for jn, jv in [
                    ("finger_joint", fj),
                    ("left_inner_knuckle_joint", -fj),
                    ("left_inner_finger_joint", fj),
                    ("right_outer_knuckle_joint", -fj),
                    ("right_inner_knuckle_joint", -fj),
                    ("right_inner_finger_joint", fj),
                ]:
                    if jn not in cmd_names:
                        cmd_names.append(jn)
                        cmd_pos.append(jv)

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = cmd_names
            cmd.position = cmd_pos
            self.cmd_pub.publish(cmd)

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        return result


def _dur_to_sec(d: Duration) -> float:
    return d.sec + d.nanosec * 1e-9


def main():
    rclpy.init()
    node = TrajectoryFollower()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
