from launch import LaunchDescription
from launch_ros.actions import Node

URDF = "/home/user/Desktop/ur_pick/ur3e_rg2.urdf"


def generate_launch_description():
    with open(URDF) as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
        ),
    ])
