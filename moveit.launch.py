"""
MoveIt2 + RViz launcher for the ur3e_rg2 robot.

Pairs with isaac_ros2_sim.py — that one runs Isaac Sim with the ROS2 bridge,
this one runs the MoveIt planning/visualization side and forwards trajectories
to /joint_command via trajectory_bridge.py.
"""

from pathlib import Path
import yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

UR_PICK = Path("/home/user/Desktop/ur_pick")
CFG = UR_PICK / "moveit_config"


def _read(p):
    return Path(p).read_text()


def _yaml(p):
    return yaml.safe_load(Path(p).read_text())


def generate_launch_description():
    robot_description = {"robot_description": _read(UR_PICK / "ur3e_rg2.urdf")}
    robot_description_semantic = {"robot_description_semantic": _read(CFG / "ur3e_rg2.srdf")}
    kinematics = {"robot_description_kinematics": _yaml(CFG / "kinematics.yaml")}
    joint_limits = {"robot_description_planning": _yaml(CFG / "joint_limits.yaml")}
    moveit_controllers = _yaml(CFG / "moveit_controllers.yaml")
    ompl = {"ompl": _yaml(CFG / "ompl_planning.yaml")}

    planning_pipeline = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        },
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics,
            joint_limits,
            moveit_controllers,
            planning_pipeline,
            ompl,
            {"publish_robot_description_semantic": True},
        ],
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", str(CFG / "moveit.rviz")],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics,
            joint_limits,
        ],
    )

    bridge = ExecuteProcess(
        cmd=["python3", str(UR_PICK / "trajectory_bridge.py")],
        name="trajectory_bridge",
        output="screen",
    )

    return LaunchDescription([rsp, move_group, rviz, bridge])
