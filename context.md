this is the exmple issac sim

DarK404 /
UR_Isaac-sim
Public

    Code
    Issues
    Pull requests
    Actions
    Projects
    Security and quality
    Insights

DarK404/UR_Isaac-sim
Name	Last commit message
	Last commit date
DarK404
DarK404
Create LICENSE
0aeae7c
 · 
2 years ago
Universal_Robots_ROS2_Description-ros2
	
ur3e with gripper
	
3 years ago
moveit_config
	
Update isaac_demo.launch.py
	
3 years ago
usd_files
	
uploading the usd files
	
3 years ago
LICENSE
	
Create LICENSE
	
2 years ago
README.md
	
Update README.md
	
3 years ago
Repository files navigation

    README
    GPL-3.0 license

UR_Isaac-sim

This repository provides a moveit_config to control a UR3e robot using Moveit2. The container includes Isaac Sim simulation for testing purposes.
Getting Started

To get started with this repository, follow these steps:

    Clone this repository to your local machine.

git clone https://github.com/DarK404/moveit2_tutorials

    Install Docker and Docker Compose if you haven't already.
    Open a terminal in the project directory and run the following command to build the Docker container:

cd moveit2_tutorials/doc/how_to_guides/isaac_panda
docker compose build

    Git clone https://github.com/DarK404/UR_Isaac-sim/tree/main and Locate the usd files in your localhost Nucleus environment under the folder name ur3e_gripper_humble_moveit2-humble.

    Launch Isaac-sim inside the launch directory

cd moveit2_tutorials/doc/how_to_guides/isaac_panda/launch
./python.sh isaac_moveit_Hackathon.py

    After the build process is complete, run the following command to start the container:

docker compose up demo_isaac

    In RVIZ, you should see a visualization of the UR3e robot. You can control the robot using Moveit2.

1.RVIZ visualization

Material Bread logo

2.Isaac Sim visualization

but I want to      use the models of thissssssssssss and thhhhhhe gripper of this 
|


tonydle /
moveit2_tutorials_ur_onrobot
Public

    Code
    Issues
    Pull requests
    Actions
    Projects
    Security and quality
    Insights

tonydle/moveit2_tutorials_ur_onrobot
Name	Last commit message
	Last commit date
tonydle
tonydle
Added YouTube video thumbnail
a609cd8
 · 
2 weeks ago
docs/gifs
	
Added all GIFs
	
2 weeks ago
ur_onrobot_hello_moveit
	
Added demo packages
	
3 weeks ago
ur_onrobot_mtc
	
Updated MTC tutorial
	
3 weeks ago
.gitignore
	
Added demo packages
	
3 weeks ago
LICENSE
	
Initial commit
	
3 weeks ago
README.md
	
Added YouTube video thumbnail
	
2 weeks ago
Repository files navigation

    README
    MIT license

moveit2_tutorials_ur_onrobot

This repo accompanies the YouTube video below. Watch the video here:

Watch the video on YouTube

Following the MoveIt 2 tutorial examples, but with a [UR manipulator + OnRobot gripper] robot instead of the Panda.

Before using this repo, make sure to have the UR_OnRobot_ROS2 package set up and working. If you have not already set that up, go there first and follow its installation instructions.

This repo contains two ROS 2 packages:

    ur_onrobot_hello_moveit
    ur_onrobot_mtc

Setup

Clone this repo into your workspace, then install any missing ROS dependencies:

rosdep install --from-paths src --ignore-src -r -y

Build and source the workspace:

colcon build --packages-select ur_onrobot_hello_moveit ur_onrobot_mtc
source install/setup.bash

Run the demos

    Start the driver with the UR3e + RG2 robot and fake hardware, but keep its RViz off:

ros2 launch ur_onrobot_control start_robot.launch.py ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false

    Start the MoveIt config for the same configuration, also with RViz off:

ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py ur_type:=ur3e onrobot_type:=rg2 launch_rviz:=false

    Start one shared RViz session with the custom config from this repo:

ros2 launch ur_onrobot_hello_moveit tutorials_rviz.launch.py

    Keep those three terminals running, then run the demos below one at a time in a fourth terminal.

Demo 1: Your First Project

ros2 run ur_onrobot_hello_moveit your_first_project

GIF placeholder: your_first_project
Demo 2: Visualizing In RViz

ros2 run ur_onrobot_hello_moveit visualizing_in_rviz

GIF placeholder: visualizing_in_rviz
Demo 3: Planning Around Objects

ros2 run ur_onrobot_hello_moveit planning_around_objects

GIF placeholder: planning_around_objects
Demo 4: Minimal MTC

ros2 run ur_onrobot_mtc minimal

GIF placeholder: minimal_mtc
Demo 5: Pick And Place MTC

ros2 launch ur_onrobot_mtc pick_and_place_demo.launch.py

GIF placeholder: pick_and_place_mtc