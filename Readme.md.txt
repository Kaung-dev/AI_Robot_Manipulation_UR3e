Open the Isaac sim with ros2 bridge enable 
in the sim open the scene.usd 

These are the robot control 
T1: ros2 launch ur_onrobot_control start_robot.launch.py ur_type:=ur3e onrobot_type:=rg2 use_fake_hardware:=true launch_rviz:=false
T2: ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py ur_type:=ur3e onrobot_type:=rg2
T3: python3 /home/user/Desktop/ur_pick/scripts/moveit_to_isaac_bridge.py 


