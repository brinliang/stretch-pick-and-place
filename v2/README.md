# run trajectory

## launch ros nodes in separate terminals

ros2 launch stretch_nav2 navigation.launch.py map:=${HELLO_FLEET_PATH}/maps/kitchen3.yaml

ros2 launch stretch_core d435i_low_resolution.launch.py

ros2 launch stretch_core stretch_aruco.launch.py


## run robot code
python3 <filename>



# record poses

### get nav pose

ros2 topic echo /amcl_pose


### get joint positions

#### absolute values for joints
ros2 topic echo /stretch/joint_states

#### get joint positions relative to marker
python3 record_ik_point.py

#### (optional) visualize positions for aruco markers
ros2 run rviz2 rviz2 -d /home/hello-robot/ament_ws/src/stretch_tutorials/rviz/aruco_detector_example.rviz
