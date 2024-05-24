# Details
This is a minimal implementation of the task of picking up an object and placing the object using Stretch. 

There are eight states in the state machine, located in pick_and_place.py.

1. START: starts all of the background processes, and localizes the robot on the map to a pre-defined position
2. NAVIGATE_TO_PICK: navigates the robot to the location of the object to pick
3. DETECT_TO_PICK: pans the camera to look for the object, marked by an aruco marker with the label 'pick'
4. PICK: picks the detected object
5. NAVIGATE_TO_PLACE: navigates the robot to the location to place the object
6. DETECT_TO_PLACE: pans the camera to look for the location to place the object, marked by an aruco marker with the label 'drop'
7. PLACE: places the object in the detected location
8. END: closes all of the background processes

There are six ROS nodes and corresponding functions to run the nodes, located in nodes.py.

1. SetInitialState: sets the initial position of the robot on the Nav2 map to a specified pose
2. NavigateToGoal: navigates the robot to a pose specified on the Nav2 map
3. DetectMarker: pans the camera and looks for an aruco marker with the specified label
4. Manipulate: moves the robot's end effector to a specified position relative to a specified frame to pick or place
5. NavigationMode: requests a service to switch the robot to the mode required for navigation
6. PositionMode: requests a service to switch the robot to the mode required for joint movements


# Usage
```python3 pick_and_place.py```
