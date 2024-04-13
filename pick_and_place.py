import smach
import subprocess
import yaml
import signal

# ros interfaces
from geometry_msgs.msg import Point, Quaternion

# ros nodes
import nodes

class Start(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt'], output_keys=['processes']) 

    def execute(self, userdata):
        # start navigation, camera, aruco detector, state publisher, and rosbag processor processes
        navigation_process = subprocess.Popen(['ros2', 'launch', 'stretch_nav2' , 'navigation.launch.py' , 'map:=/home/hello-robot/stretch_user/maps/robotics_kitchen.yaml'])
        camera_process = subprocess.Popen(['ros2', 'launch', 'stretch_core' , 'd435i_low_resolution.launch.py' ])
        detector_process = subprocess.Popen(['ros2', 'launch', 'stretch_core' , 'stretch_aruco.launch.py' ])

        userdata.processes = [navigation_process, camera_process, detector_process]

        # localize robot to pre defined location
        with open('config/pick_and_place.yaml') as f:
            initial_pose = yaml.safe_load(f)['nav_to_pick_start']

        nodes.set_initial_pose(
            Point(x=initial_pose['position']['x'], y=initial_pose['position']['y'], z=initial_pose['position']['z']),
            Quaternion(x=initial_pose['orientation']['x'], y=initial_pose['orientation']['y'], z=initial_pose['orientation']['z'], w=initial_pose['orientation']['w'])
        )

        return 'success'
    

class NavigateToPick(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'])

    def execute(self, userdata):
        # change robot to navigation mode
        nodes.switch_to_nav_mode()

        # navigate to pick location
        with open('config/pick_and_place.yaml') as f:
            pick_location = yaml.safe_load(f)['nav_to_pick_end']

        transition = nodes.nav_to_pose(
            Point(x=pick_location['position']['x'], y=pick_location['position']['y'], z=pick_location['position']['z']),
            Quaternion(x=pick_location['orientation']['x'], y=pick_location['orientation']['y'], z=pick_location['orientation']['z'], w=pick_location['orientation']['w'])
        )

        return transition


class DetectToPick(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'], 
                             output_keys=['marker_transform'])

    def execute(self, userdata):
        # change robot to position mode
        nodes.switch_to_position_mode()

        # detect marker to pick
        with open('config/pick_and_place.yaml') as f:
            pick_marker_name = yaml.safe_load(f)['detect_to_pick_name']

        transition, transform = nodes.detect_marker(pick_marker_name)
        userdata.marker_transform = transform

        return transition
    

class Pick(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'],
                             input_keys=['marker_transform'])

    def execute(self, userdata):
        # change robot to position mode
        nodes.switch_to_position_mode()

        # pick object
        with open('config/pick_and_place.yaml') as f:
            marker_point = yaml.safe_load(f)['pick_marker_point']

        transition = nodes.manipulate(marker_point, userdata.marker_transform, 'pick')

        return transition


class NavigateToPlace(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'])

    def execute(self, userdata):
        # change robot to navigation mode
        nodes.switch_to_nav_mode()

        # navigate to place location
        with open('config/pick_and_place.yaml') as f:
            place_location = yaml.safe_load(f)['nav_to_place_end']

        transition = nodes.nav_to_pose(
            Point(x=place_location['position']['x'], y=place_location['position']['y'], z=place_location['position']['z']),
            Quaternion(x=place_location['orientation']['x'], y=place_location['orientation']['y'], z=place_location['orientation']['z'], w=place_location['orientation']['w'])
        )

        return transition


class DetectToPlace(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'], 
                             output_keys=['marker_transform'])

    def execute(self, userdata):
        # change robot to position mode
        nodes.switch_to_position_mode()

        # detect marker to place
        with open('config/pick_and_place.yaml') as f:
            place_marker_name = yaml.safe_load(f)['detect_to_place_name']

        transition, transform = nodes.detect_marker(place_marker_name)
        userdata.marker_transform = transform

        return transition


class Place(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'],
                             input_keys=['marker_transform'], output_keys=['result'])

    def execute(self, userdata):
        # change robot to position mode
        nodes.switch_to_position_mode()

        # place object
        with open('config/pick_and_place.yaml') as f:
            marker_point = yaml.safe_load(f)['place_marker_point']

        transition = nodes.manipulate(marker_point, userdata.marker_transform, 'place')

        # all states are successful, set result to success
        if transition == 'success':
            userdata.result = 'success'

        return transition


class End(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'abort', 'preempt'], input_keys=['processes', 'result']) 

    def execute(self, userdata):
        # close processes
        for i in range(len(userdata.processes)):
            userdata.processes[i].send_signal(signal.SIGINT)

        return userdata.result


def main():
    sm = smach.StateMachine(outcomes=['success', 'abort', 'preempt'])
    sm.userdata.marker_transform = None
    sm.userdata.processes = None
    sm.userdata.result = 'abort'

    with sm:
        smach.StateMachine.add('START', Start(), transitions={'success':'NAVIGATE_TO_PICK', 'abort':'END', 'preempt':'preempt'})
        smach.StateMachine.add('NAVIGATE_TO_PICK', NavigateToPick(), transitions={'success':'DETECT_TO_PICK', 'abort':'END', 'preempt':'preempt', 'fail':'END'})
        smach.StateMachine.add('DETECT_TO_PICK', DetectToPick(), transitions={'success':'PICK', 'abort':'END', 'preempt':'preempt', 'fail':'END'})
        smach.StateMachine.add('PICK', Pick(), transitions={'success':'NAVIGATE_TO_PLACE', 'abort':'END', 'preempt':'preempt', 'fail':'END'})
        smach.StateMachine.add('NAVIGATE_TO_PLACE', NavigateToPlace(), transitions={'success':'DETECT_TO_PLACE', 'abort':'END', 'preempt':'preempt', 'fail':'END'})
        smach.StateMachine.add('DETECT_TO_PLACE', DetectToPlace(), transitions={'success':'PLACE', 'abort':'END', 'preempt':'preempt', 'fail':'END'})
        smach.StateMachine.add('PLACE', Place(), transitions={'success':'END', 'abort':'END', 'preempt':'preempt', 'fail':'END'})
        smach.StateMachine.add('END', End(), transitions={'success':'success', 'abort':'abort', 'preempt':'preempt'})

    outcome = sm.execute()


if __name__ == '__main__':
    main()