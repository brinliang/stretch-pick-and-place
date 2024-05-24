import rclpy
from rclpy.executors import MultiThreadedExecutor

import smach_ros
from smach import State, StateMachine

import yaml
import threading

from states import Start, LookForMarker, Navigate, ManipulateFromObjectIK, End, ManipulateFromMarkerIK, LookForObject
from stretch_actions import StretchActions


def main():

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    rclpy.init()

    executor = MultiThreadedExecutor()
    robot = StretchActions()
    executor.add_node(robot)

    sm = StateMachine(outcomes=['success', 'abort', 'preempt'])

    with sm:
        StateMachine.add('START', Start(robot, config['nav_initial']), transitions={'success':'NAVIGATE_TO_CUP', 'abort':'abort', 'preempt':'preempt', 'fail':'abort'})
        StateMachine.add('NAVIGATE_TO_CUP', Navigate(robot, config['nav_pick']), transitions={'success':'LOOK_FOR_CUP', 'preempt':'preempt', 'abort':'abort', 'fail':'abort'})
        StateMachine.add('LOOK_FOR_CUP', LookForObject(robot, 'cup', confidence=0.8, head_tilt=-0.6, pan_steps=5, left_limit=-2.5, right_limit=0.5), transitions={'success':'PICK_CUP', 'preempt':'preempt', 'abort':'abort', 'fail':'abort'})
        StateMachine.add('PICK_CUP', ManipulateFromObjectIK(robot, 'cup', config['manipulation_pick']), transitions={'success':'NAVIGATE_TO_SINK', 'preempt':'preempt', 'abort':'abort', 'fail':'abort'})
        StateMachine.add('NAVIGATE_TO_SINK', Navigate(robot, config['nav_place']), transitions={'success':'LOOK_FOR_SINK', 'preempt':'preempt', 'abort':'abort', 'fail':'abort'})
        StateMachine.add('LOOK_FOR_SINK', LookForMarker(robot, 'drop'), transitions={'success':'PLACE_CUP', 'preempt':'preempt', 'abort':'abort', 'fail':'abort'})
        StateMachine.add('PLACE_CUP', ManipulateFromMarkerIK(robot, 'drop', config['manipulation_place']), transitions={'success':'END', 'preempt':'preempt', 'abort':'abort', 'fail':'abort'})
        StateMachine.add('END', End(), transitions={'success':'success', 'abort':'abort', 'preempt':'preempt', 'fail':'abort'})


    sis = smach_ros.IntrospectionServer('state_machine', sm, '/state_machine')
    sis.start()

    threading.Thread(target=sm.execute).start()

    sis.stop()

    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()