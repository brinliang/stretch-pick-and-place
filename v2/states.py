from smach import State
from stretch_actions import StretchActions


class Start(State):
    def __init__(self, robot: StretchActions, pose):
        State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'])
        self.robot = robot
        self.pose = pose

    def execute(self, userdata):
        transition = self.robot.publish_initial_pose(self.pose)

        return transition


class End(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'abort', 'preempt', 'fail'])

    def execute(self, userdata):
        return 'success'


class LookForMarker(State):
    def __init__(self, robot: StretchActions, marker_name, head_tilt=0.0, pan_steps=9, left_limit=-3.6, right_limit=1.45):
        State.__init__(self, outcomes=['success', 'fail', 'abort', 'preempt'])
        self.robot = robot
        self.marker_name = marker_name
        self.head_tilt = head_tilt
        self.pan_steps = pan_steps
        self.left_limit = left_limit
        self.right_limit = right_limit

    def execute(self, userdata):
        transition = self.robot.look_for_marker(self.marker_name, self.head_tilt, self.pan_steps, self.left_limit, self.right_limit)

        return transition
    

class LookForObject(State):
    def __init__(self, robot: StretchActions, target_object, confidence=0.5, head_tilt=0.0, pan_steps=9, left_limit=-3.6, right_limit=1.45):
        State.__init__(self, outcomes=['success', 'fail', 'abort', 'preempt'])
        self.robot = robot
        self.target_object = target_object
        self.confidence = confidence
        self.head_tilt = head_tilt
        self.pan_steps = pan_steps
        self.left_limit = left_limit
        self.right_limit = right_limit

    def execute(self, userdata):
        transition = self.robot.look_for_object(self.target_object, self.confidence, self.head_tilt, self.pan_steps, self.left_limit, self.right_limit)

        return transition


class Navigate(State):
    def __init__(self, robot: StretchActions, pose):
        State.__init__(self, outcomes=['success', 'fail', 'abort', 'preempt'])
        self.robot = robot
        self.pose = pose

    def execute(self, userdata):
        self.robot.navigation_camera()
        transition = self.robot.navigate_to_pose(self.pose)

        return transition


class ManipulateFromMarkerIK(State):
    def __init__(self, robot, marker_name, manipulation_info):
        State.__init__(self, outcomes=['success', 'fail', 'abort', 'preempt'])
        self.robot = robot
        self.marker_name = marker_name
        self.manipulation_info = manipulation_info

    def execute(self, userdata):
        transition = self.robot.manipulate_from_marker_ik(self.marker_name, self.manipulation_info)

        return transition


class ManipulateFromObjectIK(State):
    def __init__(self, robot, target_object, manipulation_info):
        State.__init__(self, outcomes=['success', 'fail', 'abort', 'preempt'])
        self.robot = robot
        self.target_object = target_object
        self.manipulation_info = manipulation_info

    def execute(self, userdata):
        transition = self.robot.manipulate_from_object_ik(self.target_object, self.manipulation_info)

        return transition


class ManipulateWithJointValues(State):
    def __init__(self, robot, joint_values):
        State.__init__(self, outcomes=['success', 'fail', 'abort', 'preempt'])
        self.robot = robot
        self.joint_values = joint_values

    def execute(self, userdata):
        transition = self.robot.follow_joint_trajectories(self.joint_values)

        return transition



