import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from std_srvs.srv import Trigger


from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import numpy as np

import ikpy.chain
import time
import pick_and_place
import os


# ros node for setting initial pose in Nav2
class SetInitialPose(Node):
    def __init__(self, point, orientation):
        super().__init__('set_initial_pose')
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.timer = self.create_timer(2, self.publish_initial_pose)
        self.initial_point = point
        self.initial_orientation = orientation

    def publish_initial_pose(self):
        initial_pose_msg = PoseWithCovarianceStamped()

        # set frame and timestamp
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'

        # set point and orientation on map
        initial_pose_msg.pose.pose.position = self.initial_point
        initial_pose_msg.pose.pose.orientation =  self.initial_orientation

        # publish the initial pose
        self.publisher.publish(initial_pose_msg)
        self.get_logger().info('initial pose has been published')


# set initial position of robot in Nav2
def set_initial_pose(point, orientation):
    rclpy.init()
    node = SetInitialPose(point, orientation)

    # give time for startup
    for i in range(10):
        rclpy.spin_once(node)
        time.sleep(1)
    node.destroy_node()
    rclpy.shutdown()


# navigate to pose
class NavigateToGoal(Node):
    def __init__(self):
        super().__init__('nav_to_goal')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.finished = False
        self.transition = 'success'


    def send_goal(self, point, orientation):
        # specify goal with position and orientation 
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.pose.position = point
        goal_msg.pose.pose.orientation = orientation
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # send pose action client and wait for it to finish
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('goal rejected')
            return

        self.get_logger().info('goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'goal reached: {result}')
        self.finished = True


# navigate to a pose in Nav2
def nav_to_pose(point, orientation):
    rclpy.init()
    node = NavigateToGoal()
    node.send_goal(point, orientation)

    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node)
    
    transition = node.transition
    node.destroy_node()
    rclpy.shutdown()

    return transition


# ros node for finding aruco markers
class DetectMarker(Node):
    def __init__(self, marker_name):
        super().__init__('detect')
        # initialize client for moving camera
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        
        # wait for server
        server_reached = self.trajectory_client.wait_for_server(timeout_sec=60.0)
        if not server_reached:
            self.get_logger().error('Unable to connect to arm action server. Timeout exceeded.')
            exit(1)

        # name of the aruco marker to look for
        self.marker_name = marker_name

        # target frame to get transform relative to
        self.target_frame = 'base_link'

        # transform listeners for getting marker to target transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # set up a recurring timer to look for markers and move the camera
        self.marker_trans = None
        self.get_marker_transform = self.create_timer(2.0, self.get_marker_transform_callback)

        # specify left and right limits for camera pan, pan steps, and tilt
        self.camera_constraints = {
            'far_right': -3.6,
            'far_left': 1.45,
            'num_pan_steps': 7,
            'head_tilt': -0.65
        }

        # create a list of camera positions to move to
        self.camera_positions = [[self.camera_constraints['head_tilt'], x] for x in np.linspace(
            self.camera_constraints['far_right'], self.camera_constraints['far_left'], self.camera_constraints['num_pan_steps'])]

        # track node state
        self.finished = False
        self.transition = 'success'


    def get_marker_transform_callback(self):
        try:
            # look for marker
            now = Time()
            trans = self.tf_buffer.lookup_transform(self.target_frame, self.marker_name, now)
        except TransformException as ex:
            # marker is not available, move camera and try again
            self.get_logger().info(f'marker not found')
            if len(self.camera_positions) > 0:
                # move camera to next position
                goal_msg = FollowJointTrajectory.Goal()
                point = JointTrajectoryPoint()
                point.positions = self.camera_positions.pop(0)
                point.time_from_start = Duration(seconds=1.0).to_msg()
                goal_msg.trajectory.joint_names = ['joint_head_tilt', 'joint_head_pan']
                goal_msg.trajectory.points = [point]
                self.trajectory_client.send_goal_async(goal_msg)
                self.get_logger().info('moved camera')
                return
            else:
                # camera has moved to all positions and marker has not been found
                self.transition = 'fail'
                self.finished = True
                return
        
        # marker found
        self.get_logger().info(f'marker found')
        self.marker_trans = trans
        self.finished = True
        self.get_marker_transform.destroy()


# detect a marker given the marker name
def detect_marker(marker_name):
    rclpy.init()
    node = DetectMarker(marker_name)
    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node)
    transition = node.transition
    transform = node.marker_trans
    node.destroy_node()
    rclpy.shutdown()

    return transition, transform


# ros node for picking and placing a marked object
class Manipulate(Node):
    def __init__(self, marker_point, marker_trans, mode):
        super().__init__('manipulate')

        # initialize client for moving joints
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        server_reached = self.trajectory_client.wait_for_server(timeout_sec=60.0)
        if not server_reached:
            self.get_logger().error('Unable to connect to arm action server. Timeout exceeded.')
            exit(1)

        # subscription to get initial joint positions
        self.joint_states = None
        self.joint_sub = self.create_subscription(JointState, '/stretch/joint_states', self.joint_states_callback, 1)

        # target point relative to marker frame
        self.marker_point = np.array([marker_point['x'], marker_point['y'], marker_point['z'], 1])

        # transform from marker to base frame
        self.marker_trans = marker_trans

        # mode: pick or place
        self.mode = mode

        # track node state
        self.finished = False
        self.transition = 'success'
        self.goal_ready = True

        # list of joint positions to move to
        self.trajectories = []

        # execute when ready
        self.manipulate_timer = self.create_timer(1.0, self.manipulate_callback)


    def joint_states_callback(self, joint_states):
        self.joint_states = joint_states


    def manipulate_callback(self):
        # wait for joint subscriber
        if self.joint_states is None:
            return
        
        # prevent callback
        self.manipulate_timer.destroy()
        
        self.get_logger().info('computing trajectory')
        
        # compute trajectory using inverse kinematics
        urdf_filename = os.path.join(os.path.dirname(__file__), 'config/stretch.urdf')
        chain = ikpy.chain.Chain.from_urdf_file(urdf_filename)

        # transformation matrix from marker frame to base frame
        q = self.marker_trans.transform.rotation
        t = self.marker_trans.transform.translation
        transform_matrix = np.array(
            [
                [1-2*q.y**2-2*q.z**2, 2*q.x*q.y-2*q.z*q.w, 2*q.x*q.z+2*q.y*q.w, t.x],
                [2*q.x*q.y+2*q.z*q.w, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z-2*q.x*q.w, t.y],
                [2*q.x*q.z-2*q.y*q.w, 2*q.y*q.z+2*q.x*q.w, 1-2*q.x**2-2*q.y**2, t.z],
                [0, 0, 0, 1]
            ]
        )

        # initial state for inverse kinematics
        q_init = [
            0.0, 0.0, 0.0, 
            self.joint_states.position[1], 
            0.0, 
            self.joint_states.position[2], 
            self.joint_states.position[3], 
            self.joint_states.position[4], 
            self.joint_states.position[5], 
            self.joint_states.position[8], 
            0.0, 
            self.joint_states.position[9], 
            self.joint_states.position[10], 
            0.0, 0.0
        ]

        # get joint positions from inverse kinematics
        target_point = np.dot(transform_matrix, self.marker_point)[:3]
        q_soln = chain.inverse_kinematics(target_point, initial_position=q_init)

        if self.mode == 'pick':
            # open gripper and move arm to marker
            self.trajectories.append({'joint_gripper_finger_left': 0.4})
            self.trajectories.append({'translate_mobile_base': q_soln[1]})
            self.trajectories.append({'joint_lift': q_soln[3]})
            self.trajectories.append({'joint_wrist_yaw': q_soln[9]})
            self.trajectories.append({'joint_wrist_pitch': q_soln[11]})
            self.trajectories.append({'joint_wrist_roll': q_soln[12]})
            self.trajectories.append({'wrist_extension': q_soln[5] + q_soln[6] + q_soln[7] + q_soln[8]})

            # close gripper and retract arm
            self.trajectories.append({'joint_gripper_finger_left': 0.05})
            self.trajectories.append({'joint_lift': q_soln[3] + 0.05})
            self.trajectories.append({'wrist_extension': min(q_soln[5] + q_soln[6] + q_soln[7] + q_soln[8], 0.05)})

        elif self.mode == 'place':
            # move arm to marker and open
            self.trajectories.append({'translate_mobile_base': q_soln[1]})
            self.trajectories.append({'joint_lift': q_soln[3]})
            self.trajectories.append({'joint_wrist_yaw': q_soln[9]})
            self.trajectories.append({'joint_wrist_pitch': q_soln[11]})
            self.trajectories.append({'joint_wrist_roll': q_soln[12]})
            self.trajectories.append({'wrist_extension': q_soln[5] + q_soln[6] + q_soln[7] + q_soln[8]})
            self.trajectories.append({'joint_gripper_finger_left': 0.4})

            # retract arm
            self.trajectories.append({'joint_lift': q_soln[3] + 0.05})
            self.trajectories.append({'wrist_extension': min(q_soln[5] + q_soln[6] + q_soln[7] + q_soln[8], 0.05)})

        
        self.trajectory_timer = self.create_timer(0.1, self.trajectory_callback)

    def trajectory_callback(self):
        if len(self.trajectories) > 0 and self.goal_ready:
            # no active trajectories, send next trajectory
            trajectory = self.trajectories.pop(0)

            goal_msg = FollowJointTrajectory.Goal()
            point = JointTrajectoryPoint()
            point.positions = list(trajectory.values())
            point.time_from_start = Duration(seconds=1.0).to_msg()
            goal_msg.trajectory.points = [point]
            goal_msg.trajectory.joint_names = list(trajectory.keys())
            
            self._send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
            self._send_goal_future.add_done_callback(self.goal_response_callback)
            self.goal_ready = False
        elif len(self.trajectories) > 0:
            # trajectory is still active, wait
            return
        else:
            # all trajectories have been sent, finish process
            self.finished = True
            self.trajectory_timer.destroy()
            return


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result
        self.goal_ready = True


# pick or place an object given a point relative to the marker and the base to marker transform
def manipulate(marker_point, marker_trans, mode):
    rclpy.init()
    node = Manipulate(marker_point, marker_trans, mode)
    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node)
    transition = node.transition
    node.destroy_node()
    rclpy.shutdown()

    return transition


# client for changing robot mode to navigation mode
class NavigationMode(Node):
    def __init__(self):
        super().__init__('position_mode_client')
        self.cli = self.create_client(Trigger, 'switch_to_navigation_mode')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again')
        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    

# change robot mode to navigation mode
def switch_to_nav_mode():
    rclpy.init()
    node = NavigationMode()
    response = node.send_request()
    node.get_logger().info(f'{response}')
    node.destroy_node()
    rclpy.shutdown()


# client for changing robot mode to position mode
class PositionMode(Node):
    def __init__(self):
        super().__init__('position_mode_client')
        self.cli = self.create_client(Trigger, 'switch_to_position_mode')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again')
        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


# change robot mode to position mode
def switch_to_position_mode():
    rclpy.init()
    node = PositionMode()
    response = node.send_request()
    node.get_logger().info(f'{response}')
    node.destroy_node()
    rclpy.shutdown()


def main():
    pick_and_place.main()

if __name__ == '__main__':
    main()