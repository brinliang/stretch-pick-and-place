import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup

from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Point, Quaternion, PoseWithCovarianceStamped, Transform, Vector3
from sensor_msgs.msg import Image, CameraInfo

from std_srvs.srv import Trigger

from control_msgs.action import FollowJointTrajectory
from nav2_msgs.action import NavigateToPose

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from threading import Event
import time
import numpy as np
import cv2
from ultralytics import YOLOWorld
from ultralytics import FastSAM

from stretch_perception import get_target_position, target_present, get_target_position_slow
from stretch_manipulation import rotate_arm_to_target, calculate_ik, calculate_motion, calculate_ik_from_point


class StretchActions(Node):
    def __init__(self, mode='aruco'):
        super().__init__('stretch3')
        self.robot_trajectory_done = Event()
        self.robot_nav_done = Event()
        self.robot_switch_done = Event()

        self.callback_group = ReentrantCallbackGroup()
        self.user_error = False

        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10, callback_group=self.callback_group)
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory', callback_group=self.callback_group)
        self.navigation_client = ActionClient(self, NavigateToPose, '/navigate_to_pose', callback_group=self.callback_group)
        self.nav_goal_handle = None

        self.position_mode_client = self.create_client(Trigger, 'switch_to_position_mode', callback_group=self.callback_group)
        self.navigation_mode_client = self.create_client(Trigger, 'switch_to_navigation_mode', callback_group=self.callback_group)

        self.position_mode_client.wait_for_service(timeout_sec=1.0)
        self.navigation_mode_client.wait_for_service(timeout_sec=1.0)
        self.trajectory_client.wait_for_server(timeout_sec=1.0)
        self.navigation_client.wait_for_server(timeout_sec=1.0)
 
        self.joint_states = None
        self.joint_states_sub = self.create_subscription(JointState, '/stretch/joint_states', self.joint_states_callback, 1, callback_group=self.callback_group)


        # perception

        # marker detection
        self.detection_active = False

        # listen for marker transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # try to find marker positions every 0.1 seconds
        self.detected_markers = dict()
        self.detect_markers = self.create_timer(0.1, self.detect_markers_callback, callback_group=self.callback_group)


        # object detection
        
        if mode == 'object_detection':
            self.rgb = None
            self.depth = None
            self.camera_info = None
            self.base_to_camera_tf = None

            self.rgb_subscriber = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_cb, 1, callback_group=self.callback_group)
            self.depth_subscriber = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_cb, 1, callback_group=self.callback_group)
            self.camera_subscriber = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_cb, 1, callback_group=self.callback_group)

            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_timer = self.create_timer(1, self.base_to_camera_tf_cb)

            self.detector = YOLOWorld('yolov8l-world.pt')
            self.segmenter = FastSAM('FastSAM-x.pt')


    def rgb_cb(self, msg):
        self.rgb = msg


    def depth_cb(self, msg):
        self.depth = msg


    def camera_cb(self, msg):
        self.camera_info = msg


    def base_to_camera_tf_cb(self):
        try:
            now = Time()
            self.base_to_camera_tf = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        except TransformException as e:
            pass

    
    def get_target_position(self, target_class='cup', confidence=0.5):
        while self.rgb is None or self.depth is None or self.camera_info is None or self.base_to_camera_tf is None:
            time.sleep(0.1)

        # process camera message
        camera = {
            'fx': self.camera_info.k[4],
            'cx': self.camera_info.k[5],
            'fy': self.camera_info.k[0],
            'cy': self.camera_info.k[2]
        }

        # process image message
        rgb_image = cv2.rotate(cv2.cvtColor(np.reshape(np.array(self.rgb.data), (self.rgb.height, self.rgb.width, 3)), 
            cv2.COLOR_RGB2BGR), # color should be in bgr format
            cv2.ROTATE_90_CLOCKWISE) # make sure image is in correct orientation
        
        # process depth message
        depth_data = np.frombuffer(self.depth.data, dtype=np.uint16)
        depth_image = cv2.rotate(np.reshape(depth_data, (self.depth.height, self.depth.width)), cv2.ROTATE_90_CLOCKWISE)

        # run perception
        point = get_target_position(rgb_image, depth_image, camera, self.base_to_camera_tf, self.detector, self.segmenter, target_class, confidence, 'cpu')
        # point = get_target_position_slow(rgb_image, depth_image, camera, self.base_to_camera_tf, self.detector, self.segmenter, target_class, confidence, 'cpu')


        return point
    

    def look_for_object(self, target_name, confidence=0.5, head_tilt=-0.5, pan_steps=9, left_limit=-3.6, right_limit=1.45):
        '''
        Move the head camera to look for a target object in the environment.
        '''
        while self.rgb is None:
            time.sleep(0.1)

        self.switch_to_position_mode()

        camera_positions = [[head_tilt, x] for x in np.linspace(left_limit, right_limit, pan_steps)]

        for position in camera_positions:
            trajectory = {
                'joint_head_tilt': position[0],
                'joint_head_pan': position[1]
            }

            self.follow_joint_trajectory(trajectory)

            time.sleep(1)

            rgb_image = cv2.rotate(cv2.cvtColor(np.reshape(np.array(self.rgb.data), (self.rgb.height, self.rgb.width, 3)), 
                cv2.COLOR_RGB2BGR), # color should be in bgr format
                cv2.ROTATE_90_CLOCKWISE) # make sure image is in correct orientation

            if target_present(rgb_image, self.detector, target_name, confidence):
                return 'success'
        
        return 'fail'


    def rotate_arm_to_target(self, x, y):
        trajectories = rotate_arm_to_target(x, y)
        transition = self.follow_joint_trajectories(trajectories)
        return transition


    def manipulate_from_marker_ik(self, marker_name, manipulation_info):
        '''
        Move the robot using inverse kinematics relative to a marker.
        The camera should be pointing towards the marker (call look_for_marker first)
        '''
        base_to_object = self.get_marker_transform(marker_name)
        self.manipulation_camera()

        self.follow_joint_trajectories(manipulation_info['pre_ik_motion'])
        
        ik_joints = calculate_ik(
            self.joint_states, 
            manipulation_info['ik_joint_names'], 
            base_to_object, 
            manipulation_info['object_to_gripper']
        )
        motion = calculate_motion(manipulation_info['post_ik_motion'], ik_joints)

        transition = self.follow_joint_trajectories(motion)

        time.sleep(2)

        if self.user_error:
            self.user_error = False
            return 'fail'

        return transition
    

    def manipulate_from_object_ik(self, target_class, manipulation_info):
        '''
        Move the robot using inverse kinematics relative to a target object.
        The camera should be pointing towards the object (call look_for_object first)

        Args:
            target_class (str): Name
            manipulation_info (dict): {
                'pre_ik_motion' (list<dict>): [{joint_name: value}],
                'ik_joint_names' (list<str>): ['joint_name'],
                'object_to_gripper' (dict): {
                    'x': float,
                    'y': float,
                    'z': float
                },
                'post_ik_motion' (list<dict>): [{'joint': joint name, 'mode': 'absolute' or 'relative_to_object', 'value': float}]
            }
        '''
        self.follow_joint_trajectories(manipulation_info['pre_ik_motion'])
        x, y, z = self.get_target_position(target_class)
        self.rotate_arm_to_target(x, y)
        base_to_object = self.get_target_position(target_class)

        ik_joints = calculate_ik_from_point(
            self.joint_states, 
            manipulation_info['ik_joint_names'], 
            base_to_object, 
            manipulation_info['object_to_gripper']
        )

        motion = calculate_motion(manipulation_info['post_ik_motion'], ik_joints)

        transition = self.follow_joint_trajectories(motion)

        time.sleep(2)

        if self.user_error:
            self.user_error = False
            return 'fail'

        return transition                  


    def publish_initial_pose(self, pose):
        '''
        Publish an estimate of the robot position on the nav2 map.
        '''
        initial_pose_msg = PoseWithCovarianceStamped()
        point = Point()
        point.x = pose['position']['x']
        point.y = pose['position']['y']
        point.z = pose['position']['z']
        initial_pose_msg.pose.pose.position = point
        orientation = Quaternion()
        orientation.x = pose['orientation']['x']
        orientation.y = pose['orientation']['y']
        orientation.z = pose['orientation']['z']
        orientation.w = pose['orientation']['w']
        initial_pose_msg.pose.pose.orientation = orientation
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'

        for i in range(5):
            self.initial_pose_pub.publish(initial_pose_msg)
            time.sleep(1)

        return 'success'


    def switch_to_navigation_mode(self):
        '''
        Switch the robot to navigation mode for nav2.
        '''
        request = Trigger.Request()
        self.robot_ready = False

        self.robot_switch_done.clear()

        future = self.navigation_mode_client.call_async(request)
        future.add_done_callback(self.call_async_callback)

        self.robot_switch_done.wait()


    def switch_to_position_mode(self):
        '''
        Switch the robot to position mode for manipulation.
        '''
        request = Trigger.Request()
        self.robot_ready = False

        self.robot_switch_done.clear()

        future = self.position_mode_client.call_async(request)
        future.add_done_callback(self.call_async_callback)

        self.robot_switch_done.wait()

    
    def call_async_callback(self, future):
        # flag robot as ready when future is done
        self.robot_switch_done.set()


    def cancel_nav_goal_callback(self, future):
        self.robot_nav_done.set()
        return

    def detect_markers_callback(self):
        # update detected markers
        if self.detection_active:
            for marker in list(self.detected_markers.keys()):
                try:
                    now = Time()
                    trans = self.tf_buffer.lookup_transform('base_link', marker, now)
                    self.detected_markers[marker] = trans
                except TransformException as e:
                    pass
            return


    def look_for_marker(self, marker_name, head_tilt=-0.5, pan_steps=9, left_limit=-3.6, right_limit=1.45):
        '''
        Look for a marker in the environment by moving the head camera.

        Args:
            marker_name (str): Name of the marker to look for.
            head_tilt (float): Tilt of the head camera.
            pan_steps (int): Number of steps to pan the head camera.
            left_limit (float): Left limit of the pan.
            right_limit (float): Right limit of the pan.
        '''
        self.switch_to_position_mode()
        camera_positions = [[head_tilt, x] for x in np.linspace(left_limit, right_limit, pan_steps)]

        # make sure marker is not detected before starting
        time.sleep(5)

        self.detection_active = True
        self.detected_markers[marker_name] = None

        for position in camera_positions:

            time.sleep(0.5)

            if self.detected_markers[marker_name] is not None:
                self.detection_active = False
                return 'success'

            if self.user_error:
                self.detection_active = False
                self.user_error = False
                return 'fail'

            trajectory = {
                'joint_head_tilt': position[0],
                'joint_head_pan': position[1]
            }

            self.follow_joint_trajectory(trajectory)
            

            time.sleep(0.5)

            if self.detected_markers[marker_name] is not None:
                self.detection_active = False
                return 'success'
            
            if self.user_error:
                self.detection_active = False
                self.user_error = False
                return 'fail'
            
        return 'fail'


    def follow_joint_trajectory(self, trajectory):
        goal_msg = FollowJointTrajectory.Goal()
        point = JointTrajectoryPoint()
        point.positions = list(trajectory.values())
        goal_msg.trajectory.points = [point]
        goal_msg.trajectory.joint_names = list(trajectory.keys())

        self.robot_trajectory_done.clear()

        future = self.trajectory_client.send_goal_async(goal_msg)
        future.add_done_callback(self.trajectory_goal_callback)

        self.robot_trajectory_done.wait()


    def follow_joint_trajectories(self, trajectories):
        '''
        Follows a list of joint trajectories in order.
        Args:
            trajectories (list<dictionary>): List of dictionaries containing joint names and positions.
        '''
        self.switch_to_position_mode()

        for trajectory in trajectories:

            if self.user_error:
                self.user_error = False
                return 'fail'
            
            self.follow_joint_trajectory(trajectory)

            if self.user_error:
                self.user_error = False
                return 'fail'

            time.sleep(0.2)

        return 'success'


    def trajectory_goal_callback(self, future):
        # get result of goal
        goal_handle = future.result()
        # print(goal_handle)
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.trajectory_result_callback)


    def trajectory_result_callback(self, future):
        # flag robot as ready when future is done
        if future.result().result.error_code == 100:
            time.sleep(3)
        self.robot_trajectory_done.set()


    def navigate_to_pose(self, pose):
        '''
        Navigate to a given pose on a map using stretch nav2

        Args:
            pose (dict): {
                'position': {
                    'x': float,
                    'y': float,
                    'z': float
                },
                'orientation': {
                    'x': float,
                    'y': float,
                    'z': float,
                    'w': float
            }
        '''
        self.switch_to_navigation_mode()
        
        # translate pose into goal message
        goal_msg = NavigateToPose.Goal()
        point = Point()
        point.x = pose['position']['x']
        point.y = pose['position']['y']
        point.z = pose['position']['z']
        goal_msg.pose.pose.position = point
        orientation = Quaternion()
        orientation.x = pose['orientation']['x']
        orientation.y = pose['orientation']['y']
        orientation.z = pose['orientation']['z']
        orientation.w = pose['orientation']['w']
        goal_msg.pose.pose.orientation = orientation
        goal_msg.pose.header.frame_id = 'map'

        self.robot_nav_done.clear()

        future = self.navigation_client.send_goal_async(goal_msg)
        future.add_done_callback(self.nav_goal_callback)

        self.robot_nav_done.wait()

        time.sleep(1)

        if self.user_error:
            self.user_error = False
            return 'fail'

        return 'success'


    def nav_goal_callback(self, future):
        self.nav_goal_handle = future.result()
        future = self.nav_goal_handle.get_result_async()
        future.add_done_callback(self.nav_result_callback)


    def nav_result_callback(self, future):
        if not future.cancelled():
            self.robot_nav_done.set()


    def navigation_camera(self):
        '''
        Point the head camera forwards.
        '''
        self.switch_to_position_mode()
        self.follow_joint_trajectory({
            'joint_head_tilt': 0.0,
            'joint_head_pan': 0.0
        })
        

    def manipulation_camera(self):
        '''
        Point the head camera towards the arm.
        '''
        self.switch_to_position_mode()
        self.follow_joint_trajectory({
            'joint_head_tilt': -0.5,
            'joint_head_pan': -1.52,
        })


    def joint_states_callback(self, msg):
        self.joint_states = msg


    def get_marker_transform(self, marker_name):
        # average marker transform over 10 iterations for stability
        self.detection_active = True
        iters = 10

        def tf_to_dict(tf):
            return {
                'x': tf.transform.translation.x,
                'y': tf.transform.translation.y,
                'z': tf.transform.translation.z,
                'qx': tf.transform.rotation.x,
                'qy': tf.transform.rotation.y,
                'qz': tf.transform.rotation.z,
                'qw': tf.transform.rotation.w
            }
        
        sum = {
            'x': 0,
            'y': 0,
            'z': 0,
            'qx': 0,
            'qy': 0,
            'qz': 0,
            'qw': 0
        }

        for i in range(iters):
            for j in list(sum.keys()):
                sum[j] += tf_to_dict(self.detected_markers[marker_name])[j]

            time.sleep(0.4)
        
        for j in list(sum.keys()):
            sum[j] /= iters

        marker_tf_message = Transform()
        marker_vector = Vector3()
        marker_vector.x = sum['x']
        marker_vector.y = sum['y']
        marker_vector.z = sum['z']
        marker_orientation = Quaternion()
        marker_orientation.x = sum['qx']
        marker_orientation.y = sum['qy']
        marker_orientation.z = sum['qz']
        marker_orientation.w = sum['qw']
        marker_tf_message.translation = marker_vector
        marker_tf_message.rotation = marker_orientation

        self.detection_active = False

        return marker_tf_message


    def get_marker_transforms(self):
        return self.detected_markers
    

    def get_joint_states(self):
        return self.joint_states
    