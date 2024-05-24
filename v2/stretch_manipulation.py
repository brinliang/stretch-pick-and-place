import ikpy.chain
import os
import numpy as np


def joint_to_stretch_index(joint):
    joint_map = {
        'wrist_extension': 0,
        'joint_lift': 1,
        'joint_arm_l3': 2,
        'joint_arm_l2': 3,
        'joint_arm_l1': 4,
        'joint_arm_l0': 5,
        'joint_head_pan': 6,
        'joint_head_tilt': 7,
        'joint_wrist_yaw': 8,
        'joint_wrist_pitch': 9,
        'joint_wrist_roll': 10,
        'joint_gripper_finger_left': 11,
        'joint_gripper_finger_right': 12
    }

    if joint in joint_map:
        return joint_map[joint]


def joint_to_ik_index(joint):
    joint_map = {
        'translate_mobile_base': 1,
        'joint_lift': 3,
        'joint_arm_l3': 5,
        'joint_arm_l2': 6,
        'joint_arm_l1': 7,
        'joint_arm_l0': 8,
        'joint_wrist_yaw': 9,
        'joint_wrist_pitch': 11,
        'joint_wrist_roll': 12,
    }
    
    if joint == 'wrist_extension':
        return range(5,9)
    elif joint in joint_map:
        return joint_map[joint]


def stretch_joints_to_ik_joints(joint_states):
    return [
        0.0, 0.0, 0.0, 
        joint_states.position[1], 
        0.0, 
        joint_states.position[2], 
        joint_states.position[3], 
        joint_states.position[4], 
        joint_states.position[5], 
        joint_states.position[8], 
        0.0, 
        joint_states.position[9], 
        joint_states.position[10], 
        0.0, 0.0
    ]


def calculate_ik(initial_joints, used_joints, base_to_object, object_to_gripper):
    '''
    Args:
        initial_joints: sensor_msgs.msg.JointState
        used_joints: list of str
        base_to_object: geometry_msgs.msg.Transform
        object_to_gripper: dict{'x': float, 'y': float, 'z': float}
    '''
    # use stretch joint states as initial ik joint states
    q_init = stretch_joints_to_ik_joints(initial_joints)

    # use only specified joints in ik
    active_links = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    for joint in used_joints:
        if joint == 'wrist_extension':
            for i in range(5,9):
                active_links[i] = True
        else:
            active_links[joint_to_ik_index(joint)] = True

    # create chain
    urdf_filename = os.path.join(os.path.dirname(__file__), 'stretch.urdf')
    chain = ikpy.chain.Chain.from_urdf_file(urdf_filename, active_links_mask=active_links)

    # calculate transformation from object coordinate frame to base coordinate frame
    q = base_to_object.rotation
    t = base_to_object.translation
    base_to_object_matrix = np.array(
        [
            [1-2*q.y**2-2*q.z**2, 2*q.x*q.y-2*q.z*q.w, 2*q.x*q.z+2*q.y*q.w, t.x],
            [2*q.x*q.y+2*q.z*q.w, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z-2*q.x*q.w, t.y],
            [2*q.x*q.z-2*q.y*q.w, 2*q.y*q.z+2*q.x*q.w, 1-2*q.x**2-2*q.y**2, t.z],
            [0, 0, 0, 1]
        ]
    )

    # convert object to gripper dictionary to vector
    object_to_gripper_vector = np.array([object_to_gripper['x'], object_to_gripper['y'], object_to_gripper['z'], 1])

    # get gripper wrt base from object wrt base and gripper wrt object
    target_ik_point = np.dot(base_to_object_matrix, object_to_gripper_vector)[:3]

    # calculate target joints using ik
    try:
        q_soln = chain.inverse_kinematics(target_ik_point, initial_position=q_init)
    except:
        return None
    
    # return dictionary of target joints
    ik_joints = dict()
    for joint in used_joints:
        if joint in ['joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0'] and 'wrist_extension' not in list(ik_joints.keys()):
            ik_joints['wrist_extension'] = q_soln[joint]
        elif joint in ['joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0']:
            ik_joints['wrist_extension'] += q_soln[joint]
        elif joint == 'wrist_extension':
            ik_joints[joint] = q_soln[5] + q_soln[6] + q_soln[7] + q_soln[8]
        else:
            ik_joints[joint] = q_soln[joint_to_ik_index(joint)]

    return ik_joints


def calculate_motion(target_joints, ik_joints):
    '''
    
    Args:
        target_joints: list of dict{'joint': str, 'mode': str, 'value': float}
        ik_joints: dict{joint_name: value}
    '''
    trajectories = []

    for joint_info in target_joints:
        if joint_info['mode'] == 'absolute':
            trajectories.append({joint_info['joint']: joint_info['value']})
        elif joint_info['mode'] == 'relative_to_object':
            trajectories.append({joint_info['joint']: joint_info['value'] + ik_joints[joint_info['joint']]})

    return trajectories


def rotate_arm_to_target(x, y):
    rotation = np.arctan2(y, x) + np.pi / 2

    trajectories = [{
        'rotate_mobile_base': rotation,
        'joint_head_pan': -1.57
    }]

    return trajectories


def calculate_ik_from_point(initial_joints, used_joints, base_to_object, object_translation):
    '''
    Args:
        initial_joints: sensor_msgs.msg.JointState
        used_joints: list of str
        base_to_object: np.array(3)
        object_translation: dict{'x': float, 'y': float, 'z': float}
    '''
    # use stretch joint states as initial ik joint states
    q_init = stretch_joints_to_ik_joints(initial_joints)

    # use only specified joints in ik
    active_links = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    for joint in used_joints:
        if joint == 'wrist_extension':
            for i in range(5,9):
                active_links[i] = True
        else:
            active_links[joint_to_ik_index(joint)] = True

    # create chain
    urdf_filename = os.path.join(os.path.dirname(__file__), 'stretch.urdf')
    chain = ikpy.chain.Chain.from_urdf_file(urdf_filename, active_links_mask=active_links)

    print(base_to_object)

    object_translation_np = np.array([object_translation['x'], object_translation['y'], object_translation['z']])
    target_point = base_to_object + object_translation_np

    print(target_point)

    # calculate target joints using ik
    try:
        q_soln = chain.inverse_kinematics(target_point, initial_position=q_init)
    except:
        return None
    
    # return dictionary of target joints
    ik_joints = dict()
    for joint in used_joints:
        if joint in ['joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0'] and 'wrist_extension' not in list(ik_joints.keys()):
            ik_joints['wrist_extension'] = q_soln[joint]
        elif joint in ['joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0']:
            ik_joints['wrist_extension'] += q_soln[joint]
        elif joint == 'wrist_extension':
            ik_joints[joint] = q_soln[5] + q_soln[6] + q_soln[7] + q_soln[8]
        else:
            ik_joints[joint] = q_soln[joint_to_ik_index(joint)]

    return ik_joints
