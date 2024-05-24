from ultralytics.models.fastsam import FastSAMPrompt

import numpy as np
import cv2
import pickle


def target_present(rgb, detector, target_class, confidence=0.5):
    detector.set_classes([target_class])

    detections = detector.predict(rgb, conf=confidence)
    detected_classes = detections[0].boxes.cls.tolist()

    if len(detected_classes) == 0:
        return False

    return True



def get_target_position(rgb, depth, camera, base_to_camera_tf, detector, segmenter, target_class, confidence=0.5, device='cpu'):
    detector.set_classes([target_class])

    original_size = rgb.shape[:2]
    modified_size = (original_size[0] - original_size[0] % 32, original_size[1] - original_size[1] % 32)

    detections = detector.predict(rgb[:modified_size[0], :modified_size[1], :], imgsz=modified_size, conf=confidence)

    detected_classes = detections[0].boxes.cls.tolist()
    detected_boxes = detections[0].boxes.xyxy.tolist()
    target_index = 0

    if len(detected_classes) == 0:
        return 'fail'

    everything_results = segmenter(detections[0].orig_img, device=device, imgsz=modified_size)
    prompt_process = FastSAMPrompt(detections[0].orig_img, everything_results, device=device)
    segments = prompt_process.box_prompt(bbox=detected_boxes[target_index])
    
    target_segment = segments[0].masks.data[0]

    mask = np.array(1 - target_segment.cpu().detach().numpy())

    x = np.ma.masked_equal(np.array(depth[:modified_size[0], :modified_size[1]] / 1000), 0.0)
    y = ((camera['cy'] - np.repeat(range(modified_size[0]), modified_size[1]).reshape(modified_size)) * x) / camera['fy']
    z = ((np.tile(range(modified_size[1]), modified_size[0]).reshape(modified_size) - camera['cx']) * x) / camera['fx']


    mask_x = np.ma.masked_array(x, mask=mask)
    mask_y = np.ma.masked_array(y, mask=mask)
    mask_z = np.ma.masked_array(z, mask=mask)

    mean_x = np.mean(mask_x)
    mean_y = np.mean(mask_y)
    mean_z = np.mean(mask_z)

    q = base_to_camera_tf.transform.rotation
    t = base_to_camera_tf.transform.translation
    transform_matrix = np.array(
        [
            [1-2*q.y**2-2*q.z**2, 2*q.x*q.y-2*q.z*q.w, 2*q.x*q.z+2*q.y*q.w, t.x],
            [2*q.x*q.y+2*q.z*q.w, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z-2*q.x*q.w, t.y],
            [2*q.x*q.z-2*q.y*q.w, 2*q.y*q.z+2*q.x*q.w, 1-2*q.x**2-2*q.y**2, t.z],
            [0, 0, 0, 1]
        ]
    )

    center_wrt_base = np.dot(transform_matrix, np.array([mean_x, mean_y, mean_z, 1]))[:3]

    return center_wrt_base



def get_target_position_slow(rgb, depth, camera, base_to_camera_tf, detector, segmenter, target_class, confidence=0.5, device='cpu'):
    detector.set_classes([target_class])

    original_size = rgb.shape[:2]
    modified_size = (original_size[0] - original_size[0] % 32, original_size[1] - original_size[1] % 32)

    detections = detector.predict(rgb[:modified_size[0], :modified_size[1], :], imgsz=modified_size, conf=confidence)

    detected_classes = detections[0].boxes.cls.tolist()
    detected_boxes = detections[0].boxes.xyxy.tolist()
    target_index = 0

    if len(detected_classes) == 0:
        return 'fail'

    everything_results = segmenter(detections[0].orig_img, device=device, imgsz=modified_size)
    prompt_process = FastSAMPrompt(detections[0].orig_img, everything_results, device=device)
    segments = prompt_process.box_prompt(bbox=detected_boxes[target_index])
    
    segment = segments[0].masks.data[0]

    center = (0, 0, 0)
    count = 0
    for i in range(segment.shape[0]):
        for j in range(segment.shape[1]):
            if segment[i][j] > 0:
                if depth[i][j] == 0:
                    continue
                count += 1

                # convert to world coordinates
                x = depth[i][j] / 1000
                y = (camera['cy'] - i) * depth[i, j] / camera['fy'] / 1000
                z = (j - camera['cx']) * depth[i, j] / camera['fx'] / 1000

                center = (center[0] + x, center[1] + y, center[2] + z)

    center = (center[0] / count, center[1] / count, center[2] / count)

    q = base_to_camera_tf.transform.rotation
    t = base_to_camera_tf.transform.translation
    transform_matrix = np.array(
        [
            [1-2*q.y**2-2*q.z**2, 2*q.x*q.y-2*q.z*q.w, 2*q.x*q.z+2*q.y*q.w, t.x],
            [2*q.x*q.y+2*q.z*q.w, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z-2*q.x*q.w, t.y],
            [2*q.x*q.z-2*q.y*q.w, 2*q.y*q.z+2*q.x*q.w, 1-2*q.x**2-2*q.y**2, t.z],
            [0, 0, 0, 1]
        ]
    )

    center_wrt_base = np.dot(transform_matrix, np.array([center[0], center[1], center[2], 1]))[:3]

    return center_wrt_base







