import yaml

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RecordIKPoint(Node):

    def __init__(self, filepath='save.yaml', marker_name='', key_name='marker_point'):
        super().__init__('record_ik_point')

        self.target_frame = marker_name

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        time_period = 1.0
        self.timer = self.create_timer(time_period, self.on_timer)
        
        self.filepath = filepath
        self.key_name = key_name
        self.finished = False

        self.average = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.count = 0

    def on_timer(self):
        from_frame_rel = 'link_grasp_center'
        to_frame_rel = self.target_frame
        if self.count < 10:

            try:
                now = Time()
                trans = self.tf_buffer.lookup_transform(to_frame_rel, from_frame_rel, now)
            except TransformException as ex:
                self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                return
            
            self.get_logger().info(f'{trans.transform.translation}')

            self.average['x'] += trans.transform.translation.x
            self.average['y'] += trans.transform.translation.y
            self.average['z'] += trans.transform.translation.z

            self.count += 1
        else:
            self.average['x'] /= self.count
            self.average['y'] /= self.count
            self.average['z'] /= self.count

            with open(self.filepath, 'r+') as f:
                data = yaml.safe_load(f)

                if data is None:
                    data = {}
                
                if self.key_name not in data:
                    data[self.key_name] = {}

                data[self.key_name]['x'] = self.average['x']
                data[self.key_name]['y'] = self.average['y']
                data[self.key_name]['z'] = self.average['z']
                f.seek(0)
                yaml.dump(data, f)


            self.get_logger().info(f'the pose of target frame {from_frame_rel} with reference to {to_frame_rel} is: {self.average}')  

            self.finished = True 


def main():
    filepath = 'save.yaml'
    marker_name = 'hanger'
    key_name = 'pick_bottle'


    rclpy.init()
    node = RecordIKPoint(filepath, marker_name, key_name)
    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()