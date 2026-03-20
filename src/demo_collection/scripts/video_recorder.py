#!/usr/bin/env python3
"""Video recorder: save camera intrinsics/extrinsics once + RGB+Depth frames.

Commands (via /demo_commands topic):
  'r'  — start recording
  's'  — stop recording
"""

import json
import os
import time

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

VIDEO_BASE_DIR = '/home/u0177383/doi_policy/foci_real_world/dataset/video_dataset'
os.makedirs(VIDEO_BASE_DIR, exist_ok=True)

# Minimum interval between saved frames (seconds).  ~10 fps.
MIN_FRAME_INTERVAL = 0.1


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_camera_info = None
        self.is_recording = False
        self.record_dir = None
        self.frame_meta = []
        self.last_save_time = 0.0

        # Synchronized RGB-D subscription
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.rgbd_callback)

        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 lambda m: setattr(self, 'latest_camera_info', m), 10)
        self.create_subscription(String, '/demo_commands', self.command_callback, 10)

        self.get_logger().info("Ready.  Send 'r' to start, 's' to stop.")

    # ------------------------------------------------------------------ #

    def command_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'r':
            self.start_recording()
        elif cmd == 's':
            self.stop_recording()

    def rgbd_callback(self, color_msg, depth_msg):
        if not self.is_recording:
            return

        now = time.monotonic()
        if now - self.last_save_time < MIN_FRAME_INTERVAL:
            return
        self.last_save_time = now

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception as e:
            self.get_logger().warn(f'Conversion error: {e}')
            return

        idx = len(self.frame_meta)
        stamp = color_msg.header.stamp
        ts = stamp.sec + stamp.nanosec * 1e-9

        cv2.imwrite(os.path.join(self.record_dir, 'color', f'{idx:05d}.png'), color)
        cv2.imwrite(os.path.join(self.record_dir, 'depth', f'{idx:05d}.png'), depth)

        self.frame_meta.append({'frame_idx': idx, 'timestamp': ts})
        self.get_logger().info(f'Saved frame {idx}', throttle_duration_sec=1.0)

    # ------------------------------------------------------------------ #

    def start_recording(self):
        if self.is_recording:
            return

        idx = max(
            (int(n[6:9]) for n in os.listdir(VIDEO_BASE_DIR)
             if n.startswith('video_') and n[6:9].isdigit()),
            default=-1
        ) + 1
        self.record_dir = os.path.join(VIDEO_BASE_DIR, f'video_{idx:03d}')
        os.makedirs(os.path.join(self.record_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'depth'), exist_ok=True)

        self.frame_meta = []
        self.last_save_time = 0.0
        self.is_recording = True

        # Save intrinsics once
        if self.latest_camera_info is not None:
            ci = self.latest_camera_info
            intrinsics = {'width': ci.width, 'height': ci.height,
                          'K': list(ci.k), 'D': list(ci.d),
                          'distortion_model': ci.distortion_model}
            with open(os.path.join(self.record_dir, 'camera_intrinsics.json'), 'w') as f:
                json.dump(intrinsics, f, indent=2)

        # Save extrinsics once
        try:
            tf = self.tf_buffer.lookup_transform(
                'fr3_link0', 'camera_color_optical_frame',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            t, r = tf.transform.translation, tf.transform.rotation
            extrinsics = {'translation': {'x': t.x, 'y': t.y, 'z': t.z},
                          'rotation':    {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w}}
            with open(os.path.join(self.record_dir, 'camera_extrinsics.json'), 'w') as f:
                json.dump(extrinsics, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Extrinsics unavailable: {e}')

        self.get_logger().info(f'Recording -> {self.record_dir}')

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        with open(os.path.join(self.record_dir, 'frames.json'), 'w') as f:
            json.dump(self.frame_meta, f, indent=2)
        self.get_logger().info(f'Saved {len(self.frame_meta)} frames -> {self.record_dir}')
        self.record_dir = None


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_recording:
            node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
