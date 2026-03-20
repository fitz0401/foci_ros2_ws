#!/usr/bin/env python3
"""
Complete Demo Recording System for Franka Robot
Subscribes to /demo_commands (std_msgs/String) for control:
- 'g' to toggle gripper (open/close)
- 'r' to start recording demo
- 's' to stop recording demo
- 'q' to quit
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Move
from control_msgs.action import GripperCommand
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import threading
import json
import os
import cv2
from cv_bridge import CvBridge
from queue import Queue


demo_base_dir = '/home/u0177383/doi_policy/foci_real_world/dataset'
os.makedirs(demo_base_dir, exist_ok=True)


class DemoRecorder(Node):
    def __init__(self):
        super().__init__('demo_recorder')
        self.gripper_type = self.declare_parameter('gripper', 'franka').value.lower().strip()
        if self.gripper_type not in ('franka', 'robotiq'):
            self.get_logger().warn(
                f"Unknown gripper type '{self.gripper_type}', fallback to 'franka'"
            )
            self.gripper_type = 'franka'
        self.robotiq_joint_name = 'robotiq_85_left_knuckle_joint'
        self.robotiq_open_position = 0.0
        self.robotiq_closed_position = 0.8
        self.robotiq_toggle_threshold = 0.05
        self.robotiq_joint_position = 0.0
        self._robotiq_joint_warned = False
        self._robotiq_goal_in_flight = False

        # Gripper action clients
        self.franka_grasp_client = None
        self.franka_move_client = None
        self.robotiq_gripper_client = None
        if self.gripper_type == 'robotiq':
            self.robotiq_gripper_client = ActionClient(
                self,
                GripperCommand,
                '/robotiq/robotiq_gripper_controller/gripper_cmd'
            )
            gripper_joint_topic = '/robotiq/joint_states'
        else:
            self.franka_grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
            self.franka_move_client = ActionClient(self, Move, '/fr3_gripper/move')
            gripper_joint_topic = '/fr3_gripper/joint_states'

        # TF buffer for getting end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        # Subscribers
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        self.gripper_state_sub = self.create_subscription(JointState, gripper_joint_topic, self.gripper_state_callback, 10)
        self.command_sub = self.create_subscription(String, '/demo_commands', self.command_callback, 10)
        # Data storage
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_gripper_width = 0.0  # Joint position sum
        # Velocity tracking for filtering stationary frames
        self.prev_gripper_pose = None
        self.prev_timestamp = None
        # Track previous gripper state for detecting state changes
        self.prev_gripper_state = None
        # Recording state
        self.is_recording = False
        self.demo_data = []
        self.demo_count = 0
        self.should_quit = False
        # Async saving queue and thread
        self.save_queue = Queue(maxsize=100)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        # Recording timer (5 Hz)
        self.record_timer = self.create_timer(0.2, self.record_frame)
        # Gripper homing (non-blocking)
        self.gripper_ready = False
        threading.Thread(target=self._init_gripper, daemon=True).start()
        self.print_instructions()

    def _init_gripper(self):
        """Initialize gripper by opening it"""
        if self.gripper_type == 'robotiq':
            if self.move_robotiq_gripper(self.robotiq_open_position, max_effort=50.0, timeout=5.0):
                self.gripper_ready = True
                print("Robotiq gripper initialized and opened")
            return

        if self.franka_move_client.wait_for_server(timeout_sec=5.0):
            goal = Move.Goal()
            goal.width = 0.08
            goal.speed = 0.1
            self.franka_move_client.send_goal_async(goal)
            self.gripper_ready = True
            print("Franka gripper initialized and opened")

    def print_instructions(self):
        print("\n" + "="*60)
        print("DEMO RECORDING SYSTEM")
        print(f"Gripper type: {self.gripper_type}")
        print("="*60)
        print("Listening to /demo_commands topic for:")
        print("  'g' - Toggle gripper (open/close)")
        print("  'r' - START recording demo")
        print("  's' - STOP recording demo")
        print("  'q' - Quit program")
        print("="*60 + "\n")

    def color_callback(self, msg):
        self.latest_color = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def gripper_state_callback(self, msg):
        if len(msg.position) == 0:
            return

        if self.gripper_type == 'robotiq':
            try:
                idx = msg.name.index(self.robotiq_joint_name)
                self.robotiq_joint_position = float(msg.position[idx])
                self.latest_gripper_width = self.robotiq_joint_position
                self._robotiq_joint_warned = False
            except ValueError:
                if not self._robotiq_joint_warned:
                    self.get_logger().warn(
                        f"Joint '{self.robotiq_joint_name}' not found in /robotiq/joint_states"
                    )
                    self._robotiq_joint_warned = True
        else:
            self.latest_gripper_width = sum(msg.position)

    def command_callback(self, msg):
        cmd = msg.data.lower().strip()
        if cmd == 'g':
            self.toggle_gripper()
        elif cmd == 'r':
            self.start_recording()
        elif cmd == 's':
            self.stop_recording()
        elif cmd == 'q':
            print("Quit command received, shutting down...")
            self.should_quit = True

    def get_gripper_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'fr3_link0', 'fr3_hand_tcp', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            pose = {
                'position': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z
                },
                'orientation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w
                }
            }
            return pose
        except Exception as e:
            self.get_logger().warn(f'Failed to get gripper pose: {e}')
            return None

    def get_camera_extrinsics(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'fr3_link0', 'camera_color_optical_frame', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            extrinsics = {
                'translation': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z
                },
                'rotation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w
                }
            }
            return extrinsics
        except Exception as e:
            self.get_logger().warn(f'Failed to get camera extrinsics: {e}')
            return None

    def calculate_gripper_velocity(self, current_pose, current_time):
        """Calculate gripper linear velocity in m/s"""
        if self.prev_gripper_pose is None or self.prev_timestamp is None:
            self.prev_gripper_pose = current_pose
            self.prev_timestamp = current_time
            return 0.0
        
        dt = current_time - self.prev_timestamp
        if dt <= 0:
            return 0.0
        
        # Calculate linear displacement
        dx = current_pose['position']['x'] - self.prev_gripper_pose['position']['x']
        dy = current_pose['position']['y'] - self.prev_gripper_pose['position']['y']
        dz = current_pose['position']['z'] - self.prev_gripper_pose['position']['z']
        
        displacement = (dx**2 + dy**2 + dz**2)**0.5
        velocity = displacement / dt
        
        # Update previous values
        self.prev_gripper_pose = current_pose
        self.prev_timestamp = current_time
        
        return velocity

    def record_frame(self):
        if not self.is_recording:
            return
        gripper_pose = self.get_gripper_pose()
        if gripper_pose is None:
            return
        if self.latest_color is None:
            self.get_logger().warn('Waiting for sensor data...')
            return
        
        # Calculate gripper velocity
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        velocity = self.calculate_gripper_velocity(gripper_pose, current_time)
        
        # Determine gripper state from actual width/position
        if self.gripper_type == 'robotiq':
            gripper_state = 'open' if self.robotiq_joint_position <= self.robotiq_toggle_threshold else 'closed'
        else:
            gripper_state = 'open' if self.latest_gripper_width > 0.04 else 'closed'
        
        # Check if gripper state changed
        gripper_state_changed = (self.prev_gripper_state is not None and 
                                 gripper_state != self.prev_gripper_state)
        
        # Skip frame if velocity is too low AND gripper state hasn't changed
        if velocity < 0.001 and not gripper_state_changed:
            return
        
        # Update previous gripper state
        self.prev_gripper_state = gripper_state
        frame_data = {
            'timestamp': current_time,
            'gripper_pose': gripper_pose,
            'gripper_state': gripper_state,
            'velocity': float(velocity)
        }
        self.demo_data.append(frame_data)
        frame_idx = len(self.demo_data) - 1
        self.save_frame_data(frame_idx)
        if frame_idx % 5 == 0:
            self.get_logger().info(f'Recording frame {frame_idx}... (vel: {velocity:.4f} m/s)')

    def save_frame_data(self, frame_idx):
        demo_dir = f'{demo_base_dir}/demo_{self.demo_count:03d}'
        save_data = {
            'demo_dir': demo_dir,
            'frame_idx': frame_idx,
            'color_img': self.latest_color,
            'depth_img': self.latest_depth
        }
        try:
            self.save_queue.put_nowait(save_data)
        except:
            self.get_logger().warn('Save queue full, dropping frame')

    def _save_worker(self):
        while True:
            try:
                save_data = self.save_queue.get()
                if save_data is None:
                    break
                demo_dir = save_data['demo_dir']
                frame_idx = save_data['frame_idx']
                color_dir = os.path.join(demo_dir, 'color')
                depth_dir = os.path.join(demo_dir, 'depth')
                if save_data['color_img'] is not None:
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(save_data['color_img'], 'bgr8')
                        color_path = os.path.join(color_dir, f'{frame_idx:04d}.png')
                        cv2.imwrite(color_path, cv_image)
                    except Exception:
                        pass
                if save_data['depth_img'] is not None:
                    try:
                        depth_image = self.bridge.imgmsg_to_cv2(save_data['depth_img'], 'passthrough')
                        depth_path = os.path.join(depth_dir, f'{frame_idx:04d}.png')
                        cv2.imwrite(depth_path, depth_image)
                    except Exception:
                        pass
            except Exception:
                pass

    def start_recording(self):
        if self.is_recording:
            self.get_logger().warn('Already recording!')
            return
        max_idx = 0
        for name in os.listdir(demo_base_dir):
            if name.startswith('demo_') and name[5:8].isdigit():
                idx = int(name[5:8])
                if idx > max_idx:
                    max_idx = idx
        self.demo_count = max_idx + 1
        demo_dir = f'{demo_base_dir}/demo_{self.demo_count:03d}'
        os.makedirs(demo_dir, exist_ok=True)
        os.chmod(demo_dir, 0o777)
        # Create subdirectories for images
        os.makedirs(os.path.join(demo_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(demo_dir, 'depth'), exist_ok=True)
        os.chmod(os.path.join(demo_dir, 'color'), 0o777)
        os.chmod(os.path.join(demo_dir, 'depth'), 0o777)
        self.demo_data = []
        self.is_recording = True
        # Reset velocity tracking
        self.prev_gripper_pose = None
        self.prev_timestamp = None
        self.prev_gripper_state = None
        if self.latest_camera_info is not None:
            intrinsics = {
                'width': self.latest_camera_info.width,
                'height': self.latest_camera_info.height,
                'K': self.latest_camera_info.k.tolist(),
                'D': self.latest_camera_info.d.tolist(),
                'distortion_model': self.latest_camera_info.distortion_model
            }
            with open(os.path.join(demo_dir, 'camera_intrinsics.json'), 'w') as f:
                json.dump(intrinsics, f, indent=2)
            os.chmod(os.path.join(demo_dir, 'camera_intrinsics.json'), 0o777)
        extrinsics = self.get_camera_extrinsics()
        if extrinsics is not None:
            with open(os.path.join(demo_dir, 'camera_extrinsics.json'), 'w') as f:
                json.dump(extrinsics, f, indent=2)
            os.chmod(os.path.join(demo_dir, 'camera_extrinsics.json'), 0o777)
        self.get_logger().info(f'Started recording demo {self.demo_count}')
        print(f"\n>>> RECORDING DEMO {self.demo_count} <<<\n")

    def stop_recording(self):
        if not self.is_recording:
            self.get_logger().warn('Not recording!')
            return
        self.is_recording = False
        demo_dir = f'{demo_base_dir}/demo_{self.demo_count:03d}'
        with open(os.path.join(demo_dir, 'trajectory.json'), 'w') as f:
            json.dump(self.demo_data, f, indent=2)
        os.chmod(os.path.join(demo_dir, 'trajectory.json'), 0o777)
        self.get_logger().info(f'Stopped recording. Saved {len(self.demo_data)} frames to {demo_dir}/')
        print(f"\n>>> DEMO {self.demo_count} SAVED ({len(self.demo_data)} frames) <<<")

    def toggle_gripper(self):
        if not self.gripper_ready:
            print("!!! Gripper not ready")
            return
        # Determine current state from actual width
        if self.gripper_type == 'robotiq':
            if self._robotiq_goal_in_flight:
                self.get_logger().info('Robotiq gripper is moving, command ignored')
                return
            if self.robotiq_joint_position > self.robotiq_toggle_threshold:
                self.open_gripper()
            else:
                self.close_gripper()
            return

        if self.latest_gripper_width > 0.04:
            self.close_gripper()
        else:
            self.open_gripper()

    def send_gripper_command(self, width, speed=0.1):
        """Use Franka Move action for precise gripper control"""
        if self.franka_move_client is None:
            self.get_logger().warn('Franka move client not initialized')
            return
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        self.franka_move_client.send_goal_async(goal)

    def move_robotiq_gripper(self, position, max_effort=50.0, timeout=5.0):
        """Use Robotiq GripperCommand action for open/close control"""
        if self.robotiq_gripper_client is None:
            self.get_logger().warn('Robotiq gripper client not initialized')
            return False

        if self._robotiq_goal_in_flight:
            self.get_logger().info('Robotiq gripper is moving, command ignored')
            return False

        if not self.robotiq_gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Robotiq gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        send_future = self.robotiq_gripper_client.send_goal_async(goal)
        self._robotiq_goal_in_flight = True

        def _goal_response_cb(fut):
            try:
                handle = fut.result()
            except Exception as exc:
                self._robotiq_goal_in_flight = False
                self.get_logger().warn(f'Gripper goal request failed: {exc}')
                return

            if not handle or not handle.accepted:
                self._robotiq_goal_in_flight = False
                self.get_logger().error('Gripper goal rejected')
                return

            self.gripper_ready = True
            res_future = handle.get_result_async()

            def _result_cb(result_fut):
                self._robotiq_goal_in_flight = False
                try:
                    res = result_fut.result()
                    if res is None:
                        self.get_logger().warn('Gripper result is None')
                        return
                    self.get_logger().info(f'Gripper action finished with status={res.status}')
                except Exception as exc:
                    self.get_logger().warn(f'Gripper result error: {exc}')

            res_future.add_done_callback(_result_cb)

        send_future.add_done_callback(_goal_response_cb)
        return True


    def open_gripper(self):
        if self.gripper_type == 'robotiq':
            print(">>> Opening Robotiq gripper")
            self.move_robotiq_gripper(self.robotiq_open_position, max_effort=50.0)
            return

        print(">>> Opening gripper")
        self.send_gripper_command(0.08, speed=0.1)

    def close_gripper(self):
        if self.gripper_type == 'robotiq':
            print(">>> Closing Robotiq gripper")
            self.move_robotiq_gripper(self.robotiq_closed_position, max_effort=50.0)
            return

        print(">>> Closing gripper (grasp)")
        # Use Grasp action for closing
        if self.franka_grasp_client is None:
            self.get_logger().warn('Franka grasp client not initialized')
            return
        grasp_goal = Grasp.Goal()
        grasp_goal.width = 0.0
        grasp_goal.speed = 0.1
        grasp_goal.force = 20.0  # Grasp force in N
        grasp_goal.epsilon.inner = 0.005
        grasp_goal.epsilon.outer = 0.005
        self.franka_grasp_client.send_goal_async(grasp_goal)

def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    try:
        while rclpy.ok() and not node.should_quit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_recording:
            node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()