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
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import String
from controller_manager_msgs.srv import ListControllers, SwitchController
from tf2_ros import Buffer, TransformListener
import threading
import subprocess
import json
import os
import cv2
from cv_bridge import CvBridge
from queue import Queue

class DemoRecorder(Node):
    def __init__(self):
        super().__init__('demo_recorder')
        # Gripper action clients
        self.franka_grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
        self.franka_move_client = ActionClient(self, Move, '/fr3_gripper/move')
        # TF buffer for getting end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        # Subscribers
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        self.gripper_state_sub = self.create_subscription(JointState, '/fr3_gripper/joint_states', self.gripper_state_callback, 10)
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
        # Controller monitoring timer (1 Hz)
        self.controller_check_timer = self.create_timer(1.0, self.check_controller_status)
        self.impedance_controller_name = 'joint_impedance_example_controller'
        self.controller_restarting = False
        # Controller manager service clients
        self.list_controllers_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        # Gripper homing (non-blocking)
        self.gripper_ready = False
        threading.Thread(target=self._init_gripper, daemon=True).start()
        self.print_instructions()

    def _init_gripper(self):
        """Initialize gripper by opening it"""
        if self.franka_move_client.wait_for_server(timeout_sec=5.0):
            goal = Move.Goal()
            goal.width = 0.08
            goal.speed = 0.1
            self.franka_move_client.send_goal_async(goal)
            self.gripper_ready = True
            print("Gripper initialized and opened")

    def print_instructions(self):
        print("\n" + "="*60)
        print("DEMO RECORDING SYSTEM")
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
        if len(msg.position) > 0:
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
        
        # Determine gripper state from actual width (>0.04 = open, <=0.04 = closed)
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
        demo_dir = f'demo_{self.demo_count:03d}'
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
                if save_data['color_img'] is not None:
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(save_data['color_img'], 'bgr8')
                        color_path = os.path.join(demo_dir, 'color', f'{frame_idx:04d}.png')
                        cv2.imwrite(color_path, cv_image)
                    except Exception:
                        pass
                if save_data['depth_img'] is not None:
                    try:
                        depth_image = self.bridge.imgmsg_to_cv2(save_data['depth_img'], 'passthrough')
                        depth_path = os.path.join(demo_dir, 'depth', f'{frame_idx:04d}.png')
                        cv2.imwrite(depth_path, depth_image)
                    except Exception:
                        pass
            except Exception:
                pass

    def check_controller_status(self):
        """Check if impedance controller is active, restart if not"""
        if self.controller_restarting:
            return
        # Check if service is available
        if not self.list_controllers_client.wait_for_service(timeout_sec=0.5):
            return
        # Call list_controllers service
        request = ListControllers.Request()
        future = self.list_controllers_client.call_async(request)
        future.add_done_callback(self._handle_controller_list)
    
    def _handle_controller_list(self, future):
        """Handle controller list response"""
        try:
            response = future.result()
            impedance_active = False
            
            for controller in response.controller:
                if controller.name == self.impedance_controller_name:
                    if controller.state == 'active':
                        impedance_active = True
                    break
            
            # If impedance controller is not active, restart it
            if not impedance_active and not self.controller_restarting:
                self.get_logger().warn('Impedance controller not active! Auto-restarting...')
                threading.Thread(target=self._restart_impedance_controller, daemon=True).start()
        except Exception as e:
            self.get_logger().debug(f'Controller check error: {e}')
    
    def _restart_impedance_controller(self):
        """Restart impedance controller using spawner"""
        self.controller_restarting = True
        print("\n" + "="*60)
        print("[AUTO-RESTART] Impedance controller lost, restarting...")
        print("="*60)
        
        try:
            # Stop recording if active
            if self.is_recording:
                print("[AUTO-RESTART] Stopping current recording...")
                self.is_recording = False
            
            # Spawn controller using ros2 control command
            result = subprocess.run(
                ['ros2', 'control', 'load_controller', '--set-state', 'active', 
                 self.impedance_controller_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("[AUTO-RESTART] Successfully restarted impedance controller")
                print("[AUTO-RESTART] You can continue teaching...\n")
            else:
                print(f"[AUTO-RESTART] Failed to restart: {result.stderr}")
                print("[AUTO-RESTART] Manual intervention required\n")
        except Exception as e:
            print(f"[AUTO-RESTART] Error: {e}")
            print("[AUTO-RESTART] Manual restart required\n")
        finally:
            self.controller_restarting = False
    
    def start_recording(self):
        if self.is_recording:
            self.get_logger().warn('Already recording!')
            return
        self.demo_count += 1
        demo_dir = f'demo_{self.demo_count:03d}'
        os.makedirs(demo_dir, exist_ok=True)
        # Create subdirectories for images
        os.makedirs(os.path.join(demo_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(demo_dir, 'depth'), exist_ok=True)
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
        extrinsics = self.get_camera_extrinsics()
        if extrinsics is not None:
            with open(os.path.join(demo_dir, 'camera_extrinsics.json'), 'w') as f:
                json.dump(extrinsics, f, indent=2)
        self.get_logger().info(f'Started recording demo {self.demo_count}')
        print(f"\n>>> RECORDING DEMO {self.demo_count} <<<\n")

    def stop_recording(self):
        if not self.is_recording:
            self.get_logger().warn('Not recording!')
            return
        self.is_recording = False
        demo_dir = f'demo_{self.demo_count:03d}'
        with open(os.path.join(demo_dir, 'trajectory.json'), 'w') as f:
            json.dump(self.demo_data, f, indent=2)
        self.get_logger().info(f'Stopped recording. Saved {len(self.demo_data)} frames to {demo_dir}/')
        print(f"\n>>> DEMO {self.demo_count} SAVED ({len(self.demo_data)} frames) <<<")

    def reset_to_position(self, target_position):
        """Reset robot to target joint position using joint trajectory"""
        print("\n[INFO] Attempting to reset to start position...")
        
        # Check if action server is available
        print("[INFO] Checking joint trajectory action server...")
        if not self.joint_trajectory_client.wait_for_server(timeout_sec=2.0):
            print("[WARN] Joint trajectory action server not available!")
            print("[WARN] This is expected if impedance controller is active.")
            print("[WARN] Manual reset required or switch to joint_trajectory_controller.")
            return
        
        print("[INFO] Joint trajectory server found, sending reset command...")
        
        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        
        point = JointTrajectoryPoint()
        point.positions = target_position
        point.velocities = [0.0] * 7
        point.time_from_start = Duration(sec=5, nanosec=0)
        
        trajectory_msg.points = [point]
        goal_msg.trajectory = trajectory_msg
        
        future = self.joint_trajectory_client.send_goal_async(goal_msg)
        print("[INFO] Reset goal sent. Robot should move back in 5 seconds.")
        
        # Optional: wait for acceptance
        def goal_response_callback(future_result):
            goal_handle = future_result.result()
            if goal_handle.accepted:
                print("[OK] Reset goal accepted by controller")
            else:
                print("[ERROR] Reset goal rejected by controller")
        
        future.add_done_callback(goal_response_callback)

    def toggle_gripper(self):
        if not self.gripper_ready:
            print("!!! Gripper not ready")
            return
        # Determine current state from actual width
        if self.latest_gripper_width > 0.04:
            self.close_gripper()
        else:
            self.open_gripper()

    def send_gripper_command(self, width, speed=0.1):
        """Use Franka Move action for precise gripper control"""
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        self.franka_move_client.send_goal_async(goal)

    def open_gripper(self):
        print(">>> Opening gripper")
        self.send_gripper_command(0.08, speed=0.1)

    def close_gripper(self):
        print(">>> Closing gripper (grasp)")
        # Use Grasp action for closing
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