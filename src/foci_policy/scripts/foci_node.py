#!/usr/bin/env python3
"""
FOCI Node - ROS2 bridge to FOCI policy via ZMQ (Using cuMotion)
"""
import sys
import os
import rclpy
import zmq
import numpy as np
import cv2
import base64
import threading
import time
import math

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Move
from control_msgs.action import GripperCommand
from sensor_msgs.msg import Image, CameraInfo, JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation

from motion_planning.panda_control_velocity import PandaCommander
from motion_planning.transform import Transform


class FOCINode(Node):
    """ROS2 node that bridges between FOCI policy and Franka robot using cuMotion"""
    def __init__(self):
        super().__init__('foci_node')
        self.gripper_type = self.declare_parameter('gripper', 'franka').value.lower().strip()
        if self.gripper_type not in ('franka', 'robotiq'):
            self.get_logger().warn(
                f"Unknown gripper type '{self.gripper_type}', fallback to 'franka'"
            )
            self.gripper_type = 'franka'
        self.robotiq_joint_name = 'robotiq_85_left_knuckle_joint'
        self.robotiq_open_position = 0.0
        self.robotiq_closed_position = 0.8
        self.robotiq_joint_position = 0.0
        self._robotiq_joint_warned = False
        self._robotiq_goal_in_flight = False
        self.gripper_open_width = 0.08

        # ZMQ server setup (REP pattern - respond to requests)
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        self.socket.bind("tcp://*:5555")
        self.get_logger().info('ZMQ server listening on port 5555')
        self.get_logger().info(f'Using gripper type: {self.gripper_type}')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # TF buffer for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_gripper_width = 0.0
        self.cached_point_cloud = None
        
        # ROS2 subscribers
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10) 
        gripper_joint_topic = '/robotiq/joint_states' if self.gripper_type == 'robotiq' else '/fr3_gripper/joint_states'
        self.gripper_state_sub = self.create_subscription(JointState, gripper_joint_topic, self.gripper_state_callback, 10)
        
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
        else:
            self.franka_grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
            self.franka_move_client = ActionClient(self, Move, '/fr3_gripper/move')
        
        # Running flag for clean shutdown
        self.running = True
        
        # Trajectory visualization publisher
        self.trajectory_viz_pub = self.create_publisher(MarkerArray, '/foci_trajectory_viz', 10)  
        
        # Start ZMQ server in separate thread
        self.zmq_thread = threading.Thread(target=self.zmq_server_loop, daemon=True)
        self.zmq_thread.start()

        # Motion planning interface
        self.pc = PandaCommander()
    
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
    
    def get_transform(self, target_frame, source_frame):
        """Get transform from TF tree"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            # Convert to 4x4 matrix
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            # Convert quaternion to rotation matrix
            matrix = np.eye(4)
            quat = [rot.x, rot.y, rot.z, rot.w]
            rotation = Rotation.from_quat(quat)
            matrix[:3, :3] = rotation.as_matrix()
            matrix[0, 3] = trans.x
            matrix[1, 3] = trans.y
            matrix[2, 3] = trans.z
            
            return {
                'position': {'x': trans.x, 'y': trans.y, 'z': trans.z},
                'orientation': {'x': rot.x, 'y': rot.y, 'z': rot.z, 'w': rot.w},
                'matrix': matrix.tolist()
            }
        except Exception as e:
            self.get_logger().warn(f'Failed to get transform {target_frame}->{source_frame}: {e}')
            return None

    def depth_to_point_cloud(self, depth, K, cam_extrinsic, mask=None):
        """ Convert depth image to point cloud, optionally filtering by mask """
        depth = depth.astype(np.float32)
        depth_filtered = cv2.bilateralFilter(depth, d=5, sigmaColor=10, sigmaSpace=10)
        depth = depth_filtered.astype(np.float32) / 1000.0  # mm to meters
        h, w = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        # Filter valid depth points
        if mask is not None:
            valid = (mask > 0) & (depth > 0.01) & (depth < 3.0)
        else:
            valid = (depth > 0.01) & (depth < 3.0)
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth[valid]
        # Back-project to 3D
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid
        point_cloud = np.stack([x, y, z], axis=1)
        point_cloud_world = (cam_extrinsic[:3, :3] @ point_cloud.T).T + cam_extrinsic[:3, 3]
        point_cloud_world_without_table = point_cloud_world[point_cloud_world[:, 2] > 0.04]
        max_points = 2048
        if point_cloud_world_without_table.shape[0] > max_points:
            idx = np.random.choice(point_cloud_world_without_table.shape[0], max_points, replace=False)
            point_cloud_world_without_table = point_cloud_world_without_table[idx]
        return point_cloud_world_without_table
    
    def get_observation(self):
        """Collect current observation for FOCI policy"""
        # Wait for data if not available
        timeout = 5.0
        start_time = self.get_clock().now()
        while (self.latest_color is None or 
               self.latest_depth is None or 
               self.latest_camera_info is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now() - start_time).nanoseconds / 1e9 > timeout:
                return {'status': 'failed', 'message': 'Timeout waiting for sensor data'}
        # Convert images
        try:
            rgb = self.bridge.imgmsg_to_cv2(self.latest_color, 'rgb8')
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
            # Encode images to base64
            _, rgb_encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rgb_b64 = base64.b64encode(rgb_encoded).decode('utf-8')
            _, depth_encoded = cv2.imencode('.png', depth)
            depth_b64 = base64.b64encode(depth_encoded).decode('utf-8')
        except Exception as e:
            return {'status': 'failed', 'message': f'Image conversion error: {e}'}
        
        # Get camera intrinsics
        K = np.array(self.latest_camera_info.k).reshape(3, 3)
        # Get camera extrinsics (fr3_link0 -> camera_color_optical_frame)
        cam_extrinsic = self.get_transform('fr3_link0', 'camera_color_optical_frame')
        if cam_extrinsic is None:
            return {'status': 'failed', 'message': 'Failed to get camera extrinsics'}
        # Get gripper pose (fr3_link0 -> fr3_hand_tcp)
        gripper_pose = self.get_transform('fr3_link0', 'fr3_hand_tcp')
        if gripper_pose is None:
            return {'status': 'failed', 'message': 'Failed to get gripper pose'}
        # Calculate gripper open ratio (0=closed, 1=open)
        if self.gripper_type == 'robotiq':
            close_ratio = min(max(self.robotiq_joint_position / self.robotiq_closed_position, 0.0), 1.0)
            gripper_open = 1.0 - close_ratio
        else:
            gripper_open = min(self.latest_gripper_width / self.gripper_open_width, 1.0)
        return {
            'status': 'success',
            'data': {
                'rgb': rgb_b64,
                'depth': depth_b64,
                'K': K.flatten().tolist(),
                'cam_extrinsic': cam_extrinsic['matrix'],
                'gripper_pose': gripper_pose,
                'gripper_open': gripper_open
            }
        }

    def execute_trajectory(self, trajectory_data, mode='grasp'):
        try:
            poses = trajectory_data.get('poses', [])
            if not poses:
                self.get_logger().warn('Received empty trajectory')
                return {'status': 'failed', 'message': 'Empty trajectory'}
            self.get_logger().info(f'Executing trajectory with {len(poses)} waypoints')

            # 0. Direct motion without obstacle avoidance
            if mode == "lift":
                first_pose_mat = Transform.from_matrix(np.array(poses[0]))
                success = self.pc.goto_pose(first_pose_mat)
                if not success:
                    self.get_logger().error('Failed to reach first waypoint')
                    return {'status': 'failed', 'message': 'Failed to reach first waypoint'}
                else:
                    self.get_logger().info('Reached first waypoint successfully')
                time.sleep(0.5)
                return {'status': 'success', 'message': 'Trajectory executed successfully'}

            self.visualize_trajectory(poses, mode)
            # Generate point cloud from current observation
            if self.latest_depth is None or self.latest_camera_info is None:
                self.get_logger().warn('No depth or camera info available for point cloud')
                point_cloud = None
            elif self.cached_point_cloud is not None:
                self.get_logger().info('Using cached point cloud for trajectory execution')
                point_cloud = self.cached_point_cloud
            # obstacle avoidance only enabled for grasp_oa and manip_oa mode
            elif mode in ['grasp', 'manip']:
                point_cloud = None
            else:
                self.get_logger().info('Generating point cloud for trajectory execution')
                try:
                    depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
                    K = np.array(self.latest_camera_info.k).reshape(3, 3)
                    # Transform to robot base frame (fr3_link0)
                    cam_extrinsic = self.get_transform('fr3_link0', 'camera_color_optical_frame')
                    cam_extrinsic = np.array(cam_extrinsic['matrix'])
                    # Convert depth to point cloud (in world frame)
                    point_cloud = self.depth_to_point_cloud(depth, K, cam_extrinsic)
                    self.cached_point_cloud = point_cloud
                    self.get_logger().info(f'Generated point cloud with {point_cloud.shape[0]} points')
                except Exception as e:
                    self.get_logger().error(f'Failed to generate point cloud: {e}')
                    point_cloud = None

            # 1. Plan to the first pose
            first_pose_mat = Transform.from_matrix(np.array(poses[0]))
            success = self.pc.goto_pose(first_pose_mat, pcl=point_cloud)
            if not success:
                self.get_logger().error('Failed to reach first waypoint')
                return {'status': 'failed', 'message': 'Failed to reach first waypoint'}
            else:
                self.get_logger().info('Reached first waypoint successfully')
            time.sleep(0.5)

            # 2. Interpolate and execute through waypoints
            for i in range(1, len(poses)):
                waypoint_mat = Transform.from_matrix(np.array(poses[i]))
                success = self.pc.goto_pose_reactive(waypoint_mat, threshold=0.01)
                if not success:
                    self.get_logger().error(f'Failed to reach waypoint {i + 1}')
                    return {'status': 'failed', 'message': f'Failed to reach waypoint {i + 1}'}
                else:
                    self.get_logger().info(f'Reached waypoint {i + 1} successfully')
                    time.sleep(0.5)
            return {'status': 'success', 'message': 'Trajectory executed successfully'}

        except Exception as e:
            self.get_logger().error(f'Trajectory execution exception: {e}')
            return {'status': 'failed', 'message': f'Exception occurred: {str(e)}'}


    def open_gripper(self):
        """Open gripper"""
        if self.gripper_type == 'robotiq':
            ok = self.move_robotiq_gripper(self.robotiq_open_position, max_effort=50.0, timeout=5.0)
            if not ok:
                return {'status': 'failed', 'message': 'Failed to open Robotiq gripper'}
            return {'status': 'success', 'message': 'Robotiq gripper opened successfully'}

        if self.franka_move_client is None:
            return {'status': 'failed', 'message': 'Franka move client is not initialized'}
        goal = Move.Goal()
        goal.width = 0.08
        goal.speed = 0.1
        self.franka_move_client.send_goal_async(goal)
        return {'status': 'success', 'message': 'Gripper opened successfully'}
    
    def close_gripper(self):
        if self.gripper_type == 'robotiq':
            ok = self.move_robotiq_gripper(self.robotiq_closed_position, max_effort=50.0, timeout=5.0)
            if not ok:
                return {'status': 'failed', 'message': 'Failed to close Robotiq gripper'}
            return {'status': 'success', 'message': 'Robotiq gripper closed successfully'}

        # goal = Grasp.Goal()
        # goal.width = 0.0
        # goal.speed = 0.1
        # goal.force = 20.0
        # goal.epsilon.inner = 0.005
        # goal.epsilon.outer = 0.005
        # self.franka_grasp_client.send_goal_async(goal)
        success = self.pc.grasp(width=0.01, force=50.0, speed=0.05)
        if not success:
            return {'status': 'failed', 'message': 'Failed to grasp the object'}
        return {'status': 'success', 'message': 'Grasp executed successfully'}

    def move_robotiq_gripper(self, position: float, max_effort: float = 50.0, timeout: float = 5.0) -> bool:
        if self.robotiq_gripper_client is None:
            self.get_logger().warn('Robotiq client is not initialized')
            return False

        if self._robotiq_goal_in_flight:
            self.get_logger().info('Robotiq gripper is moving, command ignored')
            return False

        if not self.robotiq_gripper_client.wait_for_server(timeout_sec=timeout):
            self.get_logger().warn(f'Robotiq action server not available within {timeout:.1f}s')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)

        send_future = self.robotiq_gripper_client.send_goal_async(goal)
        self._robotiq_goal_in_flight = True

        def _goal_response_cb(fut):
            try:
                goal_handle = fut.result()
            except Exception as exc:
                self._robotiq_goal_in_flight = False
                self.get_logger().warn(f'Robotiq goal request failed: {exc}')
                return

            if not goal_handle or not goal_handle.accepted:
                self._robotiq_goal_in_flight = False
                self.get_logger().warn('Robotiq goal rejected by action server')
                return

            def _result_cb(result_future):
                self._robotiq_goal_in_flight = False
                try:
                    result_msg = result_future.result()
                    if result_msg is None:
                        self.get_logger().warn('Robotiq result is None')
                        return
                    result = result_msg.result
                    self.get_logger().info(
                        f'Robotiq result: reached_goal={result.reached_goal}, '
                        f'stalled={result.stalled}, effort={result.effort:.3f}, '
                        f'position={result.position:.3f}, status={result_msg.status}'
                    )
                except Exception as exc:
                    self.get_logger().warn(f'Robotiq result error: {exc}')

            goal_handle.get_result_async().add_done_callback(_result_cb)

        send_future.add_done_callback(_goal_response_cb)
        return True

    def reset_robot(self):
        success = self.pc.home()
        if not success:
            return {'status': 'failed', 'message': 'Failed to reset robot to home position'}
        return {'status': 'success', 'message': 'Robot reset to home position successfully'}
    
    # TODO: visualize as frame instead of markers
    def visualize_trajectory(self, poses, mode='grasp'):
        """Visualize trajectory in RViz using MarkerArray"""
        try:
            marker_array = MarkerArray()
            # Color based on mode
            if mode in ['grasp', 'grasp_oa']:
                color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.8)  # Orange
            elif mode in ['manip', 'manip_oa']:
                color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.8)  # Cyan
            else:
                return
            
            # Add sphere markers for waypoints
            for i, pose_matrix in enumerate(poses):
                marker = Marker()
                marker.header.frame_id = 'fr3_link0'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f'trajectory_{mode}'
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
                # Position from matrix
                marker.pose.position.x = float(pose_matrix[0][3])
                marker.pose.position.y = float(pose_matrix[1][3])
                marker.pose.position.z = float(pose_matrix[2][3])
                # Orientation from matrix
                R = np.array([
                    [pose_matrix[0][0], pose_matrix[0][1], pose_matrix[0][2]],
                    [pose_matrix[1][0], pose_matrix[1][1], pose_matrix[1][2]],
                    [pose_matrix[2][0], pose_matrix[2][1], pose_matrix[2][2]]
                ])
                quat = Rotation.from_matrix(R).as_quat()  # x, y, z, w
                marker.pose.orientation.x = quat[0]
                marker.pose.orientation.y = quat[1]
                marker.pose.orientation.z = quat[2]
                marker.pose.orientation.w = quat[3]
                # Size and color
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color = color
                marker_array.markers.append(marker)
            
            # Add line strip connecting waypoints
            if len(poses) > 1:
                line_marker = Marker()
                line_marker.header.frame_id = 'fr3_link0'
                line_marker.header.stamp = self.get_clock().now().to_msg()
                line_marker.ns = f'trajectory_{mode}_line'
                line_marker.id = 0
                line_marker.type = Marker.LINE_STRIP
                line_marker.action = Marker.ADD
                line_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
                
                for pose_matrix in poses:
                    from geometry_msgs.msg import Point
                    p = Point()
                    p.x = float(pose_matrix[0][3])
                    p.y = float(pose_matrix[1][3])
                    p.z = float(pose_matrix[2][3])
                    line_marker.points.append(p)
                line_marker.scale.x = 0.005  # Line width
                line_marker.color = color
                marker_array.markers.append(line_marker)
            # Publish markers
            self.trajectory_viz_pub.publish(marker_array)
            self.get_logger().debug(f'Published {len(poses)} trajectory waypoints to RViz')
            
        except Exception as e:
            self.get_logger().warn(f'Failed to visualize trajectory: {e}')
    
    
    def zmq_server_loop(self):
        """Handle ZMQ requests in separate thread"""
        while self.running:
            try:
                # Wait for request
                request = self.socket.recv_json()
                request_type = request.get('type', 'unknown')
                
                self.get_logger().info(f'Received ZMQ request: {request_type}')
                
                # Handle different request types
                if request_type == 'get_observation':
                    response = self.get_observation()
                    
                elif request_type == 'execute_trajectory':
                    mode = request.get('mode', 'grasp')
                    response = self.execute_trajectory(request, mode)
                    
                elif request_type == 'open_gripper':
                    response = self.open_gripper()
                    
                elif request_type == 'close_gripper':  
                    response = self.close_gripper()
                    
                elif request_type == 'reset_robot':
                    response = self.reset_robot()

                # Send response
                self.socket.send_json(response)
                
            except zmq.Again:
                # Timeout, check running flag and continue
                continue
            except Exception as e:
                if self.running:
                    error_response = {'status': 'failed', 'message': f'Server error: {e}'}
                    try:
                        self.socket.send_json(error_response)
                    except:
                        pass
                    self.get_logger().error(f'ZMQ server error: {e}')
                else:
                    break
    
    
    def shutdown(self):
        """Clean shutdown ZMQ thread and resources"""
        self.running = False
        time.sleep(0.5)
        self.socket.close()
        self.zmq_context.term()


def main(args=None):
    rclpy.init(args=args)
    
    # Create FOCI Node
    node = FOCINode()

    # Use MultiThreadedExecutor to manage both nodes
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(node.pc)
    
    try:
        # Run executor in separate thread
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        # Return home and open gripper at start
        node.get_logger().info('Resetting robot to home position and opening gripper...')
        node.reset_robot()
        node.open_gripper()
        node.get_logger().info('FOCI Node initialized, waiting for requests...')

        # Keep main thread alive
        while rclpy.ok():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down FOCI node...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
