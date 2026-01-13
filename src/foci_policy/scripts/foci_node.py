#!/usr/bin/env python3
"""
FOCI Node - ROS2 bridge to FOCI policy via ZMQ
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from franka_msgs.action import Grasp, Move
from controller_manager_msgs.srv import SwitchController, ListControllers
import zmq
import numpy as np
from scipy.spatial.transform import Rotation
import threading
import base64
import cv2
import time
import math


class FOCINode(Node):
    """ROS2 node that bridges between FOCI policy and Franka robot"""
    
    def __init__(self):
        super().__init__('foci_node')
        
        # ZMQ server setup (REP pattern - respond to requests)
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for clean shutdown
        self.socket.bind("tcp://*:5555")
        self.get_logger().info('ZMQ server listening on port 5555')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # TF buffer for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_gripper_state = None
        self.latest_gripper_width = 0.0
        
        # ROS2 subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', 
            self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.camera_info_callback, 10
        )
        self.gripper_state_sub = self.create_subscription(
            JointState, '/fr3_gripper/joint_states',
            self.gripper_state_callback, 10
        )
        
        # Gripper action clients
        self.franka_grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
        self.franka_move_client = ActionClient(self, Move, '/fr3_gripper/move')
        
        # MoveGroup action client for trajectory execution
        self.moveit_client = ActionClient(self, MoveGroup, '/move_action')
        
        # Controller manager clients
        self.switch_controller_client = self.create_client(
            SwitchController, '/controller_manager/switch_controller'
        )
        self.list_controllers_client = self.create_client(
            ListControllers, '/controller_manager/list_controllers'
        )
        
        # Switch to trajectory controller on startup
        self.switch_to_trajectory_controller()
        
        # Reset robot to home position
        self.reset_robot()
        
        # Open gripper on startup
        self.get_logger().info('Opening gripper...')
        self.open_gripper()
        
        # Running flag for clean shutdown
        self.running = True
        
        self.get_logger().info('FOCI Node initialized, waiting for requests...')
        
        # Start ZMQ server in separate thread
        self.zmq_thread = threading.Thread(target=self.zmq_server_loop, daemon=True)
        self.zmq_thread.start()
    
    def switch_to_trajectory_controller(self):
        """Switch from velocity controller to trajectory controller"""
        self.get_logger().info('Switching to fr3_arm_controller for MoveIt...')
        
        # Wait for controller manager services
        if not self.switch_controller_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Controller manager switch service not available')
            return False
        if not self.list_controllers_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Controller manager list service not available')
            return False
        
        # Check current controller status
        list_request = ListControllers.Request()
        future = self.list_controllers_client.call_async(list_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result():
            for c in future.result().controller:
                if c.name == 'fr3_arm_controller' and c.state == 'active':
                    self.get_logger().info('fr3_arm_controller already active')
                    return True
            
            active_controllers = [c.name for c in future.result().controller if c.state == 'active']
            self.get_logger().info(f'Currently active controllers: {active_controllers}')
        
        # Switch controllers: activate fr3_arm_controller, deactivate joint_velocity_controller
        switch_request = SwitchController.Request()
        switch_request.activate_controllers = ['fr3_arm_controller']
        switch_request.deactivate_controllers = ['joint_velocity_controller']
        switch_request.strictness = SwitchController.Request.BEST_EFFORT
        switch_request.start_asap = True
        switch_request.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        
        future = self.switch_controller_client.call_async(switch_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() and future.result().ok:
            self.get_logger().info('Successfully switched to fr3_arm_controller')
            return True
        else:
            self.get_logger().error('Failed to switch controllers')
            return False
    
    def reset_robot(self):
        """Reset robot to home configuration using MoveGroup Action"""
        try:
            self.get_logger().info('Resetting robot to home configuration...')
            
            home_position = [
                0.0,                    # joint1
                -math.pi / 4,          # joint2: -π/4
                0.0,                    # joint3
                -3 * math.pi / 4,      # joint4: -3π/4
                0.0,                    # joint5
                math.pi / 2,           # joint6: π/2
                math.pi / 4            # joint7: π/4
            ]
            
            joint_names = [
                'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
                'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
            ]
            
            # Create MoveGroup goal with joint constraints
            goal = MoveGroup.Goal()
            goal.request.group_name = 'fr3_arm'
            goal.request.num_planning_attempts = 5
            goal.request.allowed_planning_time = 5.0
            goal.request.max_velocity_scaling_factor = 0.2
            goal.request.max_acceleration_scaling_factor = 0.2
            
            # Planning options
            goal.planning_options.plan_only = False
            goal.planning_options.planning_scene_diff.is_diff = True
            goal.planning_options.planning_scene_diff.robot_state.is_diff = True
            
            # Create joint constraints
            constraints = Constraints()
            for joint_name, joint_value in zip(joint_names, home_position):
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = joint_value
                joint_constraint.tolerance_above = 0.001
                joint_constraint.tolerance_below = 0.001
                joint_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_constraint)
            
            goal.request.goal_constraints.append(constraints)
            
            # Wait for action server
            if not self.moveit_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('MoveIt action server not available')
                return False
            
            # Send goal and wait
            goal_future = self.moveit_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, goal_future, timeout_sec=10.0)
            
            goal_handle = goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error('Home position goal rejected')
                return False
            
            self.get_logger().info('Home position goal accepted, waiting for completion...')
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
            
            result = result_future.result()
            if result and result.result.error_code.val == 1:  # 1 = SUCCESS
                self.get_logger().info('Robot reset done.')
                return True
            else:
                error_code = result.result.error_code.val if result else 'unknown'
                self.get_logger().error(f'Failed to reset robot! Error: {error_code}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Exception during robot reset: {e}')
            return False
    
    def color_callback(self, msg):
        self.latest_color = msg
    
    def depth_callback(self, msg):
        self.latest_depth = msg
    
    def camera_info_callback(self, msg):
        self.latest_camera_info = msg
    
    def gripper_state_callback(self, msg):
        self.latest_gripper_state = msg
        if len(msg.position) > 0:
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
            
            matrix = np.eye(4)
            # Simple conversion (for full rotation matrix, use scipy or tf_transformations)
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
        gripper_open = min(self.latest_gripper_width / 0.08, 1.0)
        
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
        """Execute trajectory using MoveIt"""
        try:
            poses = trajectory_data.get('poses', [])
            gripper_opens = trajectory_data.get('gripper_open', [])
            
            if not poses:
                return {'status': 'failed', 'message': 'Empty trajectory'}
            
            self.get_logger().info(f'Executing trajectory with {len(poses)} waypoints')
            
            # Execute each pose sequentially
            for i, pose_matrix in enumerate(poses):
                # Convert 4x4 matrix to PoseStamped
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = 'fr3_link0'
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                
                # Extract position from matrix
                pose_stamped.pose.position.x = float(pose_matrix[0][3])
                pose_stamped.pose.position.y = float(pose_matrix[1][3])
                pose_stamped.pose.position.z = float(pose_matrix[2][3])
                
                # Extract orientation from rotation matrix
                R = np.array([
                    [pose_matrix[0][0], pose_matrix[0][1], pose_matrix[0][2]],
                    [pose_matrix[1][0], pose_matrix[1][1], pose_matrix[1][2]],
                    [pose_matrix[2][0], pose_matrix[2][1], pose_matrix[2][2]]
                ])
                quat = self._rotation_matrix_to_quaternion(R)
                pose_stamped.pose.orientation.x = quat[0]
                pose_stamped.pose.orientation.y = quat[1]
                pose_stamped.pose.orientation.z = quat[2]
                pose_stamped.pose.orientation.w = quat[3]
                
                # Create MoveGroup goal
                goal = MoveGroup.Goal()
                goal.request.group_name = 'fr3_arm'
                goal.request.num_planning_attempts = 5
                goal.request.allowed_planning_time = 5.0
                goal.request.max_velocity_scaling_factor = 0.3
                goal.request.max_acceleration_scaling_factor = 0.3
                
                # Set workspace parameters
                goal.request.workspace_parameters.header.frame_id = 'fr3_link0'
                goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
                goal.request.workspace_parameters.min_corner.x = -1.0
                goal.request.workspace_parameters.min_corner.y = -1.0
                goal.request.workspace_parameters.min_corner.z = -1.0
                goal.request.workspace_parameters.max_corner.x = 1.0
                goal.request.workspace_parameters.max_corner.y = 1.0
                goal.request.workspace_parameters.max_corner.z = 1.0
                
                # Planning options
                goal.planning_options.plan_only = False
                goal.planning_options.planning_scene_diff.is_diff = True
                goal.planning_options.planning_scene_diff.robot_state.is_diff = True
                
                # Create goal constraints
                goal.request.goal_constraints.append(Constraints())
                
                # Create position constraint with BoundingVolume
                position_constraint = PositionConstraint()
                position_constraint.header.frame_id = 'fr3_link0'
                position_constraint.link_name = 'fr3_hand_tcp'
                
                # Create a small sphere around the target position
                region = SolidPrimitive()
                region.type = SolidPrimitive.SPHERE
                region.dimensions = [0.0001]  # Very small tolerance
                position_constraint.constraint_region.primitives.append(region)
                position_constraint.constraint_region.primitive_poses.append(pose_stamped.pose)
                position_constraint.weight = 1.0
                
                # Create orientation constraint
                orientation_constraint = OrientationConstraint()
                orientation_constraint.header.frame_id = 'fr3_link0'
                orientation_constraint.link_name = 'fr3_hand_tcp'
                orientation_constraint.orientation = pose_stamped.pose.orientation
                orientation_constraint.absolute_x_axis_tolerance = 0.001
                orientation_constraint.absolute_y_axis_tolerance = 0.001
                orientation_constraint.absolute_z_axis_tolerance = 0.001
                orientation_constraint.weight = 1.0
                
                goal.request.goal_constraints[0].position_constraints.append(position_constraint)
                goal.request.goal_constraints[0].orientation_constraints.append(orientation_constraint)
                
                # Send goal using callback-based approach (thread-safe)
                if not self.moveit_client.wait_for_server(timeout_sec=5.0):
                    return {'status': 'failed', 'message': 'MoveIt action server not available'}
                
                self.get_logger().info(f'Sending waypoint {i+1}/{len(poses)} to MoveIt')
                
                # Use threading.Event for synchronization instead of spin_until_future_complete
                goal_done = threading.Event()
                goal_handle_container = [None]
                
                def goal_response_callback(future):
                    goal_handle_container[0] = future.result()
                    goal_done.set()
                
                future = self.moveit_client.send_goal_async(goal)
                future.add_done_callback(goal_response_callback)
                
                if not goal_done.wait(timeout=10.0):
                    return {'status': 'failed', 'message': f'Goal {i} send timeout'}
                
                goal_handle = goal_handle_container[0]
                if not goal_handle or not goal_handle.accepted:
                    return {'status': 'failed', 'message': f'Goal {i} rejected by MoveIt'}
                
                # Wait for execution
                self.get_logger().info(f'Waiting for waypoint {i+1} execution...')
                result_done = threading.Event()
                result_container = [None]
                
                def result_callback(future):
                    result_container[0] = future.result()
                    result_done.set()
                
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(result_callback)
                
                if not result_done.wait(timeout=30.0):
                    return {'status': 'failed', 'message': f'Waypoint {i} execution timeout'}
                
                result = result_container[0]
                if not result or result.result.error_code.val != 1:  # 1 = SUCCESS
                    error_code = result.result.error_code.val if result else 'unknown'
                    return {'status': 'failed', 'message': f'Execution failed at waypoint {i} (error: {error_code})'}
                
                self.get_logger().info(f'Waypoint {i+1} executed successfully')
                
                # Control gripper if specified
                if i < len(gripper_opens):
                    gripper_open = gripper_opens[i]
                    if gripper_open > 0.5:
                        self.open_gripper()
                    else:
                        self.close_gripper()
            
            return {'status': 'success', 'message': f'Executed {len(poses)} waypoints'}
            
        except Exception as e:
            self.get_logger().error(f'Trajectory execution error: {e}')
            return {'status': 'failed', 'message': f'Execution error: {e}'}
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        rotation = Rotation.from_matrix(R)
        quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
        return quat_xyzw.tolist()
    
    def open_gripper(self):
        """Open gripper"""
        goal = Move.Goal()
        goal.width = 0.08
        goal.speed = 0.1
        self.franka_move_client.send_goal_async(goal)
    
    def close_gripper(self):
        """Close gripper"""
        goal = Grasp.Goal()
        goal.width = 0.0
        goal.speed = 0.1
        goal.force = 20.0
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        self.franka_grasp_client.send_goal_async(goal)
    
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
                    trajectory = request.get('trajectory', {})
                    mode = request.get('mode', 'grasp')
                    response = self.execute_trajectory(trajectory, mode)
                    
                elif request_type == 'open_gripper':
                    self.open_gripper()
                    response = {'status': 'success', 'message': 'Gripper opened'}
                    
                elif request_type == 'close_gripper':
                    self.close_gripper()
                    response = {'status': 'success', 'message': 'Gripper closed'}
                    
                else:
                    response = {'status': 'failed', 'message': f'Unknown request type: {request_type}'}
                
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
        time.sleep(0.5)  # Give thread time to exit
        self.socket.close()
        self.zmq_context.term()


def main(args=None):
    rclpy.init(args=args)
    node = FOCINode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down FOCI node...")
    finally:
        # Clean shutdown of ZMQ thread and resources
        node.shutdown()
        node.destroy_node()
        # Only call rclpy.shutdown() if context is still valid
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass  # Already shutdown, ignore


if __name__ == '__main__':
    main()
