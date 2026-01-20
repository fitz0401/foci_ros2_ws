#!/usr/bin/env python3
"""
FR3 Robot Controller using ROS2 and MoveIt (ZMQ Client Version)
"""

import rclpy
import sys
import zmq
import pickle
import sensor_msgs
import time
import numpy as np
import roboticstoolbox as rtb
import math

from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from typing import List, Optional
from franka_msgs.action import Grasp, ErrorRecovery, Move
from franka_msgs.msg import FrankaRobotState, GraspEpsilon, Errors
from franka_msgs.srv import SetForceTorqueCollisionBehavior
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Path
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

from motion_planning.control import calculate_velocity
from motion_planning.transform import Transform, reorder_pose_list, Rotation, matrix_to_euler_angles


class MotionPlannerClient:
    def __init__(self, host='localhost', port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[MotionClient] Connected to Motion Server at {port}")
        
        try:
            self.socket.send(pickle.dumps({'cmd': 'ping'}))
            if self.socket.poll(2000): # 2s timeout
                self.socket.recv()
                print("[MotionClient] Server online.")
            else:
                print("[MotionClient] WARNING: Server not responding!")
        except Exception as e:
            print(f"[MotionClient] Init error: {e}")

    def update_world(self, cuboids=None, pcl=None):
        """
        :param cuboids: dict, e.g. {'table': {'dims':..., 'pose':...}}
        :param pcl: numpy array (N, 3), raw point cloud data
        """
        # Preprocessing: if cuboids are nested under the 'cuboid' key, extract them because the Server will reassemble
        cuboids_data = cuboids
        if cuboids is not None and 'cuboid' in cuboids:
            cuboids_data = cuboids['cuboid']
        req = {
            'cmd': 'update_world',
            'cuboids': cuboids_data,
            'pcl': pcl # Directly send Numpy array, pickle will handle it automatically
        }
        try:
            self.socket.send(pickle.dumps(req))
            resp = pickle.loads(self.socket.recv())
            return resp.get('success', False)
        except Exception as e:
            print(f"[MotionClient] Update World Error: {e}")
            self._reset_socket()
            return False

    def plan(self, start_joint_state, target_pose, plan_config=None):
        """
        return: (waypoints_np, success)
        """
        req = {
            'cmd': 'plan',
            'start': start_joint_state,
            'target': target_pose,
            # plan_config is optional dict for planner parameters
        }
        try:
            self.socket.send(pickle.dumps(req))
            resp = pickle.loads(self.socket.recv())
            success = resp.get('success', False)
            waypoints = resp.get('waypoints', None)
            return waypoints, success
        except Exception as e:
            print(f"[MotionClient] Plan Error: {e}")
            self._reset_socket()
            return None, False

    def _reset_socket(self):
        print("[MotionClient] Resetting socket...")
        self.socket.close()
        self.context.term()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5556")


class PandaCommander(Node):
    """ FR3 Robot Controller with ZMQ Motion Planner """
    def __init__(self, robot_name: str = "fr3"):
        super().__init__('fr3_controller')
        self.callback_group = ReentrantCallbackGroup()
        
        # Robot parameters
        self.robot_name = robot_name
        self.joint_names = [
            f'{robot_name}_joint1', f'{robot_name}_joint2', f'{robot_name}_joint3', 
            f'{robot_name}_joint4', f'{robot_name}_joint5', f'{robot_name}_joint6', 
            f'{robot_name}_joint7'
        ]
        
        # TODO: define TCP transform if needed
        self.T_body_tcp = Transform.from_dict({"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.0]})
        self.T_tcp_body = self.T_body_tcp.inverse()
        self.T_tcp_link8 = self.T_body_tcp # Alias

        # wrench 
        self.wrench_raw = np.zeros(6)
        self.wrench_baseline = np.zeros(6)
        self.wrench_filtered = np.zeros(6)
        self.baseline_samples = []
        self.baseline_ready = False
        self.baseline_sample_count = 100        # 前100帧用于基线估计
        self.ema_alpha = 0.02                   # 低通滤波系数（慢）
        self.contact_hysteresis_count = 3       # 连续帧数确认接触
        self._contact_count = 0
        
        # --- ROS Subs/Pubs ---
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1, callback_group=self.callback_group)
        self.joint_velo_pub = self.create_publisher(JointState, '/joint_velocity_controller/joint_velocity', 1, callback_group=self.callback_group)
        self.predicted_path_pub = self.create_publisher(Path, '/planned_trajectory', 1, callback_group=self.callback_group)
        self.force_sub = self.create_subscription(FrankaRobotState, "/franka_robot_state_broadcaster/robot_state", self.force_callback, 10, callback_group=self.callback_group)
        self.robot_state_sub = self.create_subscription(FrankaRobotState, '/franka_robot_state_broadcaster/robot_state', self.robot_state_cb, 10, callback_group=self.callback_group)

        # --- Clients ---
        self.grasp_client = ActionClient(self, Grasp, f'/{self.robot_name}_gripper/grasp', callback_group=self.callback_group)
        self.move_client = ActionClient(self, Move, f'/{self.robot_name}_gripper/move', callback_group=self.callback_group)
        self.error_recovery_client = ActionClient(self, ErrorRecovery, '/action_server/error_recovery', callback_group=self.callback_group)
        self.collision_client = self.create_client(SetForceTorqueCollisionBehavior, '/service_server/set_force_torque_collision_behavior')
        self.planner = MotionPlannerClient() 
        
        # PID & Limits
        self.kp = 1.0
        self.kd = 0.04
        self.dt = 0.05
        self.limit_vel = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) * 0.2
        self.limit_acc = np.array([15, 7.5, 10, 12.5, 15, 20, 20]) * 0.04
        self.goal_tolerance = 0.01
        
        self.joint_command_msg = JointState()
        self.react_control_flag = False
        self.force_limit = 10.0
        self.wrench = np.zeros(6)
        self.robot_error = False
        self.home_joints = [
            0.0,                    # joint1
            -math.pi / 4,          # joint2: -π/4
            0.0,                    # joint3
            -3 * math.pi / 4,      # joint4: -3π/4
            0.0,                    # joint5
            math.pi / 2,           # joint6: π/2
            math.pi / 4            # joint7: π/4
        ]

        self.current_joint_state = None
        
        # Kinematics Model (for Client side FK/IK utils)
        self.panda = rtb.models.Panda()

        # Init World Config (Table)
        self.table_cfg = None

        self.create_timer(0.005, self.joint_velo_publisher, callback_group=self.callback_group)
        
        # Wait for robot state
        self._wait_for_connection()
        self.set_high_collision_thresholds()


    def _wait_for_connection(self):
        timeout = 10.0
        start_time = time.time()
        while self.current_joint_state is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.current_joint_state:
            self.get_logger().info("Controller connected to Robot.")
        else:
            self.get_logger().error("Robot Joint State Timeout.")

    def set_high_collision_thresholds(self):
        """Increase collision thresholds to allow grasping/contact with environment"""
        
        # Wait for service
        if not self.collision_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Collision behavior service not available.")
            return

        request = SetForceTorqueCollisionBehavior.Request()

        # ========= 设置较高的阈值（常用配置） =========
        # request.lower_torque_thresholds_acc  = [30, 30, 30, 20, 20, 15, 10]
        # request.upper_torque_thresholds_acc  = [45, 45, 45, 35, 35, 25, 20]

        # request.lower_force_thresholds_acc   = [50, 50, 50, 30, 30, 30]
        # request.upper_force_thresholds_acc   = [80, 80, 80, 50, 50, 50]

        request.lower_torque_thresholds_nominal = [20., 20., 20., 15., 15., 10., 10.]
        request.upper_torque_thresholds_nominal = [35., 35., 35., 25., 25., 20., 15.]

        request.lower_force_thresholds_nominal  = [40., 40., 40., 25., 25., 25.]
        request.upper_force_thresholds_nominal  = [100., 100., 100., 100., 100., 100.]
        # ========================================

        future = self.collision_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("High collision thresholds successfully set!")
        else:
            self.get_logger().error("Failed to set collision thresholds.")
    

    def visualize_trajectory(self, joint_waypoints, panda):
        """Visualize planned trajectory as a Path message"""
        trajectory = []
        for joints in joint_waypoints:
            T_ee = panda.fkine(joints)
            trajectory.append(np.array(T_ee))
        path_msg = self.create_path_msg(trajectory, frame_id="fr3_link0")
        self.predicted_path_pub.publish(path_msg)


    def create_path_msg(self, trajectory, frame_id="world", ee_frame_id="cur_link", color_alpha=1.0):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        for pose_matrix in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(pose_matrix[0, 3])
            pose_stamped.pose.position.y = float(pose_matrix[1, 3])
            pose_stamped.pose.position.z = float(pose_matrix[2, 3])
            rot_matrix = pose_matrix[:3, :3]
            quat = Rotation.from_matrix(rot_matrix).as_quat()
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])
            path_msg.poses.append(pose_stamped)
        return path_msg


    def joint_velo_publisher(self):
        try:
            if self.react_control_flag:
                self.joint_command_msg.header.stamp = self.get_clock().now().to_msg()
                self.joint_velo_pub.publish(self.joint_command_msg)
        except Exception as e:
            pass

    
    def force_callback(self, msg):
        """After receiving sensor callback: save raw data and apply a simple low-pass filter (EMA) to wrench_filtered"""
        raw = np.array([
            msg.o_f_ext_hat_k.wrench.force.x,
            msg.o_f_ext_hat_k.wrench.force.y,
            msg.o_f_ext_hat_k.wrench.force.z,
            msg.o_f_ext_hat_k.wrench.torque.x,
            msg.o_f_ext_hat_k.wrench.torque.y,
            msg.o_f_ext_hat_k.wrench.torque.z,
        ])
        self.wrench_raw = raw
        # Use the first baseline_sample_count frames at startup to compute the baseline average
        if not getattr(self, 'baseline_ready', False):
            self.baseline_samples.append(raw)
            if len(self.baseline_samples) >= self.baseline_sample_count:
                self.wrench_baseline = np.mean(self.baseline_samples, axis=0)
                self.baseline_ready = True
        else:
            # During operation, slowly update baseline using EMA (optional)
            self.wrench_baseline = (1 - self.ema_alpha) * self.wrench_baseline + self.ema_alpha * raw
        # Apply EMA low-pass filter to raw signal to reduce high-frequency noise
        self.wrench_filtered = (1 - self.ema_alpha) * getattr(self, 'wrench_filtered', np.zeros(6)) + self.ema_alpha * raw


    def robot_state_cb(self, msg):
        detected_error = False
        self.robot_mode = msg.robot_mode
        if np.any([msg.collision_indicators.is_cartesian_angular_collision.x, msg.collision_indicators.is_cartesian_angular_collision.y, msg.collision_indicators.is_cartesian_angular_collision.z]):
            detected_error = True
        for s in Errors.__slots__:
            if getattr(msg.current_errors, s):
                detected_error = True
        if not self.robot_error and detected_error:
            self.robot_error = True
            self.get_logger().warn("Detected robot error")
        if self.robot_error and not detected_error:
            self.robot_error = False
            self.get_logger().info("Robot error cleared")


    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg


    def get_current_joint_position(self) -> Optional[List[float]]:
        if self.current_joint_state is None:
            return None
        joint_positions = []
        for joint_name in self.joint_names:
            try:
                idx = self.current_joint_state.name.index(joint_name)
                joint_positions.append(self.current_joint_state.position[idx])
            except ValueError:
                return None
        return joint_positions


    def get_ee_pose(self):
        joint_positions = self.get_current_joint_position()
        joint_positions = np.array(joint_positions)
        T_ee = self.panda.fkine(joint_positions)
        return np.array(T_ee)


    def wait_for_action_result(self, future, timeout: float = 30.0):
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().warn(f"Action timed out after {timeout} seconds.")
            return None
        return future.result()


    def home(self) -> bool:
        self.get_logger().info("Moving to home position...")
        current_joints = np.array(self.get_current_joint_position())
        error = self.home_joints - current_joints
        if np.linalg.norm(error) <self.goal_tolerance:
            return True
        home_pose = self.panda.fkine(np.array(self.home_joints)).A @ self.T_tcp_link8.as_matrix()
        self.goto_pose(Transform.from_matrix(home_pose))
        return self.goto_joints(self.home_joints)
    
    
    def set_table(self, table_height: float = 0., table_size: float = 0.5, T_base_task: Transform = Transform.identity()) -> None:
        pose = reorder_pose_list(T_base_task.to_list())
        pose[2] += table_height
        # Only update the local Config dictionary
        self.table_cfg = {
            "cuboid": {
                "table": {
                    "dims": [table_size, table_size, 0.01],
                    "pose": pose,
                },
            },
        }


    def planning(self, target_pose, pcl=None, plan_config=None):
        """
        Call ZMQ Client for planning, supports PCL input
        """
        target_pose = reorder_pose_list((target_pose * self.T_tcp_link8).to_list())
        
        # Prepare data
        # 1. Cuboids (Table)
        cuboids_data = self.table_cfg # Default is {'cuboid': {'table': ...}}

        # 2. PointCloud (if you have)
        # pcl must be a (N, 3) numpy array
        pcl_data = pcl 
        self.planner.update_world(cuboids=cuboids_data, pcl=pcl_data)
        
        # 3. Request planning
        cur_joint = np.array(self.get_current_joint_position())
        joint_waypoints, success = self.planner.plan(cur_joint, target_pose, plan_config=plan_config)
        return joint_waypoints

    
    def goto_joints(self, target_joints):
        arrived = False
        current_joints = np.array(self.get_current_joint_position())
        error = target_joints - current_joints
        self.react_control_flag = True
        last_velocity = np.array([0.0]*7)
        last_error = error 

        if np.linalg.norm(error) < self.goal_tolerance:
            self.react_control_flag = False
            return True
            
        while np.linalg.norm(error) > self.goal_tolerance and (self.robot_error is False):
            loop_start = time.time()
            robot_state_joint = self.get_current_joint_position()
            error = target_joints - np.array(robot_state_joint)
            error_derivative = (error - last_error) / self.dt
            raw_joint_vel = (self.kp * error) + (self.kd * error_derivative)
            
            joint_acc = (raw_joint_vel - last_velocity) / self.dt
            if (np.abs(joint_acc) > self.limit_acc).any():
                joint_acc = np.clip(joint_acc, -self.limit_acc, self.limit_acc)
                joint_vel = last_velocity + joint_acc * self.dt
            else:
                joint_vel = np.clip(raw_joint_vel, -self.limit_vel, self.limit_vel)
            
            last_velocity = joint_vel
            last_error = error
            self.joint_command_msg.name = self.joint_names
            self.joint_command_msg.velocity = joint_vel.tolist()
            elapsed = time.time() - loop_start
            if self.dt > elapsed:
                time.sleep(self.dt - elapsed)
        self.react_control_flag = False
        return True


    def goto_pose(self, target_pose: Transform, pcl=None, plan_config=None) -> bool:
        cur_pose = Transform.from_matrix(self.get_ee_pose()) * self.T_tcp_link8.inverse()
        ## whether arrive at target
        dist =  (target_pose * cur_pose.inverse())
        if np.linalg.norm(dist.translation) < 0.01 and dist.rotation.as_euler('ZYX').abs().sum() < 0.1:
            return True
        
        self.get_logger().info(f"Planning to target pose...")
        joint_waypoints = self.planning(target_pose, pcl=pcl, plan_config=plan_config)
        if joint_waypoints is None:
            self.get_logger().error("Planning failed.")
            return False
        self.visualize_trajectory(joint_waypoints, self.panda)
        return self.goto_joint_trajectory(joint_waypoints)


    def goto_pose_reactive(self, target_pose, Gain=1, Lambda=0.1, threshold=0.001, detect_force=True, pcl=None, watch_dog_limit=70):
        target_pose = target_pose * self.T_tcp_body
        self.react_control_flag = True
        arrived = False
        watch_dog = 0
        last_joint = np.array(self.get_current_joint_position())
        last_velocity = np.array([0.0]*7)
        joint_vel = np.array([0.0]*7)
        wrench_baseline = self.wrench_filtered.copy()
        while True and (self.robot_error is False):
            loop_start = time.time()
            robot_state_joint = self.get_current_joint_position()
            # Gain_mod = Gain * max((self.force_limit - np.linalg.norm(self.wrench)) / (self.force_limit - 6), 0)
            # print('Gain mod:', Gain_mod)
            raw_joint_vel, arrived = calculate_velocity(self.panda, np.array(robot_state_joint), target_pose, 
                                                    obstacles=None, Gain=Gain, Lambda=Lambda, threshold=threshold)
            joint_movement = np.linalg.norm(np.array(self.get_current_joint_position())-last_joint)
            last_joint = np.array(self.get_current_joint_position())
            if joint_movement < 0.001:
                watch_dog += 1
            else:
                watch_dog = 0
            if arrived is True or watch_dog > watch_dog_limit:
                break
            if np.linalg.norm(self.wrench_filtered-wrench_baseline) > self.force_limit and detect_force:
                print('Force limit reached, stopping reactive control.')
                break
            
            joint_acc = (raw_joint_vel - last_velocity) / self.dt
            joint_acc = np.clip(joint_acc, -self.limit_acc, self.limit_acc)
            joint_vel = last_velocity + joint_acc * self.dt
            joint_vel = np.clip(joint_vel, -self.limit_vel, self.limit_vel)
            
            self.joint_command_msg.velocity = joint_vel.tolist()
            last_velocity = joint_vel
            elapsed = time.time() - loop_start
            if self.dt > elapsed:
                time.sleep(self.dt - elapsed)
        
        for _ in range(5):
            joint_vel = joint_vel*0.5
            self.joint_command_msg.velocity = (joint_vel).tolist()
            time.sleep(0.01)
        self.react_control_flag = True
        return arrived

    
    def move_gripper(self, width, speed=0.1):
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        send_future = self.move_client.send_goal_async(goal)
        start = time.time()
        while not send_future.done() and (time.time() - start) < 2.0:
            time.sleep(0.01)
        if not send_future.done(): return False
        handle = send_future.result()
        if not handle or not handle.accepted: return False
        res_future = handle.get_result_async()
        res = self.wait_for_action_result(res_future, 2.0)
        return True


    def grasp(self, width: float = 0.01, speed: float = 0.1, force: float = 50.0, inner_epsilon: float = 0.05, outer_epsilon: float = 0.05) -> bool:
        if not self.grasp_client.wait_for_server(timeout_sec=5.0):
            return False
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon = GraspEpsilon(inner=inner_epsilon, outer=outer_epsilon)
        send_future = self.grasp_client.send_goal_async(goal)
        start = time.time()
        while not send_future.done() and (time.time() - start) < 5.0:
            time.sleep(0.01)
        if not send_future.done(): return False
        handle = send_future.result()
        if not handle or not handle.accepted: return False
        res_future = handle.get_result_async()
        res = self.wait_for_action_result(res_future, 5.0)
        return res and res.status == 4
    

    def goto_joint_trajectory(self, joint_waypoints: List[List[float]]) -> bool:
        waypoints = np.array(joint_waypoints)
        if len(waypoints) < 2: return False
        self.react_control_flag = True
        process_tolerance = 0.4
        final_tolerance = self.goal_tolerance
        last_velocity = np.array([0.0] * 7)
        last_error = np.zeros(7)
        cur_i = 0
        while cur_i < len(waypoints) - 1 and self.robot_error is False:
            loop_start = time.time()
            cur_target = waypoints[cur_i + 1]
            current_joints = np.array(self.get_current_joint_position())
            error = cur_target - current_joints
            if np.linalg.norm(error) < (process_tolerance if cur_i < len(waypoints)-2 else process_tolerance*0.5):
                cur_i += 1
            error_derivative = (error - last_error) / self.dt
            raw_joint_vel = (self.kp * error) + (self.kd * error_derivative)
            
            joint_acc = np.clip((raw_joint_vel - last_velocity) / self.dt, -self.limit_acc, self.limit_acc)
            joint_vel = np.clip(last_velocity + joint_acc * self.dt, -self.limit_vel, self.limit_vel)
            
            last_velocity = joint_vel
            last_error = error
            self.joint_command_msg.name = self.joint_names
            self.joint_command_msg.velocity = joint_vel.tolist()
            if self.dt > (time.time() - loop_start): time.sleep(self.dt - (time.time() - loop_start))
        # Final Alignment
        timeout_start = time.time()
        final_target = waypoints[-1]
        while self.robot_error is False and (time.time() - timeout_start < 3.0):
            current_joints = np.array(self.get_current_joint_position())
            error = final_target - current_joints
            if np.linalg.norm(error) < final_tolerance: break
            joint_vel = np.clip(self.kp * error, -0.2, 0.2)
            self.joint_command_msg.velocity = joint_vel.tolist()
            time.sleep(self.dt)
        
        self.joint_command_msg.velocity = [0.0]*7
        time.sleep(0.05)
        self.react_control_flag = False
        return True


    def recover(self) -> bool:
        if not self.error_recovery_client.wait_for_server(timeout_sec=5.0): return False
        goal = ErrorRecovery.Goal()
        future = self.error_recovery_client.send_goal_async(goal)
        start = time.time()
        while not future.done() and (time.time() - start) < 5.0: time.sleep(0.01)
        if not future.done(): return False
        handle = future.result()
        if not handle or not handle.accepted: return False
        res = self.wait_for_action_result(handle.get_result_async(), 5.0)
        if res and res.status == 4:
            return self.home()
        return False


# --- For Testing ---
class FrankaPoseGenerator:
    def __init__(self, robot_name="fr3"):
        self.panda = rtb.models.Panda()
        self.q_min = np.array([-2.7, -1.7, -2.8, -3.0, -2.8, 0.01, -2.8]) * 0.5
        self.q_max = np.array([ 2.7,  1.7,  2.8, -0.1,  2.8, 3.7,  2.8]) * 0.5
    
    def _to_user_format(self, pos, quat):
        user_quat = [quat[3], quat[0], quat[1], quat[2]] 
        return Transform.from_list(user_quat + list(pos))

    def get_guaranteed_reachable_pose(self):
        q_rand = np.random.uniform(self.q_min, self.q_max)
        T = self.panda.fkine(q_rand)
        pos = T.t 
        quat = Rotation.from_matrix(T.R).as_quat()
        if pos[2] < 0.05: return self.get_guaranteed_reachable_pose()
        return self._to_user_format(pos, quat)

def main():
    rclpy.init()
    panda_commander = PandaCommander(robot_name="fr3")
    from rclpy.executors import MultiThreadedExecutor
    from threading import Thread
    executor = MultiThreadedExecutor(10)
    executor.add_node(panda_commander)
    generator = FrankaPoseGenerator()
    try:
        t1 = Thread(target=executor.spin, daemon=True)
        t1.start()
        for i in range(2):
            print(f"--- Loop {i} ---")
            print("Testing Grasp...")
            panda_commander.grasp(width=0.08, force=30.0)
            time.sleep(1.0) 
            print("Testing Home...")
            panda_commander.home()
            time.sleep(1.0)
            for j in range(2):
                print(f"Testing Plan to Random Pose {j}...")
                target_pose = generator.get_guaranteed_reachable_pose()
                panda_commander.goto_pose(target_pose)
                time.sleep(1.0)
            print("Testing Grasp Close...")
            panda_commander.grasp(width=0.02, force=30.0)
            time.sleep(1.0) 
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        panda_commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()