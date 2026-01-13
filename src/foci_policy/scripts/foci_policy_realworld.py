#!/usr/bin/env python3
"""
FOCI Policy RealWorld Script
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import json
import numpy as np
import cv2
import torch
import yaml
import zmq
from pathlib import Path
from PIL import Image
import open3d as o3d

from transform import Transform, Rotation
from utils import to_o3d_pcd
from doi_policy.model.foci_actor import FOCIActor
from doi_policy.foci_real_world.pose_estimator.mask_generator import MaskGenerator
from doi_policy.foci_real_world.pose_estimator.fp_real_world_utils import (
    preprocess_depth,
)
from foundation_pose.wrapper import FoundationPoseWrapper

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class ZMQRobotInterface:
    def __init__(self, request_port=5555, response_port=5556, timeout=30000):
        """ Initialize ZMQ sockets for bidirectional communication """
        self.context = zmq.Context()
        # Socket for sending commands to robot
        self.request_socket = self.context.socket(zmq.REQ)
        self.request_socket.connect(f"tcp://localhost:{request_port}")
        self.request_socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.request_socket.setsockopt(zmq.SNDTIMEO, timeout)
        print(f"ZMQ: Connected to robot on port {request_port}")
    
    def get_observation(self):
        """
        Request current observation from robot
        Returns:
            dict: {
                'rgb': np.ndarray (H, W, 3) uint8,
                'depth': np.ndarray (H, W) uint16,
                'K': np.ndarray (3, 3) camera intrinsics,
                'cam_extrinsic': np.ndarray (4, 4) camera extrinsics,
                'gripper_pose': dict with 'position' and 'orientation',
                'gripper_open': float [0, 1]
            }
        """
        request = {'type': 'get_observation'}
        self.request_socket.send_json(request)
        
        response = self.request_socket.recv_json()
        
        if response['status'] != 'success':
            raise RuntimeError(f"Failed to get observation: {response.get('message', 'Unknown error')}")
        
        data = response['data']
        
        # Decode images from base64 or raw bytes
        rgb = self._decode_image(data['rgb'], dtype=np.uint8)
        depth = self._decode_image(data['depth'], dtype=np.uint16)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'K': np.array(data['K']).reshape(3, 3),
            'cam_extrinsic': np.array(data['cam_extrinsic']).reshape(4, 4),
            'gripper_pose': data['gripper_pose'],
            'gripper_open': data.get('gripper_open', 1.0)
        }
    
    def execute_trajectory(self, trajectory, mode='grasp', blocking=True):
        """
        Send trajectory to robot for execution
        
        Args:
            trajectory: List of 4x4 pose matrices or dict with 'poses' and 'gripper_open'
            mode: 'grasp' or 'manip'
            blocking: Wait for execution completion
            
        Returns:
            dict: {'status': 'success'/'failed', 'message': str}
        """
        # Convert trajectory to serializable format
        if isinstance(trajectory, list):
            poses_list = [pose.tolist() for pose in trajectory]
            request = {
                'type': 'execute_trajectory',
                'mode': mode,
                'poses': poses_list,
                'blocking': blocking
            }
        elif isinstance(trajectory, dict):
            poses_list = [pose.tolist() for pose in trajectory['poses']]
            request = {
                'type': 'execute_trajectory',
                'mode': mode,
                'poses': poses_list,
                'gripper_open': trajectory.get('gripper_open', None),
                'blocking': blocking
            }
        else:
            raise ValueError(f"Invalid trajectory type: {type(trajectory)}")
        
        self.request_socket.send_json(request)
        response = self.request_socket.recv_json()
        
        return response
    
    def close_gripper(self, blocking=True):
        """Close gripper"""
        request = {'type': 'close_gripper', 'blocking': blocking}
        self.request_socket.send_json(request)
        return self.request_socket.recv_json()
    
    def open_gripper(self, blocking=True):
        """Open gripper"""
        request = {'type': 'open_gripper', 'blocking': blocking}
        self.request_socket.send_json(request)
        return self.request_socket.recv_json()
    
    def _decode_image(self, data, dtype=np.uint8):
        """Decode image from base64 or bytes"""
        import base64
        if isinstance(data, str):
            # Base64 encoded
            img_bytes = base64.b64decode(data)
            img_array = np.frombuffer(img_bytes, dtype=dtype)
            # Shape information should be in metadata
            return img_array
        elif isinstance(data, dict):
            # Contains shape info
            img_bytes = base64.b64decode(data['data'])
            img_array = np.frombuffer(img_bytes, dtype=dtype)
            return img_array.reshape(data['shape'])
        else:
            raise ValueError(f"Unknown image data format: {type(data)}")
    
    def close(self):
        """Close ZMQ sockets"""
        self.request_socket.close()
        self.context.term()


class FOCIPolicyRobotController:
    """Real robot controller using FOCI policy"""
    
    def __init__(self, task_name, config_dir='../config', checkpoint_dir='../checkpoints',
                 assets_dir='../assets', device='cuda:0', voxel_size=0.004, num_points=2048):
        """
        Initialize robot controller
        
        Args:
            task_name: Task name (must be in task_config.yaml)
            config_dir: Configuration directory
            checkpoint_dir: Checkpoint directory
            assets_dir: Assets directory (meshes)
            device: torch device
            voxel_size: Voxel size for downsampling
            num_points: Number of points per object
        """
        self.task_name = task_name
        self.config_dir = Path(config_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.assets_dir = Path(assets_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.voxel_size = voxel_size
        self.num_points = num_points
        
        print(f"Initializing FOCI Policy Controller for task: {task_name}")
        print(f"Device: {self.device}")
        
        # Load task configuration
        self._load_task_config()
        
        # Initialize mask generator and Foundation Pose
        self._init_pose_estimators()
        
        # Load FOCI actors
        self._load_actors()
        
        print("✓ Initialization complete\n")
    
    def _load_task_config(self):
        """Load task configuration from task_config.yaml"""
        task_config_path = self.config_dir / 'task_config.yaml'
        if not task_config_path.exists():
            # Try in foci_real_world/configs
            task_config_path = Path(__file__).parent.parent / 'configs' / 'task_config.yaml'
        
        with open(task_config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        if self.task_name not in all_configs:
            raise ValueError(f"Task {self.task_name} not found in task_config.yaml")
        
        self.task_config = all_configs[self.task_name]
        
        # Extract object information
        self.base_object = self.task_config['objects'][0]  # pa
        self.moving_object = self.task_config['objects'][1]  # pb
        self.base_mesh = self.assets_dir / f"{self.task_config['mesh_names'][0]}.obj"
        self.moving_mesh = self.assets_dir / f"{self.task_config['mesh_names'][1]}.obj"
        self.lan_grasp = self.task_config['lan_grasp']
        self.lan_manip = self.task_config['lan_manip']
        
        # Verify meshes exist
        if not self.base_mesh.exists():
            raise FileNotFoundError(f"Base mesh not found: {self.base_mesh}")
        if not self.moving_mesh.exists():
            raise FileNotFoundError(f"Moving mesh not found: {self.moving_mesh}")
        
        print(f"Task config loaded:")
        print(f"  Base object: {self.base_object} (mesh: {self.base_mesh.name})")
        print(f"  Moving object: {self.moving_object} (mesh: {self.moving_mesh.name})")
        print(f"  Grasp instruction: '{self.lan_grasp}'")
        print(f"  Manip instruction: '{self.lan_manip}'")
    
    def _init_pose_estimators(self):
        """Initialize mask generator and Foundation Pose wrappers"""
        print("Initializing pose estimators...")
        
        # Mask generator
        self.mask_generator = MaskGenerator()
        
        # Foundation Pose for base object
        self.fp_base_wrapper = FoundationPoseWrapper(
            mesh_dir=str(self.assets_dir),
            debug_dir=None
        )
        self.fp_base_wrapper.update_grasp_obj_name(self.base_mesh.stem)
        self.fp_base_estimator = self.fp_base_wrapper.create_estimator(debug_level=-1)
        
        # Foundation Pose for moving object
        self.fp_moving_wrapper = FoundationPoseWrapper(
            mesh_dir=str(self.assets_dir),
            debug_dir=None
        )
        self.fp_moving_wrapper.update_grasp_obj_name(self.moving_mesh.stem)
        self.fp_moving_estimator = self.fp_moving_wrapper.create_estimator(debug_level=-1)
        
        print("✓ Pose estimators initialized")
    
    def _load_actors(self):
        """Load FOCI grasp and manipulation actors"""
        print("Loading FOCI actors...")
        
        # Grasp actor
        grasp_ckpt = self.checkpoint_dir / 'foci' / 'real' / 'grasp' / 'best_model.pth'
        if not grasp_ckpt.exists():
            raise FileNotFoundError(f"Grasp checkpoint not found: {grasp_ckpt}")
        
        self.grasp_actor = FOCIActor(
            checkpoint_path=str(grasp_ckpt),
            device=self.device
        )
        print(f"✓ Loaded grasp actor from {grasp_ckpt}")
        
        # Manipulation actor
        manip_ckpt = self.checkpoint_dir / 'foci' / 'real' / 'manip' / 'best_model.pth'
        if not manip_ckpt.exists():
            raise FileNotFoundError(f"Manip checkpoint not found: {manip_ckpt}")
        
        self.manip_actor = FOCIActor(
            checkpoint_path=str(manip_ckpt),
            device=self.device
        )
        print(f"✓ Loaded manip actor from {manip_ckpt}")
    
    def estimate_object_poses(self, obs):
        """
        Estimate object poses using Foundation Pose
        
        Args:
            obs: Observation dict from robot
            
        Returns:
            dict: {
                'base_pose': 4x4 matrix,
                'moving_pose': 4x4 matrix,
                'base_mask': binary mask,
                'moving_mask': binary mask
            }
        """
        rgb = obs['rgb']
        depth = obs['depth']
        K = obs['K']
        cam_extrinsic = obs['cam_extrinsic']
        
        # Preprocess depth
        if depth.dtype == np.uint16:
            depth = preprocess_depth(depth)
        
        # Get masks
        image_pil = Image.fromarray(rgb)
        
        # Base object mask
        base_boxes, base_logits, base_phrases = self.mask_generator.get_scene_object_bboxes(
            image_pil, [self.base_object], visualize=False, logdir=None
        )
        base_segmasks = self.mask_generator.get_segmentation_masks(
            image_pil, base_boxes, base_logits, base_phrases, visualize=False, save_path=None
        )
        if len(base_segmasks) == 0:
            raise RuntimeError(f"No mask found for base object: {self.base_object}")
        base_mask = base_segmasks[0].astype(np.uint8)
        
        # Moving object mask
        moving_boxes, moving_logits, moving_phrases = self.mask_generator.get_scene_object_bboxes(
            image_pil, [self.moving_object], visualize=False, logdir=None
        )
        moving_segmasks = self.mask_generator.get_segmentation_masks(
            image_pil, moving_boxes, moving_logits, moving_phrases, visualize=False, save_path=None
        )
        if len(moving_segmasks) == 0:
            raise RuntimeError(f"No mask found for moving object: {self.moving_object}")
        moving_mask = moving_segmasks[0].astype(np.uint8)
        
        # Estimate poses
        # Base object
        if self.fp_base_estimator.pose_last is None:
            base_pose_cam = self.fp_base_estimator.register(
                K=K, rgb=rgb, depth=depth, ob_mask=base_mask, iteration=20
            )
        else:
            base_pose_cam = self.fp_base_estimator.track_one(
                K=K, rgb=rgb, depth=depth, iteration=5
            )
        base_pose_world = cam_extrinsic @ base_pose_cam
        
        # Moving object
        if self.fp_moving_estimator.pose_last is None:
            moving_pose_cam = self.fp_moving_estimator.register(
                K=K, rgb=rgb, depth=depth, ob_mask=moving_mask, iteration=20
            )
        else:
            moving_pose_cam = self.fp_moving_estimator.track_one(
                K=K, rgb=rgb, depth=depth, iteration=5
            )
        moving_pose_world = cam_extrinsic @ moving_pose_cam
        
        return {
            'base_pose': base_pose_world,
            'moving_pose': moving_pose_world,
            'base_mask': base_mask,
            'moving_mask': moving_mask
        }
    
    def extract_point_clouds(self, obs, pose_result):
        """
        Extract masked point clouds for both objects
        
        Args:
            obs: Observation dict
            pose_result: Result from estimate_object_poses
            
        Returns:
            dict: {
                'pa_points': (N, 3),
                'pa_colors': (N, 3),
                'pb_points': (N, 3),
                'pb_colors': (N, 3)
            }
        """
        rgb = obs['rgb']
        depth = obs['depth']
        K = obs['K']
        cam_extrinsic = obs['cam_extrinsic']
        
        # Preprocess depth
        if depth.dtype == np.uint16:
            depth = preprocess_depth(depth)
        
        base_mask = pose_result['base_mask']
        moving_mask = pose_result['moving_mask']
        
        # Extract masked point clouds (similar to preprocess_realworld_data.py)
        def extract_masked_pcd(mask):
            depth_h, depth_w = depth.shape
            mask_h, mask_w = mask.shape
            if mask_h != depth_h or mask_w != depth_w:
                mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
            
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            v, u = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
            valid_depth = (depth > 0.01) & (depth < 3.0)
            valid_mask = mask.astype(bool) & valid_depth
            
            v_valid = v[valid_mask]
            u_valid = u[valid_mask]
            z_valid = depth[valid_mask]
            
            x = (u_valid - cx) * z_valid / fx
            y = (v_valid - cy) * z_valid / fy
            z = z_valid
            points = np.stack([x, y, z], axis=1)
            
            # Transform to world coordinates
            points = (cam_extrinsic[:3, :3] @ points.T).T + cam_extrinsic[:3, 3]
            
            colors = rgb[valid_mask].astype(np.float32) / 255.0
            return points, colors
        
        pa_points_raw, pa_colors_raw = extract_masked_pcd(base_mask)
        pb_points_raw, pb_colors_raw = extract_masked_pcd(moving_mask)
        
        # Downsample and sample to num_points
        def process_pcd(points, colors):
            if len(points) == 0:
                return np.zeros((self.num_points, 3)), np.zeros((self.num_points, 3))
            
            pcd = to_o3d_pcd(points, colors)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            if len(pcd.points) > self.num_points:
                indices = np.random.choice(len(pcd.points), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(pcd.points), self.num_points, replace=True)
            
            return np.asarray(pcd.points)[indices], np.asarray(pcd.colors)[indices]
        
        pa_points, pa_colors = process_pcd(pa_points_raw, pa_colors_raw)
        pb_points, pb_colors = process_pcd(pb_points_raw, pb_colors_raw)
        
        return {
            'pa_points': pa_points,
            'pa_colors': pa_colors,
            'pb_points': pb_points,
            'pb_colors': pb_colors
        }
    
    def predict_grasp_trajectory(self, obs, pose_result, pcd_result):
        """
        Predict grasp trajectory using FOCI grasp actor
        
        Returns:
            List of 4x4 gripper pose matrices
        """
        # Get current gripper pose
        gripper_dict = obs['gripper_pose']
        gripper_pos = np.array([
            gripper_dict['position']['x'],
            gripper_dict['position']['y'],
            gripper_dict['position']['z']
        ])
        gripper_quat = np.array([
            gripper_dict['orientation']['x'],
            gripper_dict['orientation']['y'],
            gripper_dict['orientation']['z'],
            gripper_dict['orientation']['w']
        ])
        gripper_pose = Transform(
            rotation=Rotation.from_quat(gripper_quat),
            translation=gripper_pos
        ).as_matrix()
        
        # Get object poses
        pa_pose = pose_result['base_pose']
        pb_pose = pose_result['moving_pose']
        
        # Compute relative pose (gripper relative to moving object)
        current_rel_pose = np.linalg.inv(pb_pose) @ gripper_pose
        
        # Canonicalize point clouds
        R = pb_pose[:3, :3]
        t = pb_pose[:3, 3]
        pa_points_canon = ((pcd_result['pa_points'] - t) @ R)
        pb_points_canon = ((pcd_result['pb_points'] - t) @ R)
        
        # Predict trajectory (relative to moving object)
        traj_rel = self.grasp_actor.predict(
            pa_points=pa_points_canon,
            pa_colors=pcd_result['pa_colors'],
            pb_points=pb_points_canon,
            pb_colors=pcd_result['pb_colors'],
            current_pose=current_rel_pose,
            language=self.lan_grasp
        )
        
        # Convert to world coordinates
        traj_world = []
        for rel_pose in traj_rel:
            world_pose = pb_pose @ rel_pose
            traj_world.append(world_pose)
        
        return traj_world
    
    def predict_manip_trajectory(self, obs, pose_result, pcd_result):
        """
        Predict manipulation trajectory using FOCI manip actor
        
        Returns:
            List of 4x4 moving object pose matrices
        """
        # Get object poses
        pa_pose = pose_result['base_pose']
        pb_pose = pose_result['moving_pose']
        
        # Compute relative pose (moving object relative to base)
        current_rel_pose = np.linalg.inv(pa_pose) @ pb_pose
        
        # Canonicalize point clouds
        R = pa_pose[:3, :3]
        t = pa_pose[:3, 3]
        pa_points_canon = ((pcd_result['pa_points'] - t) @ R)
        pb_points_canon = ((pcd_result['pb_points'] - t) @ R)
        
        # Predict trajectory (moving object relative to base)
        traj_rel = self.manip_actor.predict(
            pa_points=pa_points_canon,
            pa_colors=pcd_result['pa_colors'],
            pb_points=pb_points_canon,
            pb_colors=pcd_result['pb_colors'],
            current_pose=current_rel_pose,
            language=self.lan_manip
        )
        
        # Convert to world coordinates (gripper trajectory)
        # We need to compute gripper poses that maintain grasp
        gripper_dict = obs['gripper_pose']
        gripper_pos = np.array([
            gripper_dict['position']['x'],
            gripper_dict['position']['y'],
            gripper_dict['position']['z']
        ])
        gripper_quat = np.array([
            gripper_dict['orientation']['x'],
            gripper_dict['orientation']['y'],
            gripper_dict['orientation']['z'],
            gripper_dict['orientation']['w']
        ])
        current_gripper_pose = Transform(
            rotation=Rotation.from_quat(gripper_quat),
            translation=gripper_pos
        ).as_matrix()
        
        # Compute grasp transform (gripper relative to object)
        grasp_transform = np.linalg.inv(pb_pose) @ current_gripper_pose
        
        # Compute gripper trajectory
        traj_world = []
        for rel_pose in traj_rel:
            # Object pose in world
            obj_world_pose = pa_pose @ rel_pose
            # Gripper pose maintaining grasp
            gripper_world_pose = obj_world_pose @ grasp_transform
            traj_world.append(gripper_world_pose)
        
        return traj_world
    
    def visualize_prediction(self, obs, pose_result, pcd_result, trajectory, mode='grasp'):
        """Visualize predicted trajectory"""
        geometries = []
        
        # World frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(world_frame)
        
        # Point clouds
        pa_pcd = o3d.geometry.PointCloud()
        pa_pcd.points = o3d.utility.Vector3dVector(pcd_result['pa_points'])
        pa_pcd.colors = o3d.utility.Vector3dVector(pcd_result['pa_colors'])
        geometries.append(pa_pcd)
        
        pb_pcd = o3d.geometry.PointCloud()
        pb_pcd.points = o3d.utility.Vector3dVector(pcd_result['pb_points'])
        pb_pcd.colors = o3d.utility.Vector3dVector(pcd_result['pb_colors'])
        geometries.append(pb_pcd)
        
        # Object poses
        pa_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        pa_frame.transform(pose_result['base_pose'])
        geometries.append(pa_frame)
        
        pb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        pb_frame.transform(pose_result['moving_pose'])
        geometries.append(pb_frame)
        
        # Trajectory
        for i, pose in enumerate(trajectory):
            traj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            traj_frame.transform(pose)
            geometries.append(traj_frame)
        
        # Trajectory line
        if len(trajectory) > 1:
            traj_positions = [pose[:3, 3] for pose in trajectory]
            lines = [[i, i+1] for i in range(len(traj_positions)-1)]
            colors = [[1.0, 0.6, 0.0] for _ in lines]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(traj_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"FOCI Prediction - {mode.upper()} mode",
            width=1280,
            height=720
        )
    
    def execute_task(self, robot_interface, visualize=False):
        """ Execute full pick and place task
        
        Args:
            robot_interface: ZMQRobotInterface instance
            visualize: Whether to visualize predictions
            
        Returns:
            dict: Execution result
        """
        print("="*70)
        print(f"Executing task: {self.task_name}")
        print("="*70)
        
        try:
            # ========== GRASP PHASE ==========
            print("\n[1/2] GRASP PHASE")
            print("-"*70)
            
            # Get observation
            print("Getting observation...")
            obs = robot_interface.get_observation()
            print("✓ Observation received")
            
            # Estimate poses
            print("Estimating object poses...")
            pose_result = self.estimate_object_poses(obs)
            print(f"✓ Base object pose: {pose_result['base_pose'][:3, 3]}")
            print(f"✓ Moving object pose: {pose_result['moving_pose'][:3, 3]}")
            
            # Extract point clouds
            print("Extracting point clouds...")
            pcd_result = self.extract_point_clouds(obs, pose_result)
            print(f"✓ PA points: {pcd_result['pa_points'].shape}")
            print(f"✓ PB points: {pcd_result['pb_points'].shape}")
            
            # Predict grasp trajectory
            print("Predicting grasp trajectory...")
            grasp_traj = self.predict_grasp_trajectory(obs, pose_result, pcd_result)
            print(f"✓ Predicted {len(grasp_traj)} waypoints")
            
            # Visualize if requested
            if visualize:
                self.visualize_prediction(obs, pose_result, pcd_result, grasp_traj, mode='grasp')
            
            # Execute grasp trajectory
            print("Executing grasp trajectory...")
            result = robot_interface.execute_trajectory(grasp_traj, mode='grasp', blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Grasp execution failed: {result.get('message', 'Unknown error')}")
            print("✓ Grasp trajectory executed")
            
            # Close gripper
            print("Closing gripper...")
            result = robot_interface.close_gripper(blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Gripper close failed: {result.get('message', 'Unknown error')}")
            print("✓ Gripper closed")
            
            # ========== MANIPULATION PHASE ==========
            print("\n[2/2] MANIPULATION PHASE")
            print("-"*70)
            
            # Get new observation
            print("Getting observation...")
            obs = robot_interface.get_observation()
            print("✓ Observation received")
            
            # Estimate poses
            print("Estimating object poses...")
            pose_result = self.estimate_object_poses(obs)
            print(f"✓ Base object pose: {pose_result['base_pose'][:3, 3]}")
            print(f"✓ Moving object pose: {pose_result['moving_pose'][:3, 3]}")
            
            # Extract point clouds
            print("Extracting point clouds...")
            pcd_result = self.extract_point_clouds(obs, pose_result)
            print(f"✓ PA points: {pcd_result['pa_points'].shape}")
            print(f"✓ PB points: {pcd_result['pb_points'].shape}")
            
            # Predict manipulation trajectory
            print("Predicting manipulation trajectory...")
            manip_traj = self.predict_manip_trajectory(obs, pose_result, pcd_result)
            print(f"✓ Predicted {len(manip_traj)} waypoints")
            
            # Visualize if requested
            if visualize:
                self.visualize_prediction(obs, pose_result, pcd_result, manip_traj, mode='manip')
            
            # Execute manipulation trajectory
            print("Executing manipulation trajectory...")
            result = robot_interface.execute_trajectory(manip_traj, mode='manip', blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Manip execution failed: {result.get('message', 'Unknown error')}")
            print("✓ Manipulation trajectory executed")
            
            # Open gripper
            print("Opening gripper...")
            result = robot_interface.open_gripper(blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Gripper open failed: {result.get('message', 'Unknown error')}")
            print("✓ Gripper opened")
            
            print("\n" + "="*70)
            print("✓ TASK COMPLETED SUCCESSFULLY")
            print("="*70)
            
            return {'status': 'success', 'message': 'Task completed'}
            
        except Exception as e:
            print("\n" + "="*70)
            print(f"✗ TASK FAILED: {e}")
            print("="*70)
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'message': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Execute FOCI policy on real robot via ZMQ'
    )
    parser.add_argument('--task', type=str, required=True,
                       help='Task name from task_config.yaml')
    parser.add_argument('--config_dir', type=str, 
                       default='../config',
                       help='Configuration directory')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='../../checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--assets_dir', type=str,
                       default='../assets',
                       help='Assets directory (meshes)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='torch device')
    parser.add_argument('--zmq_port', type=int, default=5555,
                       help='ZMQ port for robot communication')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions before execution')
    parser.add_argument('--voxel_size', type=float, default=0.004,
                       help='Voxel size for downsampling')
    parser.add_argument('--num_points', type=int, default=2048,
                       help='Number of points per object')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    config_dir = script_dir / args.config_dir
    checkpoint_dir = script_dir / args.checkpoint_dir
    assets_dir = script_dir / args.assets_dir
    
    # Initialize controller
    controller = FOCIPolicyRobotController(
        task_name=args.task,
        config_dir=config_dir,
        checkpoint_dir=checkpoint_dir,
        assets_dir=assets_dir,
        device=args.device,
        voxel_size=args.voxel_size,
        num_points=args.num_points
    )
    
    # Initialize robot interface
    print("\nConnecting to robot via ZMQ...")
    robot = ZMQRobotInterface(request_port=args.zmq_port)
    print("✓ Connected to robot\n")
    
    try:
        # Execute task
        result = controller.execute_task(robot, visualize=args.visualize)
        
        # Print result
        if result['status'] == 'success':
            print("\n✓ Execution successful!")
            return 0
        else:
            print(f"\n✗ Execution failed: {result['message']}")
            return 1
    
    finally:
        # Cleanup
        robot.close()
        print("\nZMQ connection closed")


if __name__ == '__main__':
    exit(main())
