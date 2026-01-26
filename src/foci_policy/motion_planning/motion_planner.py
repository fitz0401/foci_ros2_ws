# Third Party
import torch
import time
import sys
import zmq
import pickle
import numpy as np

# CuRobo Imports
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Mesh, Cuboid
from utils_exp.vis import color_print

class MotionPlanner:
    def __init__(self, robot_yml="ur5e.yml", debug=True):
        self.debug = debug
        
        world_config_placeholder = {
            "cuboid": {
                "placeholder": {
                    "dims": [0.1, 0.1, 0.1],
                    "pose": [0.0, 0.0, -10.0, 1, 0, 0, 0.0],
                },
            },
        }
        
        t = time.time()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_yml,
            world_config_placeholder,
            collision_cache={"obb": 10, "mesh": 10},
            interpolation_dt=0.05,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        if self.debug:
            print(f"[MotionServer] Planner loaded in {time.time()-t:.3f}s")

    def plan(self, start_joint_state, target_pose, plan_config=None):
        goal_pose = Pose.from_list(target_pose)  
        start_joint_tensor = torch.tensor(start_joint_state, dtype=torch.float32).reshape(1, -1).cuda()
        start_state = JointState.from_position(start_joint_tensor)
        
        if plan_config is None:
            plan_config = MotionGenPlanConfig(max_attempts=60) 

        result = self.motion_gen.plan_single(start_state, goal_pose, plan_config)        
        
        success = result.success.detach().cpu().item()
        waypoints_np = None
        if success:
            traj = result.get_interpolated_plan()
            waypoints_np = traj.position.detach().cpu().numpy()
            
        return waypoints_np, success

    def update_world(self, cuboids_dict=None, pcl_array=None):
        """
        Construct WorldConfig from given cuboid dictionary and pcl numpy array
        """
        t = time.time()
        self.motion_gen.clear_world_cache()
        
        world_cfg_args = {}

        # 1. Process Cuboids
        if cuboids_dict is not None and len(cuboids_dict) > 0:
            # If cuboids are provided as a dict (e.g., {'table': {'dims':..., 'pose':...}})
            # CuRobo's WorldConfig can directly accept dict conversion
            # Or we can explicitly construct Cuboid objects, here we leverage WorldConfig's flexibility
            world_cfg_args['cuboid'] = WorldConfig.from_dict({'cuboid': cuboids_dict}).cuboid

        # 2. Process PointCloud -> Mesh
        if pcl_array is not None and len(pcl_array) > 0:
            try:
                # Ensure tensor and on cuda (preferred by CuRobo)
                # Mesh.from_pointcloud requires numpy or tensor
                # Note: pcl_array should be a (N, 3) numpy array
                
                # Create Mesh object
                # pose is the origin of the mesh, usually point cloud is in world frame, so pose is origin
                mesh_obstacle = Mesh.from_pointcloud(
                    pcl_array, 
                    pose=[0,0,0,1,0,0,0], 
                    name="scene_pcl"
                )
                world_cfg_args['mesh'] = [mesh_obstacle]
                
            except Exception as e:
                print(f"[MotionServer] Error creating mesh from PCL: {e}")

        # 3. Update CuRobo
        if len(world_cfg_args) > 0:
            try:
                # WorldConfig can accept mesh=[Obj], cuboid={dict}
                world_config = WorldConfig(**world_cfg_args)
                self.motion_gen.update_world(world_config)
                if self.debug:
                    pcl_size = len(pcl_array) if pcl_array is not None else 0
                    print(f"[MotionServer] World updated: {len(world_cfg_args)} types | PCL: {pcl_size} pts | Time: {time.time()-t:.3f}s")
                return True
            except Exception as e:
                print(f"[MotionServer] Update World Error: {e}")
                return False
        
        return True

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")
    print("[MotionServer] ZMQ REP Server bound to tcp://*:5556")

    mp = MotionPlanner(robot_yml="franka.yml", debug=True)
    print("[MotionServer] Ready.")

    while True:
        try:
            msg = socket.recv()
            request = pickle.loads(msg)
            
            cmd = request.get('cmd')
            response = {}

            if cmd == 'plan':
                start = request['start']
                target = request['target']
                waypoints, success = mp.plan(start, target)
                response = {'success': success, 'waypoints': waypoints}

            elif cmd == 'update_world':
                # Parse parameters: cuboids (dict) and pcl (numpy array)
                cuboids = request.get('cuboids', None)
                pcl = request.get('pcl', None)
                
                ok = mp.update_world(cuboids_dict=cuboids, pcl_array=pcl)
                response = {'success': ok}

            elif cmd == 'ping':
                response = {'status': 'ok'}

            socket.send(pickle.dumps(response))

        except Exception as e:
            print(f"[MotionServer] Error: {e}")
            try:
                socket.send(pickle.dumps({'success': False, 'error': str(e)}))
            except:
                pass

if __name__ == "__main__":
    run_server()