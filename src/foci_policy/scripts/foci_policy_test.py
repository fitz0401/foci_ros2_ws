#!/usr/bin/env python3
"""
Test script for FOCI policy communication
Tests ZMQ communication with foci_node without running actual policy
"""

import zmq
import numpy as np
import base64
import cv2
import time
import json


class TestFOCIPolicy:
    """Simple test client that mimics foci_policy_realworld.py"""
    
    def __init__(self, port=5555):
        # ZMQ client setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 sec timeout
        print(f"Connected to FOCI node on port {port}")
    
    def test_get_observation(self):
        """Test getting observation from robot"""
        print("\n=== Testing get_observation ===")
        request = {'type': 'get_observation'}
        
        self.socket.send_json(request)
        response = self.socket.recv_json()
        
        if response['status'] == 'success':
            data = response['data']
            print("✓ Observation received:")
            print(f"  - RGB image: {len(data['rgb'])} bytes (base64)")
            print(f"  - Depth image: {len(data['depth'])} bytes (base64)")
            print(f"  - Camera intrinsics K: {np.array(data['K']).reshape(3,3)}")
            print(f"  - Gripper pose: position={data['gripper_pose']['position']}")
            print(f"  - Gripper open: {data['gripper_open']:.2f}")
            
            # Decode and save test images
            try:
                rgb_data = base64.b64decode(data['rgb'])
                rgb_array = np.frombuffer(rgb_data, dtype=np.uint8)
                rgb_img = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
                cv2.imwrite('/tmp/test_rgb.png', rgb_img)
                print("  - Saved RGB to /tmp/test_rgb.png")
                
                depth_data = base64.b64decode(data['depth'])
                depth_array = np.frombuffer(depth_data, dtype=np.uint8)
                depth_img = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                cv2.imwrite('/tmp/test_depth.png', depth_img)
                print("  - Saved depth to /tmp/test_depth.png")
            except Exception as e:
                print(f"  ! Image decode error: {e}")
            
            return True
        else:
            print(f"✗ Failed: {response.get('message', 'Unknown error')}")
            return False
    
    def test_gripper_control(self):
        """Test gripper open/close commands"""
        print("\n=== Testing gripper control ===")
        
        # Test close
        print("Closing gripper...")
        request = {'type': 'close_gripper', 'blocking': True}
        self.socket.send_json(request)
        response = self.socket.recv_json()
        print(f"  Close response: {response}")
        
        time.sleep(2)
        
        # Test open
        print("Opening gripper...")
        request = {'type': 'open_gripper', 'blocking': True}
        self.socket.send_json(request)
        response = self.socket.recv_json()
        print(f"  Open response: {response}")
        
        return response['status'] == 'success'
    
    def test_execute_trajectory(self):
        """Test sending a simple trajectory"""
        print("\n=== Testing trajectory execution ===")
        
        # Create a simple test trajectory (just 2 waypoints)
        # These are dummy poses - adjust based on your robot's workspace
        pose1 = np.eye(4)
        pose1[0, 3] = 0.3  # x
        pose1[1, 3] = 0.0  # y
        pose1[2, 3] = 0.5  # z
        
        pose2 = np.eye(4)
        pose2[0, 3] = 0.3
        pose2[1, 3] = 0.1
        pose2[2, 3] = 0.5
        
        trajectory = {
            'poses': [pose1.tolist(), pose2.tolist()],
            'gripper_open': [1.0, 1.0]  # Keep open
        }
        
        request = {
            'type': 'execute_trajectory',
            'trajectory': trajectory,
            'mode': 'grasp'
        }
        
        print("Sending trajectory with 2 waypoints...")
        self.socket.send_json(request)
        
        print("Waiting for execution (this may take a while)...")
        response = self.socket.recv_json()
        
        if response['status'] == 'success':
            print(f"✓ Trajectory executed: {response['message']}")
            return True
        else:
            print(f"✗ Execution failed: {response.get('message', 'Unknown error')}")
            return False
    
    def close(self):
        """Close connection"""
        self.socket.close()
        self.context.term()


def main():
    print("="*60)
    print("FOCI Policy Test Client")
    print("="*60)
    
    client = TestFOCIPolicy()
    
    try:
        # Test 1: Get observation
        success1 = client.test_get_observation()
        
        time.sleep(1)
        
        # Test 2: Gripper control
        success2 = client.test_gripper_control()
        
        time.sleep(1)
        
        # Test 3: Trajectory execution (commented out by default - enable when ready)
        # print("\n!!! Trajectory test disabled by default !!!")
        # print("!!! Uncomment in code to test trajectory execution !!!")
        success3 = client.test_execute_trajectory()
        # success3 = True
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary:")
        print(f"  Observation: {'✓ PASS' if success1 else '✗ FAIL'}")
        print(f"  Gripper:     {'✓ PASS' if success2 else '✗ FAIL'}")
        print(f"  Trajectory:  SKIPPED (enable manually)")
        print("="*60)
        
    except zmq.error.Again:
        print("\n✗ Timeout: No response from foci_node")
        print("  Make sure foci_node is running!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        client.close()


if __name__ == '__main__':
    main()
