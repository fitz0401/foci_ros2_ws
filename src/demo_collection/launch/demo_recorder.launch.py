import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _tf_arguments(transform, parent, child):
    xyz, quat = transform['translation'], transform['quaternion']
    return [
        '--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
        '--qx', str(quat[0]), '--qy', str(quat[1]), '--qz', str(quat[2]), '--qw', str(quat[3]),
        '--frame-id', parent, '--child-frame-id', child,
    ]


def generate_launch_description():
    package_share = get_package_share_directory('demo_collection')
    with open(os.path.join(package_share, 'config', 'demo_collection.yaml'), encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    camera, frames, transforms = config['camera'], config['frames'], config['transforms']
    recorder_params = config['demo_recorder']['ros__parameters'].copy()
    recorder_params.update({
        'base_frame': frames['base'], 'camera_frame': frames['camera_optical'],
        'ee_frame': frames['end_effector'],
    })

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
        launch_arguments={
            'depth_module.depth_profile': str(camera['profile']),
            'rgb_camera.color_profile': str(camera['profile']),
            'enable_color': 'true', 'enable_depth': 'true',
            'pointcloud.enable': 'true', 'align_depth.enable': 'true',
            'serial_no': str(camera['serial']),
        }.items())

    return LaunchDescription([
        SetEnvironmentVariable('LRS_LOG_LEVEL', 'error'),
        SetEnvironmentVariable('LIBUSB_LOG_LEVEL', '1'),
        Node(package='joy', executable='joy_node', name='joy_node', output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='static_tf_fr3_to_ref',
             arguments=_tf_arguments(transforms['base_to_reference'], frames['base'], frames['reference'])),
        Node(package='tf2_ros', executable='static_transform_publisher', name='static_tf_ref_to_cam',
             arguments=_tf_arguments(transforms['reference_to_camera'], frames['reference'], frames['camera_link'])),
        realsense_launch,
        Node(package='rviz2', executable='rviz2', name='rviz2',
             arguments=['-d', os.path.join(package_share, 'config', 'fr3_rviz_config.rviz')], output='screen'),
        Node(package='demo_collection', executable='demo_recorder.py', name='demo_recorder',
             parameters=[recorder_params], output='screen'),
        Node(package='demo_collection', executable='keyboard_publisher.py', name='keyboard_publisher',
             output='screen', prefix='xterm -e'),
        Node(package='demo_collection', executable='joystick_publisher.py', name='joystick_publisher', output='screen'),
    ])
