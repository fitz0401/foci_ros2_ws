import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _tf_arguments(transform, parent, child):
    xyz = transform['translation']
    quat = transform['quaternion']
    return [
        '--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
        '--qx', str(quat[0]), '--qy', str(quat[1]),
        '--qz', str(quat[2]), '--qw', str(quat[3]),
        '--frame-id', parent, '--child-frame-id', child,
    ]


def generate_launch_description():
    package_share = get_package_share_directory('foci_policy')
    config_path = os.path.join(package_share, 'config', 'foci.yaml')
    with open(config_path, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    camera = config['camera']
    frames = config['frames']
    transforms = config['transforms']
    node_params = config['foci_node']['ros__parameters'].copy()
    node_params.update({
        'base_frame': frames['base'],
        'camera_frame': frames['camera_optical'],
        'ee_frame': frames['end_effector'],
    })

    realsense_share = get_package_share_directory('realsense2_camera')
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, 'launch', 'rs_launch.py')),
        launch_arguments={
            'depth_module.depth_profile': str(camera['profile']),
            'rgb_camera.color_profile': str(camera['profile']),
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'log_level': 'error',
            'serial_no': str(camera['serial']),
        }.items(),
    )

    tf_base_to_ref = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_tf_fr3_to_ref',
        arguments=_tf_arguments(
            transforms['base_to_reference'], frames['base'], frames['reference']),
    )
    tf_ref_to_camera = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_tf_ref_to_cam',
        arguments=_tf_arguments(
            transforms['reference_to_camera'], frames['reference'], frames['camera_link']),
    )

    rviz_config = os.path.join(
        get_package_share_directory('demo_collection'), 'config', 'fr3_rviz_config.rviz')
    rviz_nodes = []
    if os.path.exists(rviz_config):
        rviz_nodes.append(Node(
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', rviz_config], output='screen'))

    return LaunchDescription([
        SetEnvironmentVariable('LRS_LOG_LEVEL', 'error'),
        SetEnvironmentVariable('LIBUSB_LOG_LEVEL', '1'),
        tf_base_to_ref,
        tf_ref_to_camera,
        realsense_launch,
        Node(
            package='foci_policy', executable='foci_node.py', name='foci_node',
            parameters=[node_params], output='screen'),
        *rviz_nodes,
    ])
