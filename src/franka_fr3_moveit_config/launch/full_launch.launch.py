import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package_share = get_package_share_directory('franka_fr3_moveit_config')
    config_path = os.path.join(package_share, 'config', 'deployment.yaml')
    with open(config_path, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    arguments = {
        'robot_ip': str(config['robot_ip']),
        'use_fake_hardware': str(config['use_fake_hardware']).lower(),
        'fake_sensor_commands': str(config['fake_sensor_commands']).lower(),
    }
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, 'launch', 'robot_bringup.launch.py')),
        launch_arguments=arguments.items(),
    )
    moveit_config = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, 'launch', 'moveit_config.launch.py')),
        launch_arguments=arguments.items(),
    )
    return LaunchDescription([bringup, moveit_config])
