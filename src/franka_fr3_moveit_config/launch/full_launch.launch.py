from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('franka_fr3_moveit_config'), 'launch', 'robot_bringup.launch.py']
        )),
        launch_arguments={
            'robot_ip': '172.16.0.2',
            'use_fake_hardware': 'false',
            'fake_sensor_commands': 'false'
        }.items()
    )

    moveit_config = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('franka_fr3_moveit_config'), 'launch', 'moveit_config.launch.py']
        )),
        launch_arguments={
            'robot_ip': '172.16.0.2',
            'use_fake_hardware': 'false',
            'fake_sensor_commands': 'false'
        }.items()
    )

    return LaunchDescription([bringup, moveit_config])
