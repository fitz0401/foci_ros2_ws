from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    robot_ip_param = DeclareLaunchArgument('robot_ip', description='IP of the robot')
    use_fake_hw_param = DeclareLaunchArgument('use_fake_hardware', default_value='false')
    fake_sensor_cmds_param = DeclareLaunchArgument('fake_sensor_commands', default_value='false')

    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')

    franka_description_path = get_package_share_directory('franka_description')
    fr3_xacro = os.path.join(franka_description_path, 'robots', 'fr3', 'fr3.urdf.xacro')

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        fr3_xacro, ' hand:=true',
        ' robot_ip:=', robot_ip,
        ' use_fake_hardware:=', use_fake_hardware,
        ' fake_sensor_commands:=', LaunchConfiguration('fake_sensor_commands'),
        ' ros2_control:=true'
    ])
    robot_description = {
    'robot_description': ParameterValue(robot_description_config, value_type=str)
    }

    ros2_controllers_path = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'config', 'fr3_ros_controllers.yaml'
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, ros2_controllers_path],
        remappings=[('joint_states', 'franka/joint_states')],
        output={'stdout': 'screen', 'stderr': 'screen'},
        on_exit=Shutdown(),
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'source_list': ['franka/joint_states', 'fr3_gripper/joint_states'], 'rate': 30}],
    )

    franka_robot_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['franka_robot_state_broadcaster'],
        output='screen',
        condition=UnlessCondition(use_fake_hardware),
    )

    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])
        ]),
        launch_arguments={'robot_ip': robot_ip,
                          'use_fake_hardware': use_fake_hardware}.items(),
    )

    controllers = []
    for name in ['fr3_arm_controller', 'joint_state_broadcaster']:
        controllers.append(
            ExecuteProcess(
                cmd=['ros2 run controller_manager spawner {}'.format(name)],
                shell=True,
                output='screen'
            )
        )

    return LaunchDescription([
        robot_ip_param,
        use_fake_hw_param,
        fake_sensor_cmds_param,
        robot_state_pub,
        ros2_control_node,
        joint_state_pub,
        franka_robot_state_broadcaster,
        gripper_launch,
        *controllers
    ])
