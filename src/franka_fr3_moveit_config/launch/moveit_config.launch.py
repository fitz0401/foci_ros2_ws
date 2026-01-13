from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def load_yaml(package_name, path):
    full_path = os.path.join(get_package_share_directory(package_name), path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')

    robot_ip_arg = DeclareLaunchArgument('robot_ip', description='IP of the robot')
    use_fake_hw_arg = DeclareLaunchArgument('use_fake_hardware', default_value='false')
    fake_sensor_cmds_arg = DeclareLaunchArgument('fake_sensor_commands', default_value='false')
    db_arg = DeclareLaunchArgument('db', default_value='False')

    xacro_path = os.path.join(get_package_share_directory('franka_description'), 'robots/fr3/fr3.urdf.xacro')
    semantic_path = os.path.join(get_package_share_directory('franka_fr3_moveit_config'), 'srdf/fr3_arm.srdf.xacro')

    robot_description = {'robot_description': ParameterValue(Command([
        FindExecutable(name='xacro'), ' ', xacro_path, ' hand:=true',
        ' robot_ip:=', robot_ip,
        ' use_fake_hardware:=', use_fake_hardware,
        ' fake_sensor_commands:=', fake_sensor_commands,
        ' ros2_control:=true'
    ]), value_type=str)}

    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(
            Command([FindExecutable(name='xacro'), ' ', semantic_path]),
            value_type=str
        )
    }

    kinematics = load_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')
    ompl_yaml = load_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml')
    controllers = load_yaml('franka_fr3_moveit_config', 'config/fr3_controllers.yaml')

    ompl_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_config['move_group'].update(ompl_yaml)

    execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    monitor_params = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics,
            ompl_config,
            execution,
            {
                'moveit_simple_controller_manager': controllers,
                'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager'
            },
            monitor_params
        ]
    )

    rviz_config = os.path.join(get_package_share_directory('franka_fr3_moveit_config'), 'rviz/moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='log',
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_config,
            kinematics
        ]
    )

    return LaunchDescription([
        robot_ip_arg,
        use_fake_hw_arg,
        fake_sensor_cmds_arg,
        db_arg,
        move_group_node,
        rviz_node
    ])
