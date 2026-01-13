from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def load_yaml(package_name, path):
    full_path = os.path.join(get_package_share_directory(package_name), path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # Suppress RealSense warnings via environment variables
    suppress_realsense_warnings = SetEnvironmentVariable(
        'LRS_LOG_LEVEL', 'error'
    )
    suppress_usb_warnings = SetEnvironmentVariable(
        'LIBUSB_LOG_LEVEL', '1'  # Error only
    )
    
    # Robot description for MoveIt
    robot_ip = LaunchConfiguration('robot_ip', default='172.16.0.2')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware', default='false')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands', default='false')
    
    xacro_path = os.path.join(get_package_share_directory('franka_description'), 'robots/fr3/fr3.urdf.xacro')
    semantic_path = os.path.join(get_package_share_directory('franka_fr3_moveit_config'), 'srdf/fr3_arm.srdf.xacro')
    
    robot_description = {'robot_description': ParameterValue(Command([
        FindExecutable(name='xacro'), ' ', xacro_path, ' hand:=true',
        ' robot_ip:=', robot_ip,
        ' use_fake_hardware:=', use_fake_hardware,
        ' fake_sensor_commands:=', fake_sensor_commands,
        ' ros2_control:=false'
    ]), value_type=str)}
    
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(
            Command([FindExecutable(name='xacro'), ' ', semantic_path, ' hand:=true']),
            value_type=str
        )
    }
    
    # Load MoveIt configurations
    kinematics = load_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')
    ompl_yaml = load_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml')
    controllers = load_yaml('franka_fr3_moveit_config', 'config/moveit_controllers.yaml')
    
    # OMPL configuration
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
    
    # Trajectory execution configuration
    execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }
    
    # Planning scene monitor configuration
    monitor_params = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    
    # RealSense Camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                FindPackageShare('realsense2_camera').find('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'log_level': 'error',
        }.items()
    )

    # TF: fr3_link0 → ref_frame
    tf_fr3_to_ref = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_fr3_to_ref',
        arguments=[
            '0.735097', '0.646207', '0.664998',
            '-0.370981', '-0.863263', '0.320188', '0.120952',
            'fr3_link0', 'ref_frame'
        ]
    )

    # TF: ref_frame → camera_link
    tf_ref_to_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_ref_to_cam',
        arguments=[
            '-0.0010067', '0.014068', '-0.002151',
            '0.49272', '-0.49219', '0.50886', '0.50601',
            'ref_frame', 'camera_link'
        ]
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            FindPackageShare('demo_collection').find('demo_collection'),
            'config', 'fr3_rviz_config.rviz'
        )],
        output='screen'
    )

    # MoveIt move_group node (directly configured, not via include)
    move_group = Node(
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
                'moveit_simple_controller_manager': controllers.get('moveit_simple_controller_manager', {}),
                'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager'
            },
            monitor_params
        ]
    )

    # FOCI Node (ZMQ bridge)
    foci_node = Node(
        package='foci_policy',
        executable='foci_node.py',
        name='foci_node',
        output='screen',
    )

    return LaunchDescription([
        suppress_realsense_warnings,
        suppress_usb_warnings,
        tf_fr3_to_ref,
        tf_ref_to_cam,
        realsense_launch,
        rviz_node,
        move_group,
        foci_node,
    ])
