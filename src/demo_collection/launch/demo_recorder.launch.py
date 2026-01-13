from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Joy node for joystick input
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
    )

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

    # Spawn joint impedance controller
    spawn_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_impedance_example_controller'],
        output='screen',
    )

    # Demo Recorder
    demo_recorder = Node(
        package='demo_collection',
        executable='demo_recorder.py',
        name='demo_recorder',
        output='screen',
    )

    # Keyboard Publisher (in separate terminal)
    keyboard_publisher = Node(
        package='demo_collection',
        executable='keyboard_publisher.py',
        name='keyboard_publisher',
        output='screen',
        prefix='xterm -e',  # Run in separate terminal window for keyboard input
    )

    # Joystick Publisher (optional - comment out if no joystick)
    joystick_publisher = Node(
        package='demo_collection',
        executable='joystick_publisher.py',
        name='joystick_publisher',
        output='screen',
    )

    return LaunchDescription([
        joy_node,
        tf_fr3_to_ref,
        tf_ref_to_cam,
        realsense_launch,
        rviz_node,
        spawn_controller,
        demo_recorder,
        keyboard_publisher,
        joystick_publisher,  # Comment this line if no joystick available
    ])
