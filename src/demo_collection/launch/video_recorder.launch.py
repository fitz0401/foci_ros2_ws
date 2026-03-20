from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Suppress RealSense warnings
    suppress_realsense_warnings = SetEnvironmentVariable('LRS_LOG_LEVEL', 'error')
    suppress_usb_warnings = SetEnvironmentVariable('LIBUSB_LOG_LEVEL', '1')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
    )

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

    tf_fr3_to_ref = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_fr3_to_ref',
        arguments=[
            '0.749022', '0.656649', '0.659689',
            '-0.382599', '-0.868189', '0.293314', '0.117616',
            'fr3_link0', 'ref_frame'
        ]
    )

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

    video_recorder = Node(
        package='demo_collection',
        executable='video_recorder.py',
        name='video_recorder',
        output='screen',
    )

    joystick_publisher = Node(
        package='demo_collection',
        executable='joystick_publisher.py',
        name='joystick_publisher',
        output='screen',
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

    return LaunchDescription([
        suppress_realsense_warnings,
        suppress_usb_warnings,
        joy_node,
        rviz_node,
        tf_fr3_to_ref,
        tf_ref_to_cam,
        realsense_launch,
        video_recorder,
        joystick_publisher,
    ])
