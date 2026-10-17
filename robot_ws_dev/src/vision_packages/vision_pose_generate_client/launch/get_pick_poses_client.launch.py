from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_pose_generate_client',
            executable='get_pick_poses_client',
            name='get_pick_poses_client',
            output='screen',
            parameters=[{
                'service_name': '/vision/tomato/get_pick_poses',
                'timeout_sec': 15.0,
                'return_debug_images': True,
                'segment_debug_topic': '/vision/tomato/pick_pose/segment_debug_image',
                'pose_debug_topic': '/vision/tomato/pick_pose/debug_image',
                'debug_wait_sec': 2.0,
                'save_debug_images': True,
                'debug_output_dir': '/tmp/vision_pose_generate_client',
                'fail_on_unsuccessful_response': False,
            }],
        ),
    ])
