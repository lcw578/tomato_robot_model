from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_pose_generate_client',
            executable='pick_pose_visualizer',
            name='pick_pose_visualizer',
            output='screen',
            parameters=[{
                'service_name': '/vision/tomato/get_pick_poses',
                'color_topic': '/vision/tomato/crop/color/image_raw',
                'depth_topic': '/vision/tomato/crop/depth/image_raw',
                'camera_info_topic': '/vision/tomato/crop/camera_info',
                'segment_debug_topic': '/vision/tomato/pick_pose/segment_debug_image',
                'pose_debug_topic': '/vision/tomato/pick_pose/debug_image',
                'service_timeout_sec': 5.0,
                'return_debug_images': True,
                'window_name': 'vision_pose_generate_client',
                'open3d_window_name': 'vision_pose_generate_pose',
                'panel_width': 480,
                'display_fps': 10.0,
                'enable_open3d': True,
                'target_cloud_radius_px': 64,
                'target_cloud_depth_tolerance_m': 0.060,
                'target_cloud_use_rgb': True,
                'lock_open3d_camera_view': True,
                'open3d_view_zoom': 0.7,
                'open3d_view_forward_offset_m': 0.08,
            }],
        ),
    ])
