import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory('vision_pose_generate')
    preprocess_config = os.path.join(share_dir, 'config', 'preprocess.yaml')
    detector_config = os.path.join(share_dir, 'config', 'detector.yaml')
    pose_config = os.path.join(share_dir, 'config', 'pose_estimator.yaml')
    detection_prefix = os.environ.get('VISION_POSE_GENERATE_PYTHON')
    detection_command_prefix = f'env PYTHONNOUSERSITE=1 {detection_prefix} -s' if detection_prefix else None

    return LaunchDescription([
        Node(
            package='vision_pose_generate',
            executable='image_preprocess_node',
            name='image_preprocess_node',
            output='screen',
            parameters=[preprocess_config],
        ),
        Node(
            package='vision_pose_generate',
            executable='detection_service',
            name='detection_service',
            output='screen',
            parameters=[detector_config],
            prefix=detection_command_prefix,
        ),
        Node(
            package='vision_pose_generate',
            executable='pose_estimator_service',
            name='pose_estimator_service',
            output='screen',
            parameters=[pose_config],
        ),
        Node(
            package='vision_pose_generate',
            executable='pick_pose_service_node',
            name='pick_pose_service_node',
            output='screen',
        ),
    ])
