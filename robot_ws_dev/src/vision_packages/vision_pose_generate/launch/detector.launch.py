from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    share_dir = get_package_share_directory('vision_pose_generate')
    config = os.path.join(share_dir, 'config', 'detector.yaml')
    detection_prefix = os.environ.get('VISION_POSE_GENERATE_PYTHON')
    detection_command_prefix = f'env PYTHONNOUSERSITE=1 {detection_prefix} -s' if detection_prefix else None
    return LaunchDescription([
        Node(
            package='vision_pose_generate',
            executable='detection_service',
            name='detection_service',
            output='screen',
            parameters=[config],
            prefix=detection_command_prefix,
        )
    ])
