from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    share_dir = get_package_share_directory('vision_pose_generate')
    config = os.path.join(share_dir, 'config', 'pose_estimator.yaml')
    return LaunchDescription([
        Node(
            package='vision_pose_generate',
            executable='pose_estimator_service',
            name='pose_estimator_service',
            output='screen',
            parameters=[config],
        )
    ])
