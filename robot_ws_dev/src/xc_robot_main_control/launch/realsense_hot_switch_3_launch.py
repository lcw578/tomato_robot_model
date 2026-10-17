import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

SERIAL_NO = "_230322274449"
CAMERA_NAME = "cam_hand"

def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'serial_no': SERIAL_NO,
            'camera_name': CAMERA_NAME,
            'align_depth.enable': 'true',
            'rgb_camera.profile': '848,480,30',
        }.items()
    )

    return LaunchDescription([
        realsense_launch,
    ])
