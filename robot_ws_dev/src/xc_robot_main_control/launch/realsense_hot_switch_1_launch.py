import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """函数名词不能修改  产生launch描述"""
    # RealSense相机启动配置（移除了output参数）
    # action_realsense_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         get_package_share_directory('realsense2_camera'),
    #         '/launch/rs_launch.py'
    #     ])
    # )
    action_realsense_launch = ExecuteProcess(
        cmd=[
            'bash',
            '-c',
            # 这里的 serial_no 是初始相机序列号，你可以改成你目前想默认用的那个
            'echo $$ > /tmp/realsense_camera.pid; '
            'ros2 launch realsense2_camera rs_launch.py serial_no:=f1422449 camera_name:=camera'
        ],
        # output='screen'
    )


    return LaunchDescription([
        action_realsense_launch, 
    ])