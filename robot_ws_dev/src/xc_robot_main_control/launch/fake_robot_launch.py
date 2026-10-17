"""演示用的机器人控制，修改/home/xc/code/test/ros2/robot_ws/src/tomato_robot_ui/src/widget.cpp中的以下代码
    if (motionProcess->state() == QProcess::NotRunning)
    {
        // 修改启动命令，通过bash创建新进程组
        motionProcess->start("bash", QStringList()
                                         << "-c"
                                         << "exec ros2 launch xc_robot_main_control fake_robot_launch.py");
    }"""

import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 定义相机序列号参数
    serial_no = LaunchConfiguration('serial_no', default='"f1150496"')

    # 启动Realsense相机
    action_realsense_launch = ExecuteProcess(
        cmd=[
            'bash',
            '-c',
            'ros2 launch xc_robot_main_control realsense_hot_switch_1_launch.py'
        ],
        #工作相机编号标志 1或者2 1为左相机 2为右相机
    )
    action_simulation_img_publisher = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='image_publisher_node',
        name='simulation_img_publisher',
        # output='screen'
    )
    # 主控制节点配置
    action_fake_continuous_yolo_detection_node = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='fake_continuous_yolo_detection_node',
        name='fake_continuous_yolo_detection_node',
        # output='screen'
    )



    # YOLOv8服务节点配置
    action_yolov8_service = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='yolov8_service',
        name='yolov8_service',
        # output='screen'
    )

    # 机械臂控制服务节点配置
    action_arm_motion_service = launch_ros.actions.Node(
        package='xc_aubo_arm_control',
        executable='motion_service',
        name='motion_service',
        # output='screen'
    )

    action_arm_fake_picking = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='read_and_move_node',
        name='read_and_move_node',
        # output='screen'
    )
    

    return LaunchDescription([
        action_realsense_launch, 
        # action_simulation_img_publisher,#使用本地的图片 
        action_fake_continuous_yolo_detection_node,      
        action_yolov8_service,
        action_arm_motion_service,
        action_arm_fake_picking,
    ])