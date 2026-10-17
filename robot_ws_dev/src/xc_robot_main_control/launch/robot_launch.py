import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """函数名词不能修改  产生launch描述"""
    action_realsense_launch = ExecuteProcess(
        cmd=[
            'bash',
            '-c',
            'ros2 launch xc_robot_main_control realsense_hot_switch_1_launch.py'
        ],
        #工作相机编号标志 1或者2 1为左相机 2为右相机
    )

    # 仿真图像发布节点配置 当没有相机时使用
    action_simulation_img_publisher = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='image_publisher_node',
        name='simulation_img_publisher',
        output='screen'
    )
    # 主控制节点配置
    action_main_control = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='xc_robot_main_control_node',
        name='main_control',
        output='screen'
    )

    # YOLOv8服务节点配置
    action_yolov8_service = launch_ros.actions.Node(
        package='xc_robot_main_control',
        executable='yolov8_service',
        name='yolov8_service',
        output='screen'
    )

    realsense_hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('xc_robot_main_control'),
            '/launch/realsense_hot_switch_3_launch.py'
        ])
    )

    close_vision_pose_generate = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('vision_pose_generate'),
            '/launch/pick_pose_service.launch.py'
        ])
    )
    

    # 机械臂控制服务节点配置
    action_arm_motion_service = launch_ros.actions.Node(
        package='xc_aubo_arm_control',
        executable='motion_service',
        name='motion_service',
        output='screen'
    )

    # 底盘控制节点
    action_car_control_node = launch_ros.actions.Node(
        package='xc_car_control',
        executable='agv_control_node',
        name='agv_control_node',
        output='screen'
    )
    # 底盘导航节点
    action_car_nav_node = launch_ros.actions.Node(
        package='xc_car_control',
        executable='agv_navigation_node',
        name='agv_navigation_node',
        output='screen'
    )

    action_car_alaws_go_node = launch_ros.actions.Node(
        package='xc_car_control',
        executable='agv_always_go_straght_node',
        name='agv_always_go_straght_node',
        output='screen'
    )

    # 末端执行器控制节点（开和闭）
    action_end_effector_control_node = launch_ros.actions.Node(
        package='gripper_driver_v2',
        executable='gripper_control_node',
        name='gripper_control_node',
        output='screen'
    )

    # 末端反馈节点（注意修改末端相机的编号）
    action_end_effector_feedback_node = launch_ros.actions.Node(
        package='closed_loop_feedback_detection',
        executable='image_captured_detection_sub',
        name='image_captured_detection_sub',
        output='screen'
    )

    return LaunchDescription([
        action_realsense_launch, # 相机启动节点
        # realsense_hand, # 手上的相机启动节点
        # action_simulation_img_publisher, # 仿真图像发布节点
        # action_main_control, #主控制节点
        action_yolov8_service,     # YOLOv8服务节点
        action_arm_motion_service, # 机械臂控制服务节点
        action_car_control_node, # 底盘控制节点
        # close_vision_pose_generate,#近端识别部分
        # action_car_nav_node, # 底盘导航节点
        action_car_alaws_go_node, # 底盘一直直行节点
        # action_end_effector_control_node, # 末端执行器控制节点（开和闭）
        # action_end_effector_feedback_node, # 末端反馈节点（注意修改末端相机的编号）
    ])