#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import numpy as np
import cv2
import pyrealsense2 as rs2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import select
import sys
import tty
import termios
import math
from scipy.spatial.transform import Rotation as R
from xc_interfaces.srv import MoveJoints, MoveToPose
from xc_robot_main_control.moge.moGeDepthEstimator import MoGeDepthEstimator
import threading
from ament_index_python.packages import get_package_share_directory
import os
"""
# 修改正在运行的节点的参数
ros2 param set /aruco_detection_node is_detected true
ros2 param set /aruco_detection_node is_return2box_key true
# 查看参数值
ros2 param get /aruco_detection_node is_detected
ros2 param get /aruco_detection_node is_return2box_key
"""

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')

        self.path = "/home/xc/code/test/ros2/robot_ws/src/hand_eye_calibration_python/config/right/"# 工控机
        # self.path = "/home/homevm/code/test/robot_ws/src/hand_eye_calibration_python/config/left/"# home VMWare
        
        self.calibration_file = self.path + 'calibration_results.txt'

        self.is_use_real_depth = True # True表示使用相机深度(相机深度或MoGe深度估计器深度)，False表示使用aruco识别出来空间坐标
        self.is_use_moge_depth_estimator = False # True表示使用MoGe深度估计器，False表示不使用
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.marker_length = 0.10

        self.gripper_length = 0.0
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics = None
        self.intrinsics_ready = False
        self.bridge = CvBridge()
        
        # 声明ROS2参数
        self.declare_parameter('aruco_marker_length', 0.10)
        self.declare_parameter('gripper_length', self.gripper_length)
        self.declare_parameter('is_detected', False)        
        self.declare_parameter('is_return2box_key', False)

        self.depth_estimator=None
        if self.is_use_moge_depth_estimator==True:
            share_dir = get_package_share_directory('xc_robot_main_control')
            depth_model_path = os.path.join(share_dir, 'weights', 'moge-2-vits-normal.pt')
            self.get_logger().info(f"MoGe模型路径: {depth_model_path}")
            try:
                self.depth_estimator = MoGeDepthEstimator(model_path=depth_model_path)
                self.get_logger().info("MoGe深度估计器初始化成功")
            except Exception as e:
                self.get_logger().error(f"MoGe深度估计器初始化失败: {e}")
                self.depth_estimator = None


        
        # 获取参数值
        self.marker_length = self.get_parameter('aruco_marker_length').get_parameter_value().double_value
        self.gripper_length = self.get_parameter('gripper_length').get_parameter_value().double_value
        self.key_pressed = self.get_parameter('is_detected').get_parameter_value().bool_value
        self.is_return2box_key = self.get_parameter('is_return2box_key').get_parameter_value().bool_value
        self.get_logger().info(f'参数设置: marker_length={self.marker_length}, gripper_length={self.gripper_length}')
        
        
        self.calibration_matrix = None
        self.processing_enabled = False
        self.processing_image = False  # 用于标记是否正在处理图像
        self.is_box_arrived = False  # 用于标记是否到达了初始位置
        self.is_moveing_to_pose = False  # 用于标记是否正在执行机械臂运动

        # 键盘触发标志
        self.trigger_detection = False   # 空格键触发：前往ArUco中心
        self.trigger_return = False      # b键触发：返回初始位置

        self.last_image=None
        
        # 显示线程相关变量
        self.image_queue = []
        self.image_lock = threading.Lock()
        self.display_thread_running = True
        
        # 启动显示线程
        self.display_thread = threading.Thread(target=self.display_images, daemon=True)
        self.display_thread.start()
        
        self.reentrant_cb_group = ReentrantCallbackGroup()
        self.mutex_cb_group = MutuallyExclusiveCallbackGroup()
        
        # 创建机械臂服务调用客户端
        self.move_to_pose_client = self.create_client(MoveToPose, 'move_to_pose', callback_group=self.mutex_cb_group)
        self.move_joints_client = self.create_client(MoveJoints, 'move_joints',callback_group=self.mutex_cb_group)
        self.return_to_box_client = self.create_client(MoveToPose, 'return_to_box',callback_group=self.mutex_cb_group)
        self.return_to_midpoint_client = self.create_client(MoveToPose, 'target_return_midpoint',callback_group=self.mutex_cb_group)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 1,
            callback_group=self.mutex_cb_group)
        
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=1,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback_1)
        

        
        # self.get_logger().info('ArUco检测节点已启动，等待相机内参...')
        # self.get_logger().info('按d键开始识别ArUco标志')
        
    def camera_info_callback(self, msg):
        self.get_logger().info('收到相机信息，准备保存内参...')
        
        if self.intrinsics_ready:
            return
        
        try:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('收到相机信息，已保存内参')
            
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = msg.width
            self.intrinsics.height = msg.height
            self.intrinsics.ppx = msg.k[2]
            self.intrinsics.ppy = msg.k[5]
            self.intrinsics.fx = msg.k[0]
            self.intrinsics.fy = msg.k[4]
            
            if msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif msg.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in msg.d]
            
            self.intrinsics_ready = True
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None
            self.get_logger().info('已获取相机内参')

            
        except Exception as e:
            self.get_logger().error(f'处理相机信息失败: {e}')
    
    def image_callback_1(self, color_msg, depth_msg):
        # 检查是否正在处理图像，如果是则直接返回，避免卡顿
        # self.get_logger().info("1111111111111111")
        if not self.intrinsics_ready:
            return
        if self.processing_image:
            return
            
        try:
            self.processing_image = True  # 设置处理标志为True            
            if color_msg.width != depth_msg.width or color_msg.height != depth_msg.height:
                # self.get_logger().error('图像尺寸不相同，丢弃本帧数据')
                self.processing_image=False
                return
            
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
            
            display_image = cv_image.copy()
            self.key_pressed = self.trigger_detection
            status_text = f'Status: {"Processing" if self.key_pressed else "Waiting"}'
            cv2.putText(display_image, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_image, 'Press SPACE to detect, b to return, q to quit', (10, display_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # self.get_logger().info(f'ids: {ids}')
            
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
                
                for i, corner in enumerate(corners):
                    point = np.average(corner[0], axis=0)
                    center_x, center_y = int(point[0]), int(point[1])
                    
                    cv2.circle(display_image, (center_x, center_y), 8, (0, 255, 0), -1)
                    cv2.circle(display_image, (center_x, center_y), 12, (0, 255, 0), 2)
                    
                    marker_id = ids[i][0]
                    cv2.putText(display_image, f'ID: {marker_id}', (center_x + 15, center_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    depth_value = depth_image[center_y, center_x]
                    
                    if depth_value == 0:
                        depths = []
                        for row in range(max(0, center_x-10), min(depth_image.shape[1], center_x+11)):
                            for col in range(max(0, center_y-10), min(depth_image.shape[0], center_y+11)):
                                ele_dep = depth_image[col, row]
                                if ele_dep != 0:
                                    depths.append(ele_dep)
                        if len(depths) != 0:
                            depth_value = sum(depths)/len(depths)
                    
                    if depth_value != 0 or self.is_use_real_depth==False:
                        x,y,z=0,0,0
                        if depth_value != 0 and self.is_use_real_depth==True:
                            x, y, z = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [point[0], point[1]], depth_value)
                            x=x*0.001
                            y=y*0.001
                            z=z*0.001
                        elif depth_value != 0:
                            x, y, z = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [point[0], point[1]], depth_value)

                        info_text = f'Camera: ({x:.3f}, {y:.3f}, {z:.3f})m'
                        cv2.putText(display_image, info_text, (center_x + 15, center_y + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # 只有在按键按下时才执行process_aruco_detection函数
                        if self.key_pressed:
                            # 组合日志消息，确保对齐
                            log_message = f'ArUco中心点在相机坐标系: X={x:.4f}, Y={y:.4f}, Z={z:.4f}'
                            self.get_logger().info(log_message)
                            if self.depth_estimator is not None:
                                # 从图像中提取中心点的深度值
                                depth_map = self.depth_estimator.estimate_depth(cv_image)
                                depth_value_estimator = depth_map[center_y, center_x]
                                x1, y1, z1 = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [point[0], point[1]], depth_value_estimator)

                                log_message += f', Depth_estimator: {depth_value_estimator:.4f}'
                                self.get_logger().info(f"depth:camera:{depth_value:.4f},estimator:{depth_value_estimator:.4f}")
                            self.process_aruco_detection(x, y, z, corner[0], display_image)
                            self.trigger_detection = False  # 重置触发标志

                        else:
                            self.processing_image=False 
                        # 将图像添加到队列中，由显示线程处理
                        with self.image_lock:
                            self.image_queue.append(display_image.copy())
                    else:
                        cv2.putText(display_image, 'No depth', (center_x + 15, center_y + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if self.processing_enabled:
                            self.get_logger().error('无法获取有效的深度值')
                        # 将图像添加到队列中，由显示线程处理
                        with self.image_lock:
                            self.image_queue.append(display_image.copy())
                        self.processing_image = False  # 将处理标志设置为False
                        
                        
            else:
                if self.processing_enabled:
                    self.get_logger().error('未检测到ArUco标记')
                # 将图像添加到队列中，由显示线程处理
                with self.image_lock:
                    self.image_queue.append(display_image.copy())        
                self.processing_image = False  # 将处理标志设置为False
            
        except Exception as e:
            self.get_logger().error(f'处理图像失败: {e}')
            self.processing_image = False  # 如果失败，将处理标志设置为False
        
        


    def image_callback(self, color_msg, depth_msg):
        if not self.intrinsics_ready:
            return        
        # 检查是否正在处理图像，如果是则直接返回，避免卡顿
        if self.processing_image:
            return        
        try:
            self.processing_image = True  # 设置处理标志为True
            
            if color_msg.width != depth_msg.width or color_msg.height != depth_msg.height:
                self.get_logger().error('图像尺寸不相同，丢弃本帧数据')
                self.processing_image=False
                return
            
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough") 
            self.get_logger().info(f'图像尺寸的尺寸是:{cv_image.shape}')

            
            # 将图像添加到队列中，由显示线程处理
            with self.image_lock:
                self.image_queue.append(cv_image.copy())
            
            self.processing_image = False
            
        except Exception as e:
            self.get_logger().error(f'处理图像失败: {e}')
            self.processing_image = False
    
    def handle_key(self, key):
        """处理键盘按键"""
        key = key & 0xFF
        if key == ord(' '):  # 空格键：前往ArUco中心
            if not self.trigger_detection:
                self.trigger_detection = True
                self.get_logger().info("空格键按下，准备前往ArUco中心")
        elif key == ord('b') or key == ord('B'):  # b键：返回初始位置
            self.trigger_return = True
            self.get_logger().info("b键按下，准备返回初始位置")
        elif key == ord('q') or key == ord('Q'):  # q键：退出
            self.get_logger().info("q键按下，准备退出")
            self.display_thread_running = False

    def display_images(self):
        """显示线程函数，负责处理图像显示和键盘输入"""
        while self.display_thread_running:
            try:
                # 检查是否有图像需要显示
                if self.image_queue:
                    with self.image_lock:
                        # 获取最新的图像
                        cv_image = self.image_queue.pop()
                        self.last_image = cv_image.copy()

                    # 显示图像并捕获按键
                    cv2.imshow('ArUco Detection', cv_image)
                    key = cv2.waitKey(1)
                    self.handle_key(key)
                else:
                    # 如果没有图像，短暂休眠
                    if self.last_image is not None:
                        time.sleep(0.01)
                        cv2.imshow('ArUco Detection', self.last_image)
                        key = cv2.waitKey(1)
                        self.handle_key(key)
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'显示图像失败: {e}')
                time.sleep(0.1)
    
    def destroy_node(self):
        """节点销毁时的清理工作"""
        # 停止显示线程
        self.display_thread_running = False
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        super().destroy_node()
    
    def process_aruco_detection(self, x, y, z, corners, display_image=None):
        try:
            self.load_calibration_matrix()
            
            if self.calibration_matrix is None:
                self.get_logger().error('手眼标定矩阵未加载，无法进行坐标转换')
                return
            
            marker_corners_3d = np.array([
                [-self.marker_length/2,  self.marker_length/2, 0],
                [ self.marker_length/2,  self.marker_length/2, 0],
                [ self.marker_length/2, -self.marker_length/2, 0],
                [-self.marker_length/2, -self.marker_length/2, 0]
            ], dtype=np.float32)
            
            image_points = corners.reshape(4, 2).astype(np.float32)
            
            success, rvecs, tvecs = cv2.solvePnP(
                objectPoints=marker_corners_3d,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                if rvecs.size == 3:
                    rvecs = rvecs.reshape(3, 1)
                if tvecs.size == 3:
                    tvecs = tvecs.reshape(3, 1)
                
                R_target2cam , _ = cv2.Rodrigues(rvecs)
                T_target2cam  = tvecs
                
                # R_target2cam = R_cam2target.T
                # T_target2cam = -R_cam2target.T @ T_cam2target + T_cam2target
                
                M_target2cam = np.eye(4)
                M_target2cam[:3, :3] = R_target2cam
                M_target2cam[:3, 3] = T_target2cam.flatten()
                print(f"point_depth_camera:{x},{y},{z}")
                print(f"point_solvePnP_camera:{T_target2cam[0,0]:.6f},{T_target2cam[1,0]:.6f},{T_target2cam[2,0]:.6f}")
                
                point_camera_real = np.array([x, y, z, 1]).reshape(4, 1)#使用相机深度信息
            
                point_camera_solvePnP = np.array([T_target2cam[0, 0], T_target2cam[1, 0], T_target2cam[2, 0], 1]).reshape(4, 1)#不使用相机深度信息
                # point_depth_camera = np.array([x, y, z, 1]).reshape(4, 1)#使用相机深度信息
                # point_depth_base = self.calibration_matrix @ point_depth_camera
                point_base_real = self.calibration_matrix @ point_camera_real
                point_base_solvePnP = self.calibration_matrix @ point_camera_solvePnP

                
                
                x_base_real = point_base_real[0, 0]
                y_base_real = point_base_real[1, 0]
                z_base_real = point_base_real[2, 0]
                
                x_base_solvePnP = point_base_solvePnP[0, 0]
                y_base_solvePnP = point_base_solvePnP[1, 0]
                z_base_solvePnP = point_base_solvePnP[2, 0]
                
                R_cam2base = self.calibration_matrix[:3, :3]
                T_cam2base = self.calibration_matrix[:3, 3].reshape(3, 1)
                
                R_target2base = R_cam2base @ R_target2cam
                T_target2base = R_cam2base @ T_target2cam + T_cam2base
                
                rot = R.from_matrix(R_target2base)
                quaternion = rot.as_quat()
                
                # 使用单个日志消息输出所有信息，确保对齐
                log_message = '\n' + '=' * 60 + '\n'
                log_message += 'ArUco标志相对于机械臂底座坐标系 real_depth:\n'
                log_message += f'位置信息 (X, Y, Z):\n'
                log_message += f'  X = {x_base_real:.6f} 米\n'
                log_message += f'  Y = {y_base_real:.6f} 米\n'
                log_message += f'  Z = {z_base_real:.6f} 米\n'
                log_message += 'ArUco标志相对于机械臂底座坐标系 solvePnP:\n'
                log_message += f'位置信息 (X, Y, Z):\n'
                log_message += f'  X = {x_base_solvePnP:.6f} 米\n'
                log_message += f'  Y = {y_base_solvePnP:.6f} 米\n'
                log_message += f'  Z = {z_base_solvePnP:.6f} 米\n'
                log_message += f'姿态信息 (四元数 w, x, y, z):\n'
                log_message += f'  w = {quaternion[3]:.6f}\n'
                log_message += f'  x = {quaternion[0]:.6f}\n'
                log_message += f'  y = {quaternion[1]:.6f}\n'
                log_message += f'  z = {quaternion[2]:.6f}\n'
                log_message += '=' * 60
                self.get_logger().info(log_message)
                
                # 调用机械臂移动服务
                if self.is_use_real_depth==True:
                    self.call_move_to_pose_service(x_base_real, y_base_real, z_base_real)
                else:
                    self.call_move_to_pose_service(x_base_solvePnP, y_base_solvePnP, z_base_solvePnP)

                
                if display_image is not None:
                    point = np.average(corners, axis=0)
                    center_x, center_y = int(point[0]), int(point[1])
                    
                    base_info_text = f'Base: ({x_base_real:.3f}, {y_base_real:.3f}, {z_base_real:.3f})m'
                    cv2.putText(display_image, base_info_text, (center_x + 15, center_y + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    pnp_info_text = f'PnP: ({T_target2base[0, 0]:.3f}, {T_target2base[1, 0]:.3f}, {T_target2base[2, 0]:.3f})m'
                    cv2.putText(display_image, pnp_info_text, (center_x + 15, center_y + 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    quat_info_text = f'Quat: ({quaternion[3]:.2f}, {quaternion[0]:.2f}, {quaternion[1]:.2f}, {quaternion[2]:.2f})'
                    cv2.putText(display_image, quat_info_text, (center_x + 15, center_y + 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
            else:
                self.get_logger().error('PnP求解失败')
                
        except Exception as e:
            self.get_logger().error(f'处理ArUco检测结果失败: {e}')
    
    def load_calibration_matrix(self):
        try:
            self.calibration_matrix = np.loadtxt(self.calibration_file, delimiter=',')
            if self.calibration_matrix.shape != (4, 4):
                self.get_logger().error(f'标定矩阵格式错误，期望4x4，实际为{self.calibration_matrix.shape}')
                self.calibration_matrix = None
            else:
                self.get_logger().info('成功加载手眼标定矩阵')
                
        except FileNotFoundError:
            self.get_logger().error(f'找不到标定文件: {self.calibration_file}')
            self.calibration_matrix = None
        except Exception as e:
            self.get_logger().error(f'加载标定矩阵失败: {e}')
            self.calibration_matrix = None
    

    


    def call_move_to_pose_service(self, x, y, z):
        """调用机械臂移动服务"""

        # 设置姿态：rx=90度，ry=0度，rz=0度
        rx, ry, rz = 90.0*math.pi / 180.0, 0.0, 0.0


        if y>0:
            rz=180.0*math.pi /180.0 # 正值是X轴逆时针转动，负值是顺时针转动  机械臂rz初始角度加上该角度即可
            y=y-self.gripper_length
        else:
            rz=0.0*math.pi /180.0 # 正值是X轴逆时针转动，负值是顺时针转动  机械臂rz初始角度加上该角度即可
            y=y+self.gripper_length
        # 构建请求
        request = MoveToPose.Request()
        request.x = x
        request.y = y
        request.z = z
        request.rx=rx
        request.ry=ry
        request.rz=rz
        # 设置其他参数
        request.mid_to_target = False
        request.mid2target_angle = 0.0

        # 检查服务是否可用
        if self.move_to_pose_client.wait_for_service(0.5):
            self.get_logger().info(f"调用机械臂移动服务，目标位置: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
            self.is_moveing_to_pose = True
            self.trigger_return = False  # 清除残留的返回触发标志
            # 异步调用服务
            future_move_to_pose = self.move_to_pose_client.call_async(request)
            future_move_to_pose.add_done_callback(lambda future: self.move_to_pose_callback(future,request))
        #             future = self.move_to_pose_client.call_async(request)
        # future.add_done_callback(lambda future: self.move_to_pose_callback(future, task))
            # 等待服务执行完成
            # while self.is_moveing_to_pose:
            #     time.sleep(0.5)
        else:
            self.get_logger().error("机械臂移动服务不可用")
    
    def move_to_pose_callback(self, future,request):
        """处理机械臂移动服务的回调"""
        self.get_logger().info(f"前往目标点移动:{future.result()}")

        try:
            response = future.result()
            if response.result == 0:
                self.get_logger().info("机械臂前往目标点移动成功，按b键返回初始位置")
                # 等待b键触发返回
                while self.trigger_return == False:
                    time.sleep(0.1)

                future_return_to_box = self.return_to_box_client.call_async(request)
                future_return_to_box.add_done_callback(lambda future: self.return_to_box_callback(future))
                self.trigger_return = False  # 重置返回触发标志
            else:
                self.get_logger().error(f"机械臂移动失败，错误码: {response.result}")
                self.processing_image=False
        except Exception as e:
            self.get_logger().error(f"调用机械臂移动服务失败: {e}")
            self.processing_image=False


    

    def return_to_box_callback(self, future):
        """处理机械臂移动服务的回调"""
        try:
            response = future.result()
            if response.result == 0:
                self.get_logger().info("机械臂返回初始点成功")
            else:
                self.get_logger().error(f"机械臂移动失败，错误码: {response.result}")
        except Exception as e:
            self.get_logger().error(f"调用机械臂移动服务失败: {e}")
        finally:
            self.processing_image = False

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
