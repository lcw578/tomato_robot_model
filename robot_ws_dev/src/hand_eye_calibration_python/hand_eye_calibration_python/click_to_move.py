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
import math
from xc_interfaces.srv import MoveToPose
from xc_robot_main_control.moge.moGeDepthEstimator import MoGeDepthEstimator
import threading
from ament_index_python.packages import get_package_share_directory
import os


## 在另一个终端执行 即可返回初始点

# ros2 param set /click_to_move_node is_return true



class ClickToMoveNode(Node):
    def __init__(self):
        super().__init__('click_to_move_node')
        
        self.path = "/home/xucan/code/test/robot_ws/src/hand_eye_calibration_python/config/left/"
        
        self.calibration_file = self.path + 'calibration_results.txt'
        
        self.is_use_moge_depth_estimator = False # True表示使用MoGe深度估计器，False表示不使用
        
        self.gripper_length = 0.15
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics = None
        self.intrinsics_ready = False
        self.bridge = CvBridge()
        
        self.depth_estimator = None
        if self.is_use_moge_depth_estimator:
            share_dir = get_package_share_directory('xc_robot_main_control')
            depth_model_path = os.path.join(share_dir, 'weights', 'moge-2-vits-normal.pt')
            self.get_logger().info(f"MoGe模型路径: {depth_model_path}")
            try:
                self.depth_estimator = MoGeDepthEstimator(model_path=depth_model_path)
                self.get_logger().info("MoGe深度估计器初始化成功")
            except Exception as e:
                self.get_logger().error(f"MoGe深度估计器初始化失败: {e}")
                self.depth_estimator = None
        
        self.declare_parameter('gripper_length', 0.15)
        self.declare_parameter('is_moving', False)
        self.declare_parameter('is_return', False)
        self.declare_parameter('use_moge_depth', False)
        
        self.gripper_length = self.get_parameter('gripper_length').get_parameter_value().double_value
        self.use_moge_depth = self.get_parameter('use_moge_depth').get_parameter_value().bool_value
        
        self.calibration_matrix = None
        self.processing_enabled = False
        self.processing_image = False
        self.is_moving_to_pose = False
        
        self.last_image = None
        self.depth_image = None
        self.color_image = None
        
        self.image_queue = []
        self.image_lock = threading.Lock()
        self.display_thread_running = True
        
        self.display_thread = threading.Thread(target=self.display_images, daemon=True)
        self.display_thread.start()
        
        self.reentrant_cb_group = ReentrantCallbackGroup()
        self.mutex_cb_group = MutuallyExclusiveCallbackGroup()
        
        self.move_to_pose_client = self.create_client(MoveToPose, 'move_to_pose', callback_group=self.mutex_cb_group)
        self.return_to_box_client = self.create_client(MoveToPose, 'return_to_box', callback_group=self.mutex_cb_group)
        
        # 检测/camera/color/camera_info话题是否存在
        self.check_camera_info_topic()
        
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=1,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        
        self.click_point = None
        
        # 图片显示延时控制
        self.last_image_time = 0
        self.image_display_duration = 5.0  # 每张图片显示5秒
        
        # 点击后暂停更新图片标志
        self.pause_image_update = False
        
        self.get_logger().info('点击移动节点已启动，等待相机内参...')
        self.get_logger().info('在图像窗口中点击目标点，机械臂将移动到该位置')
        self.get_logger().info(f'图片显示延时: {self.image_display_duration}秒')
    
    def check_camera_info_topic(self):
        """检测/camera/color/camera_info话题是否存在，不存在则使用默认内参"""
        time.sleep(1)  # 等待话题列表更新
        topics = self.get_topic_names_and_types()
        camera_info_topic_exists = False
        
        for topic_name, types in topics:
            if topic_name == '/camera/color/camera_info':
                camera_info_topic_exists = True
                break
        
        if camera_info_topic_exists:
            self.get_logger().info('检测到/camera/color/camera_info话题，将订阅获取相机内参')
            self.camera_info_sub = self.create_subscription(
                CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 1,
                callback_group=self.mutex_cb_group)
        else:
            self.get_logger().warn('/camera/color/camera_info话题不存在，使用默认相机内参')
            self.set_default_intrinsics()
    
    def set_default_intrinsics(self):
        """设置默认相机内参"""
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = 1280
        self.intrinsics.height = 720
        self.intrinsics.fx = 904.136
        self.intrinsics.fy = 904.541 
        self.intrinsics.ppx = 646.646
        self.intrinsics.ppy = 352.313
        self.intrinsics.model = rs2.distortion.brown_conrady
        self.intrinsics.coeffs = [0.170956, -0.525839, -0.000931794, 0.000153935, 0.486463]
        
        # 设置相机矩阵
        self.camera_matrix = np.array([
            [904.136, 0, 646.646],
            [0, 904.541, 352.313],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([0.170956, -0.525839, -0.000931794, 0.000153935, 0.486463])
        
        self.intrinsics_ready = True
        self.get_logger().info('已设置默认相机内参')
    
    def camera_info_callback(self, msg):
        if self.intrinsics_ready:
            return
        
        try:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            
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
    
    def image_callback(self, color_msg, depth_msg):
        if not self.intrinsics_ready:
            return
        
        if self.processing_image:
            return
        
        # 如果已经点击并正在移动中，暂停更新图片
        if self.pause_image_update:
            return
        
   
        
        # 检查是否在5秒内已经处理过图片
        current_time = time.time()
        if current_time - self.last_image_time < self.image_display_duration:
            return  # 跳过，等待5秒后再处理
        
        try:
            self.processing_image = True
            
            if color_msg.width != depth_msg.width or color_msg.height != depth_msg.height:
                self.processing_image = False
                return
            
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            self.color_image = cv_image.copy()
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            display_image = cv_image.copy()
            
            status_text = f'Status: {"Moving" if self.is_moving_to_pose else "Ready"}'
            cv2.putText(display_image, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_image, 'Click to move, q to quit', (10, display_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.click_point is not None:
                cv2.circle(display_image, self.click_point, 8, (0, 0, 255), -1)
                cv2.circle(display_image, self.click_point, 12, (0, 0, 255), 2)
                
                if self.depth_image is not None:
                    depth_value = self.depth_image[self.click_point[1], self.click_point[0]]
                    if depth_value != 0:
                        x, y, z = rs2.rs2_deproject_pixel_to_point(self.intrinsics, 
                                                                  [self.click_point[0], self.click_point[1]], 
                                                                  depth_value)
                        x, y, z = x*0.001, y*0.001, z*0.001
                        
                        info_text = f'Camera: ({x:.3f}, {y:.3f}, {z:.3f})m'
                        cv2.putText(display_image, info_text, (self.click_point[0] + 15, self.click_point[1] + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            with self.image_lock:
                self.image_queue.append(display_image.copy())
            
            # 更新最后处理图片的时间戳
            self.last_image_time = time.time()
            
            self.processing_image = False
            
        except Exception as e:
            self.get_logger().error(f'处理图像失败: {e}')
            self.processing_image = False
    
    def display_images(self):
        cv2.namedWindow('Click to Move')
        cv2.setMouseCallback('Click to Move', self.mouse_callback)
        
        while self.display_thread_running:
            try:
                if self.image_queue:
                    with self.image_lock:
                        cv_image = self.image_queue.pop()
                        self.last_image = cv_image.copy()
                    
                    cv2.imshow('Click to Move', cv_image)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        self.display_thread_running = False
                        break
                else:
                    if self.last_image is not None:
                        time.sleep(0.01)
                        cv2.imshow('Click to Move', self.last_image)
                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            self.display_thread_running = False
                            break
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'显示图像失败: {e}')
                time.sleep(0.1)
    
    def reset_after_failure(self):
        """机械臂移动失败后重置状态，刷新下一张图片"""
        self.get_logger().info("机械臂移动失败，刷新下一张图片...")
        self.is_moving_to_pose = False
        self.pause_image_update = False
        self.click_point = None
        self.last_image_time = 0  # 允许立即显示新图片
    
    def draw_click_marker_and_display(self):
        """立即用当前缓存的图片绘制点击标记并显示"""
        if self.color_image is None or self.click_point is None:
            return
        
        display_image = self.color_image.copy()
        
        status_text = f'Status: Moving'
        cv2.putText(display_image, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display_image, 'Click to move, q to quit', (10, display_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制点击标记
        cv2.circle(display_image, self.click_point, 8, (0, 0, 255), -1)
        cv2.circle(display_image, self.click_point, 12, (0, 0, 255), 2)
        
        # 添加深度信息
        if self.depth_image is not None:
            depth_value = self.depth_image[self.click_point[1], self.click_point[0]]
            if depth_value != 0:
                x, y, z = rs2.rs2_deproject_pixel_to_point(self.intrinsics, 
                                                          [self.click_point[0], self.click_point[1]], 
                                                          depth_value)
                x, y, z = x*0.001, y*0.001, z*0.001
                
                info_text = f'Camera: ({x:.3f}, {y:.3f}, {z:.3f})m'
                cv2.putText(display_image, info_text, (self.click_point[0] + 15, self.click_point[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 更新缓存的图片
        self.last_image = display_image.copy()
        
        # 立即添加到显示队列
        with self.image_lock:
            self.image_queue.append(display_image.copy())
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_moving_to_pose:
                self.get_logger().info('机械臂正在移动中，请等待完成')
                return
            
            self.get_logger().info(f'点击位置: ({x}, {y})')
            self.click_point = (x, y)
            
            if self.intrinsics_ready:
                self.use_moge_depth = self.get_parameter('use_moge_depth').get_parameter_value().bool_value
                
                depth_value = 0
                depth_source = "camera"
                
                if self.use_moge_depth and self.depth_estimator is not None and self.color_image is not None:
                    depth_map = self.depth_estimator.estimate_depth(self.color_image)
                    depth_value = depth_map[y, x]
                    depth_source = "MoGe"
                    self.get_logger().info(f"使用MoGe深度估计器，深度值: {depth_value:.4f}")
                elif self.depth_image is not None:
                    depth_value = self.depth_image[y, x]
                    
                    if depth_value == 0:
                        depths = []
                        for row in range(max(0, x-10), min(self.depth_image.shape[1], x+11)):
                            for col in range(max(0, y-10), min(self.depth_image.shape[0], y+11)):
                                ele_dep = self.depth_image[col, row]
                                if ele_dep != 0:
                                    depths.append(ele_dep)
                        if len(depths) != 0:
                            depth_value = sum(depths) / len(depths)
                        else:
                            self.get_logger().error('无法获取有效的深度值')
                            return
                
                if depth_value != 0:
                    x_cam, y_cam, z_cam = rs2.rs2_deproject_pixel_to_point(
                        self.intrinsics, [x, y], depth_value)
                    x_cam, y_cam, z_cam = x_cam*0.001, y_cam*0.001, z_cam*0.001
                    
                    log_message = f'点击点在相机坐标系 (深度来源: {depth_source}): X={x_cam:.4f}, Y={y_cam:.4f}, Z={z_cam:.4f}'
                    self.get_logger().info(log_message)
                    
                    if self.use_moge_depth and self.depth_image is not None:
                        camera_depth = self.depth_image[y, x]
                        if camera_depth == 0:
                            depths = []
                            for row in range(max(0, x-10), min(self.depth_image.shape[1], x+11)):
                                for col in range(max(0, y-10), min(self.depth_image.shape[0], y+11)):
                                    ele_dep = self.depth_image[col, row]
                                    if ele_dep != 0:
                                        depths.append(ele_dep)
                            if len(depths) != 0:
                                camera_depth = sum(depths) / len(depths)
                        if camera_depth != 0:
                            self.get_logger().info(f"相机深度: {camera_depth:.4f}, MoGe深度: {depth_value:.4f}")
                    
                    # 立即用当前缓存的图片绘制点击标记并显示
                    if self.color_image is not None:
                        self.draw_click_marker_and_display()
                    
                    # 暂停图片更新
                    self.pause_image_update = True
                    self.process_click_point(x_cam, y_cam, z_cam)
    
    def process_click_point(self, x, y, z):
        try:
            self.load_calibration_matrix()
            
            if self.calibration_matrix is None:
                self.get_logger().error('手眼标定矩阵未加载，无法进行坐标转换')
                return
            
            point_camera = np.array([x, y, z, 1]).reshape(4, 1)
            point_base = self.calibration_matrix @ point_camera
            
            x_base = point_base[0, 0]
            y_base = point_base[1, 0]
            z_base = point_base[2, 0]
            
            log_message = '\n' + '=' * 60 + '\n'
            log_message += '点击点相对于机械臂底座坐标系:\n'
            log_message += f'位置信息 (X, Y, Z):\n'
            log_message += f'  X = {x_base:.6f} 米\n'
            log_message += f'  Y = {y_base:.6f} 米\n'
            log_message += f'  Z = {z_base:.6f} 米\n'
            log_message += '=' * 60
            self.get_logger().info(log_message)
            
            self.call_move_to_pose_service(x_base, y_base, z_base)
            
        except Exception as e:
            self.get_logger().error(f'处理点击点失败: {e}')
    
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
        rx, ry, rz = 90.0 * math.pi / 180.0, 0.0, 0.0
        
        if y > 0:
            rz = 180.0 * math.pi / 180.0
            y = y - self.gripper_length
        else:
            rz = 0.0 * math.pi / 180.0
            y = y + self.gripper_length
        
        request = MoveToPose.Request()
        request.x = x
        request.y = y
        request.z = z
        request.rx = rx
        request.ry = ry
        request.rz = rz
        request.mid_to_target = False
        request.mid2target_angle = 0.0
        
        if self.move_to_pose_client.wait_for_service(0.5):
            self.get_logger().info(f"调用机械臂移动服务，目标位置: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
            self.is_moving_to_pose = True
            future_move_to_pose = self.move_to_pose_client.call_async(request)
            future_move_to_pose.add_done_callback(lambda future, req=request: self.move_to_pose_callback(future, req))
        else:
            self.get_logger().error("机械臂移动服务不可用")
    
    def move_to_pose_callback(self, future, request):
        try:
            response = future.result()
            if response.result == 0:
                self.get_logger().info("机械臂前往目标点移动成功")
                
                self.get_logger().info("等待返回初始位置...")
                while not self.get_parameter('is_return').get_parameter_value().bool_value:
                    time.sleep(1)
                
                future_return_to_box = self.return_to_box_client.call_async(request)
                future_return_to_box.add_done_callback(lambda future: self.return_to_box_callback(future))
                
                self.set_parameters([rclpy.parameter.Parameter('is_return', rclpy.Parameter.Type.BOOL, False)])
            else:
                self.get_logger().error(f"机械臂移动失败，错误码: {response.result}")
                # 机械臂移动失败，刷新下一张图片
                self.reset_after_failure()
        except Exception as e:
            self.get_logger().error(f"调用机械臂移动服务失败: {e}")
            # 调用失败，刷新下一张图片
            self.reset_after_failure()
    
    def return_to_box_callback(self, future):
        try:
            response = future.result()
            if response.result == 0:
                self.get_logger().info("机械臂返回初始点成功")
                self.get_logger().info("等待下一次点击...")
            else:
                self.get_logger().error(f"机械臂移动失败，错误码: {response.result}")
        except Exception as e:
            self.get_logger().error(f"调用机械臂移动服务失败: {e}")
        finally:
            self.is_moving_to_pose = False
            # 恢复图片更新
            self.pause_image_update = False
            # 清除点击标记
            self.click_point = None
            # 重置时间戳，允许立即显示新图片
            self.last_image_time = 0
    
    def destroy_node(self):
        self.display_thread_running = False
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        cv2.destroyAllWindows()
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ClickToMoveNode()
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