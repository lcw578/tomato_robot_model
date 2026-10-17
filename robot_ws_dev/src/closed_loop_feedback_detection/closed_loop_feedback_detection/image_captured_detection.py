#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
import threading
import csv 
import time
from datetime import datetime
import pyrealsense2 as rs
from ament_index_python.packages import get_package_share_directory


from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from xc_interfaces.msg import TestResult, MidpointReached

from closed_loop_feedback_detection.min_rectangle_processor import StemDiameterProcessor
# from xc_robot_main_control.xc_robot_main_control.moge.utils3d.numpy import module_name

class IntegratedVisionControlNode(Node):
    def __init__(self):
        super().__init__('integrated_vision_control_node')
        self.bridge = CvBridge()

        # --- 1. 参数配置 ---
        self.declare_parameter('threshold_mm', 10.0)
        self.declare_parameter('pixel_per_mm', 4.865)
        self.declare_parameter('model_name', 'best20260422.pt')# 工控机
        # self.declare_parameter('model_path', '/home/xucan/code/test/robot_ws/src/closed_loop_feedback_detection/weights/gripper_stem_20260222.pt') #office wsl
        self.declare_parameter('results_dir', '/home/xc/code/test/hhp-temp') #工控机
        # self.declare_parameter('results_dir', '/home/xucan/code/test/hhp-temp')

        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('enable_roi_mask', True)
        self.declare_parameter('max_attempts', 5) # 最大尝试次数

        self.threshold_mm = self.get_parameter('threshold_mm').value
        self.px_per_mm = self.get_parameter('pixel_per_mm').value
        self.results_dir = self.get_parameter('results_dir').value
        self.save_debug = self.get_parameter('save_debug_images').value
        self.enable_roi = self.get_parameter('enable_roi_mask').value
        self.max_attempts = self.get_parameter('max_attempts').value

        self.reentrant_cb_group = ReentrantCallbackGroup()

        # 目录初始化
        self.debug_base_path = os.path.join(self.results_dir, 'debug_visuals')
        self.raw_base_path = os.path.join(self.results_dir, 'raw_images')
        os.makedirs(self.debug_base_path, exist_ok=True)
        os.makedirs(self.raw_base_path, exist_ok=True)
        
        self.csv_file = os.path.join(self.results_dir, f"secure_logic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        # --- 2. 状态变量与锁 ---
        self.midpoint_reached = False
        self.last_x1_status = 0
        self.is_busy = False 
        self.processing_lock = threading.Lock()
        self.midpoint_reached_lock = threading.Lock()

        # --- 3. 硬件与算法初始化 ---
        self.initialize_camera()
        self.initialize_yolo_model()
        self.processor_tool = StemDiameterProcessor()

        # --- 4. 订阅与发布 ---
        self.create_subscription(MidpointReached, 'midpoint_reached', self.midpoint_reached_callback, 10, callback_group=self.reentrant_cb_group)
        self.create_subscription(UInt8, 'gripper/input_x1_status', self.x1_status_callback, 10, callback_group=self.reentrant_cb_group)
        
        self.test_result_pub = self.create_publisher(TestResult, 'test_result', 10, callback_group=self.reentrant_cb_group)
        self.yolo_viz_pub = self.create_publisher(Image, 'yolo_viz_with_skeleton', 10,callback_group=self.reentrant_cb_group)

    def initialize_camera(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device("241222077007")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.get_logger().info("✅ RealSense 相机已启动")
        except Exception as e:
            self.get_logger().error(f"❌ 相机启动失败: {e}")
            self.pipeline = None

    def initialize_yolo_model(self):
        model_name = self.get_parameter('model_name').value
        share_dir = get_package_share_directory('closed_loop_feedback_detection')
        path = os.path.join(share_dir, 'weights', model_name)
        try:
            self.yolo_model = YOLO(path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.get_logger().info(f"✅ YOLO模型加载成功 ({self.device})")
            while self.pipeline is None:
                self.get_logger().warn("相机未初始化，等待中...")
                time.sleep(0.1)
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()            
            raw_color = np.asanyarray(color_frame.get_data())
            inference_img = raw_color.copy()
            # 尝试推理一次，避免第一次推理的时间过长
            results = self.yolo_model.predict(inference_img, conf=0.5, device=self.device, verbose=False, iou=0.2)
            self.get_logger().info(f"✅ YOLO模型预热成功")        

                



        except Exception as e:
            self.get_logger().error(f"❌ 模型加载失败: {e}")

    def midpoint_reached_callback(self, msg):
        with self.midpoint_reached_lock:
            self.midpoint_reached = msg.reached
            self.get_logger().warn(f"✅ 中间点状态更新: {self.midpoint_reached}")

    def x1_status_callback(self, msg):
        current_status = msg.data
        if self.last_x1_status == 0 and current_status != 0:
            with self.midpoint_reached_lock:
                midpoint_status = self.midpoint_reached
            self.get_logger().warn(f"✅ 中间点: {self.midpoint_reached}，信号点状态: {current_status}")
            if midpoint_status and not self.is_busy:
                self.publish_test_result(1) #感应到传感器就先发布一次                
                threading.Thread(target=self.async_wrapper).start()
        self.last_x1_status = current_status

    def async_wrapper(self):
        with self.processing_lock: 
            self.is_busy = True
            try:
                self.camera_process_logic()
            finally:
                self.is_busy = False

    def camera_process_logic(self):
        with self.midpoint_reached_lock:
            self.midpoint_reached = False 
        
        if self.pipeline is None:
            self.get_logger().error("相机未初始化")
            return

        success_detection = False
        self.get_logger().info(f"🚀 开始循环识别，最大尝试: {self.max_attempts}")

        for attempt in range(self.max_attempts):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue
            
            raw_color = np.asanyarray(color_frame.get_data())
            origin_img = raw_color.copy()
            h, w = raw_color.shape[:2]

            # ROI 涂黑预处理 (使用复本进行推理)
            inference_img = raw_color.copy()
            if self.enable_roi:
                inference_img[:, 0:w // 2] = 0 

            # 推理
            results = self.yolo_model.predict(inference_img, conf=0.5, device=self.device, verbose=False, iou=0.2)
            
            # 初始化变量
            current_vis_data = []
            temp_diameters_mm = []
            final_code = 1  # 默认代码
            
            now = datetime.now()
            date_folder = now.strftime("%Y%m%d")
            timestamp = now.strftime("%H%M%S_%f")[:-3]
            
            # 处理结果
            if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                # 有识别结果
                result = results[0]
                annotated_img = result.plot()
                self.diameters_mm=0.0
                
                self.get_logger().info(f"🎯 第 {attempt + 1} 次成功捕捉到目标")
                
                # 处理细节逻辑
                self.process_detection_results(result, raw_color, inference_img, timestamp)
                success_detection = True
                
                # 保存 Debug 图像
                if self.save_debug:
                    debug_daily_path = os.path.join(self.debug_base_path, date_folder)
                    os.makedirs(debug_daily_path, exist_ok=True)                
                    final_viz = self.processor_tool.visualize_results(annotated_img, current_vis_data)
                    
                    header_text = f"CODE:{final_code} | MM: {self.diameters_mm:.2f}"
                    cv2.putText(final_viz, header_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    cv2.imwrite(os.path.join(debug_daily_path, f"debug_{timestamp}_full.jpg"), final_viz)
                    cv2.imwrite(os.path.join(debug_daily_path, f"debug_{timestamp}_origin.jpg"), origin_img)                
                    self.yolo_viz_pub.publish(self.bridge.cv2_to_imgmsg(final_viz, "bgr8"))
                
                break
            else:
                # 没有识别结果
                self.get_logger().warn(f"⚠️ 第 {attempt + 1} 次尝试未识别，继续重试...")
                time.sleep(0.01) # 微小延迟等待下一帧
                
                # 保存 Debug 图像 - 使用原始图片作为识别后的图片
                if self.save_debug:
                    debug_daily_path = os.path.join(self.debug_base_path, date_folder)
                    os.makedirs(debug_daily_path, exist_ok=True)
                    
                    # 使用原始图片作为识别后的图片
                    final_viz = origin_img.copy()
                    
                    header_text = f"CODE:{final_code} | 未识别到目标"
                    cv2.putText(final_viz, header_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    cv2.imwrite(os.path.join(debug_daily_path, f"debug_{timestamp}_full.jpg"), final_viz)
                    cv2.imwrite(os.path.join(debug_daily_path, f"debug_{timestamp}_origin.jpg"), origin_img)                
                    self.yolo_viz_pub.publish(self.bridge.cv2_to_imgmsg(final_viz, "bgr8"))
            

        if not success_detection:
            self.get_logger().error("❌ 达到最大尝试次数仍无结果")
            self.publish_test_result(1) 

    def process_detection_results(self, result, raw_color, inference_img, timestamp):
        h, w = raw_color.shape[:2]
        current_vis_data = []
        temp_diameters_mm = [] 
        found_perfect_stalk = False
        found_danger = False

        for i, (mask_data, box) in enumerate(zip(result.masks.data, result.boxes)):
            cls_name = self.yolo_model.names[int(box.cls[0])]
            mask_np = (mask_data.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (w, h))
            
            # 直径计算 (骨架提取)
            skel_res = self.processor_tool.calculate_skeleton_diameter_pure(mask_resized)
            pixel_diameter = skel_res['diameter'] if skel_res['success'] else 0.0
            calc_diameter_mm = pixel_diameter / self.px_per_mm if pixel_diameter > 0 else 0.0
            self.diameters_mm=calc_diameter_mm
            temp_diameters_mm.append(f"{calc_diameter_mm:.2f}")

            # 逻辑判定
            calc_label = "stem" if calc_diameter_mm > self.threshold_mm else "stalk"
            if cls_name == "stalk" and calc_label == "stalk":
                found_perfect_stalk = True
            else:
                found_danger = True

            self.log_to_csv([datetime.now().isoformat(), cls_name, calc_label, calc_diameter_mm])
            
            if skel_res['success']:
                current_vis_data.append({
                    'success': True, 'method': 'skeleton', 'final_diameter': pixel_diameter, 
                    'points': skel_res['valid_points'], 'bbox_offset': (0, 0)
                })

        final_code = 3 if found_danger else (2 if found_perfect_stalk else 3)
        self.publish_test_result(final_code)



        

    def log_to_csv(self, row):
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'YOLO_Label', 'Calc_Label', 'Diameter_MM'])
            writer.writerow(row)

    def publish_test_result(self, val):
        msg = TestResult()
        msg.result = val
        self.get_logger().warn(f"✅ 我发布的结果是：{val}")
        for i in range(10):
            i=i+1
            self.test_result_pub.publish(msg)
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedVisionControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.pipeline: node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()