#!/usr/bin/env python3
"""
功能与 image_captured_detection.py 一致，但不再使用 pyrealsense2 SDK 直接拉流，
而是通过订阅 ROS2 RealSense 的话题获取彩色图和深度图，再借助
message_filters.ApproximateTimeSynchronizer 在 0.1s 时差窗口内对齐。
不满足配对条件的图像直接丢弃。
"""
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
import message_filters
from ament_index_python.packages import get_package_share_directory

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from xc_interfaces.msg import TestResult, MidpointReached

from closed_loop_feedback_detection.min_rectangle_processor import StemDiameterProcessor


class IntegratedVisionControlNodeSub(Node):
    def __init__(self):
        super().__init__('integrated_vision_control_node_sub')
        self.bridge = CvBridge()

        # --- 1. 参数配置 ---
        self.declare_parameter('threshold_mm', 10.0)
        self.declare_parameter('pixel_per_mm', 4.865)
        self.declare_parameter('model_name', 'gripper_stem_xiongAn_20260726.pt')  # 工控机
        self.declare_parameter('results_dir', '/home/xc/code/test/hhp-temp')  # 工控机

        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('enable_roi_mask', True)
        self.declare_parameter('max_attempts', 5)  # 最大尝试次数
        self.declare_parameter('color_topic', '/cam_hand/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/cam_hand/aligned_depth_to_color/image_raw')
        self.declare_parameter('time_slop', 0.1)

        self.threshold_mm = self.get_parameter('threshold_mm').value
        self.px_per_mm = self.get_parameter('pixel_per_mm').value
        self.results_dir = self.get_parameter('results_dir').value
        self.save_debug = self.get_parameter('save_debug_images').value
        self.enable_roi = self.get_parameter('enable_roi_mask').value
        self.max_attempts = self.get_parameter('max_attempts').value
        self.time_slop = self.get_parameter('time_slop').value
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.reentrant_cb_group = ReentrantCallbackGroup()

        # 目录初始化
        self.debug_base_path = os.path.join(self.results_dir, 'debug_visuals')
        self.raw_base_path = os.path.join(self.results_dir, 'raw_images')
        os.makedirs(self.debug_base_path, exist_ok=True)
        os.makedirs(self.raw_base_path, exist_ok=True)

        self.csv_file = os.path.join(
            self.results_dir,
            f"secure_logic_log_sub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        # --- 2. 状态变量与锁 ---
        self.midpoint_reached = False
        self.last_x1_status = 0
        self.is_busy = False
        self.processing_lock = threading.Lock()
        self.midpoint_reached_lock = threading.Lock()
        self.cooldown_time = 1.5
        self.last_process_time = 0.0
        # 用于在线程间传递同步好的图像帧
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.latest_color = None  # np.ndarray (BGR)
        self.latest_depth = None  # np.ndarray
        # 在没有外部相机时，使用一张空白图做模型预热
        self._warmup_color = np.zeros((480, 640, 3), dtype=np.uint8)

        # --- 3. 算法初始化 ---
        self.initialize_yolo_model()
        self.processor_tool = StemDiameterProcessor()

        # --- 4. 订阅与发布 ---
        self.create_subscription(
            MidpointReached, 'midpoint_reached',
            self.midpoint_reached_callback, 1,
            callback_group=self.reentrant_cb_group
        )
        self.create_subscription(
            UInt8, 'gripper/input_x1_status',
            self.x1_status_callback, 1,
            callback_group=self.reentrant_cb_group
        )

        self.test_result_pub = self.create_publisher(
            TestResult, 'test_result', 1, callback_group=self.reentrant_cb_group
        )
        self.yolo_viz_pub = self.create_publisher(
            Image, 'yolo_viz_with_skeleton', 1, callback_group=self.reentrant_cb_group
        )

        # 相机话题订阅 + 近似时间同步器
        color_sub = message_filters.Subscriber(self, Image, self.color_topic)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=1,
            slop=self.time_slop,
        )
        self.ts.registerCallback(self.sync_image_callback)

        self.get_logger().info(
            f"✅ 订阅彩色: {self.color_topic}, 深度: {self.depth_topic}, slop={self.time_slop}s"
        )

    def initialize_yolo_model(self):
        model_name = self.get_parameter('model_name').value
        share_dir = get_package_share_directory('closed_loop_feedback_detection')
        path = os.path.join(share_dir, 'weights', model_name)
        try:
            self.yolo_model = YOLO(path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.get_logger().info(f"✅ YOLO模型加载成功 ({self.device})")
            # 使用一张空白图做预热，避免第一次推理耗时过长
            results = self.yolo_model.predict(
                self._warmup_color, conf=0.5, device=self.device,
                verbose=False, iou=0.2
            )
            self.get_logger().info("✅ YOLO模型预热成功")
        except Exception as e:
            self.get_logger().error(f"❌ 模型加载失败: {e}")

    def sync_image_callback(self, color_msg, depth_msg):
        """
        近似时间同步回调：彩色与深度时间差在 slop 之内才保留，否则直接丢弃。
        """
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"⚠️ 彩色图解码失败，已丢弃: {e}")
            return
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"⚠️ 深度图解码失败，已丢弃: {e}")
            return

        with self.frame_lock:
            self.latest_color = color_img
            self.latest_depth = depth_img
        # 通知等待中的处理线程
        self.new_frame_event.set()

    def midpoint_reached_callback(self, msg):
        with self.midpoint_reached_lock:
            self.midpoint_reached = msg.reached
            self.get_logger().warn(f"✅ 中间点状态更新: {self.midpoint_reached}")

    def x1_status_callback(self, msg):
        current_status = msg.data
        if self.last_x1_status == 0 and current_status != 0:
            now = time.time()
            if now - self.last_process_time < self.cooldown_time:
                self.get_logger().info(
                    f"⏳ 冷却中，忽略触发（剩余 {self.cooldown_time - (now - self.last_process_time):.1f}s）"
                )
                return
            with self.midpoint_reached_lock:
                midpoint_status = self.midpoint_reached
            self.get_logger().warn(
                f"✅ 中间点: {self.midpoint_reached}，信号点状态: {current_status}"
            )
            if midpoint_status and not self.is_busy:
                self.last_process_time = now
                self.publish_test_result(1)
                threading.Thread(target=self.async_wrapper, daemon=True).start()
        self.last_x1_status = current_status

    def async_wrapper(self):
        with self.processing_lock:
            self.is_busy = True
            try:
                self.camera_process_logic()
            finally:
                self.is_busy = False
                # 清理可能残留的标志位
                self.new_frame_event.clear()

    def _wait_for_synced_frame(self, timeout=2.0):
        """
        阻塞等待一对满足同步条件的彩色+深度帧，超时则返回 (None, None)。
        """
        self.new_frame_event.clear()
        got = self.new_frame_event.wait(timeout=timeout)
        if not got:
            return None, None
        with self.frame_lock:
            return self.latest_color, self.latest_depth

    def collect_n_frames(self, n, per_frame_timeout=2.0):
        """
        连续采集 n 帧同步好的彩色+深度图像，返回列表。
        每帧最多等待 per_frame_timeout 秒。
        """
        frames = []
        for i in range(n):
            color_img, depth_img = self._wait_for_synced_frame(timeout=per_frame_timeout)
            if color_img is None:
                self.get_logger().warn(
                    f"⚠️ 采集第 {i + 1}/{n} 帧超时，已采集 {len(frames)} 帧"
                )
                break
            frames.append({
                'color': color_img.copy(),
                'depth': depth_img.copy(),
            })
        self.get_logger().info(f"📸 采集完成，共 {len(frames)} 帧")
        return frames

    def camera_process_logic(self):
        with self.midpoint_reached_lock:
            self.midpoint_reached = False

        success_detection = False
        self.get_logger().info(
            f"🚀 先采集 {self.max_attempts} 帧图像，再依次处理"
        )
        self.get_logger().info(f"中间点状态: {self.midpoint_reached}")
        

        frames = self.collect_n_frames(self.max_attempts, per_frame_timeout=2.0)

        if len(frames) == 0:
            self.get_logger().error("❌ 一帧图像都没采集到")
            self.publish_test_result(1)
            return

        for idx, frame in enumerate(frames):
            color_img = frame['color']
            depth_img = frame['depth']

            raw_color = color_img
            origin_img = raw_color.copy()
            h, w = raw_color.shape[:2]

            # ROI 涂黑预处理 (使用复本进行推理)
            inference_img = raw_color.copy()
            if self.enable_roi:
                # 左右各涂黑1/4，保留中间1/2
                left_width = w // 4
                right_width = w // 4
                inference_img[:, :left_width] = 0  # 左边1/4涂黑
                inference_img[:, w - right_width:] = 0  # 右边1/4涂黑

            # 推理
            results = self.yolo_model.predict(
                inference_img, conf=0.5, device=self.device, verbose=False, iou=0.2
            )

            # 初始化变量
            current_vis_data = []
            temp_diameters_mm = []
            final_code = 1  # 默认代码

            now = datetime.now()
            date_folder = now.strftime("%Y%m%d")
            timestamp = now.strftime("%H%M%S_%f")[:-3]

            # 处理结果
            if (
                len(results) > 0
                and results[0].masks is not None
                and len(results[0].masks.data) > 0
            ):
                result = results[0]
                annotated_img = result.plot()
                self.diameters_mm = 0.0

                self.get_logger().info(f"🎯 第 {idx + 1} 帧成功捕捉到目标")

                # 处理细节逻辑
                self.process_detection_results(
                    result, raw_color, inference_img, depth_img, timestamp
                )
                success_detection = True

                # 保存 Debug 图像
                if self.save_debug:
                    debug_daily_path = os.path.join(self.debug_base_path, date_folder)
                    os.makedirs(debug_daily_path, exist_ok=True)
                    final_viz = self.processor_tool.visualize_results(
                        annotated_img, current_vis_data
                    )
                    header_text = (
                        f"CODE:{final_code} | MM: {self.diameters_mm:.2f}"
                    )
                    cv2.putText(
                        final_viz, header_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                    cv2.imwrite(
                        os.path.join(debug_daily_path, f"debug_{timestamp}_full.jpg"),
                        final_viz,
                    )
                    cv2.imwrite(
                        os.path.join(debug_daily_path, f"debug_{timestamp}_origin.jpg"),
                        origin_img,
                    )
                    self.yolo_viz_pub.publish(
                        self.bridge.cv2_to_imgmsg(final_viz, "bgr8")
                    )

                break
            else:
                # 没有识别结果
                self.get_logger().warn(
                    f"⚠️ 第 {idx + 1} 帧未识别到目标，继续处理下一帧..."
                )

                # 保存 Debug 图像 - 使用原始图片作为识别后的图片
                if self.save_debug:
                    debug_daily_path = os.path.join(self.debug_base_path, date_folder)
                    os.makedirs(debug_daily_path, exist_ok=True)
                    final_viz = origin_img.copy()
                    header_text = f"CODE:{final_code} | 未识别到目标"
                    cv2.putText(
                        final_viz, header_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )
                    cv2.imwrite(
                        os.path.join(debug_daily_path, f"debug_{timestamp}_full.jpg"),
                        final_viz,
                    )
                    cv2.imwrite(
                        os.path.join(debug_daily_path, f"debug_{timestamp}_origin.jpg"),
                        origin_img,
                    )
                    self.yolo_viz_pub.publish(
                        self.bridge.cv2_to_imgmsg(final_viz, "bgr8")
                    )

        if not success_detection:
            self.get_logger().error("❌ 全部帧处理完毕仍无结果")
            self.publish_test_result(1)

    def process_detection_results(
        self, result, raw_color, inference_img, depth_img, timestamp
    ):
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
            skel_res = self.processor_tool.calculate_skeleton_diameter_pure(
                mask_resized
            )
            pixel_diameter = skel_res['diameter'] if skel_res['success'] else 0.0
            calc_diameter_mm = (
                pixel_diameter / self.px_per_mm if pixel_diameter > 0 else 0.0
            )
            self.diameters_mm = calc_diameter_mm
            temp_diameters_mm.append(f"{calc_diameter_mm:.2f}")

            # 逻辑判定
            calc_label = "trunk" if calc_diameter_mm > self.threshold_mm else "stem"
            if cls_name == "trunk" or calc_label == "trunk": #如果识别到干支或直径大于阈值，则认为是识别到危险
                found_danger = True                
            else:
                found_perfect_stalk = True
                

            self.log_to_csv(
                [datetime.now().isoformat(), cls_name, calc_label, calc_diameter_mm]
            )

            if skel_res['success']:
                current_vis_data.append({
                    'success': True,
                    'method': 'skeleton',
                    'final_diameter': pixel_diameter,
                    'points': skel_res['valid_points'],
                    'bbox_offset': (0, 0),
                })

        final_code = 3 if found_danger else (2 if found_perfect_stalk else 1)
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
            i = i + 1
            self.test_result_pub.publish(msg)
            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedVisionControlNodeSub()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
