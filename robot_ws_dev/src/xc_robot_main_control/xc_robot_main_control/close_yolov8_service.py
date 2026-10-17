#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from xc_interfaces.srv import CloseImgDetection
from xc_interfaces.msg import ArmStatus
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import time
import threading
import math
import os
from ament_index_python.packages import get_package_share_directory
import pyrealsense2 as rs2
from ultralytics import YOLO

from xc_robot_main_control.stemGrowthAndPoints import StemGrowthAndPoints


class CloseYOLOv8Service(Node):
    def __init__(self):
        super().__init__('close_yolov8_service')

        # 参数声明
        self.declare_parameter('color_topic', '/cam_hand/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/cam_hand/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/cam_hand/color/camera_info')
        self.declare_parameter('arm_status_topic', 'arm_status')
        self.declare_parameter('service_name', 'close_img_detection')
        self.declare_parameter('combined_image_topic', 'close_yolo_combined_image')
        self.declare_parameter('time_threshold', 0.05)
        self.declare_parameter('depth_scale', 0.0001)
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('hand_eye_file_path', '')
        self.declare_parameter('line_intersect_threshold', 5.0)  # line_A与stem骨骼交点的垂直距离阈值(像素)

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.arm_status_topic = self.get_parameter('arm_status_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.combined_image_topic = self.get_parameter('combined_image_topic').value
        self.time_threshold = self.get_parameter('time_threshold').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.line_intersect_threshold = float(self.get_parameter('line_intersect_threshold').value)
        yolo_model_path = self.get_parameter('yolo_model_path').value
        hand_eye_file_path = self.get_parameter('hand_eye_file_path').value

        self.bridge = CvBridge()
        self.mutex_cb_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_cb_group = ReentrantCallbackGroup()

        self.growth_point_calculator = StemGrowthAndPoints()

        # 最新帧缓存
        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.frame_lock = threading.Lock()

        # 相机内参
        self.intrinsics = None
        self.intrinsics_ready = False

        # 机械臂状态缓存
        self.latest_arm_status = None
        self.arm_status_lock = threading.Lock()

        # 手眼标定矩阵(眼在手上: 相机坐标系 -> 末端坐标系)
        self.hand_eye_matrix = self._load_hand_eye_matrix(hand_eye_file_path)

        # YOLO模型路径: 未指定时使用功能包内权重
        if not yolo_model_path:
            package_share_dir = get_package_share_directory('xc_robot_main_control')
            yolo_model_path = os.path.join(package_share_dir, 'weights', 'best.pt')
        self.yolo_model_path = yolo_model_path
        self._init_yolov8_model()

        self._setup_subscriptions()
        self._setup_services()

        # 调试图像发布者
        self.combined_pub = self.create_publisher(Image, self.combined_image_topic, 10)

        self.get_logger().info(f"近距离YOLOv8采摘点服务已启动")
        self.get_logger().info(f"  彩色图话题: {self.color_topic}")
        self.get_logger().info(f"  深度图话题: {self.depth_topic}")
        self.get_logger().info(f"  机械臂状态话题: {self.arm_status_topic}")
        self.get_logger().info(f"  服务名: {self.service_name}")
        self.get_logger().info(f"  手眼标定文件: {self.hand_eye_file_path}")
        self.get_logger().info(f"  YOLO模型: {self.yolo_model_path}")
        self.get_logger().info(f"  类别: {self.yolo_model.names}")

    def _init_yolov8_model(self):
        """初始化YOLOv8模型"""
        self.get_logger().info(f"正在加载YOLOv8模型: {self.yolo_model_path}")
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            self.get_logger().info(f"YOLOv8模型加载成功")
        except Exception as e:
            self.get_logger().error(f"YOLOv8模型加载失败: {str(e)}")
            self.yolo_model = None

    def _load_hand_eye_matrix(self, param_path):
        """加载手眼标定矩阵(4x4), 依次尝试: 功能包share目录 -> 参数指定路径 -> 源码路径"""
        candidates = []
        try:
            share_dir = get_package_share_directory('hand_eye_calibration_python')
            candidates.append(os.path.join(share_dir, 'config', 'hand', 'calibration_results.txt'))
        except Exception:
            pass
        if param_path:
            candidates.append(param_path)
        candidates.append('/home/xc_home_wsl/code/test/ROS2/robot_ws/src/hand_eye_calibration_python/config/hand/calibration_results.txt')

        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as file:
                        content = file.read().strip()
                    lines = content.split('\n')
                    matrix = np.array([[float(v) for v in line.split(',')] for line in lines], dtype=np.float64)
                    if matrix.shape != (4, 4):
                        raise ValueError(f"矩阵尺寸错误: {matrix.shape}")
                    self.hand_eye_file_path = path
                    self.get_logger().info(f"手眼标定矩阵加载成功: {path}")
                    self.get_logger().info(f"\n{matrix}")
                    return matrix
                except Exception as e:
                    self.get_logger().error(f"解析手眼标定文件失败: {path}, 错误: {str(e)}")
        self.hand_eye_file_path = ''
        self.get_logger().error("未找到有效的手眼标定文件, 请通过 hand_eye_file_path 参数指定")
        return None

    def _setup_subscriptions(self):
        # 彩色图与深度图时间同步
        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.image_callback)

        # 相机内参
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            1
        )

        # 机械臂状态(位置+四元数姿态)
        self.arm_status_sub = self.create_subscription(
            ArmStatus,
            self.arm_status_topic,
            self.arm_status_callback,
            10
        )

    def _setup_services(self):
        self.close_img_service = self.create_service(
            CloseImgDetection,
            self.service_name,
            self.handle_close_img_detection,
            callback_group=self.reentrant_cb_group
        )
        self.get_logger().info(f"近距离检测服务已创建: {self.service_name}")

    def image_callback(self, color_msg, depth_msg):
        with self.frame_lock:
            self.latest_color_msg = color_msg
            self.latest_depth_msg = depth_msg

    def camera_info_callback(self, msg):
        if self.intrinsics_ready:
            return
        try:
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = msg.width
            self.intrinsics.height = msg.height
            self.intrinsics.ppx = msg.k[2]
            self.intrinsics.ppy = msg.k[5]
            self.intrinsics.fx = msg.k[0]
            self.intrinsics.fy = msg.k[4]
            if msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif msg.distortion_model == 'kannala_brandt4':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            else:
                self.intrinsics.model = rs2.distortion.brown_conrady
            self.intrinsics.coeffs = [i for i in msg.d]
            self.intrinsics_ready = True
            self.get_logger().info(
                f"相机内参已获取: {msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}"
            )
        except Exception as e:
            self.get_logger().error(f"解析相机内参失败: {str(e)}")

    def arm_status_callback(self, msg):
        with self.arm_status_lock:
            self.latest_arm_status = msg

    # ---------- 坐标变换工具 ----------

    @staticmethod
    def quaternion_to_matrix(x, y, z, w):
        """四元数转3x3旋转矩阵"""
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return np.eye(3)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]
        ], dtype=np.float64)

    @staticmethod
    def arm_status_to_matrix(msg):
        """ArmStatus(位置+四元数) 转 4x4齐次变换矩阵 T_base_end"""
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = msg.position.x
        T[1, 3] = msg.position.y
        T[2, 3] = msg.position.z
        T[:3, :3] = CloseYOLOv8Service.quaternion_to_matrix(
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        return T

    @staticmethod
    def rotation_matrix_to_rpy(R):
        """旋转矩阵转固定轴XYZ欧拉角(roll-pitch-yaw, 弧度), 与tf2 setRPY/MoveToPose约定一致"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            rx = math.atan2(R[2, 1], R[2, 2])
            ry = math.atan2(-R[2, 0], sy)
            rz = math.atan2(R[1, 0], R[0, 0])
        else:
            rx = math.atan2(-R[1, 2], R[1, 1])
            ry = math.atan2(-R[2, 0], sy)
            rz = 0.0
        return rx, ry, rz

    # ---------- 图像帧与深度工具(参考pick_pose_service_node) ----------

    def _validate_frames(self, color_msg, depth_msg):
        color_time = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
        depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        time_diff = abs(color_time - depth_time)

        if time_diff > self.time_threshold:
            self.get_logger().warn(
                f"时间戳差异过大: {time_diff:.4f}s > {self.time_threshold}s"
            )
            return False

        if color_msg.width != depth_msg.width or color_msg.height != depth_msg.height:
            self.get_logger().warn(
                f"图像尺寸不一致: color={color_msg.width}x{color_msg.height}, "
                f"depth={depth_msg.width}x{depth_msg.height}"
            )
            return False

        return True

    def _get_latest_valid_frames(self, max_wait=2.0):
        start_time = time.time()
        while time.time() - start_time < max_wait:
            with self.frame_lock:
                color_msg = self.latest_color_msg
                depth_msg = self.latest_depth_msg

            if color_msg is None or depth_msg is None:
                time.sleep(0.01)
                continue

            if self._validate_frames(color_msg, depth_msg):
                return color_msg, depth_msg

            time.sleep(0.01)

        return None, None

    def _detect_with_yolov8(self, color_cv):
        """直接使用YOLOv8进行检测，返回标准化结果"""
        if self.yolo_model is None:
            self.get_logger().error("YOLOv8模型未初始化")
            return None

        try:
            results = self.yolo_model(color_cv)
            result = results[0]

            class DetectionResult:
                pass

            detection_result = DetectionResult()
            detection_result.class_ids = []
            detection_result.class_names = []
            detection_result.confidences = []
            detection_result.bboxes = []
            detection_result.masks = []

            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.yolo_model.names[cls] if cls < len(self.yolo_model.names) else f"class_{cls}"

                    x1, y1, x2, y2 = map(float, box.xyxy[0])

                    detection_result.class_ids.append(cls)
                    detection_result.class_names.append(class_name)
                    detection_result.confidences.append(conf)
                    detection_result.bboxes.extend([x1, y1, x2, y2])

                    if result.masks is not None and i < len(result.masks):
                        mask_data = result.masks[i].data.cpu().numpy().astype(bool)[0]
                        mask_data = cv2.resize(
                            mask_data.astype(np.uint8) * 255,
                            (color_cv.shape[1], color_cv.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        mask_msg = self.bridge.cv2_to_imgmsg(mask_data, "mono8")

                        class MaskWrapper:
                            pass
                        mask_wrapper = MaskWrapper()
                        mask_wrapper.mask = mask_msg
                        detection_result.masks.append(mask_wrapper)

            return detection_result

        except Exception as e:
            self.get_logger().error(f"YOLOv8检测异常: {str(e)}")
            return None

    def _get_neighborhood_depth(self, depth_image, mask, x, y, max_window_size=45):
        h, w = depth_image.shape[:2]
        for window_size in range(3, max_window_size + 1, 2):
            half = window_size // 2
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(w, x + half + 1)
            y2 = min(h, y + half + 1)

            roi = depth_image[y1:y2, x1:x2]
            mask_roi = mask[y1:y2, x1:x2] if mask is not None else None

            if mask_roi is not None:
                valid_depths = roi[(mask_roi > 0) & (roi > 0)]
            else:
                valid_depths = roi[roi > 0]

            if len(valid_depths) > 0:
                return float(np.mean(valid_depths)) * self.depth_scale

        return 0.0

    def _get_mask_average_depth(self, mask, depth_image):
        if mask is None or depth_image is None:
            return 0.0
        mask_uint8 = mask.astype(np.uint8)
        depth_values = depth_image[mask_uint8 > 0]
        valid_depths = depth_values[depth_values > 0]
        if len(valid_depths) == 0:
            return 0.0
        return float(np.mean(valid_depths)) * self.depth_scale

    def _get_valid_depth_value(self, depth_image, mask, x, y):
        depth_value = depth_image[y, x] * self.depth_scale

        if depth_value < 0.20 or np.isnan(depth_value) or depth_value > 1.0:
            depth_value = self._get_neighborhood_depth(
                depth_image, mask, x, y, max_window_size=45
            )

            if depth_value <= 0.20 or np.isnan(depth_value) or depth_value >= 1.0:
                depth_value = self._get_mask_average_depth(mask, depth_image)

        return depth_value

    # ---------- 检测结果分类 ----------

    def classify_detections(self, response, num_boxes, total_masks):
        """按类别名分类: stalk(果柄) / stem(主茎) / tomato(番茄)"""
        stalks, stems, tomatos = [], [], []
        for i in range(num_boxes):
            idx = i * 4
            x1 = response.bboxes[idx]
            y1 = response.bboxes[idx + 1]
            x2 = response.bboxes[idx + 2]
            y2 = response.bboxes[idx + 3]
            conf = response.confidences[i]
            class_name = response.class_names[i].lower()

            if i >= total_masks:
                self.get_logger().warn(f"检测框{i}({class_name})无对应掩码")
                continue

            if class_name == "stalk":
                stalks.append((x1, y1, x2, y2, conf, i))
            elif class_name == "stem":
                stems.append((x1, y1, x2, y2, conf, i))
            elif class_name == "tomato":
                tomatos.append((x1, y1, x2, y2, conf, i))

        return stalks, stems, tomatos

    # ---------- 果柄(stalk)关键点计算(参考stemGrowthAndPoints) ----------

    def _get_thin_skeleton(self, cv_mask):
        """掩码细化: 掩膜不连续时先膨胀, 再细化提取骨骼(返回0-1二值骨架图)"""
        if len(cv_mask.shape) == 3:
            mask_gray = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = cv_mask.copy()

        contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            self.get_logger().info("掩膜不连续，执行膨胀操作...")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dilated_mask = cv2.dilate(mask_gray, kernel)
            dilated_mask = dilated_mask // 255
        else:
            self.get_logger().info("掩膜是连续的，不需要执行膨胀操作...")
            dilated_mask = mask_gray // 255

        thin = self.growth_point_calculator.image_thin(dilated_mask * 255)
        return thin

    def _get_stalk_key_points(self, stalk, stalk_cv_mask, w, h):
        """
        计算果柄(stalk)的关键点:
        - stalk_point_top:    骨架最高点(最小y)
        - stalk_point_bottom: 通过最高点到最左/最右点距离比较得到(距离大的为最低点)
        - stalk_point_cut:    采摘点(中线与骨架交点均值, 参考determine_growth_direction_and_picking_point)
        - line_B角度:         stalk_point_top与stalk_point_bottom连线的角度
        返回: dict 或 None
        """
        thin = self._get_thin_skeleton(stalk_cv_mask)
        skeleton_points = np.column_stack(np.where(thin > 0))
        if len(skeleton_points) <= 10:
            self.get_logger().warn(f"果柄骨架点数量不足: {len(skeleton_points)}")
            return None

        # 最左、最右、最高点
        leftmost_idx = np.argmin(skeleton_points[:, 1])
        left_point = (int(skeleton_points[leftmost_idx, 1]), int(skeleton_points[leftmost_idx, 0]))
        rightmost_idx = np.argmax(skeleton_points[:, 1])
        right_point = (int(skeleton_points[rightmost_idx, 1]), int(skeleton_points[rightmost_idx, 0]))
        topmost_idx = np.argmin(skeleton_points[:, 0])
        top_point = (int(skeleton_points[topmost_idx, 1]), int(skeleton_points[topmost_idx, 0]))

        # 参考calculate_tangent_angle_with_abc: 比较最高点到最左/最右的距离, 距离大的作为最低点
        dis_left = math.sqrt((top_point[0] - left_point[0]) ** 2 + (top_point[1] - left_point[1]) ** 2)
        dis_right = math.sqrt((top_point[0] - right_point[0]) ** 2 + (top_point[1] - right_point[1]) ** 2)
        if dis_left >= dis_right:
            bottom_point = left_point
        else:
            bottom_point = right_point

        stalk_point_top = top_point
        stalk_point_bottom = bottom_point

        # line_B: 最高点与最低点连线的角度
        dx = stalk_point_bottom[0] - stalk_point_top[0]
        dy = stalk_point_bottom[1] - stalk_point_top[1]
        if dx == 0:
            line_B_angle = math.pi / 2
            line_B_slope = float('inf')
        else:
            line_B_slope = dy / dx
            line_B_angle = math.atan(line_B_slope)

        # 采摘点: 参考determine_growth_direction_and_picking_point的中线法
        x1, y1, x2, y2 = stalk[0], stalk[1], stalk[2], stalk[3]
        stalk_w = x2 - x1
        stalk_h = y2 - y1
        if stalk_h > 0.8 * stalk_w:
            # 垂直生长: 水平中线与骨架交点
            mind = int(stalk_h * 0.5 + y1)
            mind = int(max(0, min(mind, h - 1)))
            cut_x_sum = 0
            cout_cut = 0
            for xx in range(thin.shape[1]):
                if thin[mind, xx] == 1:
                    cut_x_sum += xx
                    cout_cut += 1
            if cout_cut > 0:
                stalk_point_cut = (cut_x_sum // cout_cut, mind)
            else:
                nearest_point = self.growth_point_calculator.find_nearest_point_on_skeleton(thin, 'horizontal', mind)
                stalk_point_cut = (int(nearest_point[0]), int(nearest_point[1]))
                self.get_logger().info(f"未找到中线与骨架的交点，使用最近点: {stalk_point_cut}")
        else:
            # 水平生长: 垂直中线与骨架交点
            mind = int(stalk_w * 0.5 + x1)
            mind = int(max(0, min(mind, w - 1)))
            cut_y_sum = 0
            cout_cut = 0
            for yy in range(thin.shape[0]):
                if thin[yy, mind] == 1:
                    cut_y_sum += yy
                    cout_cut += 1
            if cout_cut > 0:
                stalk_point_cut = (mind, cut_y_sum // cout_cut)
            else:
                nearest_point = self.growth_point_calculator.find_nearest_point_on_skeleton(thin, 'vertical', mind)
                stalk_point_cut = (int(nearest_point[0]), int(nearest_point[1]))
                self.get_logger().info(f"未找到中线与骨架的交点，使用最近点: {stalk_point_cut}")

        return {
            'thin': thin,
            'stalk_point_top': stalk_point_top,
            'stalk_point_bottom': stalk_point_bottom,
            'stalk_point_cut': stalk_point_cut,
            'line_B_slope': line_B_slope,
            'line_B_angle': line_B_angle,
        }

    def _find_stem_obj(self, stems, response, total_masks, stalk_point_cut, line_B_slope):
        """
        绘制经过采摘点、角度与line_B相同的直线line_A,
        查看line_A与各个stem(主茎)骨架的交点, 交点与采摘点距离最近的stem作为stem_obj
        返回: (stem_obj, 交点像素坐标, stem骨架图) 或 None
        """
        cut_x, cut_y = stalk_point_cut
        h_img, w_img = self._current_image_shape

        best_stem = None
        best_intersect = None
        best_thin = None
        best_dist = float('inf')

        for stem in stems:
            stem_idx = stem[5]
            if stem_idx >= total_masks:
                continue
            try:
                stem_cv_mask = self.bridge.imgmsg_to_cv2(response.masks[stem_idx].mask, "mono8")
            except Exception as e:
                self.get_logger().error(f"解析stem掩码失败: {e}")
                continue

            thin = self._get_thin_skeleton(stem_cv_mask)
            skeleton_points = np.column_stack(np.where(thin > 0))
            if len(skeleton_points) == 0:
                continue
            skel_xs = skeleton_points[:, 1].astype(np.float64)
            skel_ys = skeleton_points[:, 0].astype(np.float64)

            if np.isinf(line_B_slope):
                # 垂直线: 距离为x方向差
                perp_dists = np.abs(skel_xs - cut_x)
            else:
                # 直线line_A: y = slope*(x - cut_x) + cut_y
                perp_dists = np.abs(line_B_slope * skel_xs - skel_ys + (cut_y - line_B_slope * cut_x)) / math.sqrt(line_B_slope ** 2 + 1)

            min_idx = int(np.argmin(perp_dists))
            if perp_dists[min_idx] > self.line_intersect_threshold:
                continue  # 该stem与line_A无交点

            # 交点: 最近骨架点在line_A上的投影
            sx, sy_ = skel_xs[min_idx], skel_ys[min_idx]
            if np.isinf(line_B_slope):
                intersect_point = (int(cut_x), int(sy_))
            else:
                # 直线上参数化投影
                vx, vy = 1.0, line_B_slope
                t = ((sx - cut_x) * vx + (sy_ - cut_y) * vy) / (vx * vx + vy * vy)
                px = cut_x + t * vx
                py = cut_y + t * vy
                intersect_point = (int(round(px)), int(round(py)))
                intersect_point = (int(max(0, min(intersect_point[0], w_img - 1))),
                                   int(max(0, min(intersect_point[1], h_img - 1))))

            dist_to_cut = math.sqrt((intersect_point[0] - cut_x) ** 2 + (intersect_point[1] - cut_y) ** 2)
            self.get_logger().info(
                f"stem #{stem_idx} 与line_A的交点: {intersect_point}, 距采摘点距离: {dist_to_cut:.1f}px"
            )

            if dist_to_cut < best_dist:
                best_dist = dist_to_cut
                best_stem = stem
                best_intersect = intersect_point
                best_thin = thin

        if best_stem is None:
            return None
        return best_stem, best_intersect, best_thin

    def _get_stem_point_at_row(self, stem_thin, row_y, default_x):
        """
        经过给定行(水平线)与stem骨架求交点: 取该行(±2像素容差)骨架点x均值
        返回 (x, y) 或 None
        """
        h_img, w_img = stem_thin.shape[:2]
        row_y = int(max(0, min(row_y, h_img - 1)))

        for offset in (0, -1, 1, -2, 2):
            yy = row_y + offset
            if yy < 0 or yy >= h_img:
                continue
            xs = np.where(stem_thin[yy, :] == 1)[0]
            if len(xs) > 0:
                mean_x = int(np.mean(xs))
                mean_x = int(max(0, min(mean_x, w_img - 1)))
                return (mean_x, yy)
        # 兜底: 使用采摘点x
        self.get_logger().warn(f"水平线y={row_y}与stem骨架无交点, 使用默认x={default_x}")
        return (int(default_x), row_y)

    # ---------- 3D与位姿计算 ----------

    def _pixel_to_base(self, depth_cv, mask, pixel, T_base_cam, fallback_depth=None):
        """像素坐标 -> 相机坐标系 -> 机械臂底座坐标系. 返回 (p_cam, p_base) 或 (None, None)"""
        h, w = depth_cv.shape[:2]
        x, y = int(pixel[0]), int(pixel[1])
        x = int(max(0, min(x, w - 1)))
        y = int(max(0, min(y, h - 1)))

        depth_value = self._get_valid_depth_value(depth_cv, mask, x, y)
        if not (0.20 < depth_value < 1.0) or np.isnan(depth_value):
            if fallback_depth is not None and 0.20 < fallback_depth < 1.0:
                self.get_logger().warn(f"像素{pixel}无有效深度值, 使用回退深度: {fallback_depth:.3f}m")
                depth_value = fallback_depth
            else:
                self.get_logger().warn(f"像素{pixel}无有效深度值: {depth_value}")
                return None, None

        point_3d = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_value)
        homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        base_point = (T_base_cam @ homogeneous)[:3]
        return np.array(point_3d), base_point

    # ---------- 服务处理 ----------

    def handle_close_img_detection(self, request, response):
        self.get_logger().info(
            f"收到近距离检测请求: tomato_xyz(base)=({request.x:.3f}, {request.y:.3f}, {request.z:.3f})"
        )
        try:
            # 0. 基础检查
            if self.hand_eye_matrix is None:
                response.success = False
                response.message = "手眼标定矩阵未加载"
                return response
            if not self.intrinsics_ready:
                response.success = False
                response.message = "相机内参未就绪"
                return response
            with self.arm_status_lock:
                arm_status = self.latest_arm_status
            if arm_status is None:
                response.success = False
                response.message = f"尚未收到机械臂状态({self.arm_status_topic})"
                return response

            tomato_xyz = np.array([request.x, request.y, request.z], dtype=np.float64)

            # 1. 计算当前相机位姿: T_base_cam = T_base_end(机械臂当前位姿) @ T_end_cam(手眼标定)
            T_base_end = self.arm_status_to_matrix(arm_status)
            T_base_cam = T_base_end @ self.hand_eye_matrix
            T_cam_base = np.linalg.inv(T_base_cam)

            # 2. tomato中心(底座系) -> 相机坐标系
            tomato_homogeneous = np.append(tomato_xyz, 1.0)
            tomato_xyz_cam = (T_cam_base @ tomato_homogeneous)[:3]
            self.get_logger().info(
                f"tomato中心在相机坐标系下的位置: "
                f"X={tomato_xyz_cam[0]:.3f}m, Y={tomato_xyz_cam[1]:.3f}m, Z={tomato_xyz_cam[2]:.3f}m"
            )
            if tomato_xyz_cam[2] <= 0.05:
                response.success = False
                response.message = f"tomato在相机坐标系下深度异常: {tomato_xyz_cam[2]:.3f}m"
                return response

            # 3. 投影到图像坐标
            pixel = rs2.rs2_project_point_to_pixel(
                self.intrinsics,
                [float(tomato_xyz_cam[0]), float(tomato_xyz_cam[1]), float(tomato_xyz_cam[2])]
            )
            tomato_xy_img = (int(round(pixel[0])), int(round(pixel[1])))
            self.get_logger().info(f"tomato中心投影到图像坐标: {tomato_xy_img}")
            img_w = self.intrinsics.width
            img_h = self.intrinsics.height
            margin = 30
            if not (-margin <= tomato_xy_img[0] < img_w + margin and -margin <= tomato_xy_img[1] < img_h + margin):
                response.success = False
                response.message = f"tomato投影超出图像范围: {tomato_xy_img} ({img_w}x{img_h})"
                return response

            # 4. 获取最新图像帧并YOLO检测
            color_msg, depth_msg = self._get_latest_valid_frames(max_wait=2.0)
            if color_msg is None or depth_msg is None:
                response.success = False
                response.message = "无法获取有效的彩色/深度图像对"
                return response
            try:
                color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            except Exception as e:
                response.success = False
                response.message = f"图像转换失败: {str(e)}"
                return response

            yolo_response = self._detect_with_yolov8(color_cv)
            if yolo_response is None:
                response.success = False
                response.message = "YOLO检测失败"
                return response

            num_boxes = len(yolo_response.class_ids)
            total_masks = len(yolo_response.masks) if hasattr(yolo_response, 'masks') else 0
            if num_boxes == 0 or total_masks == 0:
                response.success = False
                response.message = "YOLO未检测到任何目标"
                return response
            self._current_image_shape = color_cv.shape[:2]

            stalks, stems, tomatos = self.classify_detections(yolo_response, num_boxes, total_masks)
            self.get_logger().info(f"果柄(stalk)数量: {len(stalks)} | 主茎(stem)数量: {len(stems)} | 番茄数量: {len(tomatos)}")

            if len(stalks) == 0:
                response.success = False
                response.message = "未检测到stalk(果柄)"
                return self._publish_debug_image(response, color_cv, None)
            if len(stems) == 0:
                response.success = False
                response.message = "未检测到stem(主茎)"
                return self._publish_debug_image(response, color_cv, None)

            # 5. 选择距离tomato_xyz_cam最近且在其上方的stalk
            best_stalk = None
            best_stalk_mask = None
            best_dist = float('inf')
            for stalk in stalks:
                stalk_idx = stalk[5]
                try:
                    stalk_cv_mask = self.bridge.imgmsg_to_cv2(yolo_response.masks[stalk_idx].mask, "mono8")
                except Exception as e:
                    self.get_logger().error(f"解析stalk掩码失败: {e}")
                    continue

                cx = int((stalk[0] + stalk[2]) / 2)
                cy = int((stalk[1] + stalk[3]) / 2)
                p_cam, p_base = self._pixel_to_base(depth_cv, stalk_cv_mask, (cx, cy), T_base_cam)
                if p_cam is None:
                    continue

                # 要求stalk中心在tomato上方(底座坐标系Z更高)
                if p_base[2] <= tomato_xyz[2]:
                    self.get_logger().info(
                        f"stalk #{stalk_idx} 中心不在tomato上方(stalk_Z={p_base[2]:.3f} <= tomato_Z={tomato_xyz[2]:.3f}), 跳过"
                    )
                    continue

                dist = float(np.linalg.norm(p_cam - tomato_xyz_cam))
                self.get_logger().info(
                    f"stalk #{stalk_idx} 中心: 像素=({cx},{cy}), 相机系距离tomato={dist:.3f}m"
                )
                if dist < best_dist:
                    best_dist = dist
                    best_stalk = stalk
                    best_stalk_mask = stalk_cv_mask

            if best_stalk is None:
                response.success = False
                response.message = "未找到在tomato上方且距离最近的stalk"
                return self._publish_debug_image(response, color_cv, None)
            self.get_logger().info(
                f"选定stalk #{best_stalk[5]}, 距tomato相机系距离: {best_dist:.3f}m"
            )

            # 6. 计算stalk关键点(采摘点/最高点/最低点)与line_B
            stalk_result = self._get_stalk_key_points(best_stalk, best_stalk_mask, img_w, img_h)
            if stalk_result is None:
                response.success = False
                response.message = "stalk骨架点不足, 无法计算关键点"
                return self._publish_debug_image(response, color_cv, None)

            stalk_point_cut = stalk_result['stalk_point_cut']
            stalk_point_top = stalk_result['stalk_point_top']
            stalk_point_bottom = stalk_result['stalk_point_bottom']
            line_B_slope = stalk_result['line_B_slope']
            self.get_logger().info(
                f"stalk_point_cut={stalk_point_cut}, stalk_point_top={stalk_point_top}, "
                f"stalk_point_bottom={stalk_point_bottom}, line_B角度={math.degrees(stalk_result['line_B_angle']):.2f}°"
            )

            # 7. line_A与stem求交, 选出stem_obj
            stem_result = self._find_stem_obj(stems, yolo_response, total_masks, stalk_point_cut, line_B_slope)
            if stem_result is None:
                response.success = False
                response.message = "line_A与所有stem骨架均无交点"
                return self._publish_debug_image(response, color_cv, None)
            stem_obj, stem_intersect, stem_thin = stem_result
            self.get_logger().info(f"选定stem_obj #{stem_obj[5]}, 交点: {stem_intersect}")

            # 8. 经过stalk最高点和最低点绘制水平线, 求与stem_obj骨架的交点
            stem_point_top = self._get_stem_point_at_row(stem_thin, stalk_point_top[1], stalk_point_cut[0])
            stem_point_bottom = self._get_stem_point_at_row(stem_thin, stalk_point_bottom[1], stalk_point_cut[0])
            self.get_logger().info(f"stem_point_top={stem_point_top}, stem_point_bottom={stem_point_bottom}")

            # 9. 五个点转机械臂底座坐标系
            stem_cv_mask = self.bridge.imgmsg_to_cv2(yolo_response.masks[stem_obj[5]].mask, "mono8")
            cut_depth = self._get_valid_depth_value(depth_cv, best_stalk_mask, *stalk_point_cut)

            points_pixel = {
                'stalk_point_cut': (stalk_point_cut, best_stalk_mask),
                'stalk_point_top': (stalk_point_top, best_stalk_mask),
                'stalk_point_bottom': (stalk_point_bottom, best_stalk_mask),
                'stem_point_top': (stem_point_top, stem_cv_mask),
                'stem_point_bottom': (stem_point_bottom, stem_cv_mask),
            }
            points_base = {}
            points_cam = {}
            for name, (pixel_pt, mask_pt) in points_pixel.items():
                p_cam, p_base = self._pixel_to_base(depth_cv, mask_pt, pixel_pt, T_base_cam, fallback_depth=cut_depth)
                if p_base is None:
                    response.success = False
                    response.message = f"{name}深度值无效, 无法计算3D坐标"
                    return self._publish_debug_image(response, color_cv, None)
                points_cam[name] = p_cam
                points_base[name] = p_base
                self.get_logger().info(
                    f"{name}: 像素={pixel_pt}, 相机系=({p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f})m, "
                    f"底座系=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})m"
                )

            # 10. 计算向量A(主茎方向: stem_top->stem_bottom)、向量B(果柄方向: stalk_top->stalk_bottom)、向量C(垂直于A和B)
            vec_A = points_base['stem_point_bottom'] - points_base['stem_point_top']
            vec_B = points_base['stalk_point_bottom'] - points_base['stalk_point_top']
            norm_A = np.linalg.norm(vec_A)
            norm_B = np.linalg.norm(vec_B)
            self.get_logger().info(f"向量A(主茎方向): {vec_A}, 模长={norm_A:.4f}m")
            self.get_logger().info(f"向量B(果柄方向): {vec_B}, 模长={norm_B:.4f}m")

            if norm_A < 0.005:
                self.get_logger().warn("向量A模长过小, 使用stem骨架整体最高/最低点方向替代")
                stem_skeleton_points = np.column_stack(np.where(stem_thin > 0))
                s_top_idx = np.argmin(stem_skeleton_points[:, 0])
                s_bottom_idx = np.argmax(stem_skeleton_points[:, 0])
                s_top = (int(stem_skeleton_points[s_top_idx, 1]), int(stem_skeleton_points[s_top_idx, 0]))
                s_bottom = (int(stem_skeleton_points[s_bottom_idx, 1]), int(stem_skeleton_points[s_bottom_idx, 0]))
                _, top_base = self._pixel_to_base(depth_cv, stem_cv_mask, s_top, T_base_cam, fallback_depth=cut_depth)
                _, bottom_base = self._pixel_to_base(depth_cv, stem_cv_mask, s_bottom, T_base_cam, fallback_depth=cut_depth)
                if top_base is None or bottom_base is None:
                    response.success = False
                    response.message = "向量A退化且替代点深度无效"
                    return self._publish_debug_image(response, color_cv, None)
                vec_A = bottom_base - top_base
                norm_A = np.linalg.norm(vec_A)

            if norm_B < 1e-9:
                response.success = False
                response.message = "向量B(果柄方向)退化"
                return self._publish_debug_image(response, color_cv, None)

            vec_C = np.cross(vec_A, vec_B)
            norm_C = np.linalg.norm(vec_C)
            if norm_C < 1e-9:
                self.get_logger().warn("向量A与向量B平行, 使用主茎方向与竖直方向构造法向")
                vec_C = np.cross(vec_A, np.array([0.0, 0.0, 1.0]))
                norm_C = np.linalg.norm(vec_C)
                if norm_C < 1e-9:
                    vec_C = np.array([1.0, 0.0, 0.0])
                    norm_C = 1.0
            vec_C = vec_C / norm_C
            self.get_logger().info(f"向量C(垂直于A和B, 单位化): {vec_C}")

            # 11. 末端姿态: Z轴沿向量C(接近方向), Y轴沿向量A(主茎方向, 正交化), X=Y×Z
            z_axis = vec_C
            a_unit = vec_A / norm_A
            y_axis = a_unit - np.dot(a_unit, z_axis) * z_axis
            norm_y = np.linalg.norm(y_axis)
            if norm_y < 1e-6:
                # A与C平行时, 用底座系Z轴正交化
                fallback_y = np.array([0.0, 0.0, 1.0])
                y_axis = fallback_y - np.dot(fallback_y, z_axis) * z_axis
                norm_y = np.linalg.norm(y_axis)
                if norm_y < 1e-6:
                    y_axis = np.array([0.0, 1.0, 0.0])
                    norm_y = 1.0
            y_axis = y_axis / norm_y
            x_axis = np.cross(y_axis, z_axis)
            R_base_tool = np.column_stack([x_axis, y_axis, z_axis])

            rx, ry, rz = self.rotation_matrix_to_rpy(R_base_tool)
            cut_base = points_base['stalk_point_cut']

            self.get_logger().info(
                f"最终采摘点(底座系): X={cut_base[0]:.4f}, Y={cut_base[1]:.4f}, Z={cut_base[2]:.4f}, "
                f"RX={math.degrees(rx):.2f}°, RY={math.degrees(ry):.2f}°, RZ={math.degrees(rz):.2f}°"
            )

            response.success = True
            response.message = "成功计算近端采摘位姿"
            response.x = float(cut_base[0])
            response.y = float(cut_base[1])
            response.z = float(cut_base[2])
            response.rx = float(rx)
            response.ry = float(ry)
            response.rz = float(rz)

            debug_info = {
                'stalk': best_stalk,
                'stems': stems,
                'tomato_xy_img': tomato_xy_img,
                'stalk_point_cut': stalk_point_cut,
                'stalk_point_top': stalk_point_top,
                'stalk_point_bottom': stalk_point_bottom,
                'stem_point_top': stem_point_top,
                'stem_point_bottom': stem_point_bottom,
                'line_B_slope': line_B_slope,
                'stem_obj_idx': stem_obj[5],
            }
            return self._publish_debug_image(response, color_cv, debug_info)

        except Exception as e:
            self.get_logger().error(f"近距离检测处理错误: {str(e)}")
            response.success = False
            response.message = f"处理异常: {str(e)}"
            return response

    def _publish_debug_image(self, response, color_cv, debug_info):
        """绘制调试信息并发布, 同时返回服务响应"""
        try:
            if debug_info is None:
                combined = np.hstack((color_cv, color_cv))
                combined_msg = self.bridge.cv2_to_imgmsg(combined, "bgr8")
                self.combined_pub.publish(combined_msg)
                return response

            annotated = color_cv.copy()
            stalk = debug_info['stalk']
            cv2.rectangle(annotated, (int(stalk[0]), int(stalk[1])), (int(stalk[2]), int(stalk[3])), (0, 255, 0), 2)
            for stem in debug_info['stems']:
                color = (0, 255, 255) if stem[5] == debug_info['stem_obj_idx'] else (128, 128, 0)
                cv2.rectangle(annotated, (int(stem[0]), int(stem[1])), (int(stem[2]), int(stem[3])), color, 2)

            # tomato投影点(红色)
            cv2.circle(annotated, debug_info['tomato_xy_img'], 6, (0, 0, 255), -1)
            cv2.putText(annotated, "tomato", (debug_info['tomato_xy_img'][0] + 8, debug_info['tomato_xy_img'][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # line_B(品红): stalk最高点->最低点
            cv2.line(annotated, debug_info['stalk_point_top'], debug_info['stalk_point_bottom'], (255, 0, 255), 2)

            # line_A(绿色): 经过采摘点, 斜率与line_B相同
            cut = debug_info['stalk_point_cut']
            slope = debug_info['line_B_slope']
            h_img, w_img = annotated.shape[:2]
            if np.isinf(slope):
                cv2.line(annotated, (cut[0], 0), (cut[0], h_img - 1), (0, 255, 0), 2)
            else:
                y_at_0 = int(slope * (0 - cut[0]) + cut[1])
                y_at_w = int(slope * (w_img - 1 - cut[0]) + cut[1])
                cv2.line(annotated, (0, y_at_0), (w_img - 1, y_at_w), (0, 255, 0), 2)

            # 水平线(青色): 经过stalk最高点和最低点
            cv2.line(annotated, (0, debug_info['stalk_point_top'][1]), (w_img - 1, debug_info['stalk_point_top'][1]), (255, 255, 0), 1)
            cv2.line(annotated, (0, debug_info['stalk_point_bottom'][1]), (w_img - 1, debug_info['stalk_point_bottom'][1]), (255, 255, 0), 1)

            # 五个关键点
            key_points = [
                (debug_info['stalk_point_cut'], (0, 255, 0), "cut"),
                (debug_info['stalk_point_top'], (0, 0, 255), "s_top"),
                (debug_info['stalk_point_bottom'], (255, 0, 0), "s_bottom"),
                (debug_info['stem_point_top'], (0, 165, 255), "stem_top"),
                (debug_info['stem_point_bottom'], (255, 0, 255), "stem_bottom"),
            ]
            for pt, color, label in key_points:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, color, -1)
                cv2.putText(annotated, label, (int(pt[0]) + 8, int(pt[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            combined = np.hstack((color_cv, annotated))
            combined_msg = self.bridge.cv2_to_imgmsg(combined, "bgr8")
            self.combined_pub.publish(combined_msg)
        except Exception as e:
            self.get_logger().error(f"发布调试图像失败: {str(e)}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CloseYOLOv8Service()
        node.get_logger().info("近距离YOLOv8服务节点创建成功，开始运行...")
        rclpy.spin(node)
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"服务启动失败: {str(e)}")
        else:
            print(f"服务启动失败: {str(e)}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
