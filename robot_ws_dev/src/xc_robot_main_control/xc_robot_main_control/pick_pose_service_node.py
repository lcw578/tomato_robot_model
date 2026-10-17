#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import time
import threading
from vision_tomato_interfaces.srv import GetPickPoses
from vision_tomato_interfaces.msg import PickPoseCandidate
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header

import pyrealsense2 as rs2
from ultralytics import YOLO

from xc_robot_main_control.stemGrowthAndPoints import StemGrowthAndPoints


class PickPoseServiceNode(Node):
    def __init__(self):
        super().__init__('pick_pose_service_node')

        self.declare_parameter('color_topic', '/cam_hand/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/cam_hand/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/cam_hand/color/camera_info')
        self.declare_parameter('service_name', '/vision/tomato/get_pick_poses')
        self.declare_parameter('time_threshold', 0.05)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('yolo_model_path', '/home/xc/code/test/ros2/robot_ws/src/xc_robot_main_control/weights/best.pt')

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.time_threshold = self.get_parameter('time_threshold').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.yolo_model_path = self.get_parameter('yolo_model_path').value

        self.bridge = CvBridge()

        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.frame_lock = threading.Lock()

        self.intrinsics = None
        self.intrinsics_ready = False

        self.mutex_cb_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_cb_group = ReentrantCallbackGroup()

        self.growth_point_calculator = StemGrowthAndPoints()

        # 初始化 YOLOv8 模型
        self._init_yolov8_model()

        self._setup_subscriptions()
        self._setup_services()

        self.get_logger().info(f"采摘点服务节点已启动")
        self.get_logger().info(f"  彩色图话题: {self.color_topic}")
        self.get_logger().info(f"  深度图话题: {self.depth_topic}")
        self.get_logger().info(f"  服务名: {self.service_name}")
        self.get_logger().info(f"  时间戳阈值: {self.time_threshold}s")
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

    def _setup_subscriptions(self):
        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.image_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            1
        )

    def _setup_services(self):
        self.pick_pose_service = self.create_service(
            GetPickPoses,
            self.service_name,
            self.handle_get_pick_poses,
            callback_group=self.reentrant_cb_group
        )
        self.get_logger().info(f"采摘点服务已创建: {self.service_name}")

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

            # 构建标准化的检测结果对象
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

                    # 获取边界框
                    x1, y1, x2, y2 = map(float, box.xyxy[0])

                    detection_result.class_ids.append(cls)
                    detection_result.class_names.append(class_name)
                    detection_result.confidences.append(conf)
                    detection_result.bboxes.extend([x1, y1, x2, y2])

                    # 获取mask（如果是分割模型）
                    if result.masks is not None and i < len(result.masks):
                        mask_data = result.masks[i].data.cpu().numpy().astype(bool)[0]
                        # resize到原图尺寸
                        mask_data = cv2.resize(
                            mask_data.astype(np.uint8) * 255,
                            (color_cv.shape[1], color_cv.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        # 转换为Image消息格式
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

    @staticmethod
    def calculate_iou_stems(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        return intersection_area / box1_area if box1_area > 0 else 0

    def classify_detections(self, response, num_boxes, total_masks):
        stems, tomatos, trunks = [], [], []
        for i in range(num_boxes):
            idx = i * 4
            x1 = response.bboxes[idx]
            y1 = response.bboxes[idx + 1]
            x2 = response.bboxes[idx + 2]
            y2 = response.bboxes[idx + 3]
            conf = response.confidences[i]
            class_name = response.class_names[i]

            if class_name.lower() == "stem":
                if i < total_masks:
                    stems.append((x1, y1, x2, y2, conf, i))
                else:
                    self.get_logger().warn(f"检测框{i}为果梗，但无对应掩码")
            elif class_name.lower() == "tomato":
                if i < total_masks:
                    tomatos.append((x1, y1, x2, y2, conf, i))
                else:
                    self.get_logger().warn(f"检测框{i}为番茄，但无对应掩码")
            elif class_name.lower() == "trunk":
                if i < total_masks:
                    trunks.append((x1, y1, x2, y2, conf, i))
                else:
                    self.get_logger().warn(f"检测框{i}为树干，但无对应掩码")

        return stems, tomatos, trunks

    def combine_boxes(self, img, stems, tomatos, img_size, iou_threshold=0):
        matched_stems = set()
        matched_tomatos = set()
        matched_pairs = []
        matched_rectangles = []

        for s_idx, stem in enumerate(stems):
            if s_idx in matched_stems:
                continue

            best_tomato = None
            best_tomato_idx = None
            best_iou = 0
            stem_bottom = stem[3]

            for t_idx, tomato in enumerate(tomatos):
                if t_idx in matched_tomatos or stem_bottom > (tomato[1] + tomato[3]) / 2:
                    continue

                current_iou = self.calculate_iou_stems(stem, tomato)

                if current_iou > best_iou and current_iou > iou_threshold:
                    best_iou = current_iou
                    best_tomato = tomato
                    best_tomato_idx = t_idx

            if best_tomato is not None:
                x1 = int(min(stem[0], best_tomato[0]))
                y1 = int(min(stem[1], best_tomato[1]))
                x2 = int(max(stem[2], best_tomato[2]))
                y2 = int(max(stem[3], best_tomato[3]))

                h_img, w_img = img.shape[:2]
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img - 1))
                y2 = max(0, min(y2, h_img - 1))

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                matched_rectangles.append([x1, y1, x2, y2])

                matched_stems.add(s_idx)
                matched_tomatos.add(best_tomato_idx)

                matched_pairs.append((stem, best_tomato))

        return img, matched_pairs, matched_rectangles

    def _get_valid_depth_value(self, depth_image, stem_cv_mask, x, y):
        depth_value = depth_image[y, x] * self.depth_scale

        if depth_value < 0.20 or np.isnan(depth_value) or depth_value > 1.0:
            depth_value = self._get_neighborhood_depth(
                depth_image, stem_cv_mask, x, y, max_window_size=45
            )

            if depth_value <= 0.20 or np.isnan(depth_value) or depth_value >= 1.0:
                depth_value = self._get_mask_average_depth(stem_cv_mask, depth_image)

        return depth_value

    def _compute_pick_point_3d(self, color_cv, depth_cv, yolo_response):
        if not self.intrinsics_ready:
            self.get_logger().error("相机内参未就绪，无法计算3D坐标")
            return None

        total_masks = len(yolo_response.masks) if hasattr(yolo_response, 'masks') else 0
        num_boxes = len(yolo_response.class_ids)

        if total_masks == 0 or num_boxes == 0:
            self.get_logger().warn("YOLO未检测到任何目标")
            return None

        h, w = color_cv.shape[:2]

        stems, tomatos, trunks = self.classify_detections(yolo_response, num_boxes, total_masks)
        stem_count = len(stems)
        tomato_count = len(tomatos)
        trunks_count = len(trunks)

        self.get_logger().info(
            f"果梗数量: {stem_count} | 番茄数量: {tomato_count} | 主茎数量: {trunks_count}"
        )

        if stem_count == 0:
            self.get_logger().warn("未检测到果梗，无法计算采摘点")
            return None

        if tomato_count == 0:
            self.get_logger().warn("未检测到番茄，无法匹配")
            return None

        matched_img, matched_pairs, matched_rectangles = self.combine_boxes(
            color_cv.copy(), stems, tomatos, (h, w)
        )

        matched_pairs_count = len(matched_pairs)
        self.get_logger().info(f"匹配数量: {matched_pairs_count}")

        if matched_pairs_count == 0:
            self.get_logger().warn("果梗与番茄匹配失败")
            return None

        best_candidate = None
        best_score = -1

        for pair_idx, (stem, tomato) in enumerate(matched_pairs):
            stem_idx = stem[5]
            tomato_idx = tomato[5]

            self.get_logger().info(
                f"处理匹配对 #{pair_idx + 1}/{matched_pairs_count}: "
                f"果梗索引={stem_idx}, 番茄索引={tomato_idx}"
            )

            if stem_idx >= total_masks or tomato_idx >= total_masks:
                self.get_logger().error("索引超出掩码范围")
                continue

            try:
                stem_mask_msg = yolo_response.masks[stem_idx]
                stem_cv_mask = self.bridge.imgmsg_to_cv2(stem_mask_msg.mask, "mono8")

                tomato_mask_msg = yolo_response.masks[tomato_idx]
                tomato_cv_mask = self.bridge.imgmsg_to_cv2(tomato_mask_msg.mask, "mono8")
            except Exception as e:
                self.get_logger().error(f"解析掩码失败: {e}")
                continue

            stem_center_x = (stem[0] + stem[2]) / 2
            tomato_center_x = (tomato[0] + tomato[2]) / 2

            stem_width = abs(stem[2] - stem[0])
            stem_height = abs(stem[3] - stem[1])
            stem_aspect_ratio = stem_width / stem_height if stem_height != 0 else 0

            try:
                (growth_direction, is_growth_down, picking_point,
                 rad, skeleton_leftmost_point, skeleton_rightmost_point,
                 skeleton_topmost_point, tangent_angle) = \
                    self.growth_point_calculator.determine_growth_direction_and_picking_point(
                        pair_idx,
                        stem,
                        tomato,
                        stem_cv_mask,
                        tomato_cv_mask,
                        color_cv,
                        stem_center_x,
                        tomato_center_x,
                        stem_aspect_ratio,
                        w,
                        h
                    )
            except Exception as e:
                self.get_logger().error(f"计算采摘点时出错: {e}")
                continue

            if picking_point is None or skeleton_leftmost_point is None or skeleton_rightmost_point is None:
                self.get_logger().warn(f"匹配对 #{pair_idx + 1} 无法获取采摘点或果梗边界点")
                continue

            x, y = picking_point
            x = int(max(0, min(x, w - 1)))
            y = int(max(0, min(y, h - 1)))

            depth_value = self._get_valid_depth_value(depth_cv, stem_cv_mask, x, y)

            if depth_value <= 0.20 or depth_value >= 1.0 or np.isnan(depth_value):
                self.get_logger().warn(f"匹配对 #{pair_idx + 1} 无有效深度值")
                continue

            point_3d = rs2.rs2_deproject_pixel_to_point(
                self.intrinsics, [x, y], depth_value
            )

            score = stem[4] + tomato[4]

            candidate = PickPoseCandidate()
            candidate.header = Header()
            candidate.header.stamp = self.get_clock().now().to_msg()
            candidate.header.frame_id = 'camera_color_optical_frame'
            candidate.candidate_id = 0
            candidate.image_u = float(x)
            candidate.image_v = float(y)
            candidate.depth = depth_value
            candidate.pose_xyzrpy_camera = [
                float(point_3d[0]),
                float(point_3d[1]),
                float(point_3d[2]),
                0.0, 0.0, 0.0
            ]
            candidate.pose_camera = PoseStamped()
            candidate.pose_camera.header = candidate.header
            candidate.pose_camera.pose.position.x = float(point_3d[0])
            candidate.pose_camera.pose.position.y = float(point_3d[1])
            candidate.pose_camera.pose.position.z = float(point_3d[2])
            candidate.pose_camera.pose.orientation.w = 1.0
            candidate.score = score
            candidate.maturity = 'unknown'
            candidate.has_pose_base = False
            candidate.reason = f"stem_idx={stem_idx}, tomato_idx={tomato_idx}, growth={growth_direction}"

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def handle_get_pick_poses(self, request, response):
        self.get_logger().info("收到采摘点请求")

        color_msg, depth_msg = self._get_latest_valid_frames(max_wait=2.0)
        if color_msg is None or depth_msg is None:
            response.success = False
            response.message = "无法获取有效的彩色/深度图像对"
            return response

        if not self.intrinsics_ready:
            response.success = False
            response.message = "相机内参未就绪"
            return response

        try:
            color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            response.success = False
            response.message = f"图像转换失败: {str(e)}"
            return response

        # 直接使用YOLOv8进行检测
        yolo_response = self._detect_with_yolov8(color_cv)
        if yolo_response is None:
            response.success = False
            response.message = "YOLO检测失败"
            return response

        best_candidate = self._compute_pick_point_3d(color_cv, depth_cv, yolo_response)

        if best_candidate is None:
            response.success = False
            response.message = "未能计算出有效的采摘点"
            return response

        response.success = True
        response.message = "成功获取采摘点"
        response.best = best_candidate

        self.get_logger().info(
            f"采摘点计算完成: "
            f"相机坐标系=({best_candidate.pose_xyzrpy_camera[0]:.3f}, "
            f"{best_candidate.pose_xyzrpy_camera[1]:.3f}, "
            f"{best_candidate.pose_xyzrpy_camera[2]:.3f})m, "
            f"深度={best_candidate.depth:.3f}m"
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PickPoseServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
