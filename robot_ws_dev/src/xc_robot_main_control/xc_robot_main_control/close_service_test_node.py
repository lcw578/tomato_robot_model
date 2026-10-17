#!/usr/bin/env python3
"""
close_img_detection 服务测试节点

相机内参：
内参：优先读 camera_info_path（ros2 topic echo /cam_hand/color/camera_info --once > info.yaml 导出的 yaml）；
没有就给 -p fx:=.. -p fy:=.. -p cx:=.. -p cy:=.. 作为参数


功能:
1. 从磁盘加载指定的彩色图、深度图(16位PNG, 单位mm)、相机内参(yaml或参数)、机械臂末端位姿
2. 持续发布彩色图、深度图、相机信息、机械臂位姿(arm_status), 供 close_yolov8_service 订阅
3. 弹出图像窗口, 鼠标点击任意点:
   - 反投影计算该点在相机坐标系下的XYZ
   - 结合手眼标定结果与机械臂末端位姿, 计算该点在机械臂底座坐标系下的XYZ
4. 按 's' 键: 使用最近一次点击点的底座系XYZ调用 close_img_detection 服务, 打印服务端返回的采摘位姿
5. 按 'q' / ESC 退出

示例:
ros2 run xc_robot_main_control close_service_test_node --ros-args \
  -p color_path:=/path/to/xxx_c.png \
  -p depth_path:=/path/to/xxx_d.png \
  -p camera_info_path:=/path/to/camera_info.yaml \
  -p arm_pose_rpy_deg:=[0.35,-0.25,0.45,0.0,0.0,90.0]
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from xc_interfaces.msg import ArmStatus
from xc_interfaces.srv import CloseImgDetection
import numpy as np
import cv2
import math
import os
import threading
import yaml
import pyrealsense2 as rs2


class CloseServiceTestNode(Node):
    def __init__(self):
        super().__init__('close_service_test_node')

        # ---------- 参数声明 ----------
        self.declare_parameter('color_path', '')
        self.declare_parameter('depth_path', '')
        self.declare_parameter('camera_info_path', '')  # ros2 topic echo 导出的CameraInfo yaml
        self.declare_parameter('fx', 0.0)
        self.declare_parameter('fy', 0.0)
        self.declare_parameter('cx', 0.0)
        self.declare_parameter('cy', 0.0)
        self.declare_parameter('depth_scale', 0.001)   # 深度图缩放: 16UC1(mm) -> m
        # 机械臂末端位姿, 二选一:
        # arm_pose: [x, y, z, qx, qy, qz, qw]
        self.declare_parameter('arm_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        # arm_pose_rpy_deg: [x, y, z, roll_deg, pitch_deg, yaw_deg]
        self.declare_parameter('arm_pose_rpy_deg', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('hand_eye_file_path', '')
        # 话题/服务名(与服务端默认保持一致)
        self.declare_parameter('color_topic', '/cam_hand/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/cam_hand/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/cam_hand/color/camera_info')
        self.declare_parameter('arm_status_topic', 'arm_status')
        self.declare_parameter('service_name', 'close_img_detection')
        self.declare_parameter('publish_rate', 10.0)

        self.color_path = self.get_parameter('color_path').value
        self.depth_path = self.get_parameter('depth_path').value
        self.camera_info_path = self.get_parameter('camera_info_path').value
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.arm_status_topic = self.get_parameter('arm_status_topic').value
        self.service_name = self.get_parameter('service_name').value
        publish_rate = float(self.get_parameter('publish_rate').value)

        self.bridge = CvBridge()

        # ---------- 加载图像 ----------
        if not os.path.exists(self.color_path):
            raise FileNotFoundError(f"彩色图不存在: {self.color_path}")
        if not os.path.exists(self.depth_path):
            raise FileNotFoundError(f"深度图不存在: {self.depth_path}")

        self.color_cv = cv2.imread(self.color_path, cv2.IMREAD_COLOR)
        self.depth_cv = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
        if self.color_cv is None:
            raise ValueError(f"无法读取彩色图: {self.color_path}")
        if self.depth_cv is None:
            raise ValueError(f"无法读取深度图: {self.depth_path}")
        if self.depth_cv.dtype != np.uint16:
            self.get_logger().warn(f"深度图类型为 {self.depth_cv.dtype}, 期望 uint16(mm)")
        self.img_h, self.img_w = self.color_cv.shape[:2]
        self.get_logger().info(f"彩色图: {self.color_path} ({self.img_w}x{self.img_h})")
        self.get_logger().info(f"深度图: {self.depth_path} dtype={self.depth_cv.dtype}")

        # ---------- 相机内参 ----------
        self.intrinsics = self._load_intrinsics()
        if self.intrinsics is None:
            raise RuntimeError("相机内参加载失败: 请提供 camera_info_path 或 fx/fy/cx/cy 参数")

        # ---------- 机械臂末端位姿 ----------
        self.arm_position, self.arm_quaternion, self.arm_rpy = self._load_arm_pose()

        # ---------- 手眼标定矩阵(眼在手上: 相机系 -> 末端系) ----------
        self.hand_eye_matrix = self._load_hand_eye_matrix(
            self.get_parameter('hand_eye_file_path').value
        )
        if self.hand_eye_matrix is None:
            raise RuntimeError("手眼标定矩阵加载失败")
        # T_base_cam = T_base_end(当前机械臂位姿) @ T_end_cam(手眼标定)
        self.T_base_end = self._pose_to_matrix(self.arm_position, self.arm_quaternion)
        self.T_base_cam = self.T_base_end @ self.hand_eye_matrix
        self.get_logger().info(f"T_base_end(机械臂末端位姿):\n{self.T_base_end}")
        self.get_logger().info(f"T_base_cam(相机在底座系位姿):\n{self.T_base_cam}")

        # ---------- 发布器 ----------
        self.color_pub = self.create_publisher(Image, self.color_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.arm_status_pub = self.create_publisher(ArmStatus, self.arm_status_topic, 10)

        # 预构建消息(图像只加载一次)
        self.color_msg = self.bridge.cv2_to_imgmsg(self.color_cv, encoding='bgr8')
        self.depth_msg = self.bridge.cv2_to_imgmsg(self.depth_cv, encoding='16UC1')
        self.camera_info_msg = self._build_camera_info_msg()
        self.arm_status_msg = self._build_arm_status_msg()

        # ---------- 定时发布(保证服务端同步器/内参/位姿持续可获取) ----------
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        # ---------- 服务客户端 ----------
        self.close_client = self.create_client(CloseImgDetection, self.service_name)
        self.pending_future = None

        # 最近点击结果
        self.last_click_pixel = None
        self.last_click_cam = None
        self.last_click_base = None

        self.get_logger().info("测试节点已启动: 点击图像中的点计算XYZ, 按 's' 调用服务, 按 'q' 退出")

    # ---------- 加载工具 ----------

    def _load_intrinsics(self):
        """从yaml或参数构建rs2.intrinsics"""
        intr = rs2.intrinsics()
        intr.width = self.img_w
        intr.height = self.img_h
        intr.model = rs2.distortion.brown_conrady
        intr.coeffs = [0.0] * 5

        if self.camera_info_path and os.path.exists(self.camera_info_path):
            with open(self.camera_info_path, 'r') as f:
                info = yaml.safe_load(f)
            intr.width = int(info.get('width', self.img_w))
            intr.height = int(info.get('height', self.img_h))
            k = info.get('k')
            if k and len(k) >= 9:
                intr.fx = float(k[0])
                intr.fy = float(k[4])
                intr.ppx = float(k[2])
                intr.ppy = float(k[5])
            d = info.get('d')
            if d:
                intr.coeffs = [float(v) for v in d][:5]
                while len(intr.coeffs) < 5:
                    intr.coeffs.append(0.0)
            model = str(info.get('distortion_model', ''))
            if model == 'kannala_brandt4':
                intr.model = rs2.distortion.kannala_brandt4
            self.get_logger().info(f"相机内参从yaml加载: {self.camera_info_path}")
        else:
            fx = float(self.get_parameter('fx').value)
            fy = float(self.get_parameter('fy').value)
            cx = float(self.get_parameter('cx').value)
            cy = float(self.get_parameter('cy').value)
            if fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
                self.get_logger().error("未提供 camera_info_path 且 fx/fy/cx/cy 参数无效")
                return None
            intr.fx = fx
            intr.fy = fy
            intr.ppx = cx
            intr.ppy = cy
            self.get_logger().info("相机内参从参数加载")

        self.get_logger().info(
            f"内参: {intr.width}x{intr.height}, fx={intr.fx:.2f}, fy={intr.fy:.2f}, "
            f"cx={intr.ppx:.2f}, cy={intr.ppy:.2f}"
        )
        return intr

    def _load_arm_pose(self):
        """机械臂末端位姿: 优先 arm_pose(四元数), 否则 arm_pose_rpy_deg(欧拉角, 度)"""
        arm_pose = list(self.get_parameter('arm_pose').value)
        arm_pose_rpy = list(self.get_parameter('arm_pose_rpy_deg').value)

        position = [0.0, 0.0, 0.0]
        quaternion = [0.0, 0.0, 0.0, 1.0]  # qx,qy,qz,qw
        rpy_rad = [0.0, 0.0, 0.0]

        # qw默认1.0, 用 qx/qy/qz 非零 或 qw 偏离1.0 判断是否显式设置四元数
        if abs(arm_pose[3]) > 1e-9 or abs(arm_pose[4]) > 1e-9 or abs(arm_pose[5]) > 1e-9 or abs(arm_pose[6] - 1.0) > 1e-9:
            position = [float(arm_pose[0]), float(arm_pose[1]), float(arm_pose[2])]
            quaternion = [float(arm_pose[3]), float(arm_pose[4]),
                          float(arm_pose[5]), float(arm_pose[6])]
            rpy_rad = list(self.quaternion_to_rpy(*quaternion))
            self.get_logger().info(
                f"机械臂位姿(四元数): pos={position}, q=[{quaternion[0]:.4f},"
                f"{quaternion[1]:.4f},{quaternion[2]:.4f},{quaternion[3]:.4f}]"
            )
        elif any(abs(v) > 1e-9 for v in arm_pose_rpy):
            position = [float(arm_pose_rpy[0]), float(arm_pose_rpy[1]), float(arm_pose_rpy[2])]
            rpy_rad = [math.radians(float(arm_pose_rpy[3])),
                       math.radians(float(arm_pose_rpy[4])),
                       math.radians(float(arm_pose_rpy[5]))]
            quaternion = list(self.rpy_to_quaternion(*rpy_rad))
            self.get_logger().info(
                f"机械臂位姿(欧拉角deg): pos={position}, "
                f"rpy_deg=[{arm_pose_rpy[3]:.2f},{arm_pose_rpy[4]:.2f},{arm_pose_rpy[5]:.2f}]"
            )
        else:
            self.get_logger().info("未设置机械臂末端位姿, 使用单位姿态(原点, 无旋转)")
        return position, quaternion, rpy_rad

    def _load_hand_eye_matrix(self, param_path):
        """加载手眼标定矩阵(4x4), 与服务端相同的查找顺序"""
        candidates = []
        if param_path:
            candidates.append(param_path)
        candidates.append('/home/xc_home_wsl/code/test/ROS2/robot_ws/src/hand_eye_calibration_python/config/hand/calibration_results.txt')
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory('hand_eye_calibration_python')
            candidates.insert(0, os.path.join(share_dir, 'config', 'hand', 'calibration_results.txt'))
        except Exception:
            pass

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as file:
                        content = file.read().strip()
                    lines = content.split('\n')
                    matrix = np.array([[float(v) for v in line.split(',')] for line in lines],
                                      dtype=np.float64)
                    if matrix.shape != (4, 4):
                        raise ValueError(f"矩阵尺寸错误: {matrix.shape}")
                    self.get_logger().info(f"手眼标定矩阵加载成功: {path}")
                    self.get_logger().info(f"\n{matrix}")
                    return matrix
                except Exception as e:
                    self.get_logger().error(f"解析手眼标定文件失败: {path}, 错误: {e}")
        self.get_logger().error("未找到有效的手眼标定文件, 请通过 hand_eye_file_path 参数指定")
        return None

    # ---------- 消息构建 ----------

    def _build_camera_info_msg(self):
        msg = CameraInfo()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.width = self.intrinsics.width
        msg.height = self.intrinsics.height
        msg.distortion_model = 'plumb_bob'
        msg.d = [float(v) for v in self.intrinsics.coeffs]
        k = [0.0] * 9
        k[0] = self.intrinsics.fx
        k[2] = self.intrinsics.ppx
        k[4] = self.intrinsics.fy
        k[5] = self.intrinsics.ppy
        k[8] = 1.0
        msg.k = k
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        p = [0.0] * 12
        p[0] = self.intrinsics.fx
        p[2] = self.intrinsics.ppx
        p[5] = self.intrinsics.fy
        p[6] = self.intrinsics.ppy
        p[10] = 1.0
        msg.p = p
        return msg

    def _build_arm_status_msg(self):
        msg = ArmStatus()
        msg.position.x = self.arm_position[0]
        msg.position.y = self.arm_position[1]
        msg.position.z = self.arm_position[2]
        msg.orientation.x = self.arm_quaternion[0]
        msg.orientation.y = self.arm_quaternion[1]
        msg.orientation.z = self.arm_quaternion[2]
        msg.orientation.w = self.arm_quaternion[3]
        msg.rx = float(self.arm_rpy[0])
        msg.ry = float(self.arm_rpy[1])
        msg.rz = float(self.arm_rpy[2])
        return msg

    def timer_callback(self):
        stamp = self.get_clock().now().to_msg()
        self.color_msg.header.stamp = stamp
        self.depth_msg.header.stamp = stamp
        self.color_msg.header.frame_id = 'camera_color_optical_frame'
        self.depth_msg.header.frame_id = 'camera_color_optical_frame'
        self.camera_info_msg.header.stamp = stamp
        self.arm_status_msg.header.stamp = stamp
        self.color_pub.publish(self.color_msg)
        self.depth_pub.publish(self.depth_msg)
        self.camera_info_pub.publish(self.camera_info_msg)
        self.arm_status_pub.publish(self.arm_status_msg)

    # ---------- 坐标变换 ----------

    @staticmethod
    def rpy_to_quaternion(roll, pitch, yaw):
        """固定轴XYZ欧拉角(弧度) -> 四元数(x,y,z,w), q = qz*qy*qx"""
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return qx, qy, qz, qw

    @staticmethod
    def quaternion_to_rpy(qx, qy, qz, qw):
        """四元数(x,y,z,w) -> 固定轴XYZ欧拉角(弧度)"""
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)]
        ])
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy >= 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    @staticmethod
    def _pose_to_matrix(position, quaternion):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = position[0]
        T[1, 3] = position[1]
        T[2, 3] = position[2]
        qx, qy, qz, qw = quaternion
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        T[:3, :3] = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)]
        ])
        return T

    def _get_depth_at(self, x, y, max_window=15):
        """取像素深度(米), 0深度时在邻域内搜索有效深度均值"""
        depth_m = float(self.depth_cv[y, x]) * self.depth_scale
        if depth_m > 0:
            return depth_m
        h, w = self.depth_cv.shape[:2]
        for win in range(3, max_window + 1, 2):
            half = win // 2
            x1, x2 = max(0, x - half), min(w, x + half + 1)
            y1, y2 = max(0, y - half), min(h, y + half + 1)
            roi = self.depth_cv[y1:y2, x1:x2].astype(np.float32)
            valid = roi[roi > 0]
            if len(valid) > 0:
                return float(np.mean(valid)) * self.depth_scale
        return 0.0

    def handle_click(self, x, y):
        """鼠标点击: 像素 -> 相机系XYZ -> 底座系XYZ"""
        x = int(max(0, min(x, self.img_w - 1)))
        y = int(max(0, min(y, self.img_h - 1)))

        depth_m = self._get_depth_at(x, y)
        if depth_m <= 0.05:
            self.get_logger().warn(f"点击点({x},{y})无有效深度")
            return

        point_cam = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_m)
        cam_xyz = np.array([point_cam[0], point_cam[1], point_cam[2]], dtype=np.float64)
        homogeneous = np.append(cam_xyz, 1.0)
        base_xyz = (self.T_base_cam @ homogeneous)[:3]

        self.last_click_pixel = (x, y)
        self.last_click_cam = cam_xyz
        self.last_click_base = base_xyz

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"点击像素: ({x}, {y}), 深度: {depth_m:.4f}m")
        self.get_logger().info(
            f"相机坐标系XYZ: X={cam_xyz[0]:.4f}m, Y={cam_xyz[1]:.4f}m, Z={cam_xyz[2]:.4f}m"
        )
        self.get_logger().info(
            f"底座坐标系XYZ: X={base_xyz[0]:.4f}m, Y={base_xyz[1]:.4f}m, Z={base_xyz[2]:.4f}m"
        )
        self.get_logger().info("按 's' 以此底座XYZ调用 close_img_detection 服务")

    def call_service(self):
        """使用最近一次点击的底座系XYZ调用服务"""
        if self.last_click_base is None:
            self.get_logger().warn("尚未点击任何点, 请先在图像窗口中点击目标点")
            return

        if not self.close_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f"服务 '{self.service_name}' 不可用, 请先启动 close_yolov8_service"
            )
            return

        req = CloseImgDetection.Request()
        req.x = float(self.last_click_base[0])
        req.y = float(self.last_click_base[1])
        req.z = float(self.last_click_base[2])
        self.get_logger().info(f"调用服务 {self.service_name}: tomato_xyz(base)=({req.x:.4f}, {req.y:.4f}, {req.z:.4f})")
        self.pending_future = self.close_client.call_async(req)
        self.pending_future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info("=" * 60)
            if resp.success:
                self.get_logger().info(
                    f"服务成功: 采摘点(底座系) X={resp.x:.4f}, Y={resp.y:.4f}, Z={resp.z:.4f}, "
                    f"RX={math.degrees(resp.rx):.2f}°, RY={math.degrees(resp.ry):.2f}°, RZ={math.degrees(resp.rz):.2f}°"
                )
            else:
                self.get_logger().warn(f"服务失败: {resp.message}")
        except Exception as e:
            self.get_logger().error(f"服务响应处理异常: {e}")

    # ---------- OpenCV窗口 ----------

    def run_gui(self):
        window_name = 'close_service_test (click: pick point, s: call service, q: quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.handle_click(x, y)

        cv2.setMouseCallback(window_name, on_mouse)

        self.get_logger().info("操作: 左键点击=计算该点XYZ, 's'=调用服务, 'q'/ESC=退出")
        while rclpy.ok():
            display = self.color_cv.copy()
            if self.last_click_pixel is not None:
                cv2.circle(display, self.last_click_pixel, 6, (0, 0, 255), -1)
                lines = [
                    f"pixel: {self.last_click_pixel}",
                    f"cam : ({self.last_click_cam[0]:.3f},{self.last_click_cam[1]:.3f},{self.last_click_cam[2]:.3f})m",
                    f"base: ({self.last_click_base[0]:.3f},{self.last_click_base[1]:.3f},{self.last_click_base[2]:.3f})m",
                    "press 's' to call service",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(display, text, (10, 30 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, "click a point...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self.call_service()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CloseServiceTestNode()
        # 后台线程spin, 主线程跑OpenCV窗口
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        node.run_gui()
    except Exception as e:
        print(f"测试节点启动失败: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
