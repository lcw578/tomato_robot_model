from dataclasses import dataclass
import importlib
import sys
import types

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from vision_tomato_interfaces.msg import PickPoseCandidate
from vision_tomato_interfaces.srv import GetPickPoses

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

def create_open3d_compat_stubs():
    plotly = types.ModuleType('plotly')
    plotly_go = types.ModuleType('plotly.graph_objects')
    plotly.graph_objects = plotly_go
    sys.modules['plotly'] = plotly
    sys.modules['plotly.graph_objects'] = plotly_go

    dash = types.ModuleType('dash')
    dash.html = types.ModuleType('dash.html')
    dash.dcc = types.ModuleType('dash.dcc')

    class Dash:
        pass

    dash.Dash = Dash
    sys.modules['dash'] = dash
    sys.modules['dash.html'] = dash.html
    sys.modules['dash.dcc'] = dash.dcc

    sys.modules['open3d.ml'] = types.ModuleType('open3d.ml')


def clear_open3d_modules():
    for name in list(sys.modules):
        if name == 'open3d' or name.startswith('open3d.'):
            sys.modules.pop(name, None)


def import_open3d_with_fallback():
    clear_open3d_modules()
    create_open3d_compat_stubs()
    try:
        module = importlib.import_module('open3d')
        return module, 'compat mode with stubbed plotly/dash/ml'
    except Exception as compat_exc:
        return None, repr(compat_exc)


o3d, OPEN3D_IMPORT_ERROR = import_open3d_with_fallback()


@dataclass
class ServiceState:
    request_pending: bool = False
    success: bool | None = None
    message: str = 'waiting for first service call'
    best: PickPoseCandidate | None = None


class PickPoseVisualizer(Node):
    def __init__(self):
        super().__init__('pick_pose_visualizer')
        self.declare_parameter('service_name', '/vision/tomato/get_pick_poses')
        self.declare_parameter('color_topic', '/vision/tomato/crop/color/image_raw')
        self.declare_parameter('depth_topic', '/vision/tomato/crop/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/vision/tomato/crop/camera_info')
        self.declare_parameter('segment_debug_topic', '/vision/tomato/pick_pose/segment_debug_image')
        self.declare_parameter('pose_debug_topic', '/vision/tomato/pick_pose/debug_image')
        self.declare_parameter('service_timeout_sec', 5.0)
        self.declare_parameter('return_debug_images', True)
        self.declare_parameter('window_name', 'vision_pose_generate_client')
        self.declare_parameter('open3d_window_name', 'vision_pose_generate_pose')
        self.declare_parameter('panel_width', 480)
        self.declare_parameter('display_fps', 10.0)
        self.declare_parameter('enable_open3d', True)
        self.declare_parameter('target_cloud_radius_px', 64)
        self.declare_parameter('target_cloud_depth_tolerance_m', 0.060)
        self.declare_parameter('target_cloud_use_rgb', True)
        self.declare_parameter('lock_open3d_camera_view', True)
        self.declare_parameter('open3d_view_zoom', 0.7)
        self.declare_parameter('open3d_view_forward_offset_m', 0.26)

        self.service_name = str(self.get_parameter('service_name').value)
        self.color_topic = str(self.get_parameter('color_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.segment_debug_topic = str(self.get_parameter('segment_debug_topic').value)
        self.pose_debug_topic = str(self.get_parameter('pose_debug_topic').value)
        self.service_timeout_sec = float(self.get_parameter('service_timeout_sec').value)
        self.return_debug_images = bool(self.get_parameter('return_debug_images').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.open3d_window_name = str(self.get_parameter('open3d_window_name').value)
        self.panel_width = max(240, int(self.get_parameter('panel_width').value))
        self.display_fps = max(1.0, float(self.get_parameter('display_fps').value))
        self.enable_open3d = bool(self.get_parameter('enable_open3d').value)
        self.target_cloud_radius_px = max(4, int(self.get_parameter('target_cloud_radius_px').value))
        self.target_cloud_depth_tolerance_m = float(
            self.get_parameter('target_cloud_depth_tolerance_m').value
        )
        self.target_cloud_use_rgb = bool(self.get_parameter('target_cloud_use_rgb').value)
        self.lock_open3d_camera_view = bool(self.get_parameter('lock_open3d_camera_view').value)
        self.open3d_view_zoom = float(self.get_parameter('open3d_view_zoom').value)
        self.open3d_view_forward_offset_m = float(
            self.get_parameter('open3d_view_forward_offset_m').value
        )

        if cv2 is None or np is None:
            raise RuntimeError('opencv-python and numpy are required for pick_pose_visualizer')

        self.bridge = CvBridge()
        self.client = self.create_client(GetPickPoses, self.service_name)
        self.state = ServiceState()
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_segment_debug = None
        self.latest_pose_debug = None
        self.last_trigger_id = 0
        self.result_trigger_id = 0

        self.create_subscription(Image, self.color_topic, self.color_callback, 5)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 5)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 5)
        self.create_subscription(Image, self.segment_debug_topic, self.segment_debug_callback, 5)
        self.create_subscription(Image, self.pose_debug_topic, self.pose_debug_callback, 5)

        self.display_timer = self.create_timer(1.0 / self.display_fps, self.update_display)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.init_open3d()
        self.get_logger().info(
            f'pick_pose_visualizer started: color={self.color_topic}, service={self.service_name}'
        )

    def init_open3d(self):
        self.o3d_vis = None
        self.o3d_pick_frame = None
        self.o3d_link = None
        self.o3d_target_cloud = None
        if not self.enable_open3d:
            return
        if o3d is None:
            detail = f': {OPEN3D_IMPORT_ERROR}' if OPEN3D_IMPORT_ERROR else ''
            self.get_logger().warning(f'open3d is unavailable, 3D pose view disabled{detail}')
            return
        if OPEN3D_IMPORT_ERROR:
            self.get_logger().warning(f'open3d compatibility import enabled: {OPEN3D_IMPORT_ERROR}')

        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window(
            window_name=self.open3d_window_name,
            width=960,
            height=720,
            visible=True,
        )
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.o3d_vis.add_geometry(world)
        render_option = self.o3d_vis.get_render_option()
        render_option.background_color = np.asarray([0.08, 0.08, 0.10])
        render_option.line_width = 4.0
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def color_callback(self, msg):
        self.latest_color = msg

    def segment_debug_callback(self, msg):
        self.latest_segment_debug = msg

    def pose_debug_callback(self, msg):
        self.latest_pose_debug = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def trigger_service_call(self):
        if self.state.request_pending:
            self.get_logger().warning('service request is already pending')
            return
        if not self.client.wait_for_service(timeout_sec=min(0.1, self.service_timeout_sec)):
            self.state.success = None
            self.state.message = f'service not ready: {self.service_name}'
            return

        request = GetPickPoses.Request()
        request.return_debug_images = self.return_debug_images
        future = self.client.call_async(request)
        self.last_trigger_id += 1
        self.state.request_pending = True
        self.state.message = f'service request sent by keyboard trigger #{self.last_trigger_id}'
        future.add_done_callback(self.handle_service_result)

    def handle_service_result(self, future):
        self.state.request_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.state.success = False
            self.state.best = None
            self.state.message = f'service call failed: {exc}'
            self.get_logger().error(self.state.message)
            return

        self.state.success = bool(response.success)
        self.state.message = response.message or 'ok'
        self.state.best = response.best if response.success else None
        self.result_trigger_id = self.last_trigger_id
        if response.success:
            self.get_logger().info(
                f'pick pose success: candidate={response.best.candidate_id} '
                f'score={response.best.score:.3f} maturity={response.best.maturity}'
            )
            self.update_open3d_pose(response.best)
        else:
            self.get_logger().warning(f'pick pose unsuccessful: {response.message}')
            self.clear_open3d_pose()

    def image_msg_to_bgr(self, msg):
        if msg is None:
            return None
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return np.ascontiguousarray(image)

    def make_placeholder(self, title, detail):
        panel = np.zeros((360, self.panel_width, 3), dtype=np.uint8)
        self.put_text(panel, title, (16, 40), scale=0.9, color=(0, 220, 255))
        self.put_text(panel, detail, (16, 86), scale=0.7, color=(180, 180, 180))
        return panel

    def fit_panel(self, image, title):
        if image is None:
            return self.make_placeholder(title, 'no image')

        height, width = image.shape[:2]
        scale = self.panel_width / float(width)
        panel_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (self.panel_width, panel_height))
        self.put_text(resized, title, (12, 28), scale=0.8, color=(0, 255, 255))
        return resized

    def put_text(self, image, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def pad_to_height(self, image, target_height):
        height, width = image.shape[:2]
        if height >= target_height:
            return image
        padded = np.zeros((target_height, width, 3), dtype=np.uint8)
        padded[:height, :width] = image
        return padded

    def overlay_pick_result(self, image):
        if image is None:
            return None
        if self.state.success and self.state.best is not None:
            best = self.state.best
            center = (int(round(best.image_u)), int(round(best.image_v)))
            cv2.circle(image, center, 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(
                image,
                center,
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            self.put_text(
                image,
                f'pick #{self.result_trigger_id} score={best.score:.3f}',
                (12, image.shape[0] - 48),
                scale=0.7,
                color=(40, 255, 40),
            )
            self.put_text(
                image,
                f'uv=({best.image_u:.1f}, {best.image_v:.1f}) depth={best.depth:.3f}m',
                (12, image.shape[0] - 18),
                scale=0.65,
                color=(0, 220, 255),
            )
        elif self.result_trigger_id > 0:
            self.put_text(
                image,
                f'last trigger #{self.result_trigger_id}: {self.state.message}',
                (12, image.shape[0] - 18),
                scale=0.65,
                color=(0, 120, 255),
            )
        return image

    def rpy_to_rotation(self, roll, pitch, yaw):
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rot_z @ rot_y @ rot_x

    def update_open3d_pose(self, best):
        if self.o3d_vis is None:
            return
        self.clear_open3d_pose()

        pose = np.eye(4)
        pose[:3, :3] = self.rpy_to_rotation(
            best.pose_xyzrpy_camera[3],
            best.pose_xyzrpy_camera[4],
            best.pose_xyzrpy_camera[5],
        )
        pose[:3, 3] = np.array(best.pose_xyzrpy_camera[:3], dtype=float)

        self.o3d_pick_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self.o3d_pick_frame.transform(pose)
        self.o3d_vis.add_geometry(self.o3d_pick_frame)

        points = o3d.utility.Vector3dVector([
            [0.0, 0.0, 0.0],
            list(map(float, best.pose_xyzrpy_camera[:3])),
        ])
        self.o3d_link = o3d.geometry.LineSet(
            points=points,
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        self.o3d_link.colors = o3d.utility.Vector3dVector([[1.0, 0.7, 0.1]])
        self.o3d_vis.add_geometry(self.o3d_link)
        self.update_open3d_target_cloud(best)
        self.update_open3d_camera_view(best)
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def clear_open3d_pose(self):
        if self.o3d_vis is None:
            return
        if self.o3d_pick_frame is not None:
            self.o3d_vis.remove_geometry(self.o3d_pick_frame, reset_bounding_box=False)
            self.o3d_pick_frame = None
        if self.o3d_link is not None:
            self.o3d_vis.remove_geometry(self.o3d_link, reset_bounding_box=False)
            self.o3d_link = None
        if self.o3d_target_cloud is not None:
            self.o3d_vis.remove_geometry(self.o3d_target_cloud, reset_bounding_box=False)
            self.o3d_target_cloud = None
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def depth_image_to_meters(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        return depth

    def build_target_cloud_points(self, best):
        if self.latest_depth is None or self.latest_camera_info is None:
            return None

        depth = self.depth_image_to_meters(self.latest_depth)
        color = self.image_msg_to_bgr(self.latest_color)
        info = self.latest_camera_info
        if depth.ndim != 2:
            return None

        u0 = int(round(best.image_u))
        v0 = int(round(best.image_v))
        radius = self.target_cloud_radius_px
        u_min = max(0, u0 - radius)
        u_max = min(depth.shape[1], u0 + radius + 1)
        v_min = max(0, v0 - radius)
        v_max = min(depth.shape[0], v0 + radius + 1)
        if u_min >= u_max or v_min >= v_max:
            return None

        roi = depth[v_min:v_max, u_min:u_max]
        color_roi = None
        if color is not None and color.shape[:2] == depth.shape[:2]:
            color_roi = color[v_min:v_max, u_min:u_max]
        uu, vv = np.meshgrid(
            np.arange(u_min, u_max, dtype=np.float32),
            np.arange(v_min, v_max, dtype=np.float32),
        )
        valid = np.isfinite(roi) & (roi > 0.05)
        valid &= np.abs(roi - float(best.depth)) <= self.target_cloud_depth_tolerance_m
        if not np.any(valid):
            return None

        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        z = roi[valid]
        x = (uu[valid] - cx) * z / fx
        y = (vv[valid] - cy) * z / fy
        points = np.stack([x, y, z], axis=1)
        if points.shape[0] == 0:
            return None
        colors = None
        if self.target_cloud_use_rgb and color_roi is not None:
            colors = color_roi[valid][:, ::-1].astype(np.float64) / 255.0
        return points, colors

    def update_open3d_target_cloud(self, best):
        if self.o3d_vis is None:
            return
        cloud_data = self.build_target_cloud_points(best)
        if cloud_data is None:
            self.get_logger().warning('failed to build target point cloud from depth ROI')
            return
        points, colors = cloud_data

        self.o3d_target_cloud = o3d.geometry.PointCloud()
        self.o3d_target_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is None:
            colors = np.tile(np.array([[0.15, 0.85, 0.35]], dtype=np.float64), (points.shape[0], 1))
        self.o3d_target_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.o3d_vis.add_geometry(self.o3d_target_cloud)

    def update_open3d_camera_view(self, best):
        if self.o3d_vis is None or not self.lock_open3d_camera_view:
            return

        target = np.array(best.pose_xyzrpy_camera[:3], dtype=np.float64)
        norm = np.linalg.norm(target)
        if norm < 1e-6:
            return

        target_dir = target / norm
        eye = target_dir * self.open3d_view_forward_offset_m
        forward = target - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return
        forward = forward / forward_norm

        up_hint = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if abs(np.dot(forward, up_hint)) > 0.98:
            up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        right = np.cross(forward, up_hint)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            return
        right = right / right_norm
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        rotation_wc = np.stack([right, -up, forward], axis=0)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation_wc
        extrinsic[:3, 3] = -rotation_wc @ eye

        view = self.o3d_vis.get_view_control()
        params = view.convert_to_pinhole_camera_parameters()
        params.extrinsic = extrinsic
        view.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    def draw_status_block(self, canvas):
        lines = []
        lines.append('keys: space=trigger q/esc=quit')
        if self.state.request_pending:
            lines.append('request: pending')
        else:
            lines.append('request: idle')
        lines.append(f'trigger_id: {self.last_trigger_id} result_id: {self.result_trigger_id}')

        if self.state.success is None:
            lines.append(f'status: {self.state.message}')
        elif self.state.success:
            lines.append('status: success')
            best = self.state.best
            if best is not None:
                lines.append(
                    f'candidate={best.candidate_id} peduncle={best.peduncle_id} cluster={best.cluster_id}'
                )
                lines.append(
                    f'score={best.score:.3f} maturity={best.maturity} ripe_ratio={best.ripe_ratio:.3f}'
                )
                lines.append(
                    f'uv=({best.image_u:.1f}, {best.image_v:.1f}) depth={best.depth:.3f}m'
                )
                cam = list(best.pose_xyzrpy_camera)
                lines.append(
                    f'cam xyz=({cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f})'
                )
                lines.append(
                    f'cam rpy=({cam[3]:.3f}, {cam[4]:.3f}, {cam[5]:.3f})'
                )
                if best.has_pose_base:
                    base = list(best.pose_xyzrpy_base)
                    lines.append(
                        f'base xyz=({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f})'
                    )
                if best.reason:
                    lines.append(f'reason: {best.reason}')
        else:
            lines.append('status: no valid pose')
            lines.append(f'message: {self.state.message}')

        y = 30
        for line in lines[:10]:
            self.put_text(canvas, line, (12, y), scale=0.65, color=(255, 255, 255))
            y += 30

    def update_display(self):
        color = self.image_msg_to_bgr(self.latest_color)
        color = self.overlay_pick_result(color)
        color = self.fit_panel(color, 'color')
        segment = self.fit_panel(self.image_msg_to_bgr(self.latest_segment_debug), 'segment debug')
        pose = self.fit_panel(self.image_msg_to_bgr(self.latest_pose_debug), 'pose debug')

        target_height = max(color.shape[0], segment.shape[0], pose.shape[0], 360)
        color = self.pad_to_height(color, target_height)
        segment = self.pad_to_height(segment, target_height)
        pose = self.pad_to_height(pose, target_height)

        top = np.hstack([color, segment, pose])
        bottom = np.zeros((280, top.shape[1], 3), dtype=np.uint8)
        self.draw_status_block(bottom)
        canvas = np.vstack([top, bottom])

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            self.get_logger().info('visualizer closed by user')
            raise KeyboardInterrupt
        if key == 32:
            self.get_logger().info('space pressed, trigger pick pose request')
            self.trigger_service_call()
        if self.o3d_vis is not None:
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()

    def destroy_node(self):
        if cv2 is not None:
            cv2.destroyAllWindows()
        if self.o3d_vis is not None:
            self.o3d_vis.destroy_window()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PickPoseVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
