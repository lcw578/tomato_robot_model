import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import MarkerArray

from vision_tomato_interfaces.msg import PickPoseCandidate
from vision_tomato_interfaces.srv import GeneratePickPoses, GetPickPoses, SegmentTomato

from .debug_viz import candidates_to_markers


class PickPoseServiceNode(Node):
    def __init__(self):
        super().__init__('pick_pose_service_node')
        self.declare_parameter('color_topic', '/vision/tomato/crop/color/image_raw')
        self.declare_parameter('depth_topic', '/vision/tomato/crop/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/vision/tomato/crop/camera_info')
        self.declare_parameter('segment_service', '/vision/tomato/segment')
        self.declare_parameter('pose_service', '/vision/tomato/generate_pick_poses')
        self.declare_parameter('get_pick_poses_service', '/vision/tomato/get_pick_poses')
        self.declare_parameter('service_timeout_sec', 10.0)
        self.declare_parameter('publish_results', True)

        self.callback_group = ReentrantCallbackGroup()
        self.latest_color = None
        self.latest_depth = None
        self.latest_info = None
        self.latest_lock = threading.Lock()

        self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self.color_callback,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.info_callback,
            5,
            callback_group=self.callback_group,
        )

        self.segment_client = self.create_client(
            SegmentTomato,
            self.get_parameter('segment_service').value,
            callback_group=self.callback_group,
        )
        self.pose_client = self.create_client(
            GeneratePickPoses,
            self.get_parameter('pose_service').value,
            callback_group=self.callback_group,
        )
        self.srv = self.create_service(
            GetPickPoses,
            self.get_parameter('get_pick_poses_service').value,
            self.handle_get_pick_poses,
            callback_group=self.callback_group,
        )

        self.best_pub = self.create_publisher(PoseStamped, '/vision/tomato/pick_pose/best', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/vision/tomato/pick_pose/markers', 5)
        self.segment_debug_pub = self.create_publisher(Image, '/vision/tomato/pick_pose/segment_debug_image', 5)
        self.pose_debug_pub = self.create_publisher(Image, '/vision/tomato/pick_pose/debug_image', 5)

        self.get_logger().info(
            f"pick_pose_service_node started at {self.get_parameter('get_pick_poses_service').value}"
        )

    def color_callback(self, msg):
        with self.latest_lock:
            self.latest_color = msg

    def depth_callback(self, msg):
        with self.latest_lock:
            self.latest_depth = msg

    def info_callback(self, msg):
        with self.latest_lock:
            self.latest_info = msg

    def get_latest_inputs(self):
        return self.get_latest_inputs_wait(0.25)

    def inputs_are_synced(self, color, depth, info):
        if color is None or depth is None or info is None:
            return False
        c = color.header.stamp
        d = depth.header.stamp
        i = info.header.stamp
        return c.sec == d.sec == i.sec and c.nanosec == d.nanosec == i.nanosec

    def get_latest_inputs_wait(self, timeout_sec):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        last = (None, None, None)
        while True:
            with self.latest_lock:
                last = (self.latest_color, self.latest_depth, self.latest_info)
            color, depth, info = last
            if self.inputs_are_synced(color, depth, info):
                return last
            if time.monotonic() >= deadline:
                return last
            time.sleep(0.01)

    def wait_for_future(self, future, timeout_sec):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout_sec):
            return None, 'timeout'
        try:
            return future.result(), ''
        except Exception as exc:
            return None, str(exc)

    def fill_failure(self, response, message):
        response.success = False
        response.message = message
        response.best = PickPoseCandidate()
        self.get_logger().warning(message)
        return response

    def handle_get_pick_poses(self, request, response):
        color, depth, info = self.get_latest_inputs_wait(0.25)
        if color is None or depth is None or info is None:
            return self.fill_failure(response, 'missing latest color/depth/camera_info')
        if not self.inputs_are_synced(color, depth, info):
            return self.fill_failure(response, 'latest color/depth/camera_info are not synchronized')

        timeout_sec = float(self.get_parameter('service_timeout_sec').value)
        if not self.segment_client.wait_for_service(timeout_sec=1.0):
            return self.fill_failure(response, 'segment service is not ready')
        if not self.pose_client.wait_for_service(timeout_sec=1.0):
            return self.fill_failure(response, 'pose service is not ready')

        seg_req = SegmentTomato.Request()
        seg_req.color_image = color
        seg_resp, error = self.wait_for_future(self.segment_client.call_async(seg_req), timeout_sec)
        if seg_resp is None:
            return self.fill_failure(response, f'segment service call failed: {error}')
        if not seg_resp.success:
            return self.fill_failure(response, f'segment failed: {seg_resp.message}')

        pose_req = GeneratePickPoses.Request()
        pose_req.color_image = color
        pose_req.depth_image = depth
        pose_req.camera_info = info
        pose_req.instances = seg_resp.instances
        pose_resp, error = self.wait_for_future(self.pose_client.call_async(pose_req), timeout_sec)
        if pose_resp is None:
            return self.fill_failure(response, f'pose service call failed: {error}')
        if not pose_resp.success:
            return self.fill_failure(response, 'pose generation failed')

        response.success = True
        response.best = pose_resp.best
        response.message = pose_resp.message or 'ok'
        if bool(request.return_debug_images):
            self.segment_debug_pub.publish(seg_resp.debug_image)
            self.pose_debug_pub.publish(pose_resp.debug_image)

        if bool(self.get_parameter('publish_results').value):
            self.publish_results(info, pose_resp)

        return response

    def publish_results(self, info, pose_resp):
        if not pose_resp.success:
            return
        best_pose = pose_resp.best
        self.best_pub.publish(best_pose.pose_camera)
        markers = candidates_to_markers(
            [best_pose],
            best_pose.pose_camera.header.frame_id,
            self.get_clock().now().to_msg(),
        )
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = PickPoseServiceNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
