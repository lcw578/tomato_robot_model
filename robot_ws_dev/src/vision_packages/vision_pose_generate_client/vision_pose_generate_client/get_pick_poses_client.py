from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from vision_tomato_interfaces.srv import GetPickPoses

try:
    import cv2
except ImportError:
    cv2 = None


class GetPickPosesClient(Node):
    def __init__(self):
        super().__init__('get_pick_poses_client')
        self.declare_parameter('service_name', '/vision/tomato/get_pick_poses')
        self.declare_parameter('timeout_sec', 15.0)
        self.declare_parameter('return_debug_images', True)
        self.declare_parameter('segment_debug_topic', '/vision/tomato/pick_pose/segment_debug_image')
        self.declare_parameter('pose_debug_topic', '/vision/tomato/pick_pose/debug_image')
        self.declare_parameter('debug_wait_sec', 2.0)
        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('debug_output_dir', '/tmp/vision_pose_generate_client')
        self.declare_parameter('fail_on_unsuccessful_response', False)

        self.service_name = str(self.get_parameter('service_name').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.return_debug_images = bool(self.get_parameter('return_debug_images').value)
        self.segment_debug_topic = str(self.get_parameter('segment_debug_topic').value)
        self.pose_debug_topic = str(self.get_parameter('pose_debug_topic').value)
        self.debug_wait_sec = float(self.get_parameter('debug_wait_sec').value)
        self.save_debug_images = bool(self.get_parameter('save_debug_images').value)
        self.debug_output_dir = Path(str(self.get_parameter('debug_output_dir').value))
        self.fail_on_unsuccessful_response = bool(
            self.get_parameter('fail_on_unsuccessful_response').value
        )
        self.client = self.create_client(GetPickPoses, self.service_name)
        self.bridge = CvBridge()
        self.latest_segment_debug = None
        self.latest_pose_debug = None
        self.request_started_ns = 0

        self.segment_debug_sub = self.create_subscription(
            Image,
            self.segment_debug_topic,
            self.segment_debug_callback,
            5,
        )
        self.pose_debug_sub = self.create_subscription(
            Image,
            self.pose_debug_topic,
            self.pose_debug_callback,
            5,
        )

    def segment_debug_callback(self, msg):
        self.latest_segment_debug = msg
        self.get_logger().info(
            f'received segment debug image: {msg.width}x{msg.height} encoding={msg.encoding}'
        )

    def pose_debug_callback(self, msg):
        self.latest_pose_debug = msg
        self.get_logger().info(
            f'received pose debug image: {msg.width}x{msg.height} encoding={msg.encoding}'
        )

    def message_is_current(self, msg):
        if msg is None:
            return False
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        return stamp_ns >= self.request_started_ns

    def wait_for_debug_images(self):
        if not self.return_debug_images:
            return

        deadline = self.get_clock().now().nanoseconds + int(self.debug_wait_sec * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            have_segment = self.message_is_current(self.latest_segment_debug)
            have_pose = self.message_is_current(self.latest_pose_debug)
            if have_segment and have_pose:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        have_segment = self.message_is_current(self.latest_segment_debug)
        have_pose = self.message_is_current(self.latest_pose_debug)
        self.get_logger().info(
            f'debug image status: segment={have_segment} pose={have_pose}'
        )
        if self.save_debug_images:
            self.save_debug_images_to_disk()

    def save_debug_images_to_disk(self):
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        self.save_one_debug_image(
            msg=self.latest_segment_debug,
            name='segment_debug',
        )
        self.save_one_debug_image(
            msg=self.latest_pose_debug,
            name='pose_debug',
        )

    def save_one_debug_image(self, msg, name):
        if not self.message_is_current(msg):
            self.get_logger().warning(f'no current {name} image to save')
            return
        if cv2 is None:
            self.get_logger().warning(f'cv2 is unavailable, skip saving {name} image')
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        stamp = f'{msg.header.stamp.sec}_{msg.header.stamp.nanosec:09d}'
        output_path = self.debug_output_dir / f'{stamp}_{name}.png'
        if not cv2.imwrite(str(output_path), cv_image):
            self.get_logger().error(f'failed to save {name} image: {output_path}')
            return
        self.get_logger().info(f'saved {name} image: {output_path}')

    def call_once(self):
        self.get_logger().info(f'waiting for {self.service_name} ...')
        if not self.client.wait_for_service(timeout_sec=self.timeout_sec):
            self.get_logger().error(f'service not available: {self.service_name}')
            return 1

        self.latest_segment_debug = None
        self.latest_pose_debug = None
        self.request_started_ns = self.get_clock().now().nanoseconds
        request = GetPickPoses.Request()
        request.return_debug_images = self.return_debug_images
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        if not future.done():
            self.get_logger().error('service call timeout')
            return 2

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'service call failed: {exc}')
            return 3

        self.get_logger().info(f'success={response.success}')
        self.get_logger().info(f'message={response.message}')
        if not response.success:
            self.get_logger().warning('service returned unsuccessful result')
            self.wait_for_debug_images()
            return 4 if self.fail_on_unsuccessful_response else 0

        best = response.best
        xyzrpy_camera = list(best.pose_xyzrpy_camera)
        xyzrpy_base = list(best.pose_xyzrpy_base)
        self.get_logger().info(
            'best camera xyzrpy[m,rad]='
            f'[{xyzrpy_camera[0]:.4f}, {xyzrpy_camera[1]:.4f}, {xyzrpy_camera[2]:.4f}, '
            f'{xyzrpy_camera[3]:.4f}, {xyzrpy_camera[4]:.4f}, {xyzrpy_camera[5]:.4f}]'
        )
        self.get_logger().info(
            'best base xyzrpy[m,rad]='
            f'[{xyzrpy_base[0]:.4f}, {xyzrpy_base[1]:.4f}, {xyzrpy_base[2]:.4f}, '
            f'{xyzrpy_base[3]:.4f}, {xyzrpy_base[4]:.4f}, {xyzrpy_base[5]:.4f}]'
        )
        self.get_logger().info(
            f'candidate={best.candidate_id}, peduncle={best.peduncle_id}, '
            f'cluster={best.cluster_id}, fruits={list(best.fruit_ids)}'
        )
        self.get_logger().info(
            f'score={best.score:.3f}, maturity={best.maturity}, '
            f'ripe_ratio={best.ripe_ratio:.3f}, has_pose_base={best.has_pose_base}'
        )
        self.wait_for_debug_images()
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = GetPickPosesClient()
    try:
        exit_code = node.call_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
