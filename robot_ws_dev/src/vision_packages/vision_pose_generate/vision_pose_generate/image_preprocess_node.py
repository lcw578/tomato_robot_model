import copy

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class ImagePreprocessNode(Node):
    def __init__(self):
        super().__init__('image_preprocess_node')
        self.declare_parameter('input_color_topic', '/cam_hand/color/image_rect_raw')
        self.declare_parameter('input_depth_topic', '/cam_hand/aligned_depth_to_color/image_raw')
        self.declare_parameter('input_camera_info_topic', '/cam_hand/color/camera_info')
        self.declare_parameter('output_color_topic', '/vision/tomato/crop/color/image_raw')
        self.declare_parameter('output_depth_topic', '/vision/tomato/crop/depth/image_raw')
        self.declare_parameter('output_camera_info_topic', '/vision/tomato/crop/camera_info')
        self.declare_parameter('crop_mode', 'center_square')
        self.declare_parameter('publish_rate_limit_hz', 10.0)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_info = None
        self.last_stamp_key = None

        self.color_pub = self.create_publisher(Image, self.get_parameter('output_color_topic').value, 5)
        self.depth_pub = self.create_publisher(Image, self.get_parameter('output_depth_topic').value, 5)
        self.info_pub = self.create_publisher(CameraInfo, self.get_parameter('output_camera_info_topic').value, 5)

        self.create_subscription(Image, self.get_parameter('input_color_topic').value, self.color_callback, 10)
        self.create_subscription(Image, self.get_parameter('input_depth_topic').value, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.get_parameter('input_camera_info_topic').value, self.info_callback, 10)

        rate = float(self.get_parameter('publish_rate_limit_hz').value)
        period = 1.0 / rate if rate > 0 else 0.1
        self.timer = self.create_timer(period, self.publish_latest)
        self.get_logger().info('image_preprocess_node started')

    def color_callback(self, msg):
        self.latest_color = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def info_callback(self, msg):
        self.latest_info = msg

    def compute_crop(self, width, height):
        if self.get_parameter('crop_mode').value != 'center_square':
            return 0, 0, width, height
        side = min(width, height)
        x0 = int((width - side) / 2)
        y0 = int((height - side) / 2)
        return x0, y0, side, side

    def crop_camera_info(self, info, x0, y0, width, height):
        cropped = copy.deepcopy(info)
        cropped.width = int(width)
        cropped.height = int(height)
        k = list(cropped.k)
        p = list(cropped.p)
        k[2] -= float(x0)
        k[5] -= float(y0)
        p[2] -= float(x0)
        p[6] -= float(y0)
        cropped.k = k
        cropped.p = p
        cropped.roi.x_offset = 0
        cropped.roi.y_offset = 0
        cropped.roi.width = int(width)
        cropped.roi.height = int(height)
        cropped.roi.do_rectify = True
        return cropped

    def publish_latest(self):
        if self.latest_color is None or self.latest_depth is None or self.latest_info is None:
            return

        stamp = self.latest_color.header.stamp
        stamp_key = (stamp.sec, stamp.nanosec)
        if stamp_key == self.last_stamp_key:
            return

        try:
            color = self.bridge.imgmsg_to_cv2(self.latest_color, desired_encoding='passthrough')
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warning(f'cv_bridge conversion failed: {exc}')
            return

        h, w = color.shape[:2]
        x0, y0, cw, ch = self.compute_crop(w, h)
        color_crop = color[y0:y0 + ch, x0:x0 + cw].copy()

        dh, dw = depth.shape[:2]
        if dw != w or dh != h:
            scale_x = dw / float(w)
            scale_y = dh / float(h)
            dx0 = int(round(x0 * scale_x))
            dy0 = int(round(y0 * scale_y))
            dcw = int(round(cw * scale_x))
            dch = int(round(ch * scale_y))
        else:
            dx0, dy0, dcw, dch = x0, y0, cw, ch
        depth_crop = depth[dy0:dy0 + dch, dx0:dx0 + dcw].copy()

        color_msg = self.bridge.cv2_to_imgmsg(color_crop, encoding=self.latest_color.encoding)
        color_msg.header = self.latest_color.header
        depth_msg = self.bridge.cv2_to_imgmsg(depth_crop, encoding=self.latest_depth.encoding)
        depth_msg.header = self.latest_depth.header
        depth_msg.header.frame_id = self.latest_color.header.frame_id
        info_msg = self.crop_camera_info(self.latest_info, x0, y0, cw, ch)
        info_msg.header = self.latest_color.header

        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)
        self.last_stamp_key = stamp_key


def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
