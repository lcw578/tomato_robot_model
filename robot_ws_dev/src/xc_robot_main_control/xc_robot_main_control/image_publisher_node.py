#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import glob
import cv2
import time

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        # self.declare_parameter('image_folder', '/mnt/d/temp_img/20251219')  # 工控机默认图片文件夹
        self.declare_parameter('image_folder', '/mnt/f/AI_image/tomato/20250115/D435_day/')  # 办公室电脑仿真默认图片文件夹        
        # 创建发布器
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.bridge = CvBridge()
        
        # 加载图片列表
        self.image_folder = self.get_parameter('image_folder').value
        self.image_pairs = self.load_image_pairs()
        self.current_index = 0
        
        # 创建定时器（1秒周期）
        self.timer = self.create_timer(0.04, self.timer_callback)
        # self.get_logger().info(f"已加载 {len(self.image_pairs)} 对图像，开始发布...")

    def load_image_pairs(self):
        # 获取所有彩色图和深度图文件路径
        color_images = glob.glob(os.path.join(self.image_folder, '*_c.png'))
        pairs = []
        
        for color_path in color_images:
            prefix = color_path.rsplit('_c.png', 1)[0]
            depth_path = f"{prefix}_d.png"
            if os.path.exists(depth_path):
                pairs.append((color_path, depth_path))
            else:
                self.get_logger().warn(f"未找到匹配的深度图: {depth_path}")
        
        return sorted(pairs)

    def timer_callback(self):
        if not self.image_pairs:
            self.get_logger().warn("没有可用的图像对")
            return
            
        # 获取当前图像对路径
        color_path, depth_path = self.image_pairs[self.current_index]
        
        try:
            # 读取并发布彩色图
            # print(color_path)
            color_img = cv2.imread(color_path)
            if color_img is None:
                raise ValueError(f"无法读取彩色图: {color_path}")
            
            # 读取并发布深度图（假设是16位PNG）
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise ValueError(f"无法读取深度图: {depth_path}")

            # 转换图像消息
            color_msg = self.bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="16UC1")
            
            # 添加时间戳
            timestamp = self.get_clock().now().to_msg()
            color_msg.header.stamp = timestamp
            depth_msg.header.stamp = timestamp
            
            # 发布消息
            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            
            # self.get_logger().info(f"已发布: {os.path.basename(color_path)}")
            
            # 新增：成功发布后递增索引（循环遍历）
            self.current_index = (self.current_index + 1) % len(self.image_pairs)
            
        except Exception as e:
            self.get_logger().error(f"图像发布失败: {str(e)}")
            # 跳过损坏的图像对
            self.current_index = (self.current_index + 1) % len(self.image_pairs)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
