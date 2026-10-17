#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

class DepthVisualizerNode(Node):
    def __init__(self):
        super().__init__('depth_visualizer_node')
        
        # 创建cv_bridge实例
        self.bridge = CvBridge()
        
        # 创建订阅者，订阅深度图像话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_image_callback,
            10)
        self.subscription  # 防止未使用变量警告
        
        # 创建保存目录
        self.save_dir = '/home/xc/code/test/depth_images'
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.get_logger().info('Depth Visualizer Node has been started.')
        self.get_logger().info(f'Saving images to: {self.save_dir}')

    def depth_image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 创建彩色图像用于可视化
            colored_depth = self.apply_colormap(cv_image)
            
            # 生成包含时间戳的文件名
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            filename = f'depth_image_{timestamp}.png'
            filepath = os.path.join(self.save_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, colored_depth)
            
            self.get_logger().info(f'Saved depth image: {filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def apply_colormap(self, depth_image):
        """
        将深度图像转换为彩色图像进行可视化
        数据大于0的根据数值大小使用渐变色标注，数据为0的使用黑色标注
        """
        # 创建输出图像
        colored_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
        
        # 找到大于0的像素
        mask = depth_image > 0
        
        if np.any(mask):
            # 获取非零深度值的范围
            min_val = np.min(depth_image[mask])
            max_val = np.max(depth_image[mask])
            
            # 避免除零错误
            if max_val > min_val:
                # 将深度值归一化到0-255范围
                normalized = np.zeros_like(depth_image, dtype=np.float32)
                normalized[mask] = (depth_image[mask] - min_val) / (max_val - min_val) * 255
                
                # 应用COLORMAP_JET颜色映射
                colored_nonzero = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
                
                # 将着色部分复制到输出图像
                colored_image[mask] = colored_nonzero[mask]
            else:
                # 如果所有非零值都相同，使用单一颜色
                colored_image[mask] = [0, 0, 255]  # 红色
        
        # 数据为0的部分保持黑色（默认值）
        
        return colored_image

def main(args=None):
    rclpy.init(args=args)
    
    node = DepthVisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()