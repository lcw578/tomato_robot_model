#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
from sensor_msgs.msg import Image
import cv2 as cv
from xc_interfaces.msg import Mask
import time
import torch
import os
import glob


class YOLOv8ImageFolderProcessor(Node):
    def __init__(self):
        super().__init__('yolov8_image_folder_processor')
        
        # 声明并获取参数
        self.declare_parameter('image_folder', '/home/xc/code/test/ros2/robot_ws/test_data')
        self.declare_parameter('model_path', '/home/xc/code/test/ros2/robot_ws/src/xc_robot_main_control/weights/best.pt')
        self.declare_parameter('loop_rate', 0.5)  # Hz
        
        self.image_folder = self.get_parameter('image_folder').value
        model_path = self.get_parameter('model_path').value
        loop_rate = self.get_parameter('loop_rate').value
        
        # 初始化模型
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.get_logger().info(f"模型已移动到GPU设备: {self.model.device}")
        else:
            self.get_logger().warn("CUDA不可用，模型将在CPU上运行")
            
        self.bridge = CvBridge()
        
        # 创建图像发布者
        self.combined_pub = self.create_publisher(Image, 'yolo_combined_image', 10)
        
        # 添加10种半透明颜色 (BGR格式)
        self.color_palette = [
            (0, 0, 255, 0.8),    # 红色
            (0, 255, 0, 0.8),    # 绿色
            (255, 0, 0, 0.8),    # 蓝色
            (0, 255, 255, 0.8),  # 黄色
            (255, 0, 255, 0.8),  # 品红
            (128, 0, 128, 0.3),  # 紫色
            (0, 165, 255, 0.3),  # 橙色
            (255, 255, 0, 0.3),  # 青色
            (47, 157, 39, 0.3),  # 翡翠绿
            (130, 130, 130, 0.3) # 灰色
        ]
        
        # 获取图片文件列表
        self.image_files = self._get_image_files()
        if not self.image_files:
            self.get_logger().error(f"在文件夹 {self.image_folder} 中未找到图片文件")
            return
        
        self.current_index = 0
        
        # 创建定时器
        self.timer = self.create_timer(1.0 / loop_rate, self.process_next_image)
        
        self.get_logger().info(f"YOLOv8图片文件夹处理器已启动，图片文件夹: {self.image_folder}, 图片数量: {len(self.image_files)}")
    
    def _get_image_files(self):
        """获取文件夹中的所有图片文件"""
        if not os.path.exists(self.image_folder):
            self.get_logger().error(f"图片文件夹不存在: {self.image_folder}")
            return []
            
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_folder, extension)))
            
        # 按文件名排序
        image_files.sort()
        return image_files
    
    def process_next_image(self):
        """处理下一张图片"""
        if not self.image_files:
            return
            
        # 获取当前图片路径
        image_path = self.image_files[self.current_index]
        self.get_logger().info(f"处理图片: {image_path}")
        
        try:
            # 读取图片
            color_image = cv.imread(image_path)
            if color_image is None:
                self.get_logger().error(f"无法读取图片: {image_path}")
                self.current_index = (self.current_index + 1) % len(self.image_files)
                return
            
            original_h, original_w = color_image.shape[:2]
            
            # YOLO推理
            if torch.cuda.is_available():
                results = self.model.predict(color_image, device='0')
            else:
                results = self.model.predict(color_image, device='cpu')
            
            # 初始化标注图像为原始图像
            annotated_image = color_image.copy()
            
            # 检查是否有有效结果
            if results[0]:
                for result in results:
                    # 绘制检测框
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            color = self.color_palette[cls_id % 10][:3]
                            cv.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制掩膜
                    if result.masks is not None:
                        for i, mask_data in enumerate(result.masks.data):
                            mask = mask_data.cpu().numpy().astype(np.uint8) * 255
                            # 将mask缩放到原始图像尺寸
                            mask = cv.resize(mask, (original_w, original_h), interpolation=cv.INTER_NEAREST)
                            
                            cls_id = int(result.boxes[i].cls[0]) if result.boxes is not None and i < len(result.boxes) else 0
                            color = self.color_palette[cls_id % 10]
                            bgr_color = color[:3]
                            alpha = color[3]
                            
                            colored_mask = np.zeros_like(annotated_image)
                            colored_mask[mask > 0] = bgr_color
                            annotated_image = cv.addWeighted(annotated_image, 1, colored_mask, alpha, 0)
            
            # 合并并发布图像（保持原图+标注图对比）
            combined = np.hstack((color_image, annotated_image))
            combined_msg = self.bridge.cv2_to_imgmsg(combined, "bgr8")
            self.combined_pub.publish(combined_msg)
            
            self.get_logger().info(f"已发布推理结果: {image_path}")
            
        except Exception as e:
            self.get_logger().error(f"处理图片 {image_path} 时出错: {str(e)}")
        
        # 更新索引以处理下一张图片
        self.current_index = (self.current_index + 1) % len(self.image_files)


def main(args=None):
    rclpy.init(args=args)
    try:
        processor = YOLOv8ImageFolderProcessor()
        processor.get_logger().info("YOLOv8图片文件夹处理器节点创建成功，开始运行...")
        rclpy.spin(processor)
    except Exception as e:
        processor.get_logger().error(f"节点运行失败: {str(e)}")
    finally:
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()