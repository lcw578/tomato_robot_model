#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from xc_interfaces.srv import YoloDetection
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
from sensor_msgs.msg import Image
import cv2 as cv
from xc_interfaces.msg import Mask
import time
import torch
import os
from ament_index_python.packages import get_package_share_directory

class YOLOv8Service(Node):
    def __init__(self):
        super().__init__('yolov8_service')
        package_share_dir = get_package_share_directory('xc_robot_main_control')
        # model_path = os.path.join(package_share_dir, 'weights', 'fake_tomato_best.pt')
        # model_path = os.path.join(package_share_dir, 'weights','tomato_20260726_xiongAn_more.pt')
        model_path = os.path.join(package_share_dir, 'weights','best.pt')


        self.model = YOLO(model_path)
        # self.model = YOLO('/home/xc_home_wsl/code/test/ROS2/robot_ws/src/xc_robot_main_control/weights/best.pt')

        if torch.cuda.is_available():
            self.model.to('cuda')
            self.get_logger().info(f"模型已移动到GPU设备: {self.model.device}")
        else:
            self.get_logger().warn("CUDA不可用，模型将在CPU上运行")
        self.bridge = CvBridge()
        self.srv = self.create_service(
            YoloDetection, 
            'yolo_detection', 
            self.detect_objects
        )
        # 添加图像发布者
        self.combined_pub = self.create_publisher(Image, 'yolo_combined_image', 10)
        self.get_logger().info("YOLOv8检测服务已启动")
        
        # 添加10种半透明颜色 (BGR格式，alpha=0.3)
        self.color_palette = [
            (0, 0, 255, 0.8),    # 红色
            (0, 255, 0, 0.8),    # 绿色
            (255, 0, 0, 0.8),    # 蓝色
            (0, 255, 255, 0.8),  # 黄色
            (255, 0, 255, 0.8), # 品红
            (128, 0, 128, 0.3),  # 紫色
            (0, 165, 255, 0.3),  # 橙色
            (255, 255, 0, 0.3),  # 青色
            (47, 157, 39, 0.3),  # 翡翠绿
            (130, 130, 130, 0.3) # 灰色
        ]

    def detect_objects(self, request, response):
        try:
            # self.get_logger().info("*"*40)
            
            # 1. 图像转换
            color_image = self.bridge.imgmsg_to_cv2(request.color_image, "bgr8")
            original_h, original_w = color_image.shape[:2]  # 获取原始图像尺寸
            depth_image = self.bridge.imgmsg_to_cv2(request.depth_image, "passthrough")
            response.color_image = request.color_image
            response.depth_image = request.depth_image
            # print("接收到图像：",color_image.size)
            # cv.imshow("color_image",color_image)
            
            # 2. YOLO推理
            if torch.cuda.is_available():
                results = self.model.predict(color_image, device='0')
                # self.get_logger().info(f"模型设备: {self.model.device}")
            else:
                results = self.model.predict(color_image, device='cpu')
                # self.get_logger().info(f"模型设备: {self.model.device}")
            # results = self.model.predict(source=color_image, save=True, iou=0.4)

            # print("已经推理完成！！！")
            
            # 初始化标注图像为原始图像
            annotated_image = color_image.copy()
            
            # 检查是否有有效结果
            if results[0]:
                class_ids = []
                class_names = []
                confidences = []
                bboxes = []
                masks = []

                for result in results:
                    for i, box in enumerate(result.boxes):
                        # 处理检测框 - 直接使用原始图像尺寸的坐标
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # 填充基础数据
                        class_ids.append(cls_id)
                        class_names.append(self.model.names[cls_id])
                        confidences.append(conf)
                        bboxes.extend([x1, y1, x2, y2])

                        if result.masks and i < len(result.masks):
                            if result.masks[i].data is not None:
                                # self.get_logger().info("原始mask尺寸: %s" % str(result.masks[i].data[0].shape))
                            
                                mask = result.masks[i]
                                mask_data = mask.data[0].cpu().numpy().astype(np.uint8) * 255
                                
                                # 将mask缩放到原始图像尺寸
                                mask_data = cv.resize(
                                    mask_data,
                                    (original_w, original_h),
                                    interpolation=cv.INTER_NEAREST
                                )
                                
                                mask_msg = Mask()
                                mask_msg.class_id = cls_id
                                mask_msg.confidence = conf
                                mask_msg.class_names = self.model.names[cls_id]
                                mask_msg.mask = self.bridge.cv2_to_imgmsg(mask_data, "mono8")
                                masks.append(mask_msg)

                # 设置响应数据
                response.masks = masks
                response.class_ids = np.array(class_ids, dtype=np.uint32).tolist()
                response.class_names = class_names
                response.confidences = np.array(confidences, dtype=np.float32).tolist()
                response.bboxes = np.array(bboxes, dtype=np.float32).tolist()
                
                # 3. 绘制检测框和掩膜 (使用原始尺寸)
                annotated_image = color_image.copy()
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        color = self.color_palette[cls_id % 10][:3]
                        cv.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 掩膜叠加 (已经是原始尺寸)
                for mask_msg in masks:
                    if mask_msg.class_names=="stem" or mask_msg.class_names=="trunk":
                        mask = self.bridge.imgmsg_to_cv2(mask_msg.mask, "mono8")
                        class_id = mask_msg.class_id
                        color = self.color_palette[class_id % 10]
                        bgr_color = color[:3]
                        alpha = color[3]
                        
                        colored_mask = np.zeros_like(annotated_image)
                        colored_mask[mask > 0] = bgr_color
                        annotated_image = cv.addWeighted(annotated_image, 1, colored_mask, alpha, 0)

            # 合并并发布图像（保持原图+标注图对比）
            combined = np.hstack((color_image, annotated_image))
            combined_msg = self.bridge.cv2_to_imgmsg(combined, "bgr8")
            self.combined_pub.publish(combined_msg)

            # # 保存annotated_image图片
            timestamp = str(int(time.time()))
            save_path = f"/home/xc/code/test/temp/{timestamp}_annotated_image.jpg"
            # cv.imwrite(save_path, annotated_image)
            # self.get_logger().info(f"已保存标注图像到: {save_path}")
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"目标检测错误: {str(e)}")
            return response

def main(args=None):
    rclpy.init(args=args)
    try:
        service = YOLOv8Service()
        service.get_logger().info("YOLOv8服务节点创建成功，开始运行...")
        rclpy.spin(service)
    except Exception as e:
        service.get_logger().error(f"服务启动失败: {str(e)}")
    finally:
        service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
