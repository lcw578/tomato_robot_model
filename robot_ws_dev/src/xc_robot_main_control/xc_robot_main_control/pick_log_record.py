#!/usr/bin/env python3
"""
订阅picking_reault_all话题的节点
作者: Trae AI
创建日期: 2025年
描述: 订阅DetectionResult类型的消息，您可以在此基础上添加自己的回调函数逻辑
"""

import rclpy
from rclpy.node import Node
from xc_interfaces.msg import DetectionResult
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import csv
from datetime import datetime


class PickingResultSubscriber(Node):
    """
    订阅picking_reault_all话题的节点
    """

    def __init__(self):
        super().__init__('pick_log_record')
        
        # 创建参数，允许用户设置保存文件夹路径
        self.declare_parameter('save_folder', '/home/xc/code/test/log_pick20260210/')
        
        # 获取保存文件夹路径
        self.save_folder = self.get_parameter('save_folder').get_parameter_value().string_value
        
        # 创建保存文件夹（如果不存在）
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            self.get_logger().info(f'创建保存文件夹: {self.save_folder}')
        
        # 创建CSV文件保存文件夹
        self.csv_folder = os.path.join(self.save_folder, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_csv_logs')
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
            self.get_logger().info(f'创建CSV日志文件夹: {self.csv_folder}')
        
        # CSV文件路径
        self.csv_file_path = os.path.join(self.csv_folder, f'pick_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        # 初始化CSV文件并写入表头
        self._initialize_csv_file()
        
        # 创建订阅者，订阅picking_reault_all话题
        self.subscription = self.create_subscription(
            DetectionResult,
            '/picking_reault_all',
            self.picking_result_callback,
            10  # 队列大小
        )
        
        # 创建用于cv2和ROS图像之间转换的桥
        self.bridge = CvBridge()
        
        self.get_logger().info('picking_result_subscriber节点已启动，正在监听/picking_reault_all话题...')
        
    def _initialize_csv_file(self):
        """初始化CSV文件并写入表头"""
        headers = [
            'timestamp', 'growth_directions', 'picking_points_x', 'picking_points_y', 'picking_points_z',
            'distances', 'target_arm_points_x', 'target_arm_points_y', 'target_arm_points_z',
            'watch_tomato_nums', 'try_pick_times', 'go_to_target_results',
            'gripper_feedback_results', 'gripper_close_results', 'return_to_box_results',
            'gripper_open_results', 'matched_rectangles'
        ]
        
        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        
        self.get_logger().info(f'CSV日志文件已创建: {self.csv_file_path}')
    
    def picking_result_callback(self, msg):
        """
        picking_reault_all话题的回调函数
        您可以在这里添加自己的处理逻辑
        """
        self.get_logger().info(f'收到采摘结果消息 - 时间戳: {msg.timestamp}, 处理数量: {msg.handle_num}')
        
        # 将masks绘制到color_image上的功能
        try:
            # 将ROS图像消息转换为OpenCV图像
            color_image = self.bridge.imgmsg_to_cv2(msg.color_image, desired_encoding='bgr8')
            
            # 创建一个用于绘制mask的副本
            overlay_image = color_image.copy()
            
            # 为每个mask分配一个颜色
            colors = [
                (255, 0, 0),    # 蓝色
                (0, 255, 0),    # 绿色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 洋红色
                (0, 255, 255),  # 黄色
                (128, 0, 128),  # 紫色
                (0, 128, 255),  # 橙色
                (255, 128, 0),  # 浅蓝色
                (128, 128, 0),  # 深绿色
                (0, 0, 0)       # 黑色
            ]
            
            # 绘制每个mask
            for i, mask_msg in enumerate(msg.masks):
                if hasattr(mask_msg, 'data') and mask_msg.data:
                    # 获取mask数据
                    mask_data = np.array(mask_msg.data, dtype=np.uint8)
                    
                    # 重塑mask为正确的形状（假设是宽×高）
                    if hasattr(mask_msg, 'width') and hasattr(mask_msg, 'height'):
                        mask = mask_data.reshape((mask_msg.height, mask_msg.width))
                    else:
                        # 如果没有尺寸信息，则需要其他方式确定形状
                        # 这里假设mask的形状与输入图像相似比例
                        continue
                    
                    # 获取对应的颜色（循环使用）
                    
                    class_ids = msg.class_ids[i]
                    color = colors[class_ids % len(colors)]  # 确保索引不超出范围
                    
                    # 创建彩色mask
                    colored_mask = np.zeros_like(overlay_image)
                    colored_mask[mask > 0] = color
                    
                    # 将mask叠加到原图上（透明度混合）
                    alpha = 0.3  # 透明度
                    overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask, alpha, 0)
            
            # 绘制边界框
            # 假设msg.bboxes包含成对的坐标点（x1, y1, x2, y2）
            bbox_count = len(msg.matched_rectangles) // 4  # 每4个元素为一个框
            for i in range(bbox_count):
                # 计算当前框在展平数组中的起始索引
                start_idx = i * 4
                # 在图像上绘制矩形框
                cv2.rectangle(overlay_image, 
                                (int(msg.matched_rectangles[start_idx]), int(msg.matched_rectangles[start_idx + 1])), 
                                (int(msg.matched_rectangles[start_idx + 2]), int(msg.matched_rectangles[start_idx + 3])), 
                                colors[7], 2)
                if msg.picking_points_x[i] != 0 and msg.picking_points_y[i] != 0:
                    cv2.circle(overlay_image, (int(msg.picking_points_x[i]), int(msg.picking_points_y[i])), 5, colors[10], -1)
                
                # 添加标签文本（如果有的话）
                label = f"{i}:{msg.growth_directions[i]}" 
                cv2.putText(overlay_image, label, (int(msg.matched_rectangles[start_idx]), int(msg.matched_rectangles[start_idx + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[7], 2)
            
            # 生成时间戳文件名
            image_filename = f"pick_result_{msg.timestamp}.jpg"
            image_path = os.path.join(self.save_folder, image_filename)
            
            # 保存图像
            success = cv2.imwrite(image_path, overlay_image)
            if success:
                self.get_logger().info(f'图像已保存: {image_path}')
            else:
                self.get_logger().error(f'保存图像失败: {image_path}')
            
            # 将消息数据保存到CSV文件，只保存指定字段
            self._save_to_csv_specified_fields(msg)
            
            # 显示结果图像
            cv2.imshow('Color Image with Masks and Bounding Boxes', overlay_image)
            cv2.waitKey(1)  # 必须调用此函数才能更新窗口显示
            
        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {str(e)}')
    
    def _save_to_csv_specified_fields(self, msg):
        """将DetectionResult消息内容保存到CSV文件，只保存指定字段"""
        try:
            # 准备要写入的数据，只包含指定字段
            row_data = [
                msg.timestamp,  # timestamp
                ';'.join(map(str, msg.growth_directions)),  # growth_directions，用分号连接
                ';'.join(map(str, msg.picking_points_x)),  # picking_points_x，用分号连接
                ';'.join(map(str, msg.picking_points_y)),  # picking_points_y，用分号连接
                ';'.join(map(str, msg.picking_points_z)),  # picking_points_z，用分号连接
                ';'.join(map(str, msg.distances)),  # distances，用分号连接
                ';'.join(map(str, msg.target_arm_points_x)),  # target_arm_points_x，用分号连接
                ';'.join(map(str, msg.target_arm_points_y)),  # target_arm_points_y，用分号连接
                ';'.join(map(str, msg.target_arm_points_z)),  # target_arm_points_z，用分号连接
                ';'.join(map(str, msg.watch_tomato_nums)),  # watch_tomato_nums，用分号连接
                ';'.join(map(str, msg.try_pick_times)),  # try_pick_times，用分号连接
                ';'.join(map(str, msg.go_to_target_results)),  # go_to_target_results，用分号连接
                ';'.join(map(str, msg.gripper_feedback_results)),  # gripper_feedback_results，用分号连接
                ';'.join(map(str, msg.gripper_close_results)),  # gripper_close_results，用分号连接
                ';'.join(map(str, msg.return_to_box_results)),  # return_to_box_results，用分号连接
                ';'.join(map(str, msg.gripper_open_results)),  # gripper_open_results，用分号连接
                ';'.join(map(str, msg.matched_rectangles))  # matched_rectangles，用分号连接
            ]
            
            # 追加数据到CSV文件
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
                
            self.get_logger().info(f'数据已保存到CSV: {self.csv_file_path}')
            
        except Exception as e:
            self.get_logger().error(f'保存CSV数据时出错: {str(e)}')
        

def main(args=None):
    rclpy.init(args=args)
    
    picking_result_subscriber = PickingResultSubscriber()
    
    try:
        rclpy.spin(picking_result_subscriber)
    except KeyboardInterrupt:
        picking_result_subscriber.get_logger().info('接收到中断信号，正在关闭节点...')
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    finally:
        picking_result_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()