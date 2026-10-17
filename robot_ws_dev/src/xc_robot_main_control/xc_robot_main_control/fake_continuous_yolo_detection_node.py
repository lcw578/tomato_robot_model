#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from xc_interfaces.srv import YoloDetection
from xc_interfaces.msg import Detection2UI
import threading
import time

class ContinuousYoloDetectionNode(Node):
    def __init__(self):
        super().__init__('continuous_yolo_detection_node')
        self.bridge = CvBridge()
        
        # 创建YOLO检测服务客户端
        self.yolo_client = self.create_client(YoloDetection, 'yolo_detection')
        
        # 等待服务启动
        while not self.yolo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待YOLO检测服务启动...')
        
        # 订阅彩色和深度图像话题
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        # 时间同步器，允许0.1秒的时间差
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], 
            queue_size=2, 
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        
        # 用于存储当前请求的future对象
        self.current_future = None
        self.future_lock = threading.Lock()
        self.is_processing = False
        
        self.detection2ui_pub = self.create_publisher(Detection2UI, '/detection2ui', 1)
        
        self.get_logger().info("连续YOLO检测节点已启动")

    def image_callback(self, color_msg, depth_msg):
        try:
            with self.future_lock:
                if self.is_processing:
                    return
                
                self.is_processing = True
            
            # 检查图像尺寸是否匹配
            if color_msg.width != depth_msg.width or color_msg.height != depth_msg.height:
                self.get_logger().warning("图像尺寸不匹配，丢弃本帧数据")
                self.is_processing = False
                return
                
            # 检查服务是否可用
            if not self.yolo_client.service_is_ready():
                self.get_logger().warning("YOLO检测服务不可用")
                self.is_processing = False
                return
                
            # 创建服务请求
            request = YoloDetection.Request()
            request.color_image = color_msg
            request.depth_image = depth_msg
            
            # 异步调用服务
            with self.future_lock:
                self.current_future = self.yolo_client.call_async(request)
                future = self.current_future
            
            # 添加超时处理
            timer = threading.Timer(1.0, self.timeout_callback, [future])
            timer.start()
            
            future.add_done_callback(lambda f: self.detection_callback(f, color_msg, depth_msg, timer))
            
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")
            self.is_processing = False

    def timeout_callback(self, future):
        """超时回调函数"""
        with self.future_lock:
            if not future.done():
                self.get_logger().warning("YOLO检测请求超时，取消请求")
                future.cancel()

    def detection_callback(self, future, color_msg, depth_msg, timer):
        try:
            timer.cancel()
            
            response = future.result()
            self.get_logger().info("成功接收到YOLO检测结果")
            
            detection2ui_msg = Detection2UI()
            detection2ui_msg.color_image = response.color_image
            detection2ui_msg.class_ids = response.class_ids
            detection2ui_msg.class_names = response.class_names
            detection2ui_msg.confidences = response.confidences
            detection2ui_msg.bboxes = response.bboxes
            detection2ui_msg.masks = response.masks
            detection2ui_msg.matched_rectangles = []
            
            self.detection2ui_pub.publish(detection2ui_msg)
            
        except Exception as e:
            self.get_logger().error(f"检测回调错误: {str(e)}")
        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = ContinuousYoloDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()