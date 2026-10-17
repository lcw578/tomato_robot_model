#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8

import threading
import time


from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from xc_interfaces.msg import TestResult, MidpointReached
from closed_loop_feedback_detection import min_rectangle_processor as mrp

class FakeGripperFeedbackNode(Node):
    def __init__(self):
        super().__init__('fake_gripper_feedback_node')
        self.processing_lock = threading.Lock()
        self.midpoint_reached_lock = threading.Lock()
        self.reentrant_cb_group = ReentrantCallbackGroup()

        
        # --- 4. 订阅与发布 ---
        self.create_subscription(MidpointReached, 'midpoint_reached', self.midpoint_reached_callback, 10, callback_group=self.reentrant_cb_group)
        # self.create_subscription(UInt8, 'gripper/input_x1_status', self.x1_status_callback, 10, callback_group=self.reentrant_cb_group)
        
        self.test_result_pub = self.create_publisher(TestResult, 'test_result', 10, callback_group=self.reentrant_cb_group)
        # self.yolo_viz_pub = self.create_publisher(Image, 'yolo_viz_with_skeleton', 10, callback_group=self.reentrant_cb_group)


    def midpoint_reached_callback(self, msg):
        with self.midpoint_reached_lock:
            self.midpoint_reached = msg.reached
            self.get_logger().warn(f"✅ 中间点状态更新: {self.midpoint_reached}")
        if self.midpoint_reached==True:
            self.publish_test_result(1)

    def publish_test_result(self, val):
        msg = TestResult()
        msg.result = val
        self.get_logger().warn(f"✅ 我发布的结果是：{val}")
        for i in range(10):
            i=i+1
            self.test_result_pub.publish(msg)
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = FakeGripperFeedbackNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # if node.pipeline: node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()