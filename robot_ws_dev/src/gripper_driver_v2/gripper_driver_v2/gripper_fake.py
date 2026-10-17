# gripper_control_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from xc_interfaces.srv import GripperControl
from gripper_driver_v2.gripper_485_driver import Gripper485Driver
import time

class FakeGripper(Node):
    """
    ROS2抓取器控制节点
    提供gripper/control服务，控制抓取器开合
    """
    
    def __init__(self):
        super().__init__('gripper_fake_node')
        # 创建服务（使用可重入回调组以允许并发调用）
        self.callback_group = ReentrantCallbackGroup()
        self.service = self.create_service(
            GripperControl,
            'gripper/control',
            self.handle_gripper_control,
            callback_group=self.callback_group
        )        

    
    def handle_gripper_control(self, request, response):
        """
        处理抓取器控制请求
        
        Args:
            request: GripperControl_Request
            response: GripperControl_Response
            
        Returns:
            GripperControl_Response
        """
        command = request.command.lower()
        self.get_logger().info(f"收到抓取器控制命令: {command}")
        

        
        # 设置响应
        response.success = True
        response.current_position = 1000
        response.message = "哈哈哈哈"

        
        return response   

    
    def destroy_node(self):
        """节点销毁时的清理工作"""

        self.get_logger().info("关闭抓取器控制节点...")       

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # 创建节点
        node = FakeGripper()
        
        # 使用多线程执行器以支持并发服务调用
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        # 运行节点
        node.get_logger().info("抓取器控制节点已启动，按Ctrl+C退出")
        executor.spin()
        
    except KeyboardInterrupt:
        node.get_logger().info("接收到Ctrl+C信号")
    except Exception as e:
        node.get_logger().error(f"节点运行异常: {e}")
    finally:
        # 清理
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()