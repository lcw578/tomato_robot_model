# gripper_control_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from xc_interfaces.srv import GripperControl

# from gripper_driver_v2.gripper_can_driver import GripperCANDriver, MotorStatus
from gripper_driver_v2.gripper_485_driver import Gripper485Driver
import time

class GripperControlNode(Node):
    """
    ROS2抓取器控制节点
    提供gripper/control服务，控制抓取器开合
    """
    
    def __init__(self):
        super().__init__('gripper_control_node')
        
        # 声明参数
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('motor_id', 1)
        self.declare_parameter('bitrate', 1000000)
        self.declare_parameter('open_position', 4294963200)
        self.declare_parameter('close_position', 4294965458)#假装剪
        # self.declare_parameter('close_position', 4294965658) #真实剪断
        self.declare_parameter('timeout', 5.0)
        self.declare_parameter('position_tolerance', 5)
        
        # 获取参数
        port = self.get_parameter('port').value
        motor_id = self.get_parameter('motor_id').value
        bitrate = self.get_parameter('bitrate').value
        self.timeout = self.get_parameter('timeout').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        
        # 创建CAN驱动实例
        # self.gripper_driver = GripperCANDriver(
        #     can_channel=can_channel,
        #     motor_id=motor_id,
        #     bitrate=bitrate
        # )
        self.gripper_driver = Gripper485Driver(
            port=port,
            baud_rate=bitrate
        )
        
        # 更新位置参数
        self.gripper_driver.POSITION_OPEN = self.get_parameter('open_position').value
        self.gripper_driver.POSITION_CLOSE = self.get_parameter('close_position').value
        
        # 连接CAN总线
        if not self.gripper_driver.connect():
            self.get_logger().error(f"无法连接串口{port}，节点无法启动")
            return
        
        # 初始化电机（打开扭矩开关）
        if not self.gripper_driver.initialize():
            self.get_logger().error(f"电机初始化失败（无法打开扭矩开关），串口{port}已断开")
            self.gripper_driver.disconnect()
            return
        
        # 创建服务（使用可重入回调组以允许并发调用）
        self.callback_group = ReentrantCallbackGroup()
        self.service = self.create_service(
            GripperControl,
            'gripper/control',
            self.handle_gripper_control,
            callback_group=self.callback_group
        )
        
        # 创建定时器，定期发布状态
        # self.status_timer = self.create_timer(1.0, self.publish_status)
        
        
        self.get_logger().info(f"抓取器控制服务已启动，监听话题: {self.service.srv_name}")
        self.get_logger().info(f"打开位置: {self.gripper_driver.POSITION_OPEN}")
        self.get_logger().info(f"关闭位置: {self.gripper_driver.POSITION_CLOSE}")
        self.get_logger().info(f"位置容差: {self.position_tolerance}")
        self.get_logger().info(f"超时时间: {self.timeout}秒")
    
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
        
        if command == "open":
            success, current_position, message = self.gripper_driver.open_gripper()
        elif command == "close":
            success, current_position, message = self.gripper_driver.close_gripper()
        else:
            success = False
            current_position = 0
            message = f"无效命令: {command}，支持命令: 'open', 'close'"
        
        # 设置响应
        response.success = success
        response.current_position = int(current_position)
        response.message = message
        
        if success:
            self.get_logger().info(f"命令执行成功: {message}")
        else:
            self.get_logger().warn(f"命令执行失败: {message}")
        
        return response
    
    def publish_status(self):
        """定期发布电机状态（用于调试）"""
        status = self.gripper_driver.get_current_status()
        if status:
            self.get_logger().debug(
                f"电机状态: ID={status.motor_id}, "
                f"位置={status.position}, "
                f"速度={status.velocity}, "
                f"扭矩={status.torque}, "
                f"MOS温度={status.mos_temperature}℃, "
                f"转子温度={status.rotor_temperature}℃"
            )
    
    def destroy_node(self):
        """节点销毁时的清理工作"""

        self.get_logger().info("关闭抓取器控制节点...")
        
        if hasattr(self, 'gripper_driver'):
            self.gripper_driver.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # 创建节点
        node = GripperControlNode()
        
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
