#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from xc_interfaces.srv import ControlAGV
import time

class SimpleControlClient(Node):
    def __init__(self):
        super().__init__('simple_control_client')
        self.client = self.create_client(ControlAGV, 'control_agv')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待control_agv服务可用...')
        self.get_logger().info('已连接到control_agv服务')
        
    def send_control_request(self, linear_speed, angular_speed, lift_magnetic_sensor):
        request = ControlAGV.Request()
        request.linear_speed = linear_speed
        request.angular_speed = angular_speed
        request.lift_magnetic_sensor = lift_magnetic_sensor
        
        self.get_logger().info(f'发送请求: linear_speed={linear_speed}, angular_speed={angular_speed}, lift_magnetic_sensor={lift_magnetic_sensor}')
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'收到响应: control_cmd_success={response.control_cmd_success}, status_cmd_success={response.status_cmd_success}')
            self.get_logger().info(f'AGV状态: white_card_id={response.agv_status.white_card_id}, battery_voltage={response.agv_status.battery_voltage}')
            return response
        else:
            self.get_logger().error('服务调用失败')
            return None


def main(args=None):
    rclpy.init(args=args)
    client = SimpleControlClient()
    
    try:
        while rclpy.ok():
            # 发送一个简单的控制请求
            response = client.send_control_request(0, 0, False)  # 直行速度100mm/s
            
            if response is not None:
                # 等待0.2秒后继续
                # self.get_logger().info('服务调用成功')
                time.sleep(0.2)
            else:
                # 如果服务调用失败，等待一段时间再重试
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        pass
    finally:
        # 发送停止命令
        client.send_control_request(0, 0, False)
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()