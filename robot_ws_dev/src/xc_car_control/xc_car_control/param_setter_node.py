#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from rclpy.parameter import ParameterType


class ParamSetterNode(Node):
    def __init__(self):
        super().__init__('param_setter_node')
        
        # 目标节点和参数信息
        self.target_node = '/agv_control_node'
        self.param_name = 'is_chassis_running'
        self.param_value = False
        
        # 设置参数
        self.set_target_parameter()
    
    def set_target_parameter(self):
        # 创建服务客户端
        client = self.create_client(SetParameters, f'{self.target_node}/set_parameters')
        
        # 等待服务可用
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'等待 {self.target_node} 服务可用...')
        
        # 创建参数消息
        param_msg = Parameter()
        param_msg.name = self.param_name
        param_msg.value.type = ParameterType.PARAMETER_BOOL
        param_msg.value.bool_value = self.param_value
        
        # 创建请求
        request = SetParameters.Request()
        request.parameters = [param_msg]
        
        # 发送请求
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        # 处理响应并退出
        if future.result() is not None:
            result = future.result()
            if result.results[0].successful:
                self.get_logger().info(f'成功将 {self.target_node}节点的{self.param_name} 参数设置为 {self.param_value}')
            else:
                self.get_logger().error(f'设置参数失败: {result.results[0].reason}')
        else:
            self.get_logger().error('服务调用失败')
        
        # 处理完响应后退出程序
        rclpy.shutdown()
        exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = ParamSetterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()