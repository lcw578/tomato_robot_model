#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from your_interface_pkg.srv import ServiceA, ServiceB  # 请替换为实际的服务接口
from your_interface_pkg.msg import TopicCMessage      # 请替换为实际的话题消息类型

class ServiceChainNode(Node):
    def __init__(self):
        super().__init__('service_chain_node')
        
        # 使用可重入回调组，允许在回调中执行其他回调
        self.callback_group = ReentrantCallbackGroup()
        
        # 创建服务A和B的客户端
        self.client_a = self.create_client(
            ServiceA, 
            'service_a',
            callback_group=self.callback_group
        )
        self.client_b = self.create_client(
            ServiceB, 
            'service_b',
            callback_group=self.callback_group
        )
        
        # 等待服务可用
        while not self.client_a.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务A上线...')
        while not self.client_b.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务B上线...')
        
        # 创建话题C的订阅者
        self.subscription = self.create_subscription(
            TopicCMessage,
            'topic_c',
            self.topic_c_callback,
            10,  # 队列大小
            callback_group=self.callback_group
        )
        
        self.get_logger().info('服务链节点已初始化完成，等待话题C消息...')
        
        # 标志位，用于防止在处理一个消息时处理另一个消息（可选）
        self.is_processing = False

    def topic_c_callback(self, msg):
        """
        话题C的回调函数
        当收到话题C的消息时，触发服务调用链
        """
        if self.is_processing:
            self.get_logger().warning('正在处理前一个请求，跳过新消息')
            return
            
        self.is_processing = True
        self.get_logger().info(f'收到话题C消息: {msg}')
        
        # 第一步：调用服务A
        self.call_service_a(msg)

    def call_service_a(self, topic_msg):
        """调用服务A"""
        self.get_logger().info('调用服务A...')
        
        # 创建服务A的请求（根据实际接口调整）
        request_a = ServiceA.Request()
        # 根据topic_msg设置request_a的参数
        
        # 异步调用服务A
        future_a = self.client_a.call_async(request_a)
        future_a.add_done_callback(
            lambda future: self.service_a_callback(future, topic_msg)
        )

    def service_a_callback(self, future, topic_msg):
        """服务A的响应回调"""
        try:
            response_a = future.result()
            self.get_logger().info(f'服务A返回结果: {response_a.success}')
            
            if response_a.success:
                # 服务A成功，继续调用服务B
                self.call_service_b(topic_msg, response_a)
            else:
                self.get_logger().warning('服务A返回False，停止处理')
                self.is_processing = False
                
        except Exception as e:
            self.get_logger().error(f'服务A调用失败: {str(e)}')
            self.is_processing = False

    def call_service_b(self, topic_msg, response_a):
        """调用服务B"""
        self.get_logger().info('调用服务B...')
        
        # 创建服务B的请求（根据实际接口调整）
        request_b = ServiceB.Request()
        # 根据topic_msg和response_a设置request_b的参数
        
        # 异步调用服务B
        future_b = self.client_b.call_async(request_b)
        future_b.add_done_callback(
            lambda future: self.service_b_callback(future, topic_msg, response_a)
        )

    def service_b_callback(self, future, topic_msg, response_a):
        """服务B的响应回调"""
        try:
            response_b = future.result()
            self.get_logger().info(f'服务B返回结果: {response_b}')
            self.get_logger().info('话题C回调处理完成')
            
        except Exception as e:
            self.get_logger().error(f'服务B调用失败: {str(e)}')
        
        # 处理完成，重置标志位
        self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # 创建节点
        node = ServiceChainNode()
        
        # 使用多线程执行器，允许并行处理多个回调
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('接收到终止信号')
        finally:
            node.destroy_node()
            executor.shutdown()
            
    except Exception as e:
        print(f'节点启动失败: {str(e)}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()