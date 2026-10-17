#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from xc_interfaces.srv import ControlAGV
from xc_interfaces.msg import AGVStatus
import threading
import time


class AGVControlClient(Node):
    def __init__(self):
        super().__init__('agv_always_go_straght_node')
        
        self.reentrant_cb_group = ReentrantCallbackGroup()
        
        self.cli = self.create_client(ControlAGV, 'control_agv', callback_group=self.reentrant_cb_group)
        self.declare_parameter('is_continue_run', True)
        self.is_continue_run_ = self.get_parameter('is_continue_run').value
        self.last_is_continue_run_ = False
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待control_agv服务可用...')
        
        self.req = ControlAGV.Request()
        
        self.linear_speed_lock = threading.Lock()
        self.linear_speed = -100 #100
        self.req.angular_speed = 0
        self.req.lift_magnetic_sensor = True

        self.req_stop = ControlAGV.Request()
        self.req_stop.linear_speed = 0
        self.req_stop.angular_speed = 0
        self.req_stop.lift_magnetic_sensor = True
        
        self.agv_status_sub = self.create_subscription(
            AGVStatus,
            'agv_status',
            self.agv_status_callback,
            10,
            callback_group=self.reentrant_cb_group
        )
        
        self.timer = self.create_timer(0.1, self.send_request, callback_group=self.reentrant_cb_group)
        self.future = None

    def agv_status_callback(self, msg):
        # self.get_logger().info(f'white_card_id: {msg.white_card_id}')
        if msg.white_card_id == 23:
            with self.linear_speed_lock:
                self.linear_speed = 100
            # else:
            #     self.linear_speed = 100

    def send_request(self):
        self.is_continue_run_ = self.get_parameter('is_continue_run').value
        if self.is_continue_run_ != self.last_is_continue_run_:
            self.last_is_continue_run_ = self.is_continue_run_
            self.get_logger().info(f'是否继续直行: {self.is_continue_run_}')
        if self.is_continue_run_==True:
            with self.linear_speed_lock:
                self.req.linear_speed = self.linear_speed
            self.future = self.cli.call_async(self.req)
        else:
            self.future = self.cli.call_async(self.req_stop)
        time.sleep(0.1)

    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f'服务响应 - 控制命令成功: {response.control_cmd_success}, '
                f'状态命令成功: {response.status_cmd_success}, '
                f'左轮速度: {response.agv_status.left_wheel_speed}, '
                f'右轮速度: {response.agv_status.right_wheel_speed},'
                f'RFID卡号: {response.agv_status.white_card_id}'
            )
        except Exception as e:
            self.get_logger().error(f'服务调用失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    client = AGVControlClient()
    
    executor = MultiThreadedExecutor()
    executor.add_node(client)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
