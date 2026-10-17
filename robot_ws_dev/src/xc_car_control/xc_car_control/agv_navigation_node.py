#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from xc_interfaces.srv import ControlAGV
from xc_interfaces.msg import AGVStatus
import csv
import time
import os
import threading
import queue

"""
节点实现了以下状态：

IDLE: 空闲状态，等待开始执行指令
EXECUTING_STRAIGHT: 执行直线运动指令
EXECUTING_ROTATION: 执行旋转运动指令
WAITING_FOR_TAPE: 旋转后等待磁条检测

csv_file: 导航路径CSV文件
loop_rate: 导航循环频率
command_interval: 指令发送间隔
rotation_timeout: 旋转超时时间

1：磁条传感器降下前进；
2：磁条传感器降下后退；
3：磁条传感器升起前进；
4：磁条传感器升起后退；
5：磁条传感器下降左转，以前进方向为准；
6：磁条传感器下降右转，以前进方向为准；

"""

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from xc_interfaces.srv import ControlAGV
import csv
import time
import os
import numpy as np

class AGVNavigationNode(Node):
    def __init__(self):
        super().__init__('agv_navigation_node')
        
        # 声明参数
        self.declare_parameter('csv_file', '/home/xc/code/test/ros2/robot_ws/install/xc_car_control/share/xc_car_control/navigation_file/navigation_file.csv')
        self.declare_parameter('command_interval', 0.1)  # 秒
        self.declare_parameter('rotation_timeout', 2.0)  # 秒
        self.declare_parameter('stuck_timeout', 20.0)  # 直行/后退卡住超时时间

        self.declare_parameter('is_continue_run', True)
        self.is_continue_run_ = self.get_parameter('is_continue_run').value
        
        # 获取参数
        csv_file = self.get_parameter('csv_file').value
        self.command_interval = self.get_parameter('command_interval').value
        self.rotation_timeout = self.get_parameter('rotation_timeout').value
        self.stuck_timeout = self.get_parameter('stuck_timeout').value

        # 加载导航路径
        self.navigation_path = self.load_navigation_path(csv_file)
        self.get_logger().info(f"加载了{csv_file}文件，共有 {len(self.navigation_path)} 条导航指令")
        
        # 创建服务客户端
        self.control_client = self.create_client(ControlAGV, 'control_agv')
        while not self.control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待control_agv服务可用...')
        
        # 用于同步服务调用的线程和事件
        self.service_thread = None
        self.service_result = None
        self.service_event = threading.Event()
        
        # 添加全局变量存储底盘状态
        self.agv_status_ = None
        
        # 当前状态
        self.current_index = 0
        self.last_command_time = 0
        self.rotation_start_time = 0
        self.straight_start_time = 0
        self.execution_state = "IDLE"  # IDLE, EXECUTING_STRAIGHT, EXECUTING_ROTATION, WAITING_FOR_TAPE
        
        # 初始化底盘状态
        self.initialize_agv_status()
        
        self.get_logger().info("AGV导航节点已启动")

    def load_navigation_path(self, csv_file):
        """加载CSV导航路径文件"""
        path = []
        try:
            if not os.path.exists(csv_file):
                self.get_logger().error(f"CSV文件不存在: {csv_file}")
                return path
            
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            card_id = int(row[0])
                            action = int(row[1])
                            path.append({'card_id': card_id, 'action': action})
                        except ValueError:
                            self.get_logger().warn(f"跳过无效行: {row}")
            return path
        except Exception as e:
            self.get_logger().error(f"加载CSV文件时出错: {str(e)}")
            return []

    def _service_call_worker(self, request):
        """服务调用工作线程"""
        try:
            future = self.control_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                self.service_result = future.result()
            else:
                self.service_result = None
        except Exception as e:
            self.get_logger().error(f"服务调用异常: {str(e)}")
            self.service_result = None
        finally:
            self.service_event.set()

    def call_control_service(self, linear_speed, angular_speed, lift_magnetic_sensor):
        """调用控制服务（同步方式）"""
        # self.get_logger().info("准备调用服务")
        request = ControlAGV.Request()
        request.linear_speed = linear_speed
        request.angular_speed = angular_speed
        request.lift_magnetic_sensor = lift_magnetic_sensor
        
        # 重置事件和结果
        self.service_event.clear()
        self.service_result = None
        
        # 在单独的线程中执行服务调用
        self.service_thread = threading.Thread(target=self._service_call_worker, args=(request,))
        self.service_thread.start()
        
        # 等待服务调用完成（阻塞当前线程，但不阻塞ROS2事件循环）
        self.service_event.wait(timeout=1.0)
        
        if self.service_result is not None:
            # self.get_logger().info("服务调用成功")
            # 更新全局变量存储底盘状态
            self.agv_status_ = self.service_result.agv_status
            return self.service_result
        else:
            self.get_logger().error("服务调用超时或失败")
            return None


    def initialize_agv_status(self):
        """初始化AGV状态"""
        try:
            # 尝试获取一次AGV状态
            response = self.call_control_service(0, 0, False)
            if response and response.status_cmd_success:
                self.agv_status_ = response.agv_status
                self.get_logger().info("AGV状态初始化成功")
            else:
                self.get_logger().warn("AGV状态初始化失败")
        except Exception as e:
            self.get_logger().warn(f"AGV状态初始化异常: {str(e)}")

    def execute_navigation(self):
        """执行导航"""
        self.get_logger().info("开始执行导航")
        
        # 重置状态
        self.current_index = 0
        self.execution_state = "IDLE"
        
        # 遍历所有导航指令
        while self.current_index < len(self.navigation_path):
            command = self.navigation_path[self.current_index]
            self.get_logger().info(f"执行指令 {self.current_index+1}: 白卡 {command['card_id']}, 动作 {command['action']}")
            
            # 执行指令直到完成
            self.execute_command_until_complete(command)
            
            # 移动到下一个指令
            self.current_index += 1
        
        self.get_logger().info("导航路径完成!")
        # 发送停止命令
        self.call_control_service(0, 0, False)

    def execute_command_until_complete(self, command):
        """执行单个指令直到完成"""
        action = command['action']
        
        if action in [1, 2, 3, 4]:  # 直线运动
            self.execute_straight_command_until_complete(action)
        elif action in [5, 6]:  # 旋转运动
            self.execute_rotation_command_until_complete(action)

    def execute_straight_command_until_complete(self, action):
        """执行直线运动指令直到完成"""
        self.execution_state = "EXECUTING_STRAIGHT"
        self.straight_start_time = time.time()
        self.last_command_time = 0
        
        self.get_logger().info(f"开始直行/后退指令: {'前进' if action in [1, 3] else '后退'}")
        temp_flag = 0      
        # 持续执行直到完成
        while rclpy.ok():
            current_time = time.time()
            
            # 检查是否超时
            if current_time - self.straight_start_time > self.stuck_timeout:
                # 直接使用全局变量获取状态
                temp_status = self.agv_status_
                # if self.agv_status_ is not None and not (self.agv_status_.front_magnetic_sensor or self.agv_status_.rear_magnetic_sensor) and action in [1, 2]:
                if temp_status is not None and not (temp_status.front_magnetic_sensor or temp_status.rear_magnetic_sensor) and action in [1, 2]:
                    if(temp_flag<3):
                        temp_flag += 1
                        self.get_logger().error(f"巡线直行指令超时,磁条传感器第{temp_flag}次未检测到磁条!")
                        self.get_logger().error(f"底盘状态: front_magnetic_sensor:{temp_status.front_magnetic_sensor}, rear_magnetic_sensor:{temp_status.rear_magnetic_sensor}, action:{action}")

                    else:
                        #TODO 离开了导航线,需要进入导航修正程序
                        self.handle_error(f"巡线直行指令超时，且磁条传感器未检测到磁条!")
                        self.get_logger().error(f"底盘状态: front_magnetic_sensor:{temp_status.front_magnetic_sensor}, rear_magnetic_sensor:{temp_status.rear_magnetic_sensor}, action:{action}")
                        return
            
            # 检查是否需要发送新命令
            if current_time - self.last_command_time >= self.command_interval:
                linear_speed = 100  # mm/s
                angular_speed = 0
                lift_magnetic_sensor=False
                
                if action in [3, 4]:  # 动作3和4需要升起传感器
                    lift_magnetic_sensor=True
                # lift_magnetic_sensor = action in [3, 4]  # 动作3和4需要升起传感器
                
                # 设置方向
                if action in [2, 4]:  # 后退
                    linear_speed = -linear_speed
                
                # 调用服务 TODO
                self.is_continue_run_ = self.get_parameter('is_continue_run').value
                if self.is_continue_run_==True:
                    response = self.call_control_service(linear_speed, angular_speed, lift_magnetic_sensor)
                    
                    if response and response.control_cmd_success:
                        self.last_command_time = current_time
                        self.get_logger().debug(f"是否继续导航：{self.is_continue_run_}, 发送直线命令: 速度={linear_speed}, 升降={lift_magnetic_sensor}")
                    else:
                        self.get_logger().warn("直线命令发送失败")
                else:
                    response = self.call_control_service(0, 0, lift_magnetic_sensor)
                    
                    if response and response.control_cmd_success:
                        self.last_command_time = current_time
                        self.get_logger().debug(f"是否继续导航：{self.is_continue_run_}, 发送直线命令: 速度=0, 升降={lift_magnetic_sensor}")
                    else:
                        self.get_logger().warn("直线命令发送失败")
            
            # 检查是否完成
            # 直接使用全局变量获取状态
            if self.agv_status_ is not None:
                # 检查是否到达目标白卡 (下一个导航指令的白卡ID)
                self.get_logger().info(f"当前读取到的白卡ID: {self.agv_status_.white_card_id}")
                self.get_logger().info(f"目标白卡ID: {self.navigation_path[self.current_index]['card_id']}")
                next_index = self.current_index + 1
                if next_index < len(self.navigation_path):
                    if self.agv_status_.white_card_id == self.navigation_path[next_index]['card_id']:
                        self.get_logger().info("到达目标白卡，直行指令完成")
                        break
                    elif self.agv_status_.white_card_id != self.navigation_path[self.current_index]['card_id'] and self.agv_status_.white_card_id !=0:
                        #TODO 错过了正确的卡,需要进入导航修正程序
                        break
                else:
                    # 如果是最后一个指令，保持原来的逻辑
                    if self.agv_status_.white_card_id == self.navigation_path[self.current_index]['card_id']:
                        self.get_logger().info("到达目标白卡，直行指令完成")
                        break
            
            # 短暂休眠以避免过度占用CPU
            time.sleep(0.1)
        
        self.execution_state = "IDLE"

    def execute_rotation_command_until_complete(self, action):
        """执行旋转运动指令直到完成"""
        self.execution_state = "EXECUTING_ROTATION"
        self.rotation_start_time = time.time()
        self.last_command_time = 0
        
        self.get_logger().info(f"开始旋转: {'左' if action == 5 else '右'}转")
        
        # 持续执行直到完成
        while rclpy.ok():
            current_time = time.time()
            
            # 检查是否超时
            if current_time - self.rotation_start_time >= self.rotation_timeout:
                self.execution_state = "WAITING_FOR_TAPE"
                self.get_logger().info("初始化旋转完成，等待磁条检测")
                break
            
            # 检查是否需要发送新命令
            if current_time - self.last_command_time >= self.command_interval:
                angular_speed = -100  # mm/s
                if action == 6:  # 右转
                    angular_speed = -angular_speed
                    
                response = self.call_control_service(0, angular_speed, False)
                
                if response and response.control_cmd_success:
                    self.last_command_time = current_time
                    self.get_logger().debug(f"发送旋转命令: 角速度={angular_speed}")
                else:
                    self.get_logger().warn("旋转命令发送失败")
            
            # 短暂休眠以避免过度占用CPU
            time.sleep(0.1)
        
        # 等待磁条检测
        if self.execution_state == "WAITING_FOR_TAPE":
            
            self.get_logger().info("初始化旋转完成，继续旋转11111")
            while rclpy.ok():
                current_time = time.time()
                
                # 发送旋转命令
                if current_time - self.last_command_time >= 0.03:
                    # self.call_control_service(0, 0, False)
                    angular_speed = -100  # mm/s
                    if action == 6:  # 右转
                        angular_speed = -angular_speed                    
                    response = self.call_control_service(0, angular_speed, False)
                    self.last_command_time = current_time               
                

                # 检查磁条传感器状态
                # 直接使用全局变量获取状态
                if self.agv_status_ is not None and (self.agv_status_.front_magnetic_sensor or self.agv_status_.rear_magnetic_sensor):
                    self.get_logger().info("检测到磁条，旋转完成")
                    break
                # 短暂休眠以避免过度占用CPU
                time.sleep(0.04)                
                
                
        
        self.execution_state = "IDLE"

    def handle_error(self, message):
        """处理导航错误"""
        self.get_logger().error(message)
        # 发送停止命令
        self.call_control_service(0, 0, False)
        # 停止导航
        self.current_index = len(self.navigation_path)  # 停止导航

    # 删除get_agv_status方法


def main(args=None):
    rclpy.init(args=args)
    node = AGVNavigationNode()
    try:
        # 执行导航
        node.execute_navigation()
    except KeyboardInterrupt:
        pass
    finally:
        # 确保AGV停止
        node.call_control_service(0, 0, False)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()