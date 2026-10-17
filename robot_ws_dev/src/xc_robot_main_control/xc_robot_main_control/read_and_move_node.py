#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from xc_interfaces.srv import MoveToPose    
from xc_interfaces.srv import MoveJoints
from xc_interfaces.msg import ArmStatus
import math
import time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import threading
import csv
import os
from datetime import datetime
from geometry_msgs.msg import Point, Quaternion

"""
这个节点的功能是读取config/target_positions.txt文件中的目标坐标点，
并根据坐标点控制机器人到指定目标点，然后返回果框位置。
同时记录机械臂状态到CSV文件中。
"""

class ReadAndMoveNode(Node):
    def __init__(self):
        super().__init__('read_and_move_node')
        
        self.is_go_all_points = True #如果碰撞检测失败则放弃执行这个点
        self.is_record_arm_status = False #是否记录机械臂状态
        # 创建回调组
        self.reentrant_cb_group = ReentrantCallbackGroup()  # 用于arm_status订阅
        self.mutex_cb_group = MutuallyExclusiveCallbackGroup()  # 用于机械臂运动控制
        
        # 创建服务客户端
        self.move_to_pose_client = self.create_client(MoveToPose, 'move_to_pose', callback_group=self.mutex_cb_group)
        self.return_to_box_client = self.create_client(MoveToPose, 'return_to_box', callback_group=self.mutex_cb_group)
        self.move_joints_client = self.create_client(MoveJoints, 'move_joints', callback_group=self.mutex_cb_group)

        # 创建订阅者用于接收机械臂状态
        self.arm_status_subscriber = self.create_subscription(
            ArmStatus,
            'arm_status',
            self.arm_status_callback,
            10,
            callback_group=self.reentrant_cb_group
        )
        
        # 等待服务可用
        while not self.move_to_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待move_to_pose服务可用...')
        
        while not self.return_to_box_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待return_to_box服务可用...')
        
        # 文件路径参数
        # self.declare_parameter('file_path', '/home/xucan/code/test/robot_ws/src/xc_robot_main_control/config/target_positions.txt')#office WSL
        # self.declare_parameter('file_path', '/home/homevm/code/test/robot_ws/src/xc_robot_main_control/config/target_positions.txt')#home VMWare
        self.declare_parameter('file_path', '/home/xc/code/test/ros2/robot_ws/src/xc_robot_main_control/config/target_positions.txt')#工控机
        
        self.declare_parameter('loop_execution', True) 
        self.loop_execution = self.get_parameter('loop_execution').value
        
        self.get_logger().info(f'循环执行模式: {"启用" if self.loop_execution else "禁用"}')



        self.file_path = self.get_parameter('file_path').value
        
        # 初始化线程锁
        self.lock = threading.Lock()
        self.task_queue = []
        self.current_task_index = 0
        self.is_processing = False
        
        # CSV记录相关变量
        self.csv_recording = False
        self.csv_data = []
        self.target_point_number = 0  # 目标点编号
        self.error_code = None  # 错误码
        self.csv_filename = f'arm_status_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # 读取文件并执行任务
        self.read_file_and_execute()
    
    def arm_status_callback(self, msg):
        """机械臂状态回调函数"""
        if self.csv_recording:
            # 记录当前时间、位置、姿态和关节角度
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            position = msg.position
            rx = msg.rx
            ry = msg.ry
            rz = msg.rz
            joints = list(msg.joints)
            
            # 添加到CSV数据列表
            row = [
                timestamp,
                self.target_point_number,
                position.x, position.y, position.z,
                rx, ry, rz,
                joints[0], joints[1], joints[2], 
                joints[3], joints[4], joints[5],
                self.error_code if self.error_code is not None else ''
            ]
            self.csv_data.append(row)
    
    def start_csv_recording(self, task_number):
        """开始记录CSV数据"""
        with self.lock:
            self.csv_recording = True
            self.target_point_number = task_number
            self.error_code = None
            self.csv_data = []  # 清空之前的数据
    
    def stop_csv_recording(self, error_code=None):
        """停止记录CSV数据"""
        with self.lock:
            self.error_code = error_code
            self.csv_recording = False
            
            # 将数据写入CSV文件
            if self.csv_data:
                self.save_csv_data()
    
    def save_csv_data(self):
        """保存CSV数据到文件"""
        headers = [
            'Timestamp', 'Target_Point_Number', 
            'Position_X', 'Position_Y', 'Position_Z',
            'RX', 'RY', 'RZ',
            'Joint_0', 'Joint_1', 'Joint_2', 
            'Joint_3', 'Joint_4', 'Joint_5',
            'Error_Code'
        ]
        
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.isfile(self.csv_filename)
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)  # 写入表头
            writer.writerows(self.csv_data)  # 写入数据
        
        self.get_logger().info(f'已保存 {len(self.csv_data)} 条机械臂状态数据到 {self.csv_filename}')
    
    def read_file_and_execute(self):
        """读取文件并将任务加入队列"""
        try:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释行
                    continue
                
                try:
                    # 解析每一行数据
                    data = list(map(float, line.split()))
                    if len(data) != 7:
                        self.get_logger().warn(f'第{line_num+1}行数据格式不正确，跳过: {line}')
                        continue
                    
                    x, y, z, rx_deg, ry_deg, rz_deg, mid2target_angle_deg = data
                    
                    # 将角度转换为弧度
                    rx_rad = rx_deg * math.pi / 180.0
                    ry_rad = ry_deg * math.pi / 180.0
                    rz_rad = rz_deg * math.pi / 180.0
                    mid2target_angle_rad = mid2target_angle_deg * math.pi / 180.0
                    
                    # 将任务添加到队列
                    self.task_queue.append({
                        'x': x,
                        'y': y, 
                        'z': z,
                        'rx': rx_rad,
                        'ry': ry_rad,
                        'rz': rz_rad,
                        'mid2target_angle': mid2target_angle_rad,
                        'line_num': line_num + 1
                    })
                    
                except ValueError as e:
                    self.get_logger().error(f'解析第{line_num+1}行数据时出错: {e}')
                    continue
                except Exception as e:
                    self.get_logger().error(f'处理第{line_num+1}行数据时发生未知错误: {e}')
                    continue
            
            if len(self.task_queue) > 0:
                self.get_logger().info(f'成功加载{len(self.task_queue)}个任务')
                self.process_next_task()
            else:
                self.get_logger().warn('没有有效的任务数据')
        
        except FileNotFoundError:
            self.get_logger().error(f'找不到文件: {self.file_path}')
        except Exception as e:
            self.get_logger().error(f'读取文件时发生错误: {e}')
    
    def process_next_task(self):
        """处理下一个任务"""
        with self.lock:
            if self.current_task_index >= len(self.task_queue):
                if self.loop_execution:
                    self.get_logger().info('所有任务已处理完毕，开始循环执行')
                    self.current_task_index = 0
                else:
                    self.get_logger().info('所有任务已处理完毕')
                    return
            
            if self.is_processing:
                # 如果正在处理，则稍后再试
                self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))
                self.process_next_task()
                return
            
            task = self.task_queue[self.current_task_index]
            self.is_processing = True
        
        self.get_logger().info(f'执行第{task["line_num"]}行数据: X={task["x"]}, Y={task["y"]}, Z={task["z"]}, '
                              f'RX={math.degrees(task["rx"])}°, RY={math.degrees(task["ry"])}°, RZ={math.degrees(task["rz"])}°, '
                              f'mid2target_angle={math.degrees(task["mid2target_angle"])}°')
        
        # 开始记录CSV数据
        if self.is_record_arm_status:
            self.start_csv_recording(task['line_num'])
        
        # 调用move_to_pose服务
        self.call_move_to_pose(task)
    
    def call_move_to_pose(self, task):
        """调用move_to_pose服务"""
        request = MoveToPose.Request()
        request.x = task['x']
        request.y = task['y']
        request.z = task['z']
        request.rx = task['rx']
        request.ry = task['ry']
        request.rz = task['rz']
        request.mid2target_angle = task['mid2target_angle']
        
        self.get_logger().info(f'发送move_to_pose请求: pos=({request.x}, {request.y}, {request.z}), '
                              f'rot=({math.degrees(request.rx)}, {math.degrees(request.ry)}, {math.degrees(request.rz)}), '
                              f'mid2target_angle={math.degrees(request.mid2target_angle)}°')
        
        future = self.move_to_pose_client.call_async(request)
        future.add_done_callback(lambda future: self.move_to_pose_callback(future, task))
    
    def move_to_pose_callback(self, future, task):
        """move_to_pose服务回调函数"""
        try:
            response = future.result()
            if response.result == 0:  # 0表示成功
                self.get_logger().info('move_to_pose服务调用成功')
                
                # 等待1秒
                time.sleep(1.0)
                
                # 调用return_to_box服务
                self.get_logger().info('调用return_to_box服务')
                self.call_return_to_box(task)
            else:
                self.get_logger().error(f'move_to_pose服务调用失败，错误码: {response.result}')
                # 停止CSV记录并传递错误码
                self.stop_csv_recording(error_code=response.result)
                self.on_task_completed(future.result())
        except Exception as e:
            self.get_logger().error(f'处理move_to_pose响应时发生错误: {e}')
            # 停止CSV记录并传递错误码
            self.stop_csv_recording(error_code=-1)  # -1 表示内部错误
            self.on_task_completed(future.result())
    
    def call_return_to_box(self, task):
        """调用return_to_box服务"""
        # 创建一个默认的请求，具体目标位置由服务器端决定
        task = self.task_queue[self.current_task_index]
        request = MoveToPose.Request()
        request.x = task['x']
        request.y = task['y']
        request.z = task['z']
        request.rx = task['rx']
        request.ry = task['ry']
        request.rz = task['rz']
        request.mid2target_angle = task['mid2target_angle']
        
        self.get_logger().info('发送return_to_box请求')
        
        future = self.return_to_box_client.call_async(request)
        future.add_done_callback(lambda future: self.return_to_box_callback(future, task))
    
    def return_to_box_callback(self, future, task):
        """return_to_box服务回调函数"""
        try:
            response = future.result()
            if response.result == 0:  # 0表示成功
                self.get_logger().info('return_to_box服务调用成功')
            else:
                self.get_logger().error(f'return_to_box服务调用失败，错误码: {response.result}')
        except Exception as e:
            self.get_logger().error(f'处理return_to_box响应时发生错误: {e}')
        finally:
            # 停止CSV记录，此时不传递错误码，因为这是正常完成
            self.stop_csv_recording()
            self.on_task_completed(future.result())
    
    def on_task_completed(self, response):
        """任务完成后的处理"""
        if response.result == 0:
            with self.lock:
                self.is_processing = False
                self.current_task_index += 1
            
            # 处理下一个任务
            self.process_next_task()
        else:
            self.get_logger().error(f'任务{self.current_task_index + 1}执行失败，错误码: {response.result}')
            request = MoveJoints.Request()                
            request.joints[0]= 10.32*math.pi/180
            request.joints[1]= 15.37*math.pi/180 
            request.joints[2]=134.05*math.pi/180
            request.joints[3]=117.05*math.pi/180
            request.joints[4]=86.63*math.pi/180
            request.joints[5]=0.0*math.pi/180
            move_joint2initial_future = self.move_joints_client.call_async(request)
            move_joint2initial_future.add_done_callback(
                    lambda future: self.move_joints_callback(future)
                )
    
    def move_joints_callback(self, future):
        """move_joints服务回调函数"""
        try:
            response = future.result()
            if response.result == 0:  # 0表示成功
                self.get_logger().info('move_joints服务调用成功')
                self.on_task_completed(future.result())
            else:
                self.get_logger().error(f'move_joints服务调用失败，错误码: {response.result}')
                #强制完成当前任务
                response.result = 0
                self.on_task_completed(response)
        except Exception as e:
            self.get_logger().error(f'处理move_joints响应时发生错误: {e}')
        

def main(args=None):
    rclpy.init(args=args)
    
    node = ReadAndMoveNode()
    
    # 使用MultiThreadedExecutor开始多线程
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('接收到键盘中断信号')
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()