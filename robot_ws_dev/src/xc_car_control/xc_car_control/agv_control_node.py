import struct
import threading
import time
import serial
import rclpy
from rclpy.node import Node
from xc_interfaces.srv import ControlAGV
from xc_interfaces.msg import AGVStatus


from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

# ros2 service call /control_agv xc_interfaces/srv/ControlAGV "{linear_speed: 0, angular_speed: 0, lift_magnetic_sensor: true}"
# ros2 param set /agv_control_node is_chassis_running False  暂停导航参数
# ros2 param set /agv_control_node is_chassis_running True   继续导航参数

class AGVControlNode(Node):
    def __init__(self):
        super().__init__('agv_control_node')
        
        # 声明串口参数
        self.declare_parameter('serial_port', '/dev/ttyS0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('is_chassis_running', True)

        # 初始化底盘是否运行导航
        self.is_chassis_running_ = self.get_parameter('is_chassis_running').value
        self.reentrant_cb_group = ReentrantCallbackGroup()

        
        try:
            # 获取串口参数
            serial_port = self.get_parameter('serial_port').value
            baud_rate = self.get_parameter('baud_rate').value
            
            # 初始化串口
            self.ser = serial.Serial(
                port=serial_port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            self.serial_lock = threading.Lock()
            self.get_logger().info(f"成功连接串口: {serial_port}")
        except Exception as e:
            self.get_logger().error(f"串口初始化失败: {str(e)}")
            self.ser = None
        self.agv_status_lock = threading.Lock()
        # 创建服务
        self.srv = self.create_service(
            ControlAGV, 
            'control_agv', 
            self.control_agv_callback,
            callback_group=self.reentrant_cb_group)
        
        self.agv_status_pub = self.create_publisher(AGVStatus, 'agv_status', 5, callback_group=self.reentrant_cb_group)
        self.timer_ = self.create_timer(0.1, self.publish_agv_status, callback_group=self.reentrant_cb_group)

        self.agv_status_=None
        self.get_logger().info("AGV控制服务已启动")

    
    def publish_agv_status(self):
        with self.agv_status_lock:
            if self.ser is None:
                self.get_logger().warn("串口未初始化，无法发送状态获取命令")
                return
            # 1. 发送状态获取命令
            status_frame = self.build_status_frame()
            self.ser.write(status_frame)
            # self.get_logger().info(f"已发送状态获取命令: {status_frame.hex()}")
            # 2. 等待状态响应
            status_response = self.receive_frame(timeout=0.5)
            if status_response:
                # self.get_logger().info(f"接收到状态响应帧: {status_response.hex()}")
                
                
                # 检查命令码是否正确
                if status_response[3] == 0x01:  # 状态命令码
                    # self.get_logger().info("收到有效的状态响应")
                    try:
                        agv_status = self.parse_status_frame(status_response)
                        # 3. 更新内部状态
                        self.agv_status_ = agv_status
                        # 4. 发布状态至外部话题
                        self.agv_status_pub.publish(agv_status)
                        # self.get_logger().info("成功解析状态帧并发布至话题")
                    except Exception as e:
                        self.get_logger().error(f"解析状态帧时出错: {str(e)}")
                else:
                    self.get_logger().warn(f"无效的状态命令码: {status_response[3]}")
            else:
                self.get_logger().warn("状态响应超时或无效") 
    def control_agv_callback(self, request, response):
        # 初始化响应
        response.control_cmd_success = False
        response.status_cmd_success = False
        response.agv_status = AGVStatus()
        # self.get_logger().info("收到控制请求，准备开始执行服务")
        self.is_chassis_running_ = self.get_parameter('is_chassis_running').value
        # self.get_logger().info(f"底盘是否继续导航is_chassis_running_: {self.is_chassis_running_}")
        
        # 检查串口是否可用
        if self.ser is None:
            self.get_logger().error("服务调用失败: 接口未初始化")
            return response
        
        try:
            with self.agv_status_lock:
                # 1. 构建控制命令帧 (命令码0x02)
                if self.is_chassis_running_==False and request.angular_speed==0:#旋转中不允许暂停导航
                    # self.get_logger().info(f"暂停导航")
                    control_frame = self.build_control_frame(
                    linear_speed=0,
                    angular_speed=0,
                    lift_magnetic_sensor=request.lift_magnetic_sensor                
                )
                else:
                    # self.get_logger().info(f"服务请求参数: 直行速度：{request.linear_speed}，转向速度：{request.angular_speed}，传感器状态：{request.lift_magnetic_sensor}")
                    control_frame = self.build_control_frame(
                        linear_speed=request.linear_speed,
                        angular_speed=request.angular_speed,
                        lift_magnetic_sensor=request.lift_magnetic_sensor                
                    )
                
                
                # 2. 发送控制命令
                with self.serial_lock:
                    self.ser.reset_input_buffer()
                    self.ser.write(control_frame)
                    # self.get_logger().info(f"已发送控制命令帧: {control_frame.hex()}")
                    
                    # 3. 等待控制命令应答
                    control_response = self.receive_frame(timeout=0.5)
                    if control_response:
                        # self.get_logger().info(f"接收到控制响应帧: {control_response.hex()}")
                        
                        # 检查命令码是否正确
                        if control_response[3] == 0x02:  # 控制命令应答
                            response.control_cmd_success = True
                            # self.get_logger().info("收到有效的控制命令应答")
                            

                            # 4.获取agv状态                            
                            if self.agv_status_ is not None:
                                response.status_cmd_success = True
                                response.agv_status = self.agv_status_ 
                            else:
                                response.status_cmd_success = False
                        else:
                            self.get_logger().warn(f"无效的控制命令码: {control_response[3]}")
                    else:
                        self.get_logger().warn("控制命令应答超时或无效")
        except serial.SerialException as e:
            self.get_logger().error(f"串口通信错误: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"服务处理异常: {str(e)}")
        # self.get_logger().info(f"返回服务结果: {response.status_cmd_success}")
        return response

    def build_control_frame(self, linear_speed, angular_speed, lift_magnetic_sensor):
        """构建控制命令帧 (命令码0x02)"""
        # 默认小车编号=1
        car_id = 1
        
        # 构建数据域 (16字节)
        data_field = bytearray(16)
        
        # 小车编号 (小端)
        struct.pack_into('<H', data_field, 0, car_id)
        
        # 急停关闭 (0), 自动上传关闭 (0)
        data_field[2] = 0  # 急停
        data_field[3] = 0  # 自动上传
        
        # 直行速度 (小端有符号)
        linear_speed = max(min(linear_speed, 1200), -1200)
        struct.pack_into('<h', data_field, 4, linear_speed)
        
        # 旋转速度 (小端有符号)
        angular_speed = max(min(angular_speed, 1200), -1200)
        struct.pack_into('<h', data_field, 6, angular_speed)
        
        # 超声波刹车距离 (250mm)
        struct.pack_into('<H', data_field, 8, 250)
        
        # 磁条传感器升降
        data_field[12] = 1 if lift_magnetic_sensor else 0
        
        # 构建完整帧
        return self.build_frame(0x02, data_field)

    def build_status_frame(self):
        """构建状态获取命令帧 (命令码0x01)"""
        # 默认小车编号=1
        car_id = 1
        
        # 数据域 (16字节全0)
        data_field = bytearray(16)
        
        # 设置小车编号
        struct.pack_into('<H', data_field, 0, car_id)
        
        return self.build_frame(0x01, data_field)

    def build_frame(self, command_code, data_field):
        """构建完整数据帧"""
        frame = bytearray()
        
        # 帧头
        frame.extend([0xDE, 0xED])
        
        # 帧长度 (固定0x16 = 22)
        frame.append(0x16)
        
        # 命令码
        frame.append(command_code)
        
        # 数据域 (16字节)
        frame.extend(data_field)
        
        # 计算校验和 (除校验码外所有字节求和)
        checksum = sum(frame) & 0xFFFF
        frame.extend(struct.pack('<H', checksum))
        
        return frame

    def receive_frame(self, timeout=0.4):  # 将超时时间0.4
        """接收数据帧，带超时"""
        if not self.ser:
            return None
            
        start_time = time.time()
        buffer = bytearray()
        
        while time.time() - start_time < timeout:
            try:
                # 读取所有可用数据
                bytes_to_read = self.ser.in_waiting
                if bytes_to_read > 0:
                    data = self.ser.read(bytes_to_read)
                    # self.get_logger().debug(f"接收到原始数据: {data.hex()}")
                    buffer.extend(data)
                    
                    # 尝试查找帧
                    frame = self.find_frame_in_buffer(buffer)
                    if frame:
                        # 从缓冲区中移除已处理的帧
                        frame_index = buffer.find(bytearray(frame))
                        if frame_index >= 0:
                            del buffer[:frame_index + len(frame)]
                        return frame
                time.sleep(0.01)
            except serial.SerialException as e:
                self.get_logger().error(f"读取串口时出错: {str(e)}")
                return None
        
        if buffer:
            self.get_logger().warn(f"超时，缓冲区中剩余数据: {buffer.hex()}")
        
        return None

    def find_frame_in_buffer(self, buffer):
        """在缓冲区中查找有效帧"""
        # 查找帧头 0xDE 0xED
        for i in range(len(buffer) - 2):
            if buffer[i] == 0xDE and buffer[i+1] == 0xED:
                # 检查长度是否足够
                if len(buffer) - i >= 22:
                    frame = buffer[i:i+22]
                    
                    # 验证帧长度
                    if frame[2] != 0x16:  # 长度应为22
                        self.get_logger().warn(f"无效帧长度: {frame[2]}, 应为0x16")
                        continue
                    
                    # 验证校验和
                    calculated_checksum = sum(frame[0:20]) & 0xFFFF
                    received_checksum = struct.unpack_from('<H', frame, 20)[0]
                    
                    if calculated_checksum == received_checksum:
                        self.get_logger().debug(f"找到有效帧: {frame.hex()}")
                        return frame
                    else:
                        self.get_logger().warn(f"校验和错误: 计算值={calculated_checksum}, 接收值={received_checksum}")
                else:
                    self.get_logger().debug("帧不完整，等待更多数据")
        return None

    def parse_status_frame(self, frame):
        """解析状态帧 (命令码0x01) - 严格遵循图片中的协议结构"""
        status = AGVStatus()
        
        # 提取数据域 (16字节: 索引4到19)
        data_field = frame[4:20]
        self.get_logger().debug(f"状态帧数据域: {data_field.hex()}")
        
        # 1. 小车编号 (数据域字节0-1)
        car_id = struct.unpack_from('<H', data_field, 0)[0]
        self.get_logger().debug(f"小车编号: {car_id}")
        
        # 2. 电池电量 (数据域字节2-3) - 0.1V单位
        # 图片说明: 0x00F1 = 241 = 24.1V
        battery_raw = struct.unpack_from('<H', data_field, 2)[0]
        status.battery_voltage = battery_raw * 0.1  # 转换为浮点数伏特值
        self.get_logger().debug(f"电池电压: {status.battery_voltage} V")
        
        # 3. 错误标志 (数据域字节4-5)
        status.error_flags = struct.unpack_from('<H', data_field, 4)[0]
        self.get_logger().debug(f"错误标志: {status.error_flags} ({bin(status.error_flags)})")
        
        # 4. 外设状态 (数据域字节6-7)
        peripheral_state = struct.unpack_from('<H', data_field, 6)[0]
        self.get_logger().debug(f"外设状态: {peripheral_state} ({bin(peripheral_state)})")
        
        # 解析外设状态位
        status.front_magnetic_sensor = bool(peripheral_state & (1 << 3))   # bit3
        status.rear_magnetic_sensor = bool(peripheral_state & (1 << 4))    # bit4
        status.magnetic_sensor_lift_status = bool(peripheral_state & (1 << 13))  # bit13
        self.get_logger().debug(f"前传感器: {status.front_magnetic_sensor}, 后传感器: {status.rear_magnetic_sensor}, 升降状态: {status.magnetic_sensor_lift_status}")
        
        # 5. 白卡号 (数据域字节8-9)
        status.white_card_id = struct.unpack_from('<H', data_field, 8)[0]
        self.get_logger().debug(f"白卡号: {status.white_card_id}")
        
        # 6. 超声波刹车距离 (数据域字节10-11)
        ultrasonic_brake_distance = struct.unpack_from('<H', data_field, 10)[0]
        self.get_logger().debug(f"超声波刹车距离: {ultrasonic_brake_distance} mm")
        
        # 7. 左轮速度 (数据域字节12-13)
        status.left_wheel_speed = struct.unpack_from('<h', data_field, 12)[0]
        
        # 8. 右轮速度 (数据域字节14-15)
        status.right_wheel_speed = struct.unpack_from('<h', data_field, 14)[0]
        
        self.get_logger().debug(f"左轮速度: {status.left_wheel_speed}, 右轮速度: {status.right_wheel_speed}")
        
        return status


def main(args=None):
    rclpy.init(args=args)
    node = AGVControlNode()
    # 配置多线程执行器
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点运行时错误: {str(e)}")
    finally:
        if hasattr(node, 'ser') and node.ser and node.ser.is_open:
            node.ser.close()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()