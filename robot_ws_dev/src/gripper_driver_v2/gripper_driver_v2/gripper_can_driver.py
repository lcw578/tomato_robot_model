# gripper_can_driver.py
import can
import time
import struct
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class MotorStatus:
    """电机状态数据结构"""
    motor_id: int
    error_code: int
    position: int
    velocity: int
    torque: int
    mos_temperature: int
    rotor_temperature: int
    timestamp: float

class GripperCANDriver:
    """
    CAN总线抓取器驱动
    基于图片中的协议实现：
    1. 打开扭矩开关协议（8字节固定数据）
    2. 写寄存器协议（7字节数据）
    """
    
    def __init__(self, can_channel: str = 'can0', motor_id: int = 1, bitrate: int = 1000000):
        """
        初始化CAN总线连接
        
        Args:
            can_channel: CAN接口名称，如'can0'
            motor_id: 电机CAN ID
            bitrate: 波特率，默认为1Mbps
        """
        self.can_channel = can_channel
        self.motor_id = motor_id
        self.bitrate = bitrate
        # 是否始终打印位置
        self.show_pos_always = False
        
        # CAN总线接口
        self.bus = None
        
        # 状态监控
        self.current_status = None
        self.status_lock = threading.Lock()
        self.send_msg_lock = threading.Lock()
        self.set_or_get_vel_pos_lock = threading.Lock()
        
        self.running = False
        self.status_thread = None
        self.now_vel = 0
        self.now_pos = 0
        
        # 位置值定义
        self.POSITION_OPEN = 305     # 打开位置
        self.POSITION_CLOSE = 2985       # 关闭位置

        
        # 寄存器地址
        self.REG_TARGET_POSITION = 532  # 目标位置寄存器地址
        
        # 打开电机扭矩开关 协议常量
        self.OPEN_TORQUE_SWITCH_CMD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])

        self.CLOSE_TORQUE_SWITCH_CMD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
        # 设置为位置模式
        self.POSITION_MODEL_CMD = bytes([0x21, 0x00, 0x01, 0x03, 0xFF, 0xFF, 0xFF])

        # 读取速度
        self.GET_VEL_POS_CMD = bytes([0x24, 0x02, 0x08])

    
    def connect(self) -> bool:
        """
        连接CAN总线
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 使用SocketCAN接口
            self.bus = can.interface.Bus(
                channel=self.can_channel,
                bustype='socketcan',
                bitrate=self.bitrate
            )
            print(f"已连接到CAN总线 {self.can_channel}，波特率 {self.bitrate}")
            return True
        except Exception as e:
            print(f"连接CAN总线失败: {e}")
            return False
    
    def disconnect(self):
        # 关闭电机扭矩开关
        self.close_torque_switch()
        time.sleep(0.5)
        """断开CAN总线连接"""
        self.running = False
        if self.status_thread:
            self.status_thread.join(timeout=2.0)
        
        if self.bus:
            self.bus.shutdown()
            self.bus = None
    
    def _send_can_frame(self, can_id: int, data: bytes, timeout: float = 0.1) -> bool:
        """
        发送CAN帧
        
        Args:
            can_id: CAN标识符
            data: 数据字节
            timeout: 发送超时时间
            
        Returns:
            bool: 发送是否成功
        """
        if not self.bus:
            print("CAN总线未连接")
            return False
        
        try:
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False  # 使用标准帧(11位)
            )
            # print(f"发送CAN帧: ID={can_id}, 数据={data.hex()}")
            self.bus.send(msg, timeout=timeout)
            return True
        except can.CanError as e:
            print(f"发送CAN帧失败: {e}")
            return False
    
    def _receive_can_frame(self, expected_id: Optional[int] = None, timeout: float = 0.1) -> Optional[can.Message]:
        """
        接收CAN帧
        
        Args:
            expected_id: 期望的CAN ID，为None时接收任何帧
            timeout: 接收超时时间
            
        Returns:
            Optional[can.Message]: 接收到的CAN消息，超时返回None
        """
        if not self.bus:
            return None
        
        try:
            msg = self.bus.recv(timeout=timeout)
            if msg is None:
                return None
            
            if expected_id is not None and msg.arbitration_id != expected_id:
                return None
            
            return msg
        except can.CanError as e:
            print(f"接收CAN帧失败: {e}")
            return None
    
    def _handle_vel_pos_response(self,data:bytes):
        """
        解析8字节数据，前4字节为速度（小端格式），后4字节为位置（小端格式）
        
        Args:
            data: 8字节数据
        """
        if len(data) < 8:
            print(f"数据长度不足，期望8字节，实际{len(data)}字节")
            return False
        # print(f"收到速度位置数据: {data.hex()}")
        # 解析前4字节为速度（小端格式）
        vel_bytes = data[0:4]
        
        # 解析后4字节为位置（小端格式）
        pos_bytes = data[4:8]
        with self.set_or_get_vel_pos_lock:
            self.now_vel = int.from_bytes(vel_bytes, byteorder='little', signed=True)
            self.now_pos = int.from_bytes(pos_bytes, byteorder='little', signed=True)            
            if self.show_pos_always:
                print(f"解析到速度: {self.now_vel}, 位置: {self.now_pos}")
        return True



    def _parse_torque_switch_response(self, data: bytes) -> Optional[MotorStatus]:
        """
        解析打开扭矩开关的返回数据
        
        根据图片协议：
        DATA0: ID | (ERR << 4)
        DATA1: POS[15:8] (位置高8位)
        DATA2: POS[7:0]  (位置低8位)
        DATA3: VEL[11:4] (速度高8位)
        DATA4: VEL[3:0] | T[11:8] (速度低4位 | 扭矩高4位)
        DATA5: T[7:0] (扭矩低8位)
        DATA6: T_MOS (MOS管温度)
        DATA7: T_Rotor (转子温度)
        
        注意：位置为16位有符号，速度和扭矩为12位有符号
        
        Args:
            data: 8字节数据
            
        Returns:
            Optional[MotorStatus]: 解析后的状态，失败返回None
        """
        if len(data) != 8:
            return None
        
        try:
            # 解析DATA0: ID | (ERR << 4)
            data0 = data[0]
            motor_id = data0 & 0x0F
            error_code = (data0 >> 4) & 0x0F
            
            # 解析位置(16位有符号)
            position_raw = (data[1] << 8) | data[2]
            if position_raw & 0x8000:  # 检查符号位
                position_raw = position_raw - 0x10000
            
            # 解析速度(12位有符号)
            vel_high = data[3]
            vel_low = (data[4] >> 4) & 0x0F
            velocity_raw = (vel_high << 4) | vel_low
            if velocity_raw & 0x800:  # 检查12位有符号
                velocity_raw = velocity_raw - 0x1000
            
            # 解析扭矩(12位有符号)
            torque_high = data[4] & 0x0F
            torque_low = data[5]
            torque_raw = (torque_high << 8) | torque_low
            if torque_raw & 0x800:  # 检查12位有符号
                torque_raw = torque_raw - 0x1000
            
            # 温度
            mos_temp = data[6]
            rotor_temp = data[7]
            
            return MotorStatus(
                motor_id=motor_id,
                error_code=error_code,
                position=position_raw,
                velocity=velocity_raw,
                torque=torque_raw,
                mos_temperature=mos_temp,
                rotor_temperature=rotor_temp,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"解析扭矩开关响应失败: {e}")
            return None
    
    def _write_register(self, address: int, value: int, value_len: int = 4) -> bool:
        """
        向寄存器写入值（按照图片中的协议）
        
        协议格式（7字节）：
        DATA0: Address(L) - 地址低字节
        DATA1: Address(H) - 地址高字节
        DATA2: Len - 写入字节长度(1-4)
        DATA3-DATA6: Value - 写入的值，不足4字节用0xFF填充
        
        Args:
            address: 寄存器地址
            value: 要写入的值
            value_len: 值的字节长度(1-4)
            
        Returns:
            bool: 写入是否成功
        """
        if value_len < 1 or value_len > 4:
            print(f"无效的value_len: {value_len}，必须在1-4之间")
            return False
        
        # 构造7字节数据
        data = bytearray(7)
        
        # 地址（小端模式，低字节在前）
        data[0] = address & 0xFF          # Address(L)
        data[1] = (address >> 8) & 0xFF   # Address(H)
        
        # 长度
        data[2] = value_len
        
        # 值（小端模式，低字节在前）
        for i in range(value_len):
            byte_shift = i * 8
            data[3 + i] = (value >> byte_shift) & 0xFF
        
        # 不足4字节的部分用0xFF填充
        for i in range(value_len, 4):
            data[3 + i] = 0xFF
        print(f"写入寄存器: 数据={data.hex}")
        # 发送CAN帧
        can_id = self.motor_id
        with self.send_msg_lock:
            if not self._send_can_frame(can_id, bytes(data)):
                return False
            
            # 等待响应（1字节，0x01成功，0x00失败）
            response = self._receive_can_frame(expected_id=can_id, timeout=0.5)
        
        if response is None:
            print("等待写寄存器响应超时")
            return False
        print(f"写寄存器响应: {response.data}")
        if len(response.data) >= 1 and response.data[0] == 0x01:
            return True
        else:
            print(f"写寄存器失败，响应: {response.data}")
            return False
    
    def start_status_monitoring(self, interval: float = 0.1):
        """
        启动状态监控线程
        
        Args:
            interval: 状态查询间隔（秒）
        """
        if self.status_thread and self.status_thread.is_alive():
            return
        
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    # 获取电机当前位置和速度
                    with self.send_msg_lock:
                        if self._send_can_frame(self.motor_id, self.GET_VEL_POS_CMD):
                            # 接收响应
                            response = self._receive_can_frame(expected_id=self.motor_id, timeout=0.1)
                    if response:
                        result=self._handle_vel_pos_response(response.data)

                except Exception as e:
                    print(f"状态监控错误: {e}")
                
                time.sleep(interval)
        
        self.status_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.status_thread.start()
        print("状态监控已启动")
    
    def get_current_status(self) -> Optional[MotorStatus]:
        """
        获取当前电机状态
        
        Returns:
            Optional[MotorStatus]: 当前状态，无数据返回None
        """
        with self.status_lock:
            return self.current_status
    
    def initialize(self) -> bool:
        """
        初始化电机（打开扭矩开关）
        
        根据图片协议，发送8字节固定数据打开扭矩开关
        
        Returns:
            bool: 初始化是否成功
        """
        print("初始化电机（打开扭矩开关）...")
        
        # 发送打开扭矩开关命令
        with self.send_msg_lock:
            if not self._send_can_frame(self.motor_id, self.POSITION_MODEL_CMD):
                print("发送位置模式命令失败")
                return False
            
            response = self._receive_can_frame(expected_id=self.motor_id, timeout=1.0)

            if response is None:
                print("未收到位置模式响应")
                return False           

            
            if not self._send_can_frame(self.motor_id, self.OPEN_TORQUE_SWITCH_CMD):
                print("发送扭矩开关命令失败")
                return False        
            # 等待并验证响应
            response = self._receive_can_frame(expected_id=self.motor_id, timeout=1.0)
        if response is None:
            print("未收到扭矩开关响应")
            return False
        
        # if len(response.data) != 8:
        #     print(f"扭矩开关响应长度错误: {len(response.data)} 字节，期望8字节")
        #     return False
        
        # 解析响应
        status = self._parse_torque_switch_response(response.data)
        if status is None:
            print("解析扭矩开关响应失败")
            return False
        if status.error_code != 0:
            print(f"电机报告错误: 错误码 {status.error_code}")
            return False
        print(f"------电机状态参数-------")
        print(f"控制器ID：{status.motor_id}")
        print(f"错误码：{status.error_code}")
        print(f"当前位置：{status.position}")
        print(f"当前速度：{status.velocity}")
        print(f"当前扭矩：{status.torque}")
        print(f"MOS温度：{status.mos_temperature}℃")
        print(f"转子温度：{status.rotor_temperature}℃")
        
        # 启动状态监控
        self.start_status_monitoring()
        
        return True
    
    def set_position(self, position: int, timeout: float = 5.0) -> Tuple[bool, int, str]:
        """
        设置目标位置
        
        Args:
            position: 目标位置值
            timeout: 超时时间（秒）
            
        Returns:
            Tuple[bool, int, str]: (是否成功, 最终位置, 消息)
        """
        print(f"设置目标位置: {position}")
        
        # 向532寄存器写入位置值（4字节，小端模式）
        result=self._write_register(self.REG_TARGET_POSITION, position, value_len=4)
        if  result==False:
            return False, 0, "写入寄存器失败"
        else:
            with self.set_or_get_vel_pos_lock:
                pos_error=abs(self.now_pos-position)
            while pos_error>20:
                time.sleep(0.2)
                with self.set_or_get_vel_pos_lock:
                    pos_error=abs(self.now_pos-position)
            return True, position, "写入寄存器成功"
        
    
    def open_gripper(self, timeout: float = 5.0) -> Tuple[bool, int, str]:
        """打开抓取器"""
        return self.set_position(self.POSITION_OPEN, timeout)
    
    def close_gripper(self, timeout: float = 5.0) -> Tuple[bool, int, str]:
        """关闭抓取器"""
        return self.set_position(self.POSITION_CLOSE, timeout)
    
    def close_torque_switch(self, timeout: float = 5.0) -> Tuple[bool, int, str]:
        """关闭电机扭矩开关"""
        with self.send_msg_lock:
            if not self._send_can_frame(self.motor_id, self.CLOSE_TORQUE_SWITCH_CMD):
                print("发送关闭扭矩开关命令失败")
                
            else:
                print("成功关闭扭矩开关")
