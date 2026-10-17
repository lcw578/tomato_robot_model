import can
import time
import struct

class VKRobotCAN:
    def __init__(self, channel='can0', bitrate=1000000):  # 文档默认是1Mbps
        self.bus = None
        self.channel = channel
        self.bitrate = bitrate
        self.setup_can()
    
    def setup_can(self):
        """设置CAN总线"""
        try:
            # 创建CAN总线接口
            self.bus = can.interface.Bus(
                channel=self.channel,
                bustype='socketcan',
                bitrate=self.bitrate
            )
            print(f"CAN总线初始化成功 - 通道: {self.channel}, 波特率: {self.bitrate}bps")
        except Exception as e:
            print(f"CAN总线初始化失败: {e}")
            # 尝试使用虚拟接口（用于测试）
            try:
                self.bus = can.interface.Bus(
                    channel='vcan0',
                    bustype='socketcan',
                    bitrate=self.bitrate
                )
                print("使用虚拟CAN接口 vcan0")
            except:
                print("无法创建CAN接口，请检查系统配置")
                raise
    
    def build_search_frame(self, target_id=0x01):
        """构建设备搜索帧（消息类型0x00）"""
        # 根据文档构建29位扩展帧ID
        # Bit28~24: 消息类型 (0x00)
        # Bit23~8: 数据区2 (0x0000)
        # Bit7~0: 目标电机ID
        can_id = (0x00 << 24) | (0x0000 << 8) | target_id<<7
        print(f"can_id: {can_id}")

        
        # 数据区为空（8个字节的0）
        data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        
        message = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=True  # 使用扩展帧格式
        )
        print(f"message: {message}")
        
        return message
    
    def parse_search_response(self, message):
        """解析设备搜索应答帧"""
        if message is None:
            return None
        
        # 解析29位CAN ID
        can_id = message.arbitration_id
        
        # 提取各个字段
        message_type = (can_id >> 24) & 0x1F  # Bit28~24
        data_area2 = (can_id >> 8) & 0xFFFF   # Bit23~8
        target_can_id = can_id & 0xFF         # Bit7~0
        
        # 检查消息类型是否为0x00（设备搜索应答）
        if message_type != 0x00:
            return None
        
        # 解析数据区
        data = message.data
        
        # 数据区格式：
        # Byte0~1: 电机型号
        # Byte2~4: 固件版本号
        # Byte5~7: 无
        
        # 电机型号（小端序）
        motor_model = struct.unpack('<H', bytes(data[0:2]))[0]
        
        # 固件版本号（小端序，3字节）
        # 注意：struct不支持直接解包3字节，需要手动处理
        firmware_bytes = bytes(data[2:5]) + b'\x00'  # 补齐到4字节
        firmware_version = struct.unpack('<I', firmware_bytes)[0] & 0xFFFFFF
        
        # 从data_area2中提取目标电机ID（Bit15~8）
        target_motor_id = (data_area2 >> 8) & 0xFF
        
        result = {
            'message_type': message_type,
            'target_motor_id': target_motor_id,
            'target_can_id': target_can_id,
            'motor_model': motor_model,
            'firmware_version': firmware_version,
            'raw_data': data,
            'can_id_hex': hex(can_id)
        }
        
        return result
    
    def search_devices(self, timeout=2.0):
        """搜索设备并返回结果"""
        print("开始搜索VK Robot设备...")
        
        # 发送设备搜索命令（广播搜索，目标ID=0x00）
        search_frame = self.build_search_frame(target_id=0x01)
        
        try:
            self.bus.send(search_frame)
            print("设备搜索命令已发送")
        except Exception as e:
            print(f"发送搜索命令失败: {e}")
            return []
        
        # 等待并接收响应
        devices = []
        start_time = time.time()
        
        print("等待设备响应...")
        
        while time.time() - start_time < timeout:
            try:
                message = self.bus.recv(timeout=0.1)  # 100ms超时
                
                if message is not None:
                    # 解析响应
                    device_info = self.parse_search_response(message)
                    
                    if device_info is not None:
                        print(f"发现设备: CAN_ID={hex(device_info['target_can_id'])}")
                        devices.append(device_info)
                        
            except can.CanError as e:
                print(f"接收数据时出错: {e}")
                continue
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue
        
        return devices
    
    def print_device_info(self, device_info):
        """打印设备信息"""
        print("\n" + "="*50)
        print("VK Robot 设备信息")
        print("="*50)
        print(f"CAN ID: {device_info['target_can_id']} ({device_info['can_id_hex']})")
        print(f"电机ID: {device_info['target_motor_id']}")
        print(f"电机型号: {device_info['motor_model']} (0x{device_info['motor_model']:04X})")
        print(f"固件版本: {device_info['firmware_version']} (0x{device_info['firmware_version']:06X})")
        
        # 解析固件版本号（通常格式：主版本.次版本.修订版本）
        major = (device_info['firmware_version'] >> 16) & 0xFF
        minor = (device_info['firmware_version'] >> 8) & 0xFF
        patch = device_info['firmware_version'] & 0xFF
        print(f"固件版本详情: {major}.{minor}.{patch}")
        
        print(f"原始数据: {[hex(x) for x in device_info['raw_data']]}")
        print("="*50)
    
    def close(self):
        """关闭CAN总线"""
        if self.bus:
            self.bus.shutdown()
            print("CAN总线已关闭")

def main():
    # 创建VK Robot CAN接口实例
    # 注意：根据问题描述，波特率应该是10Mbps，但文档写的是1Mbps
    # 这里使用1Mbps，如果需要10Mbps，将bitrate改为10000000
    vk_can = VKRobotCAN(channel='can0', bitrate=1000000)
    
    try:
        # 搜索设备
        devices = vk_can.search_devices(timeout=3.0)
        
        if devices:
            print(f"\n共发现 {len(devices)} 个设备:")
            for i, device in enumerate(devices, 1):
                print(f"\n设备 {i}:")
                vk_can.print_device_info(device)
        else:
            print("\n未发现任何设备")
            print("可能的原因:")
            print("1. CAN总线连接问题")
            print("2. 设备未上电或未连接")
            print("3. 设备不在私有协议模式")
            print("4. CAN通道名称不正确")
            
    except Exception as e:
        print(f"程序执行出错: {e}")
    
    finally:
        vk_can.close()

if __name__ == "__main__":
    # 检查系统CAN支持
    try:
        import can
        main()
    except ImportError:
        print("请先安装python-can库: pip install python-can")
    except KeyboardInterrupt:
        print("\n程序被用户中断")