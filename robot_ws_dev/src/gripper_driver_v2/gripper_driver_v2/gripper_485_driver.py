import serial
import struct
import time

class Gripper485Driver:
    _crc_table = [
        0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
        0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
        0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
        0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
        0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
        0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
        0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
        0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
        0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
        0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
        0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
        0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
        0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
        0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
        0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
        0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
        0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
        0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
        0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
        0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
        0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
        0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
        0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
        0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
        0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
        0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
        0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
        0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
        0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
        0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
        0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
        0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
    ]
    
    def __init__(self, port: str, baud_rate: int = 9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.waiting_for_response = False
        
        self.POSITION_OPEN = 305
        self.POSITION_CLOSE = 2985
        self.CURRENT_REGISTER = 546
        
        self.MESSAGE_HEADER = bytes([0xFF, 0xFF, 0xFD])
        self.RESERVED = 0x00
        self.READ_CMD = 0x02
        self.WRITE_CMD = 0x03
        
    def _calculate_crc(self, data: bytes) -> int:
        crc = 0x0000
        for byte in data:
            i = ((crc >> 8) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self._crc_table[i]) & 0xFFFF
        return crc
    
    def _build_write_command(self, device_id: int, register_addr: int, value: int) -> bytes:
        addr_low = register_addr & 0xFF
        addr_high = (register_addr >> 8) & 0xFF
        
        value_byte0 = value & 0xFF
        value_byte1 = (value >> 8) & 0xFF
        value_byte2 = (value >> 16) & 0xFF
        value_byte3 = (value >> 24) & 0xFF
        
        parameters = bytes([addr_low, addr_high, value_byte0, value_byte1, value_byte2, value_byte3])
        
        length = len(parameters) + 3
        length_low = length & 0xFF
        length_high = (length >> 8) & 0xFF
        
        data_without_crc = bytes([device_id, length_low, length_high, self.WRITE_CMD]) + parameters
        
        crc = self._calculate_crc(self.MESSAGE_HEADER + bytes([self.RESERVED]) + data_without_crc)
        crc_low = crc & 0xFF
        crc_high = (crc >> 8) & 0xFF
        
        return self.MESSAGE_HEADER + bytes([self.RESERVED]) + data_without_crc + bytes([crc_low, crc_high])

    def _build_read_command(self, device_id: int, register_addr: int, data_length: int) -> bytes:
        parameters = struct.pack("<HH", register_addr, data_length)
        length = len(parameters) + 3
        data_without_crc = struct.pack(
            "<BHB", device_id, length, self.READ_CMD
        ) + parameters

        crc = self._calculate_crc(
            self.MESSAGE_HEADER + bytes([self.RESERVED]) + data_without_crc
        )

        return (
            self.MESSAGE_HEADER
            + bytes([self.RESERVED])
            + data_without_crc
            + struct.pack("<H", crc)
        )
    
    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0.1
            )
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def initialize(self) -> bool:
        if not self.ser or not self.ser.is_open:
            return False
        
        if self.waiting_for_response:
            print("Waiting for response, cannot send command")
            return False
        
        try:
            self.waiting_for_response = True
            torque_enable_cmd = bytes([0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x07, 0x00, 0x03, 0x00, 0x02, 0x01, 0xFF, 0x73, 0x59])
            # print(f"initialize发送扭矩使能指令: {torque_enable_cmd.hex()}")
            self.ser.write(torque_enable_cmd)
            
            response = self.ser.read(256)
            if response:
                # print(f"initialize返回指令: {response.hex()}")
                pass
            
            self.waiting_for_response = False
            success, position, msg = self.read_position()
            if success:
                print(f"当前位置: {position}")
            else:
                print(f"Failed to read position: {msg}")
            return True
        except Exception as e:
            print(f"Failed to initialize: {e}")
            self.waiting_for_response = False
            return False
    
    def open_gripper(self) -> tuple:
        if not self.ser or not self.ser.is_open:
            return False, None, "Serial port not open"
        
        if self.waiting_for_response:
            print("Waiting for response, cannot send command")
            return False, None, "Waiting for response"
        
        try:
            t_start = time.perf_counter()
            self.waiting_for_response = True
            cmd = self._build_write_command(device_id=1, register_addr=0x0214, value=self.POSITION_OPEN)
            # print(f"open_gripper发送打开指令: {cmd.hex()}")            
            self.ser.write(cmd)
            
            response = self.ser.read(256)
            if response:
                # print(f"open_gripper返回指令: {response.hex()}")
                pass
            
            dt_ms = (time.perf_counter() - t_start) * 1000
            print(f"open_gripper 指令发送: {dt_ms:.3f} ms")
            

            self.waiting_for_response = False
            success, position, msg = self.read_position()

            dt_ms = (time.perf_counter() - t_start) * 1000-dt_ms
            print(f"open_gripper 指令读取耗时: {dt_ms:.3f} ms")
            # if success:
            #     print(f"Current position: {position}")
            # else:
            #     print(f"Failed to read position: {msg}")
            return success, position, msg
        except Exception as e:
            print(f"Failed to open gripper: {e}")
            self.waiting_for_response = False
            return False, None, str(e)
    
    def close_gripper(self) -> tuple:
        if not self.ser or not self.ser.is_open:
            return False, None, "Serial port not open"
        
        if self.waiting_for_response:
            print("Waiting for response, cannot send command")
            return False, None, "Waiting for response"
        
        try:
            self.waiting_for_response = True
            t_start = time.perf_counter()
            cmd = self._build_write_command(device_id=1, register_addr=0x0214, value=self.POSITION_CLOSE)
            # print(f"close_gripper发送关闭指令: {cmd.hex()}")
            self.ser.write(cmd)
            dt_ms = (time.perf_counter() - t_start) * 1000
            print(f"close_gripper 指令发送: {dt_ms:.3f} ms")
            response = self.ser.read(256)
            if response:
                # print(f"close_gripper返回指令: {response.hex()}")
                pass

            self.waiting_for_response = False
            success, position, msg = self.read_position()
            dt_ms = (time.perf_counter() - t_start) * 1000-dt_ms
            print(f"close_gripper 指令读取耗时: {dt_ms:.3f} ms")
            # if success:
            #     print(f"Current position: {position}")
            # else:
            #     print(f"Failed to read position: {msg}")
            return success, position, msg
        except Exception as e:
            print(f"Failed to close gripper: {e}")
            self.waiting_for_response = False
            return False, None, str(e)
    
    def read_position(self) -> tuple:
        if not self.ser or not self.ser.is_open:
            return False, None, "Serial port not open"
        
        if self.waiting_for_response:
            print("Waiting for response, cannot send command")
            return False, None, "Waiting for response"
        
        try:
            self.waiting_for_response = True
            read_cmd = bytes([0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x07, 0x00, 0x02, 0x28, 0x02, 0x04, 0x00, 0x05, 0x65])
            # print(f"read_position发送查询指令: {read_cmd.hex()}")
            self.ser.write(read_cmd)
            
            response = self.ser.read(256)
            if response:
                # print(f"read_position返回指令: {response.hex()}")
                pass
                
                if len(response) >= 9:
                    err_code = response[8]
                    if err_code == 0x00:
                        if len(response) >= 13:
                            position = (response[12] << 24) | (response[11] << 16) | (response[10] << 8) | response[9]
                            self.waiting_for_response = False
                            return True, position, "Success"
                        else:
                            self.waiting_for_response = False
                            return False, None, "Response too short"
                    else:
                        self.waiting_for_response = False
                        return False, err_code, f"Error code: {err_code}"
                else:
                    self.waiting_for_response = False
                    return False, None, "Response too short"
            else:
                self.waiting_for_response = False
                return False, None, "No response"
        except Exception as e:
            print(f"Failed to read position: {e}")
            self.waiting_for_response = False
            return False, None, str(e)

    def read_current(self) -> tuple:
        """Read the signed 16-bit current value from register 546."""
        if not self.ser or not self.ser.is_open:
            return False, None, "Serial port not open"

        if self.waiting_for_response:
            print("Waiting for response, cannot send command")
            return False, None, "Waiting for response"

        try:
            self.waiting_for_response = True
            read_cmd = self._build_read_command(
                device_id=1,
                register_addr=self.CURRENT_REGISTER,
                data_length=2
            )
            self.ser.write(read_cmd)

            response = self.ser.read(256)
            if not response:
                return False, None, "No response"

            if len(response) < 9:
                return False, None, "Response too short"

            err_code = response[8]
            if err_code != 0x00:
                return False, err_code, f"Error code: {err_code}"

            if len(response) < 11:
                return False, None, "Response too short"

            current = struct.unpack("<h", response[9:11])[0]
            return True, current, "Success"
        except Exception as e:
            print(f"Failed to read current: {e}")
            return False, None, str(e)
        finally:
            self.waiting_for_response = False
    
    def disconnect(self) -> bool:
        if not self.ser or not self.ser.is_open:
            return False
        
        if self.waiting_for_response:
            print("Waiting for response, cannot send command")
            return False
        
        try:
            self.waiting_for_response = True
            torque_disable_cmd = bytes([0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x07, 0x00, 0x03, 0x00, 0x02, 0x00, 0xFF, 0x70, 0xDF])
            # print(f"disconnect发送扭矩关闭指令: {torque_disable_cmd.hex()}")
            self.ser.write(torque_disable_cmd)
            
            response = self.ser.read(256)
            if response:
                # print(f"disconnect返回指令: {response.hex()}")
                pass
            
            time.sleep(0.5)
            self.ser.write(torque_disable_cmd)
            
            response = self.ser.read(256)
            if response:
                # print(f"disconnect第二次返回指令: {response.hex()}")
                pass
            
            time.sleep(0.5)
            self.ser.close()
            self.waiting_for_response = False
            return True
        except Exception as e:
            print(f"Failed to disconnect: {e}")
            self.waiting_for_response = False
            return False
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def test_gripper():
    
    
    driver = Gripper485Driver(port='/dev/ttyACM0', baud_rate=1000000)
    
    print("Connecting...")
    success = driver.connect()
    if not success:
        print("Failed to connect")
        return
    
    print("Initializing gripper...")
    success = driver.initialize()
    if not success:
        print("Failed to initialize gripper")
        return
    time.sleep(0.1)
    
    print("Opening gripper...")
    success, position, msg = driver.open_gripper()
    if success:
        print(f"Current position: {position}")
    else:
        print(f"Failed to open gripper: {msg}")
    
    print("Waiting 2 seconds...")
    time.sleep(2)
    
    print("Closing gripper...")
    success, position, msg = driver.close_gripper()
    if success:
        print(f"Current position: {position}")
    else:
        print(f"Failed to close gripper: {msg}")

    print("Reading current...")
    success, current, msg = driver.read_current()
    if success:
        print(f"Current: {current}")
    else:
        print(f"Failed to read current: {msg}")
    
    # print("Reading position...")
    # success, position, msg = driver.read_position()
    # if success:
    #     print(f"Current position: {position}")
    # else:
    #     print(f"Failed to read position: {msg}")
    
    print("Disconnecting...")
    driver.disconnect()
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_gripper()
