import serial
import struct
import binascii

# --- 配置参数 ---
SERIAL_PORT = 'COM7'  # 你的 CH340 端口
BAUD_RATE = 115200    # 默认波特率

def parse_yahboom_imu():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
        print(f" 成功连接到 亚博智能 IMU: {SERIAL_PORT}")
        print("正在按照官方协议 (0x7E 0x23) 进行解析...")
        print("等待接收数据... (请转动 IMU)\n")
        
        while True:
            # 1. 寻找包头1: 0x7E
            if ser.read(1) == b'\x7e':
                # 2. 寻找包头2: 0x23
                if ser.read(1) == b'\x23':
                    
                    # 3. 读取长度字段 (1 byte)
                    length_byte = ser.read(1)
                    if not length_byte: continue
                    length = length_byte[0]
                    
                    # 4. 读取剩下的 payload 数据 (包括功能字、数据、校验位)
                    # 长度是从包头1到校验位的总长度。
                    # 已经读了包头1(1), 包头2(1), 长度(1)，所以还剩 length - 3 个字节
                    bytes_to_read = length - 3
                    if bytes_to_read <= 0: continue
                        
                    payload = ser.read(bytes_to_read)
                    if len(payload) != bytes_to_read: continue
                    
                    # 提取功能字
                    func_code = payload[0]
                    
                    # 根据表格：功能字 0x26 表示欧拉角返回
                    if func_code == 0x26:
                        # 数据部分是从 payload[1] 开始，一共 12 个字节 (Roll 4, Pitch 4, Yaw 4)
                        data_bytes = payload[1:13]
                        if len(data_bytes) == 12:
                            # 按照小端浮点数 '<f' 进行解析
                            roll = struct.unpack('<f', data_bytes[0:4])[0]
                            pitch = struct.unpack('<f', data_bytes[4:8])[0]
                            yaw = struct.unpack('<f', data_bytes[8:12])[0]
                            
                            # 协议中说单位是弧度，为了人类可读，我们先转换成角度展示
                            import math
                            roll_deg = math.degrees(roll)
                            pitch_deg = math.degrees(pitch)
                            yaw_deg = math.degrees(yaw)
                            
                            print(f"\r[欧拉角] Roll: {roll_deg:7.2f}° | Pitch: {pitch_deg:7.2f}° | Yaw: {yaw_deg:7.2f}°", end="")
                            

    except serial.SerialException as e:
        print(f"\n 串口错误: {e}")
    except KeyboardInterrupt:
        print("\n\n程序已手动停止。")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    parse_yahboom_imu()