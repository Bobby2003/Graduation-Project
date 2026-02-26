import time
import struct
import numpy as np
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

# 定义传感器类型常量
ONI_SENSOR_GYRO = 5
ONI_SENSOR_ACCEL = 6


class LazyManIMUTester:
    def __init__(self):
        self.sdk_path = get_default_sdk_path()
        self.camera_sdk = OrbbecCameraSDK(self.sdk_path)
        self.is_gyro_started = False

    def init_device(self):
        """初始化 SDK 并打开设备"""
        if not self.camera_sdk.initialize():
            print("[-] SDK 初始化失败")
            return False

        if not self.camera_sdk.open_device():
            print("[-] 未发现相机设备")
            return False

        print("[+] 相机连接成功")
        return True

    def test_imu(self):
        """尝试开启并读取陀螺仪数据"""
        # 1. 尝试创建陀螺仪流
        # 注意：如果设备不支持，这里通常会返回 False
        res = self.camera_sdk.create_stream(ONI_SENSOR_GYRO)
        if not res:
            print("[-] 错误：此设备不支持硬件陀螺仪，或者固件未开启 IMU 功能。")
            return

        # 2. 启动流
        self.camera_sdk.start_stream()
        print("[+] 陀螺仪流已启动，开始采集数据...")
        print("    请尝试旋转或晃动相机...")
        print("-" * 50)

        try:
            while True:
                # 3. 读取原始帧数据
                # 在 OpenNI2 中，IMU 帧通常是 3 个 float 值 (X, Y, Z) 加上时间戳
                # 我们假设 SDK 中的 capture_frame 会返回 (data, timestamp)
                # 如果您的 SDK 有专门获取 IMU 的函数，请替换它
                frame_data = self.camera_sdk.capture_gyro_frame()  # 假设支持此方法

                if frame_data is not None:
                    gyro_x, gyro_y, gyro_z = frame_data

                    # 打印实时角速度
                    print(f"\r[陀螺仪] X: {gyro_x:>8.3f} | Y: {gyro_y:>8.3f} | Z: {gyro_z:>8.3f} (deg/s)", end="")
                else:
                    # 如果没有直接的方法，可能需要通过通用的 read 接口获取
                    # print("\r正在等待陀螺仪数据推送...", end="")
                    pass

                time.sleep(0.01)  # 100Hz 刷新

        except KeyboardInterrupt:
            print("\n\n[!] 测试由用户终止")
        finally:
            self.camera_sdk.cleanup()


if __name__ == "__main__":
    tester = LazyManIMUTester()
    if tester.init_device():
        tester.test_imu()