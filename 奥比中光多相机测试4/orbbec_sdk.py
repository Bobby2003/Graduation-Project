import ctypes
import numpy as np
import cv2
import sys
import time
import traceback

# ==================== 新的帧结构体定义 ====================

# 参考OpenNI2标准头文件中的帧结构体定义
class OniFrame(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dataSize", ctypes.c_int),            # 数据大小（字节）
        ("data", ctypes.c_void_p),             # 数据指针
        ("sensorType", ctypes.c_int),          # 传感器类型
        ("timestamp", ctypes.c_uint),          # 时间戳（毫秒）
        ("frameIndex", ctypes.c_int),          # 帧索引
        ("width", ctypes.c_int),               # 宽度
        ("height", ctypes.c_int),              # 高度
        ("cropOriginX", ctypes.c_int),         # 裁剪原点X
        ("cropOriginY", ctypes.c_int),         # 裁剪原点Y
        ("croppingEnabled", ctypes.c_int),     # 是否启用裁剪
        ("stride", ctypes.c_int),              # 每行字节数
        ("videoMode_pixelFormat", ctypes.c_int),  # 像素格式
        ("videoMode_resolutionX", ctypes.c_int),  # 分辨率X
        ("videoMode_resolutionY", ctypes.c_int),  # 分辨率Y
        ("videoMode_fps", ctypes.c_int),       # 帧率
    ]


class EnhancedOrbbecSDK:
    """增强的奥比中光SDK，使用更准确的帧结构体"""

    def __init__(self):
        # 加载DLL
        self._lib_path = "C:/Users/Bobby2003/Desktop/相机驱动/奥比中光Win64-Release/sdk/libs/OpenNI2.dll"
        self._lib = ctypes.CDLL(self._lib_path)
        self._setup_functions()

    def _setup_functions(self):
        """设置必要的函数原型"""
        # 基础函数
        self._lib.oniInitialize.argtypes = [ctypes.c_int]
        self._lib.oniInitialize.restype = ctypes.c_int

        self._lib.oniShutdown.argtypes = []
        self._lib.oniShutdown.restype = None

        # 设备管理
        class OniDeviceInfo(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ("uri", ctypes.c_char * 256),
                ("vendor", ctypes.c_char * 256),
                ("name", ctypes.c_char * 256),
                ("usbVendorId", ctypes.c_ushort),
                ("usbProductId", ctypes.c_ushort),
            ]

        self.OniDeviceInfo = OniDeviceInfo

        self._lib.oniGetDeviceList.argtypes = [
            ctypes.POINTER(ctypes.POINTER(OniDeviceInfo)),
            ctypes.POINTER(ctypes.c_int)
        ]
        self._lib.oniGetDeviceList.restype = ctypes.c_int

        self._lib.oniReleaseDeviceList.argtypes = [ctypes.POINTER(OniDeviceInfo)]
        self._lib.oniReleaseDeviceList.restype = None

        self._lib.oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self._lib.oniDeviceOpen.restype = ctypes.c_int

        self._lib.oniDeviceClose.argtypes = [ctypes.c_void_p]
        self._lib.oniDeviceClose.restype = ctypes.c_int

        # 流管理
        self._lib.oniDeviceCreateStream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self._lib.oniDeviceCreateStream.restype = ctypes.c_int

        self._lib.oniStreamStart.argtypes = [ctypes.c_void_p]
        self._lib.oniStreamStart.restype = ctypes.c_int

        self._lib.oniStreamStop.argtypes = [ctypes.c_void_p]
        self._lib.oniStreamStop.restype = None

        self._lib.oniStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.oniStreamDestroy.restype = None

        # 帧读取
        self._lib.oniStreamReadFrame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self._lib.oniStreamReadFrame.restype = ctypes.c_int

        # 帧释放
        self._lib.oniFrameRelease.argtypes = [ctypes.c_void_p]
        self._lib.oniFrameRelease.restype = None

        # 扩展错误
        self._lib.oniGetExtendedError.argtypes = []
        self._lib.oniGetExtendedError.restype = ctypes.c_char_p

    def initialize(self):
        """初始化SDK"""
        status = self._lib.oniInitialize(2)  # ONI_VERSION_2_0
        return status == 0

    def get_device_list(self):
        """获取设备列表"""
        p_devices = ctypes.POINTER(self.OniDeviceInfo)()
        device_count = ctypes.c_int(0)

        status = self._lib.oniGetDeviceList(ctypes.byref(p_devices), ctypes.byref(device_count))
        if status != 0:
            error_msg = self._lib.oniGetExtendedError()
            print(f"❌ 获取设备列表失败: {status}, 错误: {error_msg}")
            return []

        devices = []
        for i in range(device_count.value):
            device = p_devices[i]
            devices.append({
                'index': i,
                'uri': device.uri.decode('utf-8', errors='ignore'),
                'name': device.name.decode('utf-8', errors='ignore'),
                'vendor': device.vendor.decode('utf-8', errors='ignore')
            })

        self._lib.oniReleaseDeviceList(p_devices)
        return devices

    def read_frame_with_debug(self, stream_handle, frame_count=1):
        """带调试信息的帧读取"""
        for i in range(frame_count):
            print(f"\n读取帧 {i+1}...")

            frame_ptr = ctypes.c_void_p()
            status = self._lib.oniStreamReadFrame(stream_handle, ctypes.byref(frame_ptr))

            if status != 0:
                error_msg = self._lib.oniGetExtendedError()
                print(f"❌ 读取帧失败: {status}, 错误: {error_msg}")
                continue

            if not frame_ptr.value:
                print("❌ 帧指针为空")
                continue

            print(f"✅ 帧指针: {frame_ptr.value}")

            try:
                # 将帧指针转换为结构体
                frame = ctypes.cast(frame_ptr, ctypes.POINTER(OniFrame)).contents

                print("帧结构体信息:")
                print(f"  dataSize: {frame.dataSize}")
                print(f"  data指针: {frame.data}")
                print(f"  sensorType: {frame.sensorType}")
                print(f"  timestamp: {frame.timestamp}")
                print(f"  frameIndex: {frame.frameIndex}")
                print(f"  width: {frame.width}")
                print(f"  height: {frame.height}")
                print(f"  stride: {frame.stride}")
                print(f"  pixelFormat: {frame.videoMode_pixelFormat}")

                # 尝试读取数据
                if frame.data and frame.dataSize > 0:
                    # 根据传感器类型处理数据
                    if frame.sensorType == 3:  # 深度传感器
                        print("  处理深度数据...")

                        # 计算期望的数据大小
                        expected_size = frame.width * frame.height * 2  # 16位深度
                        print(f"  期望大小: {expected_size} 字节")
                        print(f"  实际大小: {frame.dataSize} 字节")

                        if frame.dataSize >= expected_size:
                            # 读取深度数据
                            data_array = ctypes.cast(frame.data, ctypes.POINTER(ctypes.c_ubyte * frame.dataSize))
                            depth_data = np.frombuffer(data_array.contents, dtype=np.uint16)

                            # 重塑为图像
                            if len(depth_data) == frame.width * frame.height:
                                depth_image = depth_data.reshape(frame.height, frame.width)
                                print(f"✅ 深度图像形状: {depth_image.shape}")

                                # 显示统计信息
                                valid_pixels = np.sum(depth_image > 0)
                                print(f"  有效像素: {valid_pixels}")
                                print(f"  深度范围: {depth_image.min()}-{depth_image.max()} mm")

                                return depth_image
                            else:
                                print(f"❌ 数据大小不匹配: {len(depth_data)} != {frame.width * frame.height}")
                        else:
                            print("❌ 数据大小不足")
                    else:
                        print(f"  传感器类型: {frame.sensorType} (非深度)")
                else:
                    print("❌ 数据指针为空或数据大小为0")

            except Exception as e:
                print(f"❌ 解析帧结构体失败: {e}")
                traceback.print_exc()

            finally:
                # 释放帧
                self._lib.oniFrameRelease(frame_ptr)
                print("  帧已释放")

        return None

    def test_camera(self, device_index=0):
        """测试相机"""
        print(f"\n测试相机 {device_index}...")

        # 获取设备列表
        devices = self.get_device_list()
        if not devices or device_index >= len(devices):
            print("❌ 设备不存在")
            return False

        print(f"✅ 找到设备: {devices[device_index]['name']}")

        # 打开设备
        device_handle = ctypes.c_void_p()
        uri = devices[device_index]['uri'].encode('utf-8')
        status = self._lib.oniDeviceOpen(uri, ctypes.byref(device_handle))

        if status != 0:
            error_msg = self._lib.oniGetExtendedError()
            print(f"❌ 打开设备失败: {status}, 错误: {error_msg}")
            return False

        print("✅ 设备打开成功")

        # 创建深度流
        stream_handle = ctypes.c_void_p()
        status = self._lib.oniDeviceCreateStream(device_handle, 3, ctypes.byref(stream_handle))  # 3 = DEPTH

        if status != 0:
            error_msg = self._lib.oniGetExtendedError()
            print(f"❌ 创建深度流失败: {status}, 错误: {error_msg}")
            self._lib.oniDeviceClose(device_handle)
            return False

        print("✅ 深度流创建成功")

        # 启动流
        status = self._lib.oniStreamStart(stream_handle)
        if status != 0:
            error_msg = self._lib.oniGetExtendedError()
            print(f"❌ 启动流失败: {status}, 错误: {error_msg}")
            self._lib.oniStreamDestroy(stream_handle)
            self._lib.oniDeviceClose(device_handle)
            return False

        print("✅ 流启动成功")

        # 等待一会儿，让流稳定
        time.sleep(0.5)

        # 读取帧
        depth_image = self.read_frame_with_debug(stream_handle, 5)

        if depth_image is not None:
            print(f"\n✅ 成功获取深度图像: {depth_image.shape}")

            # 显示图像
            try:
                # 归一化深度图用于显示
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                cv2.imshow("深度图", depth_normalized)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"显示图像失败: {e}")

        # 清理
        print("\n清理资源...")
        self._lib.oniStreamStop(stream_handle)
        self._lib.oniStreamDestroy(stream_handle)
        self._lib.oniDeviceClose(device_handle)

        return depth_image is not None


def enhanced_test():
    """增强测试"""
    print("=" * 60)
    print("增强的奥比中光SDK测试")
    print("=" * 60)

    sdk = EnhancedOrbbecSDK()

    if not sdk.initialize():
        print("❌ SDK初始化失败")
        return

    print("✅ SDK初始化成功")

    # 获取设备列表
    devices = sdk.get_device_list()
    print(f"\n找到 {len(devices)} 个设备:")
    for i, dev in enumerate(devices):
        print(f"  设备 {i}: {dev['name']} - {dev['uri']}")

    # 测试第一个设备
    if devices:
        success = sdk.test_camera(0)
        if success:
            print("\n✅ 相机测试成功")
        else:
            print("\n❌ 相机测试失败")

    # 清理
    sdk._lib.oniShutdown()
    print("\n✅ 测试完成")


if __name__ == "__main__":
    enhanced_test()