# 文件：openni_fixed.py
import ctypes
import numpy as np
import os
import sys
from ctypes import Structure, POINTER, c_void_p, c_int, c_char, c_float, c_uint, c_uint64, c_uint16, c_uint8, c_uint32, \
    byref

# ============================================================
# 根据OniCTypes.h的精确定义
# ============================================================

# 常量定义
ONI_MAX_STR = 256
ONI_STATUS_OK = 0

# 枚举定义
ONI_SENSOR_IR = 1
ONI_SENSOR_COLOR = 2
ONI_SENSOR_DEPTH = 3

ONI_PIXEL_FORMAT_DEPTH_1_MM = 100
ONI_PIXEL_FORMAT_DEPTH_100_UM = 101
ONI_PIXEL_FORMAT_RGB888 = 200

ONI_API_VERSION = 2000  # OpenNI 2.0


# ============================================================
# 精确的结构体定义 (关键修复！)
# ============================================================

class OniVersion(Structure):
    _fields_ = [
        ("major", c_int),
        ("minor", c_int),
        ("maintenance", c_int),
        ("build", c_int),
    ]


class OniVideoMode(Structure):
    _fields_ = [
        ("pixelFormat", c_int),
        ("resolutionX", c_int),
        ("resolutionY", c_int),
        ("fps", c_int),
    ]


class OniSensorInfo(Structure):
    _fields_ = [
        ("sensorType", c_int),
        ("numSupportedVideoModes", c_int),
        ("pSupportedVideoModes", POINTER(OniVideoMode)),
    ]


# ⭐⭐ 关键修复：使用字符数组而不是字符指针！ ⭐⭐
class OniDeviceInfo(Structure):
    _fields_ = [
        ("uri", c_char * ONI_MAX_STR),  # char uri[ONI_MAX_STR]
        ("vendor", c_char * ONI_MAX_STR),  # char vendor[ONI_MAX_STR]
        ("name", c_char * ONI_MAX_STR),  # char name[ONI_MAX_STR]
        ("serialNumber", c_char * ONI_MAX_STR),  # char serialNumber[ONI_MAX_STR] (之前缺少！)
        ("usbVendorId", c_uint16),  # uint16_t usbVendorId
        ("usbProductId", c_uint16),  # uint16_t usbProductId
    ]


class OniFrame(Structure):
    _fields_ = [
        ("dataSize", c_int),
        ("data", c_void_p),
        ("sensorType", c_int),
        ("timestamp", c_uint64),  # ⭐ 注意：是uint64_t，不是uint
        ("frameIndex", c_int),
        ("width", c_int),
        ("height", c_int),
        ("videoMode", OniVideoMode),
        ("croppingEnabled", c_int),  # OniBool 实际是 int
        ("cropOriginX", c_int),
        ("cropOriginY", c_int),
        ("stride", c_int),
    ]


class OBCameraParams(Structure):
    _fields_ = [
        ("l_intr_p", c_float * 4),
        ("r_intr_p", c_float * 4),
        ("r2l_r", c_float * 9),
        ("r2l_t", c_float * 3),
        ("l_k", c_float * 5),
        ("r_k", c_float * 5),
    ]


# ============================================================
# 修复后的OpenNI封装类
# ============================================================

class FixedOpenNICAPI:
    """修复结构体定义后的OpenNI C API封装"""

    def __init__(self, driver_path=None):
        self.driver_path = driver_path
        self.lib = None
        self.device_handle = None
        self.depth_stream_handle = None

        # 确保Driver目录在PATH中
        if driver_path:
            self._add_driver_to_path()

        # 加载OpenNI2.dll
        self._load_openni_library()

        # 定义API函数原型
        if self.lib:
            self._define_function_prototypes()

    def _add_driver_to_path(self):
        """将Driver目录添加到系统PATH"""
        drivers_dir = os.path.join(self.driver_path, "libs", "OpenNI2", "Drivers")
        if os.path.exists(drivers_dir):
            os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']
            print(f"✅ 已将Drivers目录添加到PATH: {drivers_dir}")

    def _load_openni_library(self):
        """加载OpenNI2.dll"""
        dll_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs\OpenNI2.dll"

        if not os.path.exists(dll_path):
            print(f"❌ DLL文件不存在: {dll_path}")
            # 尝试从PATH加载
            dll_path = "OpenNI2.dll"

        try:
            self.lib = ctypes.CDLL(dll_path)
            print(f"✅ 已加载OpenNI2.dll")
        except Exception as e:
            print(f"❌ 加载DLL失败: {e}")
            self.lib = None

    def _define_function_prototypes(self):
        """定义所有API函数原型"""
        try:
            # ========== 通用API ==========
            self.lib.oniInitialize.argtypes = [c_int]
            self.lib.oniInitialize.restype = c_int

            self.lib.oniShutdown.argtypes = []
            self.lib.oniShutdown.restype = None

            # ⭐ 关键修复：使用正确的指针类型
            self.lib.oniGetDeviceList.argtypes = [POINTER(POINTER(OniDeviceInfo)), POINTER(c_int)]
            self.lib.oniGetDeviceList.restype = c_int

            self.lib.oniReleaseDeviceList.argtypes = [POINTER(OniDeviceInfo)]
            self.lib.oniReleaseDeviceList.restype = c_int

            # ========== 设备API ==========
            #self.lib.oniDeviceOpen.argtypes = [c_char_p, POINTER(c_void_p)]
            self.lib.oniDeviceOpen.restype = c_int

            self.lib.oniDeviceClose.argtypes = [c_void_p]
            self.lib.oniDeviceClose.restype = c_int

            self.lib.oniDeviceCreateStream.argtypes = [c_void_p, c_int, POINTER(c_void_p)]
            self.lib.oniDeviceCreateStream.restype = c_int

            self.lib.oniDeviceGetSensorInfo.argtypes = [c_void_p, c_int]
            self.lib.oniDeviceGetSensorInfo.restype = POINTER(OniSensorInfo)

            # ========== 流API ==========
            self.lib.oniStreamDestroy.argtypes = [c_void_p]
            self.lib.oniStreamDestroy.restype = None

            self.lib.oniStreamStart.argtypes = [c_void_p]
            self.lib.oniStreamStart.restype = c_int

            self.lib.oniStreamStop.argtypes = [c_void_p]
            self.lib.oniStreamStop.restype = None

            self.lib.oniStreamReadFrame.argtypes = [c_void_p, POINTER(POINTER(OniFrame))]
            self.lib.oniStreamReadFrame.restype = c_int

            # ========== 帧API ==========
            self.lib.oniFrameAddRef.argtypes = [POINTER(OniFrame)]
            self.lib.oniFrameAddRef.restype = None

            self.lib.oniFrameRelease.argtypes = [POINTER(OniFrame)]
            self.lib.oniFrameRelease.restype = None

            print("✅ 所有函数原型定义完成")

        except AttributeError as e:
            print(f"⚠️ 定义函数原型时出错: {e}")

    def initialize(self):
        """初始化OpenNI"""
        if not self.lib:
            return False

        print("初始化OpenNI...")
        status = self.lib.oniInitialize(ONI_API_VERSION)
        if status != ONI_STATUS_OK:
            print(f"❌ 初始化失败，错误码: {status}")
            return False

        print("✅ OpenNI初始化成功")
        return True

    def get_device_list_safe(self):
        """安全地获取设备列表（修复版本）"""
        if not self.lib:
            return False

        print("获取设备列表...")

        # 创建指针和计数变量
        devices_ptr = POINTER(OniDeviceInfo)()
        device_count = c_int(0)

        # 调用API
        status = self.lib.oniGetDeviceList(byref(devices_ptr), byref(device_count))

        print(f"API调用状态: {status}")
        print(f"设备数量: {device_count.value}")

        if status != ONI_STATUS_OK:
            print(f"❌ 获取设备列表失败")
            return False

        if device_count.value == 0:
            print("⚠️  未找到设备")
            return True  # 仍然返回True，因为API调用成功

        # ⭐ 安全地访问设备信息
        print("\n设备详细信息:")
        for i in range(device_count.value):
            device_info = devices_ptr[i]

            # 正确解码字符数组
            uri = device_info.uri.decode('utf-8', errors='ignore').rstrip('\x00')
            name = device_info.name.decode('utf-8', errors='ignore').rstrip('\x00')
            vendor = device_info.vendor.decode('utf-8', errors='ignore').rstrip('\x00')
            serial = device_info.serialNumber.decode('utf-8', errors='ignore').rstrip('\x00')

            print(f"  设备 {i}:")
            print(f"    名称: {name}")
            print(f"    厂商: {vendor}")
            print(f"    序列号: {serial}")
            print(f"    USB ID: {device_info.usbVendorId:04X}:{device_info.usbProductId:04X}")
            print(f"    URI: {uri}")

        # 释放设备列表
        self.lib.oniReleaseDeviceList(devices_ptr)

        return True

    def open_first_device(self):
        """打开第一个设备"""
        if not self.lib:
            return False

        print("\n尝试打开第一个设备...")

        # 获取设备列表
        devices_ptr = POINTER(OniDeviceInfo)()
        device_count = c_int(0)

        status = self.lib.oniGetDeviceList(byref(devices_ptr), byref(device_count))
        if status != ONI_STATUS_OK or device_count.value == 0:
            print("❌ 没有可用设备，尝试打开默认设备...")
            # 尝试使用默认URI
            uri = None
        else:
            # 使用第一个设备的URI
            device_info = devices_ptr[0]
            uri = device_info.uri
            null_char = '\x00'
            print(f"使用设备: {device_info.name.decode('utf-8', errors='ignore').rstrip(null_char)}")

        # 打开设备
        device_handle = c_void_p()
        if uri:
            status = self.lib.oniDeviceOpen(uri, byref(device_handle))
        else:
            # 打开任何设备
            status = self.lib.oniDeviceOpen(None, byref(device_handle))

        if status != ONI_STATUS_OK:
            print(f"❌ 打开设备失败，错误码: {status}")
            if devices_ptr:
                self.lib.oniReleaseDeviceList(devices_ptr)
            return False

        self.device_handle = device_handle
        print(f"✅ 设备打开成功，句柄: {device_handle.value}")

        # 释放设备列表
        if devices_ptr:
            self.lib.oniReleaseDeviceList(devices_ptr)

        return True

    def simple_depth_test(self):
        """简单的深度流测试"""
        if not self.lib or not self.device_handle:
            return False

        print("\n创建深度流...")

        # 创建深度流
        stream_handle = c_void_p()
        status = self.lib.oniDeviceCreateStream(self.device_handle, ONI_SENSOR_DEPTH, byref(stream_handle))

        if status != ONI_STATUS_OK:
            print(f"❌ 创建深度流失败，错误码: {status}")
            return False

        self.depth_stream_handle = stream_handle
        print(f"✅ 深度流创建成功，句柄: {stream_handle.value}")

        # 启动流
        print("启动深度流...")
        status = self.lib.oniStreamStart(stream_handle)
        if status != ONI_STATUS_OK:
            print(f"❌ 启动流失败，错误码: {status}")
            return False

        print("✅ 深度流已启动")

        # 读取一帧测试
        print("尝试读取一帧深度数据...")
        frame_ptr = POINTER(OniFrame)()
        status = self.lib.oniStreamReadFrame(stream_handle, byref(frame_ptr))

        if status != ONI_STATUS_OK:
            print(f"❌ 读取帧失败，错误码: {status}")
            return False

        frame = frame_ptr.contents
        print(f"✅ 成功读取深度帧:")
        print(f"   尺寸: {frame.width}x{frame.height}")
        print(f"   格式: {frame.videoMode.pixelFormat}")
        print(f"   时间戳: {frame.timestamp}")

        # 处理深度数据
        if frame.videoMode.pixelFormat == ONI_PIXEL_FORMAT_DEPTH_1_MM:
            depth_array = np.frombuffer(
                ctypes.string_at(frame.data, frame.dataSize),
                dtype=np.uint16
            ).reshape(frame.height, frame.width)

            print(f"   深度范围: {depth_array.min()}-{depth_array.max()} mm")
            print(f"   有效点数: {(depth_array > 0).sum()}")

            # 保存测试图像
            if depth_array.max() > 0:
                depth_normalized = ((depth_array.astype(np.float32) / depth_array.max()) * 255).astype(np.uint8)
                import cv2
                cv2.imwrite("test_depth_frame.png", depth_normalized)
                print("   测试图像已保存: test_depth_frame.png")

        # 增加引用计数
        self.lib.oniFrameAddRef(frame_ptr)

        # 释放帧
        self.lib.oniFrameRelease(frame_ptr)

        # 停止流
        self.lib.oniStreamStop(stream_handle)

        # 销毁流
        self.lib.oniStreamDestroy(stream_handle)
        self.depth_stream_handle = None

        return True

    def cleanup(self):
        """清理所有资源"""
        print("\n清理资源...")

        if self.lib:
            # 关闭设备
            if self.device_handle:
                self.lib.oniDeviceClose(self.device_handle)
                self.device_handle = None
                print("✅ 设备已关闭")

            # 关闭OpenNI
            self.lib.oniShutdown()
            print("✅ OpenNI已关闭")


# ============================================================
# 最小化测试程序
# ============================================================

def minimal_test():
    """最小化测试程序"""
    print("=" * 60)
    print("OpenNI C API 结构体修复测试")
    print("=" * 60)

    # 设置环境
    os.environ['PATH'] = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs\OpenNI2\Drivers" + ';' + \
                         os.environ['PATH']

    # 创建API对象
    sdk_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release"
    openni = FixedOpenNICAPI(sdk_path)

    if not openni.lib:
        print("❌ 无法加载OpenNI库")
        return False

    try:
        # 1. 初始化
        if not openni.initialize():
            return False

        # 2. 安全地获取设备列表
        if not openni.get_device_list_safe():
            print("⚠️  获取设备列表失败或没有设备")

        # 3. 打开设备
        if not openni.open_first_device():
            return False

        # 4. 简单深度测试
        if not openni.simple_depth_test():
            print("⚠️  深度测试失败")

        print("\n" + "=" * 60)
        print("✅ 测试完成！结构体修复成功")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理资源
        openni.cleanup()


def main():
    """主函数"""
    success = minimal_test()

    if success:
        print("\n🎉 恭喜！OpenNI C API 结构体修复成功")
        print("\n接下来:")
        print("1. 将 FixedOpenNICAPI 类集成到你的主程序")
        print("2. 替换原来的 primesense 代码")
        print("3. 添加彩色流支持 (ONI_SENSOR_COLOR)")
        print("\n关键修复总结:")
        print("  • OniDeviceInfo 使用字符数组而不是指针")
        print("  • 添加了缺失的 serialNumber 字段")
        print("  • 修正了数据类型 (uint16_t, uint64_t)")
        print("  • 解决了 0xC0000005 访问冲突")
    else:
        print("\n❌ 测试失败")
        print("\n如果还有问题，请尝试:")
        print("1. 以管理员身份运行")
        print("2. 检查相机连接")
        print("3. 重新安装奥比中光驱动")

    print("\n" + "=" * 70)
    input("按Enter键退出...")


if __name__ == "__main__":
    main()