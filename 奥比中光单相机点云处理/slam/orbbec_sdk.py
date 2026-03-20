"""
奥比中光相机SDK接口层
基于OpenNI C API，解决0xC0000374堆损坏问题
独立模块，便于其他代码调用
"""

import ctypes
import numpy as np
import os
import sys
from typing import Optional, Tuple, Dict, Any


class OrbbecCameraSDK:
    """奥比中光相机SDK接口层 - 独立模块"""

    # 常量定义
    ONI_MAX_STR = 256
    ONI_STATUS_OK = 0
    ONI_STATUS_ERROR = 1
    ONI_STATUS_TIME_OUT = 7

    ONI_SENSOR_IR = 1
    ONI_SENSOR_COLOR = 2
    ONI_SENSOR_DEPTH = 3

    ONI_PIXEL_FORMAT_DEPTH_1_MM = 100
    ONI_PIXEL_FORMAT_DEPTH_100_UM = 101
    ONI_PIXEL_FORMAT_RGB888 = 200
    ONI_PIXEL_FORMAT_YUV422 = 201
    ONI_PIXEL_FORMAT_GRAY8 = 202
    ONI_PIXEL_FORMAT_GRAY16 = 203

    ONI_API_VERSION = 2000  # OpenNI 2.0

    def __init__(self, sdk_path: Optional[str] = None):
        """
        初始化相机SDK接口

        Args:
            sdk_path: SDK路径，默认使用内置路径
        """
        self.sdk_path = sdk_path
        self.lib = None
        self.device_handle = None
        self.depth_stream_handle = None
        self.color_stream_handle = None
        self.ir_stream_handle = None
        self.is_initialized = False

        # 定义结构体
        self._define_structures()

        # 初始化SDK
        self._initialize_sdk()

    def set_registration_mode(self, mode: int) -> bool:
        """
        设置图像注册模式（对齐深度与彩色）
        mode: 0 = OFF, 1 = DEPTH_TO_COLOR (推荐), 2 = COLOR_TO_DEPTH
        """
        if not self.lib or not self.device_handle:
            return False

        # OpenNI2 C API 中对应函数名是 oniDeviceSetImageRegistrationMode
        if not hasattr(self.lib, 'oniDeviceSetImageRegistrationMode'):
            print("⚠️ 当前OpenNI2.dll不支持 oniDeviceSetImageRegistrationMode")
            return False

        try:
            func = self.lib.oniDeviceSetImageRegistrationMode
            func.argtypes = [ctypes.c_void_p, ctypes.c_int]
            func.restype = ctypes.c_int

            status = func(self.device_handle, mode)
            if status == self.ONI_STATUS_OK:
                print(f"✅ 已设置注册模式: {mode}")
                return True
            else:
                print(f"❌ 注册模式设置失败: {status}")
                return False
        except Exception as e:
            print(f"❌ 调用 oniDeviceSetImageRegistrationMode 失败: {e}")
            return False

    def _define_structures(self):
        """定义OpenNI结构体"""

        # OniDeviceInfo结构体
        class OniDeviceInfo(ctypes.Structure):
            _fields_ = [
                ("uri", ctypes.c_char * self.ONI_MAX_STR),
                ("vendor", ctypes.c_char * self.ONI_MAX_STR),
                ("name", ctypes.c_char * self.ONI_MAX_STR),
                ("serialNumber", ctypes.c_char * self.ONI_MAX_STR),
                ("usbVendorId", ctypes.c_uint16),
                ("usbProductId", ctypes.c_uint16),
            ]

        # OniVideoMode结构体
        class OniVideoMode(ctypes.Structure):
            _fields_ = [
                ("pixelFormat", ctypes.c_int),
                ("resolutionX", ctypes.c_int),
                ("resolutionY", ctypes.c_int),
                ("fps", ctypes.c_int),
            ]

        # OniFrame结构体
        class OniFrame(ctypes.Structure):
            _fields_ = [
                ("dataSize", ctypes.c_int),
                ("data", ctypes.c_void_p),
                ("sensorType", ctypes.c_int),
                ("timestamp", ctypes.c_uint64),
                ("frameIndex", ctypes.c_int),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int),
                ("videoMode", OniVideoMode),
                ("croppingEnabled", ctypes.c_int),
                ("cropOriginX", ctypes.c_int),
                ("cropOriginY", ctypes.c_int),
                ("stride", ctypes.c_int),
            ]

        self.OniDeviceInfo = OniDeviceInfo
        self.OniVideoMode = OniVideoMode
        self.OniFrame = OniFrame

    def _initialize_sdk(self):
        """初始化SDK"""
        # 设置DLL搜索路径
        self._setup_dll_path()

        # 加载OpenNI2.dll
        if not self._load_openni_library():
            raise RuntimeError("无法加载OpenNI2.dll，请检查SDK路径")

        # 定义函数原型
        self._define_function_prototypes()

    def _setup_dll_path(self):
        """设置DLL路径"""
        # 如果提供了SDK路径，将其Drivers目录添加到PATH
        if self.sdk_path and os.path.exists(self.sdk_path):
            drivers_dir = os.path.join(self.sdk_path, "libs", "OpenNI2", "Drivers")
            if os.path.exists(drivers_dir):
                os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']
                print(f"✅ 已将Drivers目录添加到PATH: {drivers_dir}")

    def _load_openni_library(self) -> bool:
        """加载OpenNI2.dll"""
        # 可能的DLL路径
        dll_search_paths = []

        # 如果提供了SDK路径，优先搜索
        if self.sdk_path:
            dll_search_paths.extend([
                os.path.join(self.sdk_path, "libs", "OpenNI2.dll"),
                os.path.join(self.sdk_path, "libs", "OpenNI2", "Redist", "OpenNI2.dll"),
                os.path.join(self.sdk_path, "libs", "OpenNI2", "Drivers", "OpenNI2.dll"),
            ])

        # 添加默认路径
        dll_search_paths.extend([
            r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs\OpenNI2.dll",
            "OpenNI2.dll",  # 从系统PATH加载
        ])

        for dll_path in dll_search_paths:
            if dll_path == "OpenNI2.dll" or os.path.exists(dll_path):
                try:
                    self.lib = ctypes.CDLL(dll_path)
                    if dll_path != "OpenNI2.dll":
                        print(f"✅ 已从指定路径加载: {dll_path}")
                    else:
                        print(f"✅ 已从系统PATH加载: OpenNI2.dll")
                    return True
                except Exception as e:
                    print(f"⚠️  加载 {dll_path} 失败: {e}")

        return False

    def _define_function_prototypes(self):
        """定义所有API函数原型"""
        if not self.lib:
            return

        try:
            # ========== 通用API ==========
            self.lib.oniInitialize.argtypes = [ctypes.c_int]
            self.lib.oniInitialize.restype = ctypes.c_int

            self.lib.oniShutdown.argtypes = []
            self.lib.oniShutdown.restype = None

            # 设备列表API
            self.lib.oniGetDeviceList.argtypes = [
                ctypes.POINTER(ctypes.POINTER(self.OniDeviceInfo)),
                ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.oniGetDeviceList.restype = ctypes.c_int

            self.lib.oniReleaseDeviceList.argtypes = [ctypes.POINTER(self.OniDeviceInfo)]
            self.lib.oniReleaseDeviceList.restype = ctypes.c_int

            # ========== 设备API ==========
            self.lib.oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            self.lib.oniDeviceOpen.restype = ctypes.c_int

            self.lib.oniDeviceClose.argtypes = [ctypes.c_void_p]
            self.lib.oniDeviceClose.restype = ctypes.c_int

            self.lib.oniDeviceCreateStream.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            self.lib.oniDeviceCreateStream.restype = ctypes.c_int

            # ========== 流API ==========
            self.lib.oniStreamDestroy.argtypes = [ctypes.c_void_p]
            self.lib.oniStreamDestroy.restype = None

            self.lib.oniStreamStart.argtypes = [ctypes.c_void_p]
            self.lib.oniStreamStart.restype = ctypes.c_int

            self.lib.oniStreamStop.argtypes = [ctypes.c_void_p]
            self.lib.oniStreamStop.restype = None

            self.lib.oniStreamReadFrame.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.POINTER(self.OniFrame))
            ]
            self.lib.oniStreamReadFrame.restype = ctypes.c_int

            # ========== 帧API ==========
            self.lib.oniFrameAddRef.argtypes = [ctypes.POINTER(self.OniFrame)]
            self.lib.oniFrameAddRef.restype = None

            self.lib.oniFrameRelease.argtypes = [ctypes.POINTER(self.OniFrame)]
            self.lib.oniFrameRelease.restype = None

            print("✅ OpenNI C API函数原型定义完成")

        except AttributeError as e:
            print(f"⚠️ 定义函数原型时出错: {e}")
            raise

    # ========== 公共接口方法 ==========

    def initialize(self) -> bool:
        """初始化OpenNI SDK"""
        if not self.lib:
            return False

        print("初始化OpenNI SDK...")
        status = self.lib.oniInitialize(self.ONI_API_VERSION)

        if status != self.ONI_STATUS_OK:
            print(f"❌ 初始化失败，错误码: {status}")
            return False

        self.is_initialized = True
        print("✅ OpenNI SDK初始化成功")
        return True

    def get_device_list(self) -> Optional[Dict[int, Dict[str, Any]]]:
        """获取设备列表

        Returns:
            dict: 设备字典 {索引: {名称, 厂商, URI, ...}}
        """
        if not self.lib or not self.is_initialized:
            return None

        devices_ptr = ctypes.POINTER(self.OniDeviceInfo)()
        device_count = ctypes.c_int(0)

        status = self.lib.oniGetDeviceList(ctypes.byref(devices_ptr), ctypes.byref(device_count))
        if status != self.ONI_STATUS_OK:
            print(f"❌ 获取设备列表失败，错误码: {status}")
            return None

        devices = {}
        for i in range(device_count.value):
            device_info = devices_ptr[i]
            devices[i] = {
                'uri': device_info.uri.decode('utf-8', errors='ignore').rstrip('\x00'),
                'vendor': device_info.vendor.decode('utf-8', errors='ignore').rstrip('\x00'),
                'name': device_info.name.decode('utf-8', errors='ignore').rstrip('\x00'),
                'serial_number': device_info.serialNumber.decode('utf-8', errors='ignore').rstrip('\x00'),
                'usb_vendor_id': device_info.usbVendorId,
                'usb_product_id': device_info.usbProductId,
            }
            print(f"  [{i}] {devices[i]['name']} ({devices[i]['vendor']})")

        # 释放设备列表
        self.lib.oniReleaseDeviceList(devices_ptr)

        return devices

    def open_device(self, device_index: int = 0, uri: Optional[str] = None) -> bool:
        """打开设备

        Args:
            device_index: 设备索引
            uri: 设备URI（如果为None则使用设备索引）

        Returns:
            bool: 是否成功
        """
        if not self.lib or not self.is_initialized:
            return False

        # 获取设备URI
        if uri is None:
            devices = self.get_device_list()
            if not devices or device_index not in devices:
                print("❌ 未找到指定设备，尝试打开默认设备...")
                uri = None
            else:
                uri = devices[device_index]['uri'].encode('utf-8')

        # 打开设备
        device_handle = ctypes.c_void_p()
        status = self.lib.oniDeviceOpen(uri, ctypes.byref(device_handle))

        if status != self.ONI_STATUS_OK:
            print(f"❌ 打开设备失败，错误码: {status}")
            return False

        self.device_handle = device_handle
        print(f"✅ 设备打开成功，句柄: {device_handle.value}")
        return True

    def create_stream(self, sensor_type: int = ONI_SENSOR_DEPTH) -> bool:
        """创建数据流

        Args:
            sensor_type: 传感器类型（深度/彩色/红外）

        Returns:
            bool: 是否成功
        """
        if not self.lib or not self.device_handle:
            return False

        sensor_names = {
            self.ONI_SENSOR_DEPTH: "深度",
            self.ONI_SENSOR_COLOR: "彩色",
            self.ONI_SENSOR_IR: "红外"
        }

        print(f"创建 {sensor_names.get(sensor_type, '未知')} 传感器流...")

        stream_handle = ctypes.c_void_p()
        status = self.lib.oniDeviceCreateStream(
            self.device_handle,
            sensor_type,
            ctypes.byref(stream_handle)
        )

        if status != self.ONI_STATUS_OK:
            print(f"❌ 创建流失败，错误码: {status}")
            return False

        # 保存流句柄
        if sensor_type == self.ONI_SENSOR_DEPTH:
            self.depth_stream_handle = stream_handle
        elif sensor_type == self.ONI_SENSOR_COLOR:
            self.color_stream_handle = stream_handle
        elif sensor_type == self.ONI_SENSOR_IR:
            self.ir_stream_handle = stream_handle

        print(f"✅ 流创建成功，句柄: {stream_handle.value}")
        return True

    def start_stream(self, stream_handle: Optional[ctypes.c_void_p] = None) -> bool:
        """启动数据流

        Args:
            stream_handle: 流句柄（如果为None则使用深度流）

        Returns:
            bool: 是否成功
        """
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if not self.lib or not stream_handle:
            return False

        status = self.lib.oniStreamStart(stream_handle)
        if status != self.ONI_STATUS_OK:
            print(f"❌ 启动流失败，错误码: {status}")
            return False

        print("✅ 流启动成功")
        return True

    def read_frame(self, stream_handle: Optional[ctypes.c_void_p] = None,
                   timeout: int = 1000) -> Optional[Dict[str, Any]]:
        """读取一帧数据

        Args:
            stream_handle: 流句柄（如果为None则使用深度流）
            timeout: 超时时间（毫秒）

        Returns:
            dict: 帧数据信息
        """
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if not self.lib or not stream_handle:
            return None

        frame_ptr = ctypes.POINTER(self.OniFrame)()
        status = self.lib.oniStreamReadFrame(stream_handle, ctypes.byref(frame_ptr))

        if status != self.ONI_STATUS_OK:
            if status == self.ONI_STATUS_TIME_OUT:
                print("⚠️  读取帧超时")
            else:
                print(f"❌ 读取帧失败，错误码: {status}")
            return None

        frame = frame_ptr.contents

        # 处理帧数据
        frame_info = self._process_frame_data(frame, frame_ptr)

        return frame_info

    def _process_frame_data(self, frame: 'OniFrame', frame_ptr) -> Dict[str, Any]:
        """处理帧数据"""
        frame_info = {
            'width': frame.width,
            'height': frame.height,
            'timestamp': frame.timestamp,
            'frame_index': frame.frameIndex,
            'sensor_type': frame.sensorType,
            'pixel_format': frame.videoMode.pixelFormat,
            'frame_ptr': frame_ptr,
        }

        # 根据像素格式处理数据
        if frame.videoMode.pixelFormat == self.ONI_PIXEL_FORMAT_DEPTH_1_MM:
            # 深度数据 (单位: mm)
            depth_data = ctypes.string_at(frame.data, frame.dataSize)
            depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(frame.height, frame.width)
            frame_info['data'] = depth_array
            frame_info['data_type'] = 'depth'
            frame_info['unit'] = 'mm'

        elif frame.videoMode.pixelFormat == self.ONI_PIXEL_FORMAT_RGB888:
            # RGB彩色数据
            color_data = ctypes.string_at(frame.data, frame.dataSize)
            color_array = np.frombuffer(color_data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            frame_info['data'] = color_array
            frame_info['data_type'] = 'color'

        elif frame.videoMode.pixelFormat == self.ONI_PIXEL_FORMAT_GRAY8:
            # 灰度数据
            gray_data = ctypes.string_at(frame.data, frame.dataSize)
            gray_array = np.frombuffer(gray_data, dtype=np.uint8).reshape(frame.height, frame.width)
            frame_info['data'] = gray_array
            frame_info['data_type'] = 'grayscale'

        else:
            print(f"⚠️  不支持的像素格式: {frame.videoMode.pixelFormat}")
            self.lib.oniFrameRelease(frame_ptr)
            return None

        # 增加引用计数
        self.lib.oniFrameAddRef(frame_ptr)

        return frame_info

    def release_frame(self, frame_info: Dict[str, Any]) -> None:
        """释放帧资源

        Args:
            frame_info: 帧数据信息
        """
        if frame_info and 'frame_ptr' in frame_info:
            self.lib.oniFrameRelease(frame_info['frame_ptr'])

    def capture_depth_frame(self, timeout: int = 1000) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """捕获深度帧（简化接口）

        Args:
            timeout: 超时时间（毫秒）

        Returns:
            tuple: (深度数据数组, 帧信息字典) 或 None
        """
        frame_info = self.read_frame(self.depth_stream_handle, timeout)
        if not frame_info:
            return None

        depth_data = frame_info['data']

        # 自动释放帧资源（可选，也可以让调用者手动释放）
        self.release_frame(frame_info)

        return depth_data, frame_info

    def stop_stream(self, stream_handle: Optional[ctypes.c_void_p] = None) -> None:
        """停止数据流

        Args:
            stream_handle: 流句柄（如果为None则使用深度流）
        """
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if self.lib and stream_handle:
            self.lib.oniStreamStop(stream_handle)
            print("✅ 流已停止")

    def destroy_stream(self, stream_handle: Optional[ctypes.c_void_p] = None) -> None:
        """销毁数据流

        Args:
            stream_handle: 流句柄（如果为None则使用深度流）
        """
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if self.lib and stream_handle:
            self.lib.oniStreamDestroy(stream_handle)
            # 清理对应的句柄
            if stream_handle == self.depth_stream_handle:
                self.depth_stream_handle = None
            elif stream_handle == self.color_stream_handle:
                self.color_stream_handle = None
            elif stream_handle == self.ir_stream_handle:
                self.ir_stream_handle = None
            print("✅ 流已销毁")

    def close_device(self) -> None:
        """关闭设备"""
        if self.lib and self.device_handle:
            self.lib.oniDeviceClose(self.device_handle)
            self.device_handle = None
            print("✅ 设备已关闭")

    def shutdown(self) -> None:
        """关闭SDK"""
        if self.lib and self.is_initialized:
            self.lib.oniShutdown()
            self.is_initialized = False
            print("✅ OpenNI SDK已关闭")

    def cleanup(self) -> None:
        """完整清理所有资源"""
        print("\n清理SDK资源...")

        # 停止并销毁深度流
        if self.depth_stream_handle:
            self.stop_stream(self.depth_stream_handle)
            self.destroy_stream(self.depth_stream_handle)

        # 停止并销毁彩色流
        if self.color_stream_handle:
            self.stop_stream(self.color_stream_handle)
            self.destroy_stream(self.color_stream_handle)

        # 停止并销毁红外流
        if self.ir_stream_handle:
            self.stop_stream(self.ir_stream_handle)
            self.destroy_stream(self.ir_stream_handle)

        # 关闭设备
        self.close_device()

        # 关闭SDK
        self.shutdown()

        print("✅ 所有SDK资源已清理")


# ========== 工具函数 ==========

def get_default_sdk_path() -> str:
    """获取默认SDK路径"""
    default_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    if os.path.exists(default_path):
        return default_path
    return ""


def test_sdk_connection() -> bool:
    """测试SDK连接（简易测试）"""
    try:
        sdk = OrbbecCameraSDK(get_default_sdk_path())
        if not sdk.initialize():
            return False

        devices = sdk.get_device_list()
        if not devices:
            print("⚠️  未检测到设备")

        sdk.cleanup()
        return True

    except Exception as e:
        print(f"❌ SDK连接测试失败: {e}")
        return False


if __name__ == "__main__":
    # 模块自测试
    print("=" * 60)
    print("奥比中光SDK接口层 - 自测试")
    print("=" * 60)

    success = test_sdk_connection()

    if success:
        print("\n✅ SDK接口层测试通过")
    else:
        print("\n❌ SDK接口层测试失败")