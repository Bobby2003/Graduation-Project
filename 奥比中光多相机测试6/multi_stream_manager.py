"""
多流相机管理器 - 单实例管理多个相机流
解决多相机资源冲突问题
"""

import ctypes
import numpy as np
import time
import os
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

# ========== 定义 OpenNI 结构体（放在类外部，以便类型注解使用）==========
ONI_MAX_STR = 256

class OniDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("uri", ctypes.c_char * ONI_MAX_STR),
        ("vendor", ctypes.c_char * ONI_MAX_STR),
        ("name", ctypes.c_char * ONI_MAX_STR),
        ("serialNumber", ctypes.c_char * ONI_MAX_STR),
        ("usbVendorId", ctypes.c_uint16),
        ("usbProductId", ctypes.c_uint16),
    ]

class OniVideoMode(ctypes.Structure):
    _fields_ = [
        ("pixelFormat", ctypes.c_int),
        ("resolutionX", ctypes.c_int),
        ("resolutionY", ctypes.c_int),
        ("fps", ctypes.c_int),
    ]

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


@dataclass
class CameraStream:
    """单个相机流的信息"""
    index: int
    device_handle: ctypes.c_void_p
    stream_handle: ctypes.c_void_p
    position: str = "unknown"
    last_frame_time: float = 0
    frame_count: int = 0
    is_active: bool = True


class MultiStreamCameraManager:
    """多流相机管理器 - 单实例管理所有相机"""

    def __init__(self, sdk_path: Optional[str] = None):
        self.sdk_path = sdk_path
        self.lib = None
        self.context_initialized = False
        self.streams: List[CameraStream] = []

        # 🔧 新增：存储颜色流
        self.color_streams: List[ctypes.c_void_p] = []

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化单例 OpenNI 环境"""
        print("初始化多流相机管理器...")

        # 加载 OpenNI2.dll
        dll_search_paths = []

        if self.sdk_path:
            dll_search_paths.append(os.path.join(self.sdk_path, "libs", "OpenNI2.dll"))

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
                    break
                except Exception as e:
                    print(f"⚠️  加载 {dll_path} 失败: {e}")

        if not self.lib:
            print("❌ 无法加载 OpenNI2.dll")
            return

        # 设置 DLL 搜索路径
        if self.sdk_path:
            drivers_dir = os.path.join(self.sdk_path, "libs", "OpenNI2", "Drivers")
            if os.path.exists(drivers_dir):
                os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']

        # 定义函数原型
        self._define_function_prototypes()

        # 初始化 OpenNI
        try:
            status = self.lib.oniInitialize(2000)  # OpenNI 2.0 API
            if status == 0:
                self.context_initialized = True
                print("✅ OpenNI 上下文初始化成功（单实例）")
            else:
                print(f"❌ OpenNI 初始化失败，状态码: {status}")
        except Exception as e:
            print(f"❌ 初始化异常: {e}")

    # 在 MultiStreamCameraManager 类中添加以下方法
    def capture_all_streams_with_color(self) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """从所有流捕获深度和颜色帧（串行采集）"""
        depth_frames = [None] * len(self.streams)
        color_frames = [None] * len(self.streams)

        for i, stream in enumerate(self.streams):
            if not stream.is_active:
                continue

            result = self.capture_frame_with_color(i, timeout_ms=3000)
            if result:
                depth_array, color_array = result
                depth_frames[i] = depth_array
                color_frames[i] = color_array

                # 释放帧资源
                # 注意：这里需要合适的机制来释放frame_ptr，但简化处理

            # 流间延迟，避免 USB 总线拥堵
            if i < len(self.streams) - 1:
                time.sleep(0.1)  # 100ms 延迟

        return depth_frames, color_frames

    def cleanup(self):
        """清理所有资源"""
        print("\n清理多流管理器资源...")

        # 停止和销毁所有流
        for i, stream in enumerate(self.streams):
            print(f"  清理流 {i}...")
            try:
                # 停止深度流
                if stream.stream_handle:
                    self.lib.oniStreamStop(stream.stream_handle)

                # 停止颜色流
                if hasattr(stream, 'color_stream_handle') and stream.color_stream_handle:
                    self.lib.oniStreamStop(stream.color_stream_handle)
                    self.lib.oniStreamDestroy(stream.color_stream_handle)

                # 销毁深度流
                self.lib.oniStreamDestroy(stream.stream_handle)

                # 关闭设备
                if stream.device_handle:
                    self.lib.oniDeviceClose(stream.device_handle)

                print(f"    流 {i} 已清理")
            except Exception as e:
                print(f"    流 {i} 清理异常: {e}")

        self.streams.clear()
        self.color_streams.clear()

        # 关闭 OpenNI 上下文
        if self.context_initialized:
            self.lib.oniShutdown()
            self.context_initialized = False
            print("✅ OpenNI 上下文已关闭")

        print("✅ 所有资源已清理")

    def capture_color_frame(self, stream_index, timeout_ms=1000):
        """采集颜色帧"""
        try:
            if stream_index >= len(self.streams):
                return None

            stream = self.streams[stream_index]
            if not stream or not hasattr(stream, 'color_stream'):
                return None

            # 从颜色流读取帧
            color_frame = stream.color_stream.read_frame(timeout_ms)
            if color_frame:
                # 转换为numpy数组
                color_array = np.asanyarray(color_frame.get_data())
                return color_array, color_frame

        except Exception as e:
            print(f"采集颜色帧失败: {e}")

        return None

    def _define_function_prototypes(self):
        """定义函数原型"""
        # 基础函数
        self.lib.oniInitialize.argtypes = [ctypes.c_int]
        self.lib.oniInitialize.restype = ctypes.c_int
        self.lib.oniShutdown.argtypes = []
        self.lib.oniShutdown.restype = None

        # 设备管理
        self.lib.oniGetDeviceList.argtypes = [
            ctypes.POINTER(ctypes.POINTER(OniDeviceInfo)),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.oniGetDeviceList.restype = ctypes.c_int
        self.lib.oniReleaseDeviceList.argtypes = [ctypes.POINTER(OniDeviceInfo)]
        self.lib.oniReleaseDeviceList.restype = ctypes.c_int

        # 设备操作
        self.lib.oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.oniDeviceOpen.restype = ctypes.c_int
        self.lib.oniDeviceClose.argtypes = [ctypes.c_void_p]
        self.lib.oniDeviceClose.restype = ctypes.c_int

        # 流操作
        self.lib.oniDeviceCreateStream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.oniDeviceCreateStream.restype = ctypes.c_int
        self.lib.oniStreamStart.argtypes = [ctypes.c_void_p]
        self.lib.oniStreamStart.restype = ctypes.c_int
        self.lib.oniStreamStop.argtypes = [ctypes.c_void_p]
        self.lib.oniStreamStop.restype = None
        self.lib.oniStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.oniStreamDestroy.restype = None

        # 帧读取（关键：包含 timeout 参数）
        self.lib.oniStreamReadFrame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(OniFrame)),
            ctypes.c_int  # timeout 参数
        ]
        self.lib.oniStreamReadFrame.restype = ctypes.c_int

        # 帧管理
        self.lib.oniFrameAddRef.argtypes = [ctypes.POINTER(OniFrame)]
        self.lib.oniFrameAddRef.restype = None
        self.lib.oniFrameRelease.argtypes = [ctypes.POINTER(OniFrame)]
        self.lib.oniFrameRelease.restype = None

        print("✅ OpenNI C API函数原型定义完成")

    def discover_devices(self) -> List[Dict[str, Any]]:
        """发现可用设备"""
        if not self.context_initialized:
            return []

        devices_ptr = ctypes.POINTER(OniDeviceInfo)()
        device_count = ctypes.c_int(0)

        status = self.lib.oniGetDeviceList(ctypes.byref(devices_ptr), ctypes.byref(device_count))
        if status != 0:
            print(f"获取设备列表失败，错误码: {status}")
            return []

        devices = []
        for i in range(device_count.value):
            info = devices_ptr[i]
            device = {
                'index': i,
                'uri': info.uri.decode('utf-8', errors='ignore').rstrip('\x00'),
                'name': info.name.decode('utf-8', errors='ignore').rstrip('\x00'),
                'vendor': info.vendor.decode('utf-8', errors='ignore').rstrip('\x00'),
                'serial': info.serialNumber.decode('utf-8', errors='ignore').rstrip('\x00'),
                'usb_vid': info.usbVendorId,
                'usb_pid': info.usbProductId,
            }
            devices.append(device)
            print(f"  [{i}] {device['name']} ({device['vendor']})")

        self.lib.oniReleaseDeviceList(devices_ptr)
        return devices

    def setup_streams(self, num_streams: int = 3, enable_color: bool = True) -> bool:
        """设置多个相机流（深度+颜色）"""
        devices = self.discover_devices()
        if len(devices) < num_streams:
            print(f"⚠️  发现 {len(devices)} 个设备，但需要 {num_streams} 个")
            num_streams = len(devices)

        if num_streams == 0:
            print("❌ 没有可用的设备")
            return False

        print(f"\n开始串行初始化 {num_streams} 个相机流（深度 + 颜色）...")

        for i in range(num_streams):
            print(f"\n--- 初始化相机流 {i} ---")

            # 1. 打开设备
            device_handle = ctypes.c_void_p()
            uri = devices[i]['uri'].encode('utf-8') if i < len(devices) else None

            status = self.lib.oniDeviceOpen(uri, ctypes.byref(device_handle))
            if status != 0:
                print(f"❌ 打开设备 {i} 失败，状态码: {status}")
                continue

            # 2. 创建深度流
            depth_stream_handle = ctypes.c_void_p()
            status = self.lib.oniDeviceCreateStream(device_handle, 3,
                                                    ctypes.byref(depth_stream_handle))  # 3 = ONI_SENSOR_DEPTH
            if status != 0:
                print(f"❌ 创建深度流 {i} 失败，状态码: {status}")
                self.lib.oniDeviceClose(device_handle)
                continue

            # 3. 启动深度流
            status = self.lib.oniStreamStart(depth_stream_handle)
            if status != 0:
                print(f"❌ 启动深度流 {i} 失败，状态码: {status}")
                self.lib.oniStreamDestroy(depth_stream_handle)
                self.lib.oniDeviceClose(device_handle)
                continue

            # 4. 创建颜色流（如果启用）
            color_stream_handle = None
            if enable_color:
                try:
                    color_stream_handle = ctypes.c_void_p()
                    status = self.lib.oniDeviceCreateStream(device_handle, 2,
                                                            ctypes.byref(color_stream_handle))  # 2 = ONI_SENSOR_COLOR
                    if status != 0:
                        print(f"⚠️  创建颜色流 {i} 失败，状态码: {status}")
                        color_stream_handle = None
                    else:
                        # 启动颜色流
                        status = self.lib.oniStreamStart(color_stream_handle)
                        if status != 0:
                            print(f"⚠️  启动颜色流 {i} 失败，状态码: {status}")
                            self.lib.oniStreamDestroy(color_stream_handle)
                            color_stream_handle = None
                        else:
                            print(f"✅ 颜色流 {i} 启动成功")
                except Exception as e:
                    print(f"⚠️  初始化颜色流 {i} 时出错: {e}")
                    color_stream_handle = None

            # 5. 保存流信息
            stream = CameraStream(
                index=i,
                device_handle=device_handle,
                stream_handle=depth_stream_handle,
                position=['left', 'center', 'right'][i] if i < 3 else f"pos_{i}"
            )

            # 🔧 新增：保存颜色流句柄
            stream.color_stream_handle = color_stream_handle

            self.streams.append(stream)
            self.color_streams.append(color_stream_handle)

            print(f"✅ 相机流 {i} 初始化完成 - 深度: 有, 颜色: {'有' if color_stream_handle else '无'}")

            # 6. 关键：流间延迟，让硬件稳定
            if i < num_streams - 1:
                delay = 1.0  # 1秒延迟
                print(f"⏳ 等待 {delay} 秒初始化下一个流...")
                time.sleep(delay)

        return len(self.streams) > 0

    def capture_frame_with_color(self, stream_idx: int, timeout_ms: int = 2000) -> Optional[
        Tuple[np.ndarray, np.ndarray, Dict]]:
        """从指定流捕获深度和颜色帧"""
        if stream_idx >= len(self.streams):
            return None

        stream = self.streams[stream_idx]
        if not stream.is_active:
            return None

        print(f"\n[流{stream_idx}] 开始采集深度和颜色，超时={timeout_ms}ms")
        start_time = time.time()

        # 创建帧指针
        depth_frame_ptr = ctypes.POINTER(OniFrame)()
        color_frame_ptr = ctypes.POINTER(OniFrame)()

        try:
            # 采集深度帧
            status = self.lib.oniStreamReadFrame(stream.stream_handle, ctypes.byref(depth_frame_ptr), timeout_ms)
            if status != 0:
                print(f"[流{stream_idx}] 深度帧读取失败，错误码: {status}")
                return None

            depth_frame = depth_frame_ptr.contents
            depth_data = self._process_frame_data(depth_frame, depth_frame_ptr, is_depth=True)

            # 采集颜色帧（如果可用）
            color_data = None
            if hasattr(stream, 'color_stream_handle') and stream.color_stream_handle:
                status = self.lib.oniStreamReadFrame(stream.color_stream_handle, ctypes.byref(color_frame_ptr),
                                                     timeout_ms)
                if status == 0:
                    color_frame = color_frame_ptr.contents
                    color_data = self._process_frame_data(color_frame, color_frame_ptr, is_depth=False)
                else:
                    print(f"[流{stream_idx}] 颜色帧读取失败，错误码: {status}")

            elapsed = time.time() - start_time

            # 更新流状态
            stream.last_frame_time = time.time()
            stream.frame_count += 1

            print(f"[流{stream_idx}] 采集成功，耗时={elapsed:.2f}秒")
            return depth_data, color_data

        except Exception as e:
            print(f"[流{stream_idx}] 捕获异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_frame_data(self, frame: OniFrame, frame_ptr, is_depth: bool = True) -> np.ndarray:
        """处理帧数据并转换为numpy数组"""
        # 转换数据
        data_bytes = ctypes.string_at(frame.data, frame.dataSize)

        if is_depth:
            # 深度数据 (uint16)
            if frame.videoMode.pixelFormat == 100:  # ONI_PIXEL_FORMAT_DEPTH_1_MM
                array = np.frombuffer(data_bytes, dtype=np.uint16).reshape(frame.height, frame.width)
            else:
                raise ValueError(f"不支持的深度像素格式: {frame.videoMode.pixelFormat}")
        else:
            # 颜色数据
            if frame.videoMode.pixelFormat == 200:  # ONI_PIXEL_FORMAT_RGB888
                # RGB888格式，需要转换为BGR
                array = np.frombuffer(data_bytes, dtype=np.uint8).reshape(frame.height, frame.width, 3)
                # RGB -> BGR
                array = array[:, :, ::-1]
            elif frame.videoMode.pixelFormat == 202:  # ONI_PIXEL_FORMAT_GRAY8
                # 灰度图，转换为BGR
                gray = np.frombuffer(data_bytes, dtype=np.uint8).reshape(frame.height, frame.width)
                array = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"不支持的彩色像素格式: {frame.videoMode.pixelFormat}")

        # 增加引用计数（释放时使用）
        self.lib.oniFrameAddRef(frame_ptr)

        return array

    def capture_frame(self, stream_idx: int, timeout_ms: int = 2000) -> Optional[Tuple[np.ndarray, Dict]]:
        """从指定流捕获一帧（带详细调试）"""
        if stream_idx >= len(self.streams):
            return None

        stream = self.streams[stream_idx]
        if not stream.is_active:
            return None

        print(f"\n[流{stream_idx}] 开始采集，超时={timeout_ms}ms")
        start_time = time.time()

        # 创建帧指针
        frame_ptr = ctypes.POINTER(OniFrame)()

        try:
            # 关键：调用带超时的 oniStreamReadFrame
            print(f"[流{stream_idx}] 调用 oniStreamReadFrame...")
            status = self.lib.oniStreamReadFrame(stream.stream_handle, ctypes.byref(frame_ptr), timeout_ms)
            elapsed = time.time() - start_time

            print(f"[流{stream_idx}] oniStreamReadFrame 返回，状态码={status}，耗时={elapsed:.2f}秒")

            if status != 0:  # 0 = ONI_STATUS_OK
                if status == 7:  # ONI_STATUS_TIME_OUT
                    print(f"[流{stream_idx}] 读取超时")
                else:
                    print(f"[流{stream_idx}] 读取失败，错误码: {status}")
                return None

            # 处理帧数据
            frame = frame_ptr.contents
            frame_data = self._process_frame(frame, frame_ptr)

            # 更新流状态
            stream.last_frame_time = time.time()
            stream.frame_count += 1

            print(f"[流{stream_idx}] 采集成功，帧索引={frame.frameIndex}，尺寸={frame.width}x{frame.height}")
            return frame_data

        except Exception as e:
            print(f"[流{stream_idx}] 捕获异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_frame(self, frame: OniFrame, frame_ptr) -> Tuple[np.ndarray, Dict]:
        """处理帧数据"""
        # 转换深度数据
        if frame.videoMode.pixelFormat == 100:  # ONI_PIXEL_FORMAT_DEPTH_1_MM
            depth_data = ctypes.string_at(frame.data, frame.dataSize)
            depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(frame.height, frame.width)

            frame_info = {
                'width': frame.width,
                'height': frame.height,
                'timestamp': frame.timestamp,
                'frame_index': frame.frameIndex,
                'pixel_format': frame.videoMode.pixelFormat,
                'frame_ptr': frame_ptr,
            }

            # 增加引用计数
            self.lib.oniFrameAddRef(frame_ptr)

            return depth_array, frame_info

        print(f"⚠️ 不支持的像素格式: {frame.videoMode.pixelFormat}")
        return None, {}

    def release_frame(self, frame_info: Dict):
        """释放帧资源"""
        if frame_info and 'frame_ptr' in frame_info:
            self.lib.oniFrameRelease(frame_info['frame_ptr'])

    def capture_all_streams(self) -> List[Optional[np.ndarray]]:
        """从所有流捕获一帧（串行采集，避免冲突）"""
        frames = [None] * len(self.streams)

        for i, stream in enumerate(self.streams):
            if not stream.is_active:
                continue

            result = self.capture_frame(i, timeout_ms=3000)
            if result:
                depth_array, frame_info = result
                frames[i] = depth_array
                self.release_frame(frame_info)

            # 流间延迟，避免 USB 总线拥堵
            if i < len(self.streams) - 1:
                time.sleep(0.1)  # 100ms 延迟

        return frames

    def get_stream_positions(self) -> List[str]:
        """获取所有流的位置信息"""
        return [stream.position for stream in self.streams]

    def get_active_stream_count(self) -> int:
        """获取活动流数量"""
        return sum(1 for stream in self.streams if stream.is_active)

    def deactivate_stream(self, stream_idx: int):
        """停用指定流（临时故障处理）"""
        if stream_idx < len(self.streams):
            self.streams[stream_idx].is_active = False
            print(f"已停用流 {stream_idx}")

    def reactivate_stream(self, stream_idx: int):
        """重新激活指定流"""
        if stream_idx < len(self.streams):
            self.streams[stream_idx].is_active = True
            print(f"已重新激活流 {stream_idx}")

    def cleanup(self):
        """清理所有资源"""
        print("\n清理多流管理器资源...")

        # 停止和销毁所有流
        for i, stream in enumerate(self.streams):
            print(f"  清理流 {i}...")
            try:
                if stream.stream_handle:
                    self.lib.oniStreamStop(stream.stream_handle)
                    self.lib.oniStreamDestroy(stream.stream_handle)
                if stream.device_handle:
                    self.lib.oniDeviceClose(stream.device_handle)
                print(f"    流 {i} 已清理")
            except Exception as e:
                print(f"    流 {i} 清理异常: {e}")

        self.streams.clear()

        # 关闭 OpenNI 上下文
        if self.context_initialized:
            self.lib.oniShutdown()
            self.context_initialized = False
            print("✅ OpenNI 上下文已关闭")

        print("✅ 所有资源已清理")


def test_multi_stream_manager():
    """测试多流管理器"""
    print("=" * 60)
    print("测试多流相机管理器")
    print("=" * 60)

    # 创建管理器实例
    sdk_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    manager = MultiStreamCameraManager(sdk_path)

    if not manager.context_initialized:
        print("❌ 管理器初始化失败")
        return False

    # 设置3个流
    if not manager.setup_streams(3):
        print("❌ 设置流失败")
        manager.cleanup()
        return False

    print(f"\n✅ 成功设置 {len(manager.streams)} 个流")
    print(f"流位置: {manager.get_stream_positions()}")

    # 测试采集3帧
    for frame_num in range(3):
        print(f"\n📸 测试第 {frame_num+1} 帧采集:")

        frames = manager.capture_all_streams()

        for i, frame in enumerate(frames):
            if frame is not None:
                valid_points = np.sum(frame > 0)
                print(f"  流{i}: ✅ 采集成功，尺寸={frame.shape}，有效点={valid_points}")
            else:
                print(f"  流{i}: ❌ 采集失败")

        time.sleep(0.5)  # 帧间延迟

    # 清理
    manager.cleanup()
    print("\n🎉 多流管理器测试完成")
    return True


if __name__ == "__main__":
    test_multi_stream_manager()