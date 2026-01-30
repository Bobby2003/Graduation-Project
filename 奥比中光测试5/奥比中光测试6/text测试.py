"""
奥比中光Astra相机 - 彩色3D重建系统
基于OpenNI C API的修复版本，解决0xC0000374堆损坏问题
"""

import cv2
import numpy as np
import open3d as o3d
import json
import os
import gc
import traceback
import time
import ctypes
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# OpenNI C API 封装类 (直接嵌入，避免导入问题)
# ============================================================

class FixedOpenNICAPI:
    """修复结构体定义后的OpenNI C API封装"""

    def __init__(self, driver_path=None):
        self.driver_path = driver_path
        self.lib = None
        self.device_handle = None
        self.depth_stream_handle = None
        self.color_stream_handle = None

        # 常量定义
        self.ONI_MAX_STR = 256
        self.ONI_STATUS_OK = 0
        self.ONI_SENSOR_IR = 1
        self.ONI_SENSOR_COLOR = 2
        self.ONI_SENSOR_DEPTH = 3
        self.ONI_PIXEL_FORMAT_DEPTH_1_MM = 100
        self.ONI_PIXEL_FORMAT_RGB888 = 200
        self.ONI_API_VERSION = 2000

        # 定义结构体
        self._define_structures()

        # 初始化
        if driver_path:
            self._add_driver_to_path()

        self._load_openni_library()

        if self.lib:
            self._define_function_prototypes()

    def _define_structures(self):
        """定义OpenNI结构体"""

        # OniDeviceInfo结构体 (关键：使用字符数组而不是指针)
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

    def _add_driver_to_path(self):
        """将Driver目录添加到系统PATH"""
        if self.driver_path:
            drivers_dir = os.path.join(self.driver_path, "libs", "OpenNI2", "Drivers")
            if os.path.exists(drivers_dir):
                os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']
                print(f"✅ 已将Drivers目录添加到PATH: {drivers_dir}")

    def _load_openni_library(self):
        """加载OpenNI2.dll"""
        dll_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs\OpenNI2.dll"

        if not os.path.exists(dll_path):
            print(f"❌ DLL文件不存在: {dll_path}")
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
            # 通用API
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

            # 设备API
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

            # 流API
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

            # 帧API
            self.lib.oniFrameAddRef.argtypes = [ctypes.POINTER(self.OniFrame)]
            self.lib.oniFrameAddRef.restype = None

            self.lib.oniFrameRelease.argtypes = [ctypes.POINTER(self.OniFrame)]
            self.lib.oniFrameRelease.restype = None

            print("✅ OpenNI C API函数原型定义完成")

        except AttributeError as e:
            print(f"⚠️ 定义函数原型时出错: {e}")

    def initialize(self):
        """初始化OpenNI"""
        if not self.lib:
            return False

        print("初始化OpenNI...")
        status = self.lib.oniInitialize(self.ONI_API_VERSION)
        if status != self.ONI_STATUS_OK:
            print(f"❌ 初始化失败，错误码: {status}")
            return False

        print("✅ OpenNI初始化成功")
        return True

    def get_device_list(self):
        """获取设备列表"""
        if not self.lib:
            return False

        devices_ptr = ctypes.POINTER(self.OniDeviceInfo)()
        device_count = ctypes.c_int(0)

        status = self.lib.oniGetDeviceList(ctypes.byref(devices_ptr), ctypes.byref(device_count))
        if status != self.ONI_STATUS_OK:
            print(f"❌ 获取设备列表失败，错误码: {status}")
            return False

        print(f"发现 {device_count.value} 台设备:")
        for i in range(device_count.value):
            device_info = devices_ptr[i]
            name = device_info.name.decode('utf-8', errors='ignore').rstrip('\x00')
            vendor = device_info.vendor.decode('utf-8', errors='ignore').rstrip('\x00')
            print(f"  [{i}] {name} ({vendor})")

        # 释放设备列表
        self.lib.oniReleaseDeviceList(devices_ptr)

        return True

    def open_device(self, device_index=0):
        """打开设备"""
        if not self.lib:
            return False

        # 获取设备列表
        devices_ptr = ctypes.POINTER(self.OniDeviceInfo)()
        device_count = ctypes.c_int(0)

        status = self.lib.oniGetDeviceList(ctypes.byref(devices_ptr), ctypes.byref(device_count))
        if status != self.ONI_STATUS_OK or device_count.value == 0:
            print("❌ 没有可用设备，尝试打开默认设备...")
            uri = None
        else:
            if device_index >= device_count.value:
                print(f"❌ 设备索引 {device_index} 超出范围")
                return False
            device_info = devices_ptr[device_index]
            uri = device_info.uri
            name = device_info.name.decode('utf-8', errors='ignore').rstrip('\x00')
            print(f"打开设备: {name}")

        # 打开设备
        device_handle = ctypes.c_void_p()
        if uri:
            status = self.lib.oniDeviceOpen(uri, ctypes.byref(device_handle))
        else:
            status = self.lib.oniDeviceOpen(None, ctypes.byref(device_handle))

        if status != self.ONI_STATUS_OK:
            print(f"❌ 打开设备失败，错误码: {status}")
            if devices_ptr:
                self.lib.oniReleaseDeviceList(devices_ptr)
            return False

        self.device_handle = device_handle
        print(f"✅ 设备打开成功")

        # 释放设备列表
        if devices_ptr:
            self.lib.oniReleaseDeviceList(devices_ptr)

        return True

    def create_stream(self, sensor_type=3):
        """创建流 (3=深度, 2=彩色)"""
        if not self.lib or not self.device_handle:
            print("❌ 设备未打开")
            return False

        sensor_names = {1: "红外", 2: "彩色", 3: "深度"}
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

        if sensor_type == self.ONI_SENSOR_DEPTH:
            self.depth_stream_handle = stream_handle
        elif sensor_type == self.ONI_SENSOR_COLOR:
            self.color_stream_handle = stream_handle

        print(f"✅ 流创建成功")
        return True

    def start_stream(self, stream_handle=None):
        """启动流"""
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if not self.lib or not stream_handle:
            print("❌ 流句柄无效")
            return False

        status = self.lib.oniStreamStart(stream_handle)
        if status != self.ONI_STATUS_OK:
            print(f"❌ 启动流失败，错误码: {status}")
            return False

        print("✅ 流启动成功")
        return True

    def read_depth_frame(self, timeout=1000):
        """读取深度帧"""
        if not self.lib or not self.depth_stream_handle:
            print("❌ 深度流未就绪")
            return None

        frame_ptr = ctypes.POINTER(self.OniFrame)()
        status = self.lib.oniStreamReadFrame(self.depth_stream_handle, ctypes.byref(frame_ptr))

        if status != self.ONI_STATUS_OK:
            if status == 7:  # ONI_STATUS_TIME_OUT
                print("⚠️  读取帧超时")
            else:
                print(f"❌ 读取帧失败，错误码: {status}")
            return None

        frame = frame_ptr.contents

        # 检查像素格式
        if frame.videoMode.pixelFormat != self.ONI_PIXEL_FORMAT_DEPTH_1_MM:
            print(f"⚠️  不支持的像素格式: {frame.videoMode.pixelFormat}")
            self.lib.oniFrameRelease(frame_ptr)
            return None

        # 获取深度数据 (单位: mm)
        depth_data = ctypes.string_at(frame.data, frame.dataSize)
        depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(frame.height, frame.width)

        # 增加引用计数
        self.lib.oniFrameAddRef(frame_ptr)

        frame_info = {
            'data': depth_array,
            'width': frame.width,
            'height': frame.height,
            'timestamp': frame.timestamp,
            'frame_index': frame.frameIndex,
            'frame_ptr': frame_ptr
        }

        return frame_info

    def release_frame(self, frame_info):
        """释放帧资源"""
        if frame_info and 'frame_ptr' in frame_info:
            self.lib.oniFrameRelease(frame_info['frame_ptr'])

    def stop_stream(self, stream_handle=None):
        """停止流"""
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if self.lib and stream_handle:
            self.lib.oniStreamStop(stream_handle)
            print("✅ 流已停止")

    def destroy_stream(self, stream_handle=None):
        """销毁流"""
        if stream_handle is None:
            stream_handle = self.depth_stream_handle

        if self.lib and stream_handle:
            self.lib.oniStreamDestroy(stream_handle)
            if stream_handle == self.depth_stream_handle:
                self.depth_stream_handle = None
            elif stream_handle == self.color_stream_handle:
                self.color_stream_handle = None
            print("✅ 流已销毁")

    def close_device(self):
        """关闭设备"""
        if self.lib and self.device_handle:
            self.lib.oniDeviceClose(self.device_handle)
            self.device_handle = None
            print("✅ 设备已关闭")

    def shutdown(self):
        """关闭OpenNI"""
        if self.lib:
            self.lib.oniShutdown()
            print("✅ OpenNI已关闭")

    def cleanup(self):
        """完整清理所有资源"""
        print("\n执行OpenNI资源清理...")

        # 停止并销毁深度流
        if self.depth_stream_handle:
            self.stop_stream(self.depth_stream_handle)
            self.destroy_stream(self.depth_stream_handle)

        # 停止并销毁彩色流
        if self.color_stream_handle:
            self.stop_stream(self.color_stream_handle)
            self.destroy_stream(self.color_stream_handle)

        # 关闭设备
        self.close_device()

        # 关闭OpenNI
        self.shutdown()

        print("✅ 所有OpenNI资源已清理")


# ============================================================
# 彩色3D重建器 (使用FixedOpenNICAPI)
# ============================================================

class Color3DReconstructor:
    """彩色3D重建器 - 基于OpenNI C API的修复版本"""

    def __init__(self, driver_path=""):
        self.driver_path = driver_path

        # 使用修复后的OpenNI C API
        self.openni = FixedOpenNICAPI(driver_path)
        self.camera_initialized = False
        self.frame_counter = 0

        # OpenCV RGB摄像头
        self.cv_camera = None
        self.cv_camera_initialized = False

        # TSDF体积
        self.tsdf_volume = None

        # 相机参数
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525.0, 525.0, 319.5, 239.5
        )

        # 优化的彩色重建参数
        self.params = {
            # 深度参数
            'depth_scale': 0.001,
            'depth_clip_min': 0.6,  # 30cm
            'depth_clip_max': 8.0,  # 2米，更好的彩色效果

            # TSDF参数 - 优化彩色重建
            'tsdf_voxel_length': 0.008,  # 更小的体素，更精细的颜色
            'tsdf_sdf_trunc': 0.04,

            # 重建参数
            'max_iterations': 10,  # 减少迭代次数，提高稳定性
            'frame_interval': 0.5,  # 增加间隔，减少负载

            # 摄像头参数
            'camera_index': 0,
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 15,

            # 颜色增强参数
            'color_enhance': True,
            'color_contrast': 1.2,
            'color_brightness': 10,

            # 网格优化
            'mesh_simplify': True,
            'target_vertices': 20000,
            'smooth_mesh': True,
            'smooth_iterations': 1,

            # 导出设置 - 确保所有格式都有颜色
            'export_ply': True,
            'export_obj': True,
            'export_stl': True,
            'export_glb': False,  # GLB格式支持颜色和纹理

            # 保存设置
            'save_every_frame': True,
            'save_intermediate': True,
        }

        self.output_dir = None
        self.iteration = 0
        self.mesh_count = 0

        print("🎨 彩色3D重建器初始化完成 (使用OpenNI C API)")
        print("   专为彩色模型导出优化")

    def setup_camera(self):
        """设置相机 - 使用OpenNI C API"""
        print("=" * 50)
        print("设置相机 (使用OpenNI C API)")
        print("=" * 50)

        try:
            # 1. 初始化OpenNI
            if not self.openni.initialize():
                print("❌ OpenNI初始化失败")
                return False

            # 2. 获取设备列表
            self.openni.get_device_list()

            # 3. 打开设备 (使用第一个设备)
            if not self.openni.open_device(device_index=0):
                print("❌ 打开设备失败")
                return False

            # 4. 创建深度流
            if not self.openni.create_stream(self.openni.ONI_SENSOR_DEPTH):
                print("❌ 创建深度流失败")
                self.openni.cleanup()
                return False

            # 5. 启动深度流
            if not self.openni.start_stream():
                print("❌ 启动深度流失败")
                self.openni.destroy_stream()
                self.openni.close_device()
                return False

            # 6. 设置RGB摄像头 (保持不变)
            print("设置RGB摄像头...")
            self.cv_camera = cv2.VideoCapture(self.params['camera_index'])

            if not self.cv_camera.isOpened():
                print("尝试备用摄像头索引...")
                for idx in [1, 2, 3]:
                    self.cv_camera = cv2.VideoCapture(idx)
                    if self.cv_camera.isOpened():
                        print(f"✅ 使用摄像头索引 {idx}")
                        break

            if self.cv_camera.isOpened():
                self.cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.params['camera_width'])
                self.cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params['camera_height'])
                self.cv_camera.set(cv2.CAP_PROP_FPS, self.params['camera_fps'])

                # 测试读取
                ret, frame = self.cv_camera.read()
                if ret and frame is not None:
                    self.cv_camera_initialized = True
                    print(f"✅ RGB摄像头就绪 ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    print("❌ RGB摄像头读取失败")
            else:
                print("⚠️ 未找到可用的RGB摄像头")

            self.camera_initialized = True
            return True

        except Exception as e:
            print(f"❌ 相机设置失败: {e}")
            traceback.print_exc()
            return False

    def cleanup(self):
        """清理资源"""
        print("\n清理所有资源...")
        try:
            # 清理OpenNI资源
            if hasattr(self, 'openni') and self.openni:
                self.openni.cleanup()

            # 清理OpenCV摄像头
            if self.cv_camera:
                self.cv_camera.release()
                print("✅ OpenCV摄像头已释放")

            # 清理Open3D资源
            self.tsdf_volume = None

            # 强制垃圾回收
            gc.collect()

            self.camera_initialized = False
            print("✅ 所有资源已清理")

        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}")

    def capture_frames(self):
        """采集深度和颜色帧 - 使用OpenNI C API"""
        try:
            # 采集深度帧
            frame_info = self.openni.read_depth_frame(timeout=1000)
            if not frame_info:
                print("⚠️  深度帧读取失败")
                return None, None

            # 深度数据单位转换: mm -> m
            depth_mm = frame_info['data'].astype(np.float32)
            depth = depth_mm * 0.001  # 转换为米

            # ⭐ 关键：必须释放帧资源！
            self.openni.release_frame(frame_info)

            # 采集颜色帧
            color = None
            if self.cv_camera_initialized:
                ret, frame = self.cv_camera.read()
                if ret and frame is not None:
                    # 调整大小以匹配深度
                    if frame.shape[:2] != (480, 640):
                        frame = cv2.resize(frame, (640, 480))
                    color = frame.copy()

                    # 颜色增强
                    if self.params['color_enhance']:
                        color = self.enhance_color(color)

            self.frame_counter += 1
            return depth, color

        except Exception as e:
            print(f"❌ 帧采集失败: {e}")
            return None, None

    def enhance_color(self, image):
        """增强颜色"""
        try:
            # 转换为浮点数进行计算
            img_float = image.astype(np.float32) / 255.0

            # 对比度调整
            img_float = np.clip((img_float - 0.5) * self.params['color_contrast'] + 0.5, 0, 1)

            # 亮度调整
            img_float = np.clip(img_float + self.params['color_brightness'] / 255.0, 0, 1)

            # 转回8位
            enhanced = (img_float * 255).astype(np.uint8)
            return enhanced

        except:
            return image

    def process_depth(self, depth):
        """处理深度图"""
        if depth is None:
            return None

        # 范围裁剪
        mask = (depth >= self.params['depth_clip_min']) & (depth <= self.params['depth_clip_max'])
        depth_processed = depth.copy()
        depth_processed[~mask] = 0

        # 简单滤波
        if np.sum(mask) > 100:
            try:
                depth_processed = cv2.medianBlur(depth_processed, 3)
            except:
                pass

        return depth_processed

    def init_tsdf(self):
        """初始化TSDF体积"""
        print("初始化TSDF体积...")

        try:
            # 使用支持颜色的TSDF
            self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.params['tsdf_voxel_length'],
                sdf_trunc=self.params['tsdf_sdf_trunc'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            print(f"✅ TSDF体积初始化完成")
            print(f"   体素大小: {self.params['tsdf_voxel_length']}米")
            return True

        except Exception as e:
            print(f"❌ TSDF初始化失败: {e}")
            return False

    def fuse_frame(self, depth, color):
        """融合一帧"""
        if depth is None:
            return False

        try:
            # 创建深度图像 (单位: mm)
            depth_mm = (depth * 1000).astype(np.uint16)
            depth_image = o3d.geometry.Image(depth_mm)

            # 创建颜色图像
            if color is not None:
                # BGR转RGB
                color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                color_image = o3d.geometry.Image(color_rgb)
            else:
                # 使用默认颜色
                default_color = np.full((480, 640, 3), 200, dtype=np.uint8)
                color_image = o3d.geometry.Image(default_color)

            # 创建RGBD图像
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1000.0,
                depth_trunc=self.params['depth_clip_max'],
                convert_rgb_to_intensity=False
            )

            # 融合到TSDF
            extrinsic = np.eye(4)
            self.tsdf_volume.integrate(rgbd_image, self.intrinsic, extrinsic)

            return True

        except Exception as e:
            print(f"❌ 融合失败: {e}")
            return False

    def extract_color_mesh(self):
        """提取带颜色的网格"""
        if self.tsdf_volume is None:
            return None

        try:
            # 提取原始网格
            mesh = self.tsdf_volume.extract_triangle_mesh()

            if mesh is None or len(mesh.vertices) == 0:
                return None

            # 镜像修正（Astra相机）
            vertices = np.asarray(mesh.vertices)
            vertices[:, 0] = -vertices[:, 0]  # 左右镜像
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            # 确保有顶点颜色
            if not mesh.has_vertex_colors():
                print("⚠️ 网格没有颜色，添加默认颜色")
                mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色

            # 计算法线
            mesh.compute_vertex_normals()

            # 网格优化
            if self.params['mesh_simplify'] and len(mesh.vertices) > self.params['target_vertices']:
                target_triangles = int(len(mesh.triangles) * 0.5)
                mesh = mesh.simplify_quadric_decimation(target_triangles)
                mesh.compute_vertex_normals()

            # 网格平滑
            if self.params['smooth_mesh']:
                mesh = mesh.filter_smooth_simple(number_of_iterations=self.params['smooth_iterations'])
                mesh.compute_vertex_normals()

            return mesh

        except Exception as e:
            print(f"❌ 网格提取失败: {e}")
            return None

    def save_color_mesh(self, mesh, iteration, is_final=False):
        """保存彩色网格"""
        if mesh is None or len(mesh.vertices) == 0:
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if is_final:
                save_dir = os.path.join(self.output_dir, "final_models")
            else:
                save_dir = os.path.join(self.output_dir, "intermediate_models")

            os.makedirs(save_dir, exist_ok=True)

            saved_files = []

            # 保存PLY格式 - 最佳彩色支持
            if self.params['export_ply']:
                try:
                    ply_filename = f"color_model_f{iteration:03d}_{timestamp}.ply" if not is_final else f"final_color_model_{timestamp}.ply"
                    ply_path = os.path.join(save_dir, ply_filename)

                    # 使用二进制格式保存，支持顶点颜色
                    o3d.io.write_triangle_mesh(
                        ply_path,
                        mesh,
                        write_ascii=False,
                        compressed=True
                    )
                    saved_files.append(('PLY', ply_path))
                    print(f"   ✅ PLY格式已保存: {os.path.basename(ply_path)}")
                except Exception as e:
                    print(f"   ❌ PLY保存失败: {e}")

            # 保存OBJ格式 - 带顶点颜色
            if self.params['export_obj']:
                try:
                    obj_filename = f"color_model_f{iteration:03d}_{timestamp}.obj" if not is_final else f"final_color_model_{timestamp}.obj"
                    obj_path = os.path.join(save_dir, obj_filename)

                    # OBJ格式保存顶点颜色
                    o3d.io.write_triangle_mesh(
                        obj_path,
                        mesh,
                        write_vertex_normals=True,
                        write_vertex_colors=True
                    )
                    saved_files.append(('OBJ', obj_path))
                    print(f"   ✅ OBJ格式已保存: {os.path.basename(obj_path)}")
                except Exception as e:
                    print(f"   ❌ OBJ保存失败: {e}")

            # 保存STL格式 - 不带颜色
            if self.params['export_stl']:
                try:
                    stl_filename = f"model_f{iteration:03d}_{timestamp}.stl" if not is_final else f"final_model_{timestamp}.stl"
                    stl_path = os.path.join(save_dir, stl_filename)

                    # STL不支持颜色，保存几何
                    o3d.io.write_triangle_mesh(stl_path, mesh)
                    saved_files.append(('STL', stl_path))
                    print(f"   ✅ STL格式已保存: {os.path.basename(stl_path)}")
                except Exception as e:
                    print(f"   ❌ STL保存失败: {e}")

            # 保存GLB格式 - 更好的颜色支持
            if self.params['export_glb']:
                try:
                    import trimesh
                    glb_filename = f"color_model_f{iteration:03d}_{timestamp}.glb" if not is_final else f"final_color_model_{timestamp}.glb"
                    glb_path = os.path.join(save_dir, glb_filename)

                    # 转换为trimesh保存GLB
                    tri_mesh = trimesh.Trimesh(
                        vertices=np.asarray(mesh.vertices),
                        faces=np.asarray(mesh.triangles),
                        vertex_colors=np.asarray(mesh.vertex_colors)
                    )
                    tri_mesh.export(glb_path)
                    saved_files.append(('GLB', glb_path))
                    print(f"   ✅ GLB格式已保存: {os.path.basename(glb_path)}")
                except ImportError:
                    print("   ⚠️ trimesh未安装，跳过GLB格式")
                except Exception as e:
                    print(f"   ❌ GLB保存失败: {e}")

            self.mesh_count += 1
            return True

        except Exception as e:
            print(f"❌ 保存网格失败: {e}")
            return False

    def save_frame_images(self, iteration, depth, color):
        """保存帧图像"""
        try:
            images_dir = os.path.join(self.output_dir, "captured_images")
            os.makedirs(images_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存深度图
            if depth is not None:
                depth_colored = self.depth_to_colormap(depth)
                depth_path = os.path.join(images_dir, f"depth_{iteration:03d}_{timestamp}.png")
                cv2.imwrite(depth_path, depth_colored)

            # 保存颜色图
            if color is not None:
                color_path = os.path.join(images_dir, f"color_{iteration:03d}_{timestamp}.png")
                cv2.imwrite(color_path, color)

            return True

        except Exception as e:
            print(f"⚠️ 保存图像失败: {e}")
            return False

    def depth_to_colormap(self, depth):
        """深度图转伪彩色"""
        if depth is None:
            return None

        try:
            depth_mm = depth * 1000
            depth_normalized = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            return depth_colored
        except:
            return None

    def run_color_reconstruction(self):
        """运行彩色重建"""
        print("=" * 70)
        print("奥比中光Astra相机 - 彩色3D重建 (OpenNI C API版本)")
        print("专门导出带颜色的3D模型")
        print("=" * 70)

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"color_3d_reconstruction_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")

        # 保存配置
        config_file = os.path.join(self.output_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)

        # 设置相机
        if not self.setup_camera():
            print("❌ 相机设置失败")
            return False

        # 初始化TSDF
        if not self.init_tsdf():
            print("❌ TSDF初始化失败")
            self.cleanup()
            return False

        print("\n" + "=" * 50)
        print("开始彩色3D重建")
        print("=" * 50)

        self.iteration = 0
        start_time = time.time()

        try:
            for i in range(self.params['max_iterations']):
                self.iteration = i + 1
                print(f"\n📸 第 {self.iteration}/{self.params['max_iterations']} 帧")

                # 等待帧间隔
                if i > 0:
                    time.sleep(self.params['frame_interval'])

                # 采集帧
                depth_raw, color_raw = self.capture_frames()
                if depth_raw is None:
                    print("❌ 深度采集失败，跳过")
                    continue

                # 处理深度
                depth = self.process_depth(depth_raw)
                if depth is None or np.sum(depth > 0) < 1000:
                    print(f"❌ 有效深度点不足: {np.sum(depth > 0)}")
                    continue

                print(f"   深度点: {np.sum(depth > 0)}个")
                print(f"   颜色: {'✓' if color_raw is not None else '✗'}")

                # 融合帧
                if self.fuse_frame(depth, color_raw):
                    print("✅ 帧融合成功")

                    # 保存中间结果
                    if self.params['save_intermediate'] and (i % 2 == 0 or i == self.params['max_iterations'] - 1):
                        mesh = self.extract_color_mesh()
                        if mesh is not None:
                            print(f"💾 保存中间模型 ({len(mesh.vertices)}顶点)...")
                            self.save_color_mesh(mesh, self.iteration, is_final=False)
                            del mesh

                    # 保存图像
                    if self.params['save_every_frame']:
                        self.save_frame_images(self.iteration, depth, color_raw)

                    # 进度显示
                    elapsed = time.time() - start_time
                    avg_time = elapsed / self.iteration
                    remaining = (self.params['max_iterations'] - self.iteration) * avg_time
                    print(f"⏱️  预计剩余: {remaining:.1f}秒")

                else:
                    print("❌ 帧融合失败")

                # 清理内存
                gc.collect()

        except KeyboardInterrupt:
            print("\n🔴 用户中断重建")
        except Exception as e:
            print(f"\n❌ 重建过程中出错: {e}")
            traceback.print_exc()

        # 最终处理
        print("\n" + "=" * 50)
        print("导出最终彩色模型")
        print("=" * 50)

        if self.tsdf_volume is not None and self.iteration > 0:
            print("🎨 提取最终彩色网格...")
            final_mesh = self.extract_color_mesh()

            if final_mesh is not None:
                print(f"   顶点数: {len(final_mesh.vertices)}")
                print(f"   面片数: {len(final_mesh.triangles)}")
                print(f"   有颜色: {final_mesh.has_vertex_colors()}")

                # 保存最终模型
                print("💾 保存最终彩色模型...")
                success = self.save_color_mesh(final_mesh, self.iteration, is_final=True)

                if success:
                    print("✅ 最终彩色模型保存成功")

                    # 显示模型信息
                    self.display_model_info(final_mesh)

                    # 可选：保存模型预览图
                    self.save_model_preview(final_mesh)

                del final_mesh
            else:
                print("❌ 无法提取最终网格")

        # 生成报告
        self.generate_report()

        # 清理资源
        self.cleanup()

        print("\n" + "=" * 70)
        print("彩色3D重建完成")
        print(f"处理帧数: {self.iteration}")
        print(f"模型数量: {self.mesh_count}")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print("=" * 70)

        return True

    def display_model_info(self, mesh):
        """显示模型信息"""
        try:
            vertices = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)

            print("\n📊 模型统计信息:")
            print(f"   顶点范围: X [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
            print(f"             Y [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
            print(f"             Z [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

            if colors.shape[0] > 0:
                print(f"   颜色范围: R [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
                print(f"             G [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
                print(f"             B [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")

            # 计算模型尺寸
            bbox = mesh.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            print(f"   模型尺寸: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} 米")

        except Exception as e:
            print(f"⚠️ 显示模型信息失败: {e}")

    def save_model_preview(self, mesh):
        """保存模型预览图"""
        try:
            preview_dir = os.path.join(self.output_dir, "previews")
            os.makedirs(preview_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)  # 不显示窗口

            # 添加网格
            vis.add_geometry(mesh)

            # 设置视图
            vis.get_render_option().mesh_show_back_face = True
            vis.get_render_option().light_on = True
            vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

            # 从不同角度保存图像
            angles = [0, 45, 90, 135, 180, 225, 270, 315]

            for i, angle in enumerate(angles):
                # 设置相机位置
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                ctr.rotate(angle * 10.0, 0.0)  # 10度步长

                # 捕获图像
                image_path = os.path.join(preview_dir, f"preview_angle_{angle:03d}_{timestamp}.png")
                vis.capture_screen_image(image_path, do_render=True)

            vis.destroy_window()
            print(f"📸 模型预览图已保存: {preview_dir}")

        except Exception as e:
            print(f"⚠️ 保存预览图失败: {e}")

    def generate_report(self):
        """生成重建报告"""
        try:
            report_path = os.path.join(self.output_dir, "彩色重建报告.txt")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("奥比中光Astra相机 - 彩色3D重建报告\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"重建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理帧数: {self.iteration}\n")
                f.write(f"输出模型: {self.mesh_count}个\n")
                f.write(f"输出目录: {self.output_dir}\n\n")

                f.write("重建参数:\n")
                for key, value in self.params.items():
                    f.write(f"  {key}: {value}\n")

                f.write("\n导出格式:\n")
                f.write(f"  PLY格式: {'✓' if self.params['export_ply'] else '✗'} (推荐，带颜色)\n")
                f.write(f"  OBJ格式: {'✓' if self.params['export_obj'] else '✗'} (带顶点颜色)\n")
                f.write(f"  STL格式: {'✓' if self.params['export_stl'] else '✗'} (仅几何)\n")
                f.write(f"  GLB格式: {'✓' if self.params['export_glb'] else '✗'} (带颜色和纹理)\n")

                f.write("\n📋 使用说明:\n")
                f.write("1. PLY文件: 在MeshLab、Blender、3D Viewer中打开查看颜色\n")
                f.write("2. OBJ文件: 在支持顶点颜色的软件中查看颜色\n")
                f.write("3. STL文件: 仅包含几何，适用于3D打印\n")
                f.write("4. 所有文件保存在 'final_models' 文件夹中\n")

            print(f"📄 重建报告已保存: {report_path}")

        except Exception as e:
            print(f"⚠️ 生成报告失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("奥比中光Astra相机彩色3D重建系统")
    print("基于OpenNI C API修复版本 - 解决0xC0000374堆损坏问题")
    print("专门导出带颜色的3D模型 - PLY/OBJ/STL格式")
    print("=" * 70)

    # 检查依赖
    try:
        print("✅ OpenCV:", cv2.__version__)
        print("✅ NumPy:", np.__version__)
        print("✅ Open3D:", o3d.__version__)
    except Exception as e:
        print(f"❌ 依赖检查失败: {e}")
        return

    # 设置驱动路径
    driver_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"

    if not os.path.exists(driver_path):
        print(f"⚠️ 驱动路径不存在，尝试自动查找...")
        driver_path = ""

    # 创建重建器
    reconstructor = Color3DReconstructor(driver_path)

    # 运行重建
    print("\n🚀 开始彩色3D重建...")
    print("🎯 目标: 导出带颜色的3D模型")
    print("=" * 70)

    try:
        success = reconstructor.run_color_reconstruction()

        if success:
            print("\n🎉 彩色3D重建成功！")
            print("💡 所有带颜色的模型已保存在输出目录中")

            # 显示重要文件
            print("\n📁 重要文件:")
            print("  final_models/       - 最终彩色模型")
            print("  intermediate_models/ - 中间模型")
            print("  captured_images/    - 采集的图像")
            print("  previews/          - 模型预览图")
            print("  彩色重建报告.txt   - 详细报告")

        else:
            print("\n⚠️ 重建过程中出现问题")
            print("💡 部分数据已保存，请检查输出目录")

    except Exception as e:
        print(f"\n❌ 运行重建时出错: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("📋 提示:")
    print("1. PLY格式在MeshLab中打开可查看颜色")
    print("2. 确保重建对象光照充足")
    print("3. 相机与对象距离建议0.3-2米")
    print("4. 此版本使用OpenNI C API，已解决堆损坏问题")
    print("=" * 70)

    input("\n按Enter键退出...")


if __name__ == "__main__":
    # 设置环境变量
    os.environ['PATH'] = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs\OpenNI2\Drivers" + ';' + \
                         os.environ['PATH']

    main()