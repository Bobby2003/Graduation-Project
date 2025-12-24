"""
基于Python视觉里程计的实时3D扫描系统
作者：Bobby2003
日期：2024
专为Windows系统优化，无需C++编译
"""

import cv2
import numpy as np
import open3d as o3d
import threading
import queue
import time
import copy
import os
import json
import pickle
import warnings
import sys
from datetime import datetime
from collections import deque
from primesense import openni2
from primesense import _openni2 as c_api

# 导入自定义模块
from visual_odometry import VisualOdometry
from pointcloud_utils import PointCloudProcessor
from dataset_exporter import DatasetExporter

# 忽略特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ==================== 1. 增强型深度流管理器 ====================
# ==================== 1. 增强型深度流管理器 ====================
class EnhancedDepthStreamManager:
    def __init__(self, use_infrared=True, device_index=0):
        self.device = None
        self.depth_stream = None
        self.ir_stream = None  # 改为红外流
        self.running = False
        self.use_infrared = use_infrared  # 改为红外

        # 改进的队列管理
        self.depth_queue = queue.Queue(maxsize=30)
        self.ir_queue = queue.Queue(maxsize=30) if use_infrared else None  # 改为红外队列
        self.frame_count = 0

        # 相机参数（默认值，可从配置文件加载）
        self.fx = 475.0
        self.fy = 475.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 0.001
        self.depth_width = 640
        self.depth_height = 480

        # 同步参数
        self.sync_tolerance = 0.05  # 50ms
        self.last_sync_time = time.time()

        # 统计信息
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_fps': 0,
            'start_time': 0
        }

    def initialize(self, config_file=None):
        """初始化深度流，支持从配置文件加载参数"""
        try:
            print("🚀 初始化奥比中光深度相机...")

            # 尝试加载配置文件
            #if config_file and os.path.exists(config_file):
            #    self._load_config(config_file)

            # 初始化OpenNI2
            try:
                openni2.initialize("")
                print("✅ OpenNI2初始化成功")
            except Exception as e:
                print(f"⚠️ OpenNI2初始化警告: {e}")
                # 尝试重新初始化
                try:
                    openni2.unload()
                    openni2.initialize()
                except:
                    print("❌ 无法初始化OpenNI2")
                    return False

            # 打开设备
            try:
                devices = []
                try:
                    # 尝试枚举设备
                    device_list = openni2.Device.enumerate_uris()
                    devices = [uri.decode('utf-8') for uri in device_list]
                except:
                    pass

                if devices:
                    print(f"发现 {len(devices)} 个设备")
                    for i, dev in enumerate(devices):
                        print(f"  [{i}] {dev}")

                    if device_index < len(devices):
                        self.device = openni2.Device.open(devices[device_index])
                    else:
                        self.device = openni2.Device.open_any()
                else:
                    self.device = openni2.Device.open_any()

            except Exception as e:
                print(f"❌ 无法打开设备: {e}")
                # 可能是权限问题，尝试以不同方式打开
                try:
                    self.device = openni2.Device.open_any()
                except Exception as e2:
                    print(f"❌ 备用打开方式也失败: {e2}")
                    return False

            # 获取设备信息
            try:
                dev_info = self.device.get_device_info()
                device_name = dev_info.name.decode('utf-8', errors='ignore')
                print(f"✅ 设备已连接: {device_name}")

                # 检测是否为Astra相机
                if "astra" in device_name.lower() or "Astra" in device_name:
                    print("✅ 检测到Astra相机，启用深度+红外模式")
                    # Astra相机通常无法同时开启深度和彩色流
                    self.use_infrared = True  # 强制使用红外
            except:
                print("✅ 设备已连接（获取信息失败）")

            # 打印相机参数
            print(f"📷 相机参数: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

            return True

        except Exception as e:
            print(f"❌ 深度流初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_stream(self):
        """启动深度和红外流"""
        try:
            print("🔄 启动深度流...")

            # 创建深度流
            self.depth_stream = self.device.create_depth_stream()

            # 配置深度流
            depth_mode = c_api.OniVideoMode(
                pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=self.depth_width,
                resolutionY=self.depth_height,
                fps=30
            )
            self.depth_stream.set_video_mode(depth_mode)
            self.depth_stream.start()

            # 启动红外流（如果启用）
            if self.use_infrared:
                print("🔄 启动红外流...")
                try:
                    self.ir_stream = self.device.create_ir_stream()
                    self.ir_stream.start()
                    print("✅ 红外流启动成功")

                    # 注意：Astra相机无法同时开启深度和彩色流，但可以开启深度和红外流
                    # 所以我们不再尝试启用同步

                except Exception as e:
                    print(f"⚠️ 启动红外流失败: {e}")
                    self.use_infrared = False

            self.running = True
            self.stats['start_time'] = time.time()

            # 启动采集线程
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            print(f"[采集管理器] 采集线程已启动")

            return True

        except Exception as e:
            print(f"❌ 启动流失败: {e}")
            return False

    def _capture_loop(self):
        """采集循环 - 改为深度+红外"""
        print("[采集线程] 线程循环开始")
        while self.running:
            try:
                # 1. 读取深度帧
                depth_frame = self.depth_stream.read_frame()
                if depth_frame is None:
                    print("[采集线程] 警告: depth_stream.read_frame() 返回了 None")
                    continue

                depth_data = depth_frame.get_buffer_as_uint16()
                depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
                depth_timestamp = time.time()

                # 2. 读取红外帧（如果启用）
                ir_array = None
                ir_timestamp = None
                if self.use_infrared and self.ir_stream:
                    try:
                        ir_frame = self.ir_stream.read_frame()
                        if ir_frame:
                            ir_data = ir_frame.get_buffer_as_uint16()
                            ir_array = np.frombuffer(ir_data, dtype=np.uint16).reshape(self.depth_height,
                                                                                       self.depth_width)

                            # 将红外图像转换为8位用于显示
                            ir_8bit = self._normalize_ir_image(ir_array)
                            ir_timestamp = time.time()
                    except Exception as e:
                        print(f"[采集线程] 读取红外帧时出错: {e}")

                # 3. 放入队列
                if not self.depth_queue.full():
                    self.depth_queue.put((depth_array, depth_timestamp))

                if self.use_infrared and ir_array is not None and not self.ir_queue.full():
                    self.ir_queue.put((ir_8bit, ir_timestamp))

                self.frame_count += 1

            except Exception as e:
                print(f"[采集线程] 循环内发生异常: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"[采集线程] 线程循环结束")

    def _normalize_ir_image(self, ir_image):
        """将红外图像归一化到0-255"""
        if ir_image is None or ir_image.size == 0:
            return None

        # 移除极端值
        valid_values = ir_image[ir_image > 0]
        if len(valid_values) == 0:
            return np.zeros_like(ir_image, dtype=np.uint8)

        # 计算百分位数，避免极端值影响
        low = np.percentile(valid_values, 5)
        high = np.percentile(valid_values, 95)

        # 裁剪并归一化
        ir_clipped = np.clip(ir_image, low, high)
        ir_normalized = cv2.normalize(ir_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return ir_normalized

    def get_synchronized_frames(self, timeout=0.1):
        """获取同步的深度和红外帧"""
        try:
            if not self.use_infrared:
                # 只返回深度帧
                if not self.depth_queue.empty():
                    return self.depth_queue.get_nowait(), None, None, None
                return None, None, None, None

            # 尝试获取同步帧
            max_attempts = 10
            for attempt in range(max_attempts):
                if self.depth_queue.empty() or self.ir_queue.empty():
                    time.sleep(0.001)
                    continue

                # 获取最新的深度帧
                depth_frame, depth_time = self.depth_queue.get_nowait()

                # 寻找时间最接近的红外帧
                best_ir = None
                best_ir_time = None
                best_time_diff = float('inf')

                # 检查红外队列中的所有帧
                temp_queue = queue.Queue()
                while not self.ir_queue.empty():
                    ir_frame, ir_time = self.ir_queue.get_nowait()
                    time_diff = abs(ir_time - depth_time)

                    if time_diff < best_time_diff and time_diff < self.sync_tolerance:
                        best_time_diff = time_diff
                        best_ir = ir_frame
                        best_ir_time = ir_time

                    temp_queue.put((ir_frame, ir_time))

                # 放回未使用的红外帧
                while not temp_queue.empty():
                    self.ir_queue.put(temp_queue.get())

                if best_ir is not None:
                    return depth_frame, best_ir, depth_time, best_ir_time
                else:
                    # 放回深度帧继续尝试
                    self.depth_queue.put((depth_frame, depth_time))
                    time.sleep(0.005)

            # 未找到同步帧
            return None, None, None, None

        except queue.Empty:
            return None, None, None, None
        except Exception as e:
            print(f"[同步] 获取同步帧失败: {e}")
            return None, None, None, None

    def get_stats(self):
        """获取统计信息"""
        current_time = time.time()
        elapsed = current_time - self.stats['start_time'] if self.stats['start_time'] > 0 else 0

        stats = self.stats.copy()
        stats['elapsed_time'] = elapsed
        stats['queue_size'] = self.depth_queue.qsize()
        stats['ir_queue_size'] = self.ir_queue.qsize() if self.use_infrared else 0

        return stats

    def stop(self):
        """停止所有流"""
        print("🛑 正在停止深度流...")
        self.running = False

        # 等待采集线程结束
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)

        # 停止流
        streams = []
        if self.depth_stream:
            streams.append(('深度', self.depth_stream))
        if self.ir_stream:
            streams.append(('红外', self.ir_stream))

        for name, stream in streams:
            try:
                stream.stop()
                print(f"✅ {name}流已停止")
            except Exception as e:
                print(f"⚠️ 停止{name}流失败: {e}")

        # 关闭设备
        if self.device:
            try:
                self.device.close()
                print("✅ 设备已关闭")
            except Exception as e:
                print(f"⚠️ 关闭设备失败: {e}")

        # 卸载OpenNI2
        try:
            openni2.unload()
            print("✅ OpenNI2已卸载")
        except:
            pass

        print("🎯 深度流管理器已完全停止")


# ==================== 2. 可视化管理器 ====================

class VisualizerManager:
    def __init__(self):
        self.vis = None
        self.running = False
        self.window_width = 1280
        self.window_height = 720

        # 可视化选项
        self.show_coordinate_axis = True
        self.show_grid = True
        self.show_trajectory = True
        self.point_size = 1.5
        self.background_color = [0.05, 0.05, 0.1]

        # 存储的几何体
        self.geometries = {}
        self.trajectory_points = []

    def initialize(self, window_name="3D实时建模"):
        """初始化可视化窗口"""
        try:
            print("🔄 初始化Open3D可视化窗口...")

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=window_name,
                width=self.window_width,
                height=self.window_height,
                visible=True
            )

            # 设置渲染选项
            render_opt = self.vis.get_render_option()
            render_opt.background_color = np.array(self.background_color)
            render_opt.point_size = self.point_size
            render_opt.light_on = True
            render_opt.mesh_show_wireframe = False
            render_opt.mesh_show_back_face = False

            # 添加坐标系
            if self.show_coordinate_axis:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.3, origin=[0, 0, 0]
                )
                self.vis.add_geometry(coordinate_frame)
                self.geometries['axis'] = coordinate_frame

            # 添加网格地面
            if self.show_grid:
                grid = self._create_ground_grid()
                self.vis.add_geometry(grid)
                self.geometries['grid'] = grid

            self.running = True
            print("✅ Open3D可视化窗口已启动")

            # 设置初始视角
            self._set_default_view()

            return True

        except Exception as e:
            print(f"❌ 初始化可视化失败: {e}")
            return False

    def _create_ground_grid(self, size=2.0, step=0.2):
        """创建地面网格"""
        points = []
        lines = []
        colors = []

        # 创建网格线
        n_lines = int(size / step)

        # X方向线
        for i in range(-n_lines, n_lines + 1):
            x = i * step
            points.append([x, -size, 0])
            points.append([x, size, 0])
            lines.append([len(points) - 2, len(points) - 1])
            colors.append([0.3, 0.3, 0.3])

        # Y方向线
        for i in range(-n_lines, n_lines + 1):
            y = i * step
            points.append([-size, y, 0])
            points.append([size, y, 0])
            lines.append([len(points) - 2, len(points) - 1])
            colors.append([0.3, 0.3, 0.3])

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector(colors)

        return grid

    def _set_default_view(self):
        """设置默认视角"""
        try:
            ctr = self.vis.get_view_control()

            # 设置相机参数
            ctr.set_front([0, -0.5, -1])  # 看向前方
            ctr.set_up([0, -1, 0.5])  # 上方向
            ctr.set_zoom(0.8)  # 缩放

            # 设置视野范围
            ctr.set_lookat([0, 0, 0])  # 观察点

        except:
            pass  # 有些版本可能不支持这些设置

    def update_pointcloud(self, pointcloud, name="pointcloud"):
        """更新点云显示"""
        if not self.running or pointcloud is None or len(pointcloud.points) == 0:
            return False

        try:
            # 移除旧的点云
            if name in self.geometries:
                self.vis.remove_geometry(self.geometries[name], reset_bounding_box=False)

            # 添加新的点云
            self.vis.add_geometry(pointcloud, reset_bounding_box=False)
            self.geometries[name] = pointcloud

            return True

        except Exception as e:
            print(f"[可视化] 更新点云失败: {e}")
            return False

    def update_trajectory(self, trajectory_points):
        """更新相机轨迹"""
        if not self.running or len(trajectory_points) < 2:
            return False

        try:
            # 移除旧的轨迹
            if 'trajectory' in self.geometries:
                self.vis.remove_geometry(self.geometries['trajectory'], reset_bounding_box=False)

            # 创建轨迹线
            points = np.array(trajectory_points)
            lines = [[i, i + 1] for i in range(len(points) - 1)]

            if len(points) > 0 and len(lines) > 0:
                trajectory = o3d.geometry.LineSet()
                trajectory.points = o3d.utility.Vector3dVector(points)
                trajectory.lines = o3d.utility.Vector2iVector(lines)

                # 设置颜色（从绿到红表示时间）
                colors = []
                for i in range(len(lines)):
                    ratio = i / len(lines) if len(lines) > 0 else 0
                    colors.append([ratio, 1 - ratio, 0])  # RGB: 绿 -> 红

                trajectory.colors = o3d.utility.Vector3dVector(colors)

                # 添加轨迹
                self.vis.add_geometry(trajectory, reset_bounding_box=False)
                self.geometries['trajectory'] = trajectory

                # 保存轨迹点
                self.trajectory_points = trajectory_points

            return True

        except Exception as e:
            print(f"[可视化] 更新轨迹失败: {e}")
            return False

    def add_camera_frustum(self, pose, color=[1, 0, 0], size=0.1):
        """添加相机视锥体"""
        try:
            # 从位姿创建相机视锥体
            # pose: [x, y, z, qx, qy, qz, qw]

            # 创建相机视锥体（简化版本）
            points = [
                [0, 0, 0],  # 相机位置
                [-size, -size, size * 2],  # 近平面左下
                [size, -size, size * 2],  # 近平面右下
                [size, size, size * 2],  # 近平面右上
                [-size, size, size * 2]  # 近平面左上
            ]

            # 应用位姿变换（这里简化处理）
            # 实际应用中需要应用完整的位姿变换

            # 创建线集
            lines = [
                [0, 1], [0, 2], [0, 3], [0, 4],  # 从相机到近平面
                [1, 2], [2, 3], [3, 4], [4, 1]  # 近平面边框
            ]

            frustum = o3d.geometry.LineSet()
            frustum.points = o3d.utility.Vector3dVector(points)
            frustum.lines = o3d.utility.Vector2iVector(lines)
            frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

            # 添加几何体
            frustum_name = f"camera_{len(self.geometries)}"
            self.vis.add_geometry(frustum, reset_bounding_box=False)
            self.geometries[frustum_name] = frustum

            return True

        except Exception as e:
            print(f"[可视化] 添加相机视锥体失败: {e}")
            return False

    def clear_all(self):
        """清除所有几何体（保留坐标系和网格）"""
        try:
            # 保留的几何体
            keep_geometries = ['axis', 'grid']

            # 移除其他几何体
            for name in list(self.geometries.keys()):
                if name not in keep_geometries:
                    self.vis.remove_geometry(self.geometries[name], reset_bounding_box=False)
                    del self.geometries[name]

            # 清空轨迹点
            self.trajectory_points = []

            print("✅ 可视化已清除")
            return True

        except Exception as e:
            print(f"[可视化] 清除失败: {e}")
            return False

    def render(self):
        """渲染当前帧"""
        if self.running:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                return True
            except Exception as e:
                print(f"[可视化] 渲染失败: {e}")
                return False
        return False

    def capture_screenshot(self, filename=None):
        """捕获屏幕截图"""
        if not self.running:
            return False

        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/screenshot_{timestamp}.png"

            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # 捕获截图
            self.vis.capture_screen_image(filename)
            print(f"📸 截图已保存: {filename}")
            return True

        except Exception as e:
            print(f"[可视化] 截图失败: {e}")
            return False

    def stop(self):
        """停止可视化"""
        if self.vis:
            try:
                self.vis.destroy_window()
                self.running = False
                print("✅ Open3D可视化窗口已关闭")
            except:
                pass


# ==================== 3. 主应用程序 ====================
class RealTime3DScanner:
    def __init__(self, config_file=None):
        self.config_file = config_file

        # 初始化各模块 - 改为使用深度+红外
        self.depth_manager = EnhancedDepthStreamManager(use_infrared=True)
        self.visual_odometry = VisualOdometry()
        self.pointcloud_processor = PointCloudProcessor()
        self.visualizer = VisualizerManager()
        self.dataset_exporter = DatasetExporter()

        # 状态变量
        self.running = False
        self.recording = False
        self.scanning = False
        self.paused = False

        # 性能监控
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_update_time = time.time()

        # 数据存储
        self.scanned_frames = []
        self.trajectory = []
        self.current_pose = None

        # 控制参数
        self.update_interval = 0.1  # 10 FPS更新
        self.vo_update_interval = 0.2  # 视觉里程计5 FPS更新
        self.max_frames = 1000  # 最大采集帧数

        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/scans", exist_ok=True)
        os.makedirs("output/datasets", exist_ok=True)

        print("🚀 实时3D扫描系统初始化完成")
        print("📌 模式: 深度+红外 (Astra相机兼容模式)")

    def run(self):
        """运行主程序"""
        print("=" * 70)
        print("实时3D扫描系统 (Python视觉里程计版)")
        print("专为奥比中光Astra相机优化 - 深度+红外模式")
        print("=" * 70)

        # 1. 初始化深度相机
        print("\n1. 初始化深度相机...")
        if not self.depth_manager.initialize(self.config_file):
            print("❌ 深度相机初始化失败")
            return

        # 2. 启动深度流
        print("\n2. 启动深度和红外流...")
        if not self.depth_manager.start_stream():
            print("❌ 深度流启动失败")
            self.depth_manager.stop()
            return

        # 3. 初始化视觉里程计
        print("\n3. 初始化视觉里程计...")
        if not self.visual_odometry.initialize():
            print("⚠️ 视觉里程计初始化失败，将继续运行但不进行位姿估计")

        # 4. 初始化可视化
        print("\n4. 初始化可视化...")
        if not self.visualizer.initialize("实时3D扫描"):
            print("❌ 可视化初始化失败")
            self.depth_manager.stop()
            return

        # 5. 创建OpenCV窗口
        cv2.namedWindow("深度图", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("深度图", 640, 480)

        cv2.namedWindow("红外图", cv2.WINDOW_NORMAL)  # 改为红外图
        cv2.resizeWindow("红外图", 640, 480)

        cv2.namedWindow("控制面板", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("控制面板", 400, 300)

        # 6. 打印控制说明
        self._print_controls()

        # 7. 主循环
        print("\n" + "=" * 70)
        print("系统运行中...")
        print("=" * 70)

        self.running = True
        self._main_loop()

        # 8. 清理
        self._cleanup()

    def _main_loop(self):
        """主循环"""
        last_vo_time = 0
        last_pose = None

        try:
            while self.running:
                current_time = time.time()

                try:
                    # 获取同步的深度和红外帧
                    depth_frame, ir_frame, depth_time, ir_time = self.depth_manager.get_synchronized_frames(
                        timeout=0.001)

                except queue.Empty:
                    # 队列为空，等待下一帧
                    time.sleep(0.005)
                    continue
                except Exception as e:
                    print(f"[主循环] 获取帧数据异常: {e}")
                    time.sleep(0.01)
                    continue

                # 2. 显示图像
                self._display_images(depth_frame, ir_frame)  # 改为显示红外图

                # 3. 显示控制面板
                control_panel = self._create_control_panel()
                cv2.imshow("控制面板", control_panel)

                # 4. 视觉里程计处理（使用红外图像）
                if (ir_frame is not None and self.visual_odometry.initialized and
                        current_time - last_vo_time > self.vo_update_interval):

                    # 处理当前帧（红外图像已经是8位灰度图）
                    pose = self.visual_odometry.process_frame(ir_frame)

                    if pose is not None:
                        self.current_pose = pose

                        # 记录轨迹
                        if len(pose) >= 3:
                            self.trajectory.append(pose[:3])
                            last_pose = pose

                    last_vo_time = current_time

                # 5. 点云处理和可视化
                if current_time - self.last_update_time > self.update_interval and depth_frame is not None:
                    # 处理深度帧（使用当前位姿）
                    if self.pointcloud_processor.process_frame(depth_frame, self.current_pose):
                        # 获取当前点云
                        current_pcd = self.pointcloud_processor.get_pointcloud()

                        # 更新可视化
                        if current_pcd is not None and len(current_pcd.points) > 0:
                            self.visualizer.update_pointcloud(current_pcd)

                            # 更新轨迹
                            if len(self.trajectory) > 1:
                                self.visualizer.update_trajectory(self.trajectory)

                        # 记录扫描帧（如果正在扫描）
                        if self.scanning and len(self.scanned_frames) < self.max_frames:
                            frame_data = {
                                'depth': depth_frame.copy(),
                                'infrared': ir_frame.copy() if ir_frame is not None else None,
                                'pose': self.current_pose.copy() if self.current_pose is not None else None,
                                'timestamp': current_time
                            }
                            self.scanned_frames.append(frame_data)

                    self.last_update_time = current_time

                # 6. 渲染3D视图
                self.visualizer.render()

                # 7. 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard(key)

                # 8. 更新FPS
                self._update_fps()

                # 9. 检查是否达到最大帧数
                if self.scanning and len(self.scanned_frames) >= self.max_frames:
                    print(f"⚠️ 达到最大帧数限制 ({self.max_frames})，自动停止扫描")
                    self.scanning = False

        except KeyboardInterrupt:
            print("\n🔴 用户中断 (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()

    def _display_images(self, depth_frame, ir_frame):
        """显示深度图和红外图"""
        # 显示深度图（伪彩色）
        if depth_frame is not None:
            # 创建深度图显示
            depth_vis = self._depth_to_colormap(depth_frame)

            # 添加文本
            text = f"Depth: {depth_frame.shape}"
            cv2.putText(depth_vis, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示深度值范围
            if depth_frame.max() > 0:
                depth_text = f"Range: {depth_frame.min()}-{depth_frame.max()} mm"
                cv2.putText(depth_vis, depth_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("深度图", depth_vis)

        # 显示红外图
        if ir_frame is not None:
            # 将单通道红外图转为3通道用于显示
            if len(ir_frame.shape) == 2:
                ir_display = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            else:
                ir_display = ir_frame.copy()

            # 添加文本
            text = f"Infrared: {ir_frame.shape}"
            cv2.putText(ir_display, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("红外图", ir_display)

    def _depth_to_colormap(self, depth_image):
        """深度图转伪彩色图"""
        # 移除无效深度值
        depth_valid = depth_image.copy()
        depth_valid[depth_valid == 0] = depth_valid[depth_valid > 0].min() if np.any(depth_valid > 0) else 1

        # 归一化到0-255
        depth_normalized = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 将无效深度标记为黑色
        depth_colored[depth_image == 0] = [0, 0, 0]

        return depth_colored

    def _create_control_panel(self):
        """创建控制面板图像"""
        panel = np.zeros((300, 400, 3), dtype=np.uint8)

        # 标题
        title = "3D扫描控制系统 (深度+红外)"
        cv2.putText(panel, title, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 系统状态
        stats = self.depth_manager.get_stats()
        status_lines = [
            f"系统状态: {'运行中' if self.running else '停止'}",
            f"FPS: {self.fps:.1f}",
            f"采集帧数: {self.frame_count}",
            f"扫描状态: {'🟢 扫描中' if self.scanning else '⚪ 待机'}",
            f"扫描帧数: {len(self.scanned_frames)}/{self.max_frames}",
            f"轨迹点数: {len(self.trajectory)}",
            f"点云点数: {self.pointcloud_processor.total_points}",
            f"VO状态: {'🟢' if self.visual_odometry.initialized else '🔴'}",
            f"模式: 深度+红外",
        ]

        y_offset = 60
        for line in status_lines:
            color = (200, 200, 100) if '🟢' in line else (200, 200, 200)
            cv2.putText(panel, line, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        # 控制键说明
        hints = [
            "控制键:",
            "  Q: 退出程序",
            "  SPACE: 开始/停止扫描",
            "  S: 保存当前点云",
            "  O: 保存网格模型",
            "  C: 清除所有数据",
            "  T: 保存轨迹",
            "  D: 导出数据集",
            "  P: 暂停/继续",
            "  V: 保存截图",
            "  I: 显示系统信息",
        ]

        y_offset += 10
        for hint in hints:
            cv2.putText(panel, hint, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
            y_offset += 20

        return panel

    def _print_controls(self):
        """打印控制说明"""
        controls = """
        ==================== 控制说明 ====================
        深度图/红外图窗口:
          Q / ESC: 退出程序

        3D窗口控制:
          鼠标左键拖拽: 旋转视角
          鼠标右键拖拽: 平移视角
          鼠标滚轮: 缩放

        功能键:
          SPACE: 开始/停止扫描
          S: 保存当前点云为PLY文件
          O: 保存当前点云为OBJ网格文件
          C: 清除所有数据（点云、轨迹、扫描帧）
          T: 保存相机轨迹
          D: 导出完整数据集（用于后续处理）
          P: 暂停/继续处理
          V: 保存3D视图截图
          I: 显示系统信息

        操作建议:
        1. 按SPACE开始扫描，缓慢移动设备围绕物体旋转
        2. 扫描完成后按O保存高质量网格
        3. 按D导出完整数据集供COLMAP或3DGS处理
        ===============================================
        """
        print(controls)



    def _main_loop(self):
        """主循环"""
        last_vo_time = 0
        last_pose = None

        try:
            while self.running:
                current_time = time.time()
                try:
                    # 使用极短的超时，或直接使用非阻塞获取
                    # 方法A：如果你的 get_synchronized_frames 支持 timeout 参数
                    depth_frame, color_frame, depth_time, color_time = self.depth_manager.get_synchronized_frames(
                        timeout=0.001)

                    # 方法B：如果不支持，先尝试非阻塞获取深度帧
                    # depth_frame, depth_time = self.depth_manager.get_depth_frame(timeout=0.001)
                    # color_frame, color_time = None, None
                    # ===== 新增调试信息 =====
                    print(f"[DEBUG] 获取结果: depth={depth_frame is not None}, color={color_frame is not None}")
                    if depth_frame is not None:
                        print(f"      深度图形状: {depth_frame.shape}, 值范围: [{depth_frame.min()}, {depth_frame.max()}]")
                    if color_frame is not None:
                        print(f"      彩色图形状: {color_frame.shape}")
                    # ===== 调试结束 =====


                except queue.Empty:
                    # 队列为空，没有新数据，这是正常情况！
                    # 进行一个非常短暂的休眠，让出CPU，避免100%空转
                    print(
                        f"[主循环] 队列为空。深度队列大小: {self.depth_manager.depth_queue.qsize()}, 彩色队列大小: {self.depth_manager.color_queue.qsize() if self.depth_manager.use_color else 0}")
                    time.sleep(0.005)
                    print(f"[DEBUG] 队列为空，未获取到帧。深度队列大小: {self.depth_manager.depth_queue.qsize()}")
                    time.sleep(0.005)
                    continue
                except Exception as e:
                    print(f"[主循环] 获取帧数据异常: {e}")
                    time.sleep(0.01)
                    continue

                # 2. 显示图像
                self._display_images(depth_frame, color_frame)

                # 3. 显示控制面板
                control_panel = self._create_control_panel()
                cv2.imshow("控制面板", control_panel)

                # 4. 视觉里程计处理（控制频率）
                if (color_frame is not None and self.visual_odometry.initialized and
                        current_time - last_vo_time > self.vo_update_interval):

                    # 处理当前帧
                    pose = self.visual_odometry.process_frame(color_frame)

                    if pose is not None:
                        self.current_pose = pose

                        # 记录轨迹
                        if len(pose) >= 3:
                            self.trajectory.append(pose[:3])
                            last_pose = pose

                    last_vo_time = current_time

                # 5. 点云处理和可视化（控制频率）
                if current_time - self.last_update_time > self.update_interval:
                    # 处理深度帧（使用当前位姿）
                    if self.pointcloud_processor.process_frame(depth_frame, self.current_pose):
                        # 获取当前点云
                        current_pcd = self.pointcloud_processor.get_pointcloud()

                        # 更新可视化
                        if current_pcd is not None and len(current_pcd.points) > 0:
                            self.visualizer.update_pointcloud(current_pcd)

                            # 更新轨迹
                            if len(self.trajectory) > 1:
                                self.visualizer.update_trajectory(self.trajectory)

                        # 记录扫描帧（如果正在扫描）
                        if self.scanning and len(self.scanned_frames) < self.max_frames:
                            frame_data = {
                                'depth': depth_frame.copy(),
                                'color': color_frame.copy() if color_frame is not None else None,
                                'pose': self.current_pose.copy() if self.current_pose is not None else None,
                                'timestamp': current_time
                            }
                            self.scanned_frames.append(frame_data)

                    self.last_update_time = current_time

                # 6. 渲染3D视图
                #self.visualizer.render()

                # 7. 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard(key)

                # 8. 更新FPS
                self._update_fps()

                # 9. 检查是否达到最大帧数
                if self.scanning and len(self.scanned_frames) >= self.max_frames:
                    print(f"⚠️ 达到最大帧数限制 ({self.max_frames})，自动停止扫描")
                    self.scanning = False

        except KeyboardInterrupt:
            print("\n🔴 用户中断 (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()



    '''
    def _main_loop(self):
        """主循环 - 使用非阻塞模式避免卡死"""
        last_vo_time = 0
        last_pose = None

        try:
            while self.running:
                current_time = time.time()

                # ===== 核心修改点：非阻塞获取帧 =====
                try:
                    # 使用极短的超时，或直接使用非阻塞获取
                    # 方法A：如果你的 get_synchronized_frames 支持 timeout 参数
                    depth_frame, color_frame, depth_time, color_time = self.depth_manager.get_synchronized_frames(
                        timeout=0.001)

                    # 方法B：如果不支持，先尝试非阻塞获取深度帧
                    # depth_frame, depth_time = self.depth_manager.get_depth_frame(timeout=0.001)
                    # color_frame, color_time = None, None

                except queue.Empty:
                    # 队列为空，没有新数据，这是正常情况！
                    # 进行一个非常短暂的休眠，让出CPU，避免100%空转
                    time.sleep(0.005)  # 休眠5毫秒
                    continue  # 跳过本次循环的剩余部分
                except Exception as e:
                    print(f"[主循环] 获取帧数据异常: {e}")
                    time.sleep(0.01)
                    continue
                # ===== 修改结束 =====

                # 后续的显示、处理等代码保持不变...
                # 但为了测试，你可以先将所有处理代码都注释掉，只保留最基础的显示

                # 测试1: 只显示深度图 (最简测试)
                if depth_frame is not None:
                    depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                    depth_colored = cv2.applyColorMap(depth_display.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imshow("深度图", depth_colored)

                # 测试2: 只显示彩色图 (如果存在)
                if color_frame is not None:
                    cv2.imshow("彩色图", color_frame)

                # 处理键盘输入 (必须保持，这是退出途径)
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard(key)

                # 可以在这里添加一个简单的心跳打印，确认循环在跑
                if int(current_time) % 2 == 0:  # 每2秒打印一次
                    print(f"[心跳] 主循环运行中...")

        except KeyboardInterrupt:
            print("\n🔴 用户中断")
        except Exception as e:
            print(f"\n❌ 主循环异常: {e}")
            import traceback
            traceback.print_exc()

    '''

    def _display_images(self, depth_frame, color_frame):
        """显示深度图和彩色图"""
        # 显示深度图（伪彩色）
        if depth_frame is not None:
            depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_display.astype(np.uint8), cv2.COLORMAP_JET)

            # 添加文本
            text = f"Depth: {depth_frame.shape}"
            cv2.putText(depth_colored, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("深度图", depth_colored)

        # 显示彩色图
        if color_frame is not None:
            color_display = color_frame.copy()

            # 添加文本
            text = f"Color: {color_frame.shape}"
            cv2.putText(color_display, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("彩色图", color_display)


    def _create_control_panel(self):
        """创建控制面板图像"""
        panel = np.zeros((300, 400, 3), dtype=np.uint8)

        # 标题
        title = "3D扫描控制系统"
        cv2.putText(panel, title, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 系统状态
        stats = self.depth_manager.get_stats()
        status_lines = [
            f"系统状态: {'运行中' if self.running else '停止'}",
            f"FPS: {self.fps:.1f}",
            f"采集帧数: {self.frame_count}",
            f"扫描状态: {'🟢 扫描中' if self.scanning else '⚪ 待机'}",
            f"扫描帧数: {len(self.scanned_frames)}/{self.max_frames}",
            f"轨迹点数: {len(self.trajectory)}",
            f"点云点数: {self.pointcloud_processor.total_points}",
            f"VO状态: {'🟢' if self.visual_odometry.initialized else '🔴'}",
        ]

        y_offset = 60
        for line in status_lines:
            color = (200, 200, 100) if '🟢' in line else (200, 200, 200)
            cv2.putText(panel, line, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        # 控制键说明
        hints = [
            "控制键:",
            "  Q: 退出程序",
            "  SPACE: 开始/停止扫描",
            "  S: 保存当前点云",
            "  O: 保存网格模型",
            "  C: 清除所有数据",
            "  T: 保存轨迹",
            "  D: 导出数据集",
            "  P: 暂停/继续",
            "  V: 保存截图",
        ]

        y_offset += 10
        for hint in hints:
            cv2.putText(panel, hint, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
            y_offset += 20

        return panel

    def _handle_keyboard(self, key):
        """处理键盘输入"""
        # 退出
        if key == ord('q') or key == 27:
            print("\n收到退出指令...")
            self.running = False

        # 开始/停止扫描
        elif key == ord(' '):
            self.scanning = not self.scanning
            if self.scanning:
                print("🟢 开始扫描...")
                self.scanned_frames = []  # 清空之前的扫描
            else:
                print("⏹️ 停止扫描")
                print(f"  扫描了 {len(self.scanned_frames)} 帧数据")

        # 保存点云
        elif key == ord('s'):
            if self.pointcloud_processor.total_points > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/scans/pointcloud_{timestamp}.ply"

                if self.pointcloud_processor.save_pointcloud(filename):
                    print(f"✅ 点云已保存: {filename}")
                else:
                    print("❌ 保存点云失败")
            else:
                print("❌ 当前无点云数据")

        # 保存网格
        elif key == ord('o'):
            if self.pointcloud_processor.total_points > 1000:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/scans/mesh_{timestamp}.obj"

                if self.pointcloud_processor.reconstruct_mesh(filename):
                    print(f"✅ 网格已保存: {filename}")
                else:
                    print("❌ 保存网格失败")
            else:
                print("❌ 点云点数不足")

        # 清除数据
        elif key == ord('c'):
            self.pointcloud_processor.clear()
            self.scanned_frames = []
            self.trajectory = []
            self.visualizer.clear_all()
            self.visual_odometry.reset()
            print("✅ 所有数据已清除")

        # 保存轨迹
        elif key == ord('t'):
            if len(self.trajectory) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/trajectory_{timestamp}.json"

                trajectory_data = {
                    'trajectory': self.trajectory,
                    'frames': len(self.scanned_frames),
                    'timestamp': timestamp
                }

                try:
                    with open(filename, 'w') as f:
                        json.dump(trajectory_data, f, indent=2)
                    print(f"✅ 轨迹已保存: {filename}")
                except Exception as e:
                    print(f"❌ 保存轨迹失败: {e}")
            else:
                print("❌ 无轨迹数据")

        # 导出数据集
        elif key == ord('d'):
            if len(self.scanned_frames) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_dir = f"output/datasets/dataset_{timestamp}"

                if self.dataset_exporter.export_dataset(self.scanned_frames, dataset_dir):
                    print(f"✅ 数据集已导出: {dataset_dir}")
                else:
                    print("❌ 导出数据集失败")
            else:
                print("❌ 无扫描数据可导出")

        # 暂停/继续
        elif key == ord('p'):
            self.paused = not self.paused
            print(f"⏸️  {'暂停' if self.paused else '继续'}")

        # 保存截图
        elif key == ord('v'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/screenshot_{timestamp}.png"
            if self.visualizer.capture_screenshot(filename):
                print(f"✅ 截图已保存: {filename}")

        # 显示信息
        elif key == ord('i'):
            self._show_info()

    def _update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

            # 显示状态
            if self.scanning:
                print(
                    f"\r🟢 扫描中... 帧数: {len(self.scanned_frames)}/{self.max_frames} | FPS: {self.fps:.1f} | 点数: {self.pointcloud_processor.total_points}",
                    end="")
            else:
                print(f"\r⚪ 待机... FPS: {self.fps:.1f} | 点数: {self.pointcloud_processor.total_points}", end="")

    def _show_info(self):
        """显示系统信息"""
        stats = self.depth_manager.get_stats()

        print("\n" + "=" * 60)
        print("系统信息:")
        print(f"  当前FPS: {self.fps:.1f}")
        print(f"  采集总帧数: {stats.get('frames_captured', 0)}")
        print(f"  点云点数: {self.pointcloud_processor.total_points}")
        print(f"  扫描帧数: {len(self.scanned_frames)}")
        print(f"  轨迹长度: {len(self.trajectory)}")
        print(f"  VO状态: {'已初始化' if self.visual_odometry.initialized else '未初始化'}")
        print(f"  队列大小: {stats.get('queue_size', 0)}")
        print(f"  运行时间: {stats.get('elapsed_time', 0):.1f}秒")
        print("=" * 60)

    def _print_controls(self):
        """打印控制说明"""
        controls = """
        ==================== 控制说明 ====================
        深度图/彩色图窗口:
          Q / ESC: 退出程序

        3D窗口控制:
          鼠标左键拖拽: 旋转视角
          鼠标右键拖拽: 平移视角
          鼠标滚轮: 缩放

        功能键:
          SPACE: 开始/停止扫描
          S: 保存当前点云为PLY文件
          O: 保存当前点云为OBJ网格文件
          C: 清除所有数据（点云、轨迹、扫描帧）
          T: 保存相机轨迹
          D: 导出完整数据集（用于后续处理）
          P: 暂停/继续处理
          V: 保存3D视图截图
          I: 显示系统信息

        操作建议:
        1. 按SPACE开始扫描，缓慢移动设备围绕物体旋转
        2. 扫描完成后按O保存高质量网格
        3. 按D导出完整数据集供COLMAP或3DGS处理
        ===============================================
        """
        print(controls)

    def _cleanup(self):
        """清理资源"""
        print("\n正在关闭系统...")

        # 自动保存未保存的数据
        if len(self.scanned_frames) > 0 and not self.dataset_exporter.is_exported:
            print("💾 自动保存扫描数据...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_dir = f"output/datasets/autosave_{timestamp}"
            self.dataset_exporter.export_dataset(self.scanned_frames, dataset_dir)

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()

        # 停止可视化
        self.visualizer.stop()

        # 停止深度流
        self.depth_manager.stop()

        print("✅ 系统已安全关闭")


# ==================== 4. 辅助函数 ====================
def check_dependencies():
    """检查必要的依赖库"""
    required_libraries = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('open3d', 'open3d'),
        # 不再单独检查 'openni2'，而是检查 'primesense'
        ('primesense', 'primesense')  # ✅ 检查 primesense 包本身
    ]

    missing_libs = []

    for lib_name, pip_name in required_libraries:
        try:
            __import__(lib_name)
            print(f"✅ {lib_name} 已安装")
        except ImportError:
            print(f"❌ {lib_name} 未安装")
            missing_libs.append(pip_name)

    if missing_libs:
        print(f"\n请安装缺失的库: pip install {' '.join(missing_libs)}")
        return False

    return True


def create_default_config():
    """创建默认配置文件（针对Astra相机）"""
    config = {
        "camera_params": {
            "fx": 475.0,
            "fy": 475.0,
            "cx": 320.0,
            "cy": 240.0,
            "depth_scale": 0.001,
            "min_depth": 0.2,
            "max_depth": 2.5
        },
        "visual_odometry": {
            "max_features": 1000,
            "min_matches": 20,
            "ransac_threshold": 1.0
        },
        "pointcloud": {
            "voxel_size": 0.01,
            "max_points": 100000,
            "downsample_rate": 2
        },
        "device_settings": {
            "use_infrared": true,  # Astra相机使用红外流
            "use_color": false     # Astra相机不支持同时深度+彩色
        }
    }

    os.makedirs("config", exist_ok=True)

    with open("config/default.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Astra相机默认配置文件已创建: config/default.json")
    return "config/default.json"

def main():
    """主函数"""
    print("=" * 70)
    print("Windows平台实时3D扫描系统")
    print("专为奥比中光Astra相机优化 - 深度+红外模式")
    print("=" * 70)

    # 检查依赖
    print("\n检查依赖库...")
    if not check_dependencies():
        print("\n❌ 缺少必要的依赖库，请先安装")
        return

    # 创建默认配置（如果不存在）
    config_file = "config/default.json"
    if not os.path.exists(config_file):
        config_file = create_default_config()

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='实时3D扫描系统')
    parser.add_argument('--config', type=str, default=config_file, help='配置文件路径')
    parser.add_argument('--use-ir', action='store_true', default=True, help='使用红外流')
    parser.add_argument('--no-vo', action='store_true', help='禁用视觉里程计')
    parser.add_argument('--max-frames', type=int, default=1000, help='最大采集帧数')

    args = parser.parse_args()

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        args.config = create_default_config()

    # 运行系统
    try:
        scanner = RealTime3DScanner(config_file=args.config)
        scanner.max_frames = args.max_frames

        if args.no_vo:
            print("⚠️ 视觉里程计已禁用")

        scanner.run()

    except KeyboardInterrupt:
        print("\n\n🔴 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()