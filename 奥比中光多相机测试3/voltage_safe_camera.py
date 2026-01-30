"""
电压安全相机系统 - 针对奥比中光Astra相机电压不足问题
采用两两交替工作策略，避免同时开启三个相机
"""

"""
电压安全相机系统 - 针对奥比中光Astra相机电压不足问题
采用两两交替工作策略，避免同时开启三个相机
"""
import open3d as o3d
import cv2
import numpy as np
import threading
import time
import queue
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any  # 添加类型导入

# 检查OpenNI2
try:
    from primesense import openni2

    OPENNI2_AVAILABLE = True
except ImportError:
    OPENNI2_AVAILABLE = False
    print("⚠️ OpenNI2导入失败，深度功能将受限")

from config import MultiCameraConfig
from utils import MultiCameraUtils


class VoltageSafeCamera:
    """电压安全的单个相机控制器"""

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.initialized = False
        self.active = False

        # OpenNI2资源
        self.device = None
        self.depth_stream = None

        # OpenCV彩色相机
        self.color_camera = None

        # 线程安全锁
        self.lock = threading.RLock()

        # 缓存
        self.latest_color = None
        self.latest_depth = None
        self.last_update = 0

        # 统计
        self.frame_count = 0
        self.error_count = 0

    def initialize(self) -> bool:
        """初始化相机（不激活流）"""
        if not OPENNI2_AVAILABLE:
            print(f"相机 {self.camera_id}: OpenNI2不可用")
            return False

        with self.lock:
            try:
                print(f"相机 {self.camera_id}: 初始化...")

                # 尝试打开OpenNI2设备
                for attempt in range(3):
                    try:
                        self.device = openni2.Device.open_any()
                        if self.device:
                            print(f"相机 {self.camera_id}: OpenNI2设备打开成功")
                            break
                    except Exception as e:
                        if attempt == 2:
                            print(f"相机 {self.camera_id}: OpenNI2设备打开失败: {e}")
                            return False
                        time.sleep(0.5)

                # 尝试打开彩色相机
                self.initialize_color_camera()

                self.initialized = True
                print(f"✅ 相机 {self.camera_id} 初始化成功")
                return True

            except Exception as e:
                print(f"❌ 相机 {self.camera_id} 初始化失败: {e}")
                traceback.print_exc()
                return False

    def initialize_color_camera(self) -> bool:
        """初始化彩色相机"""
        # 尝试不同的索引
        color_indices = [
            self.camera_id,  # 直接索引
            self.camera_id * 2,  # 偶数索引
            self.camera_id * 2 + 1,  # 奇数索引
            0, 1, 2, 3  # 通用索引
        ]

        for idx in color_indices:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # 测试读取
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # 设置分辨率
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                                MultiCameraConfig.CAPTURE_CONFIG['color_resolution'][0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                                MultiCameraConfig.CAPTURE_CONFIG['color_resolution'][1])

                        self.color_camera = cap
                        print(f"相机 {self.camera_id}: 彩色相机打开成功 (索引{idx})")
                        return True
                    else:
                        cap.release()
            except:
                continue

        print(f"相机 {self.camera_id}: 彩色相机初始化失败，仅使用深度")
        return False

    def activate(self) -> bool:
        """激活相机（开始采集）"""
        if not self.initialized or self.active:
            return False

        with self.lock:
            try:
                print(f"相机 {self.camera_id}: 激活...")

                # 创建并启动深度流
                if self.device and not self.depth_stream:
                    self.depth_stream = self.device.create_depth_stream()
                    self.depth_stream.start()

                    # 预热：读取并丢弃前几帧
                    for _ in range(3):
                        try:
                            self.depth_stream.read_frame()
                        except:
                            pass
                        time.sleep(0.01)

                    print(f"相机 {self.camera_id}: 深度流启动")

                self.active = True
                self.frame_count = 0
                self.error_count = 0

                # 激活后延时
                time.sleep(0.05)

                print(f"✅ 相机 {self.camera_id} 激活完成")
                return True

            except Exception as e:
                print(f"❌ 相机 {self.camera_id} 激活失败: {e}")
                return False

    def deactivate(self) -> bool:
        """停用相机（停止采集）"""
        if not self.active:
            return False

        with self.lock:
            try:
                print(f"相机 {self.camera_id}: 停用...")

                # 停止深度流
                if self.depth_stream:
                    try:
                        self.depth_stream.stop()
                    except:
                        pass
                    self.depth_stream = None

                self.active = False
                self.latest_color = None
                self.latest_depth = None

                # 小延时，确保资源释放
                time.sleep(0.02)

                print(f"✅ 相机 {self.camera_id} 停用完成")
                return True

            except Exception as e:
                print(f"相机 {self.camera_id} 停用失败: {e}")
                return False

    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """捕获一帧"""
        if not self.active:
            return None, None

        with self.lock:
            try:
                color_frame = None
                depth_frame = None

                # 捕获深度帧
                if self.depth_stream:
                    try:
                        frame_data = self.depth_stream.read_frame()
                        depth_buffer = frame_data.get_buffer_as_uint16()
                        depth_frame = np.frombuffer(depth_buffer, dtype=np.uint16).reshape(480, 640)
                    except Exception as e:
                        self.error_count += 1
                        if self.error_count % 10 == 0:
                            print(f"相机 {self.camera_id} 深度采集错误: {e}")

                # 捕获彩色帧
                if self.color_camera:
                    try:
                        ret, color_frame = self.color_camera.read()
                        if ret and color_frame is not None:
                            # 调整尺寸
                            if color_frame.shape[:2] != (480, 640):
                                color_frame = cv2.resize(color_frame, (640, 480))
                    except Exception as e:
                        self.error_count += 1

                # 更新缓存
                if color_frame is not None or depth_frame is not None:
                    self.latest_color = color_frame
                    self.latest_depth = depth_frame
                    self.last_update = time.time()
                    self.frame_count += 1

                return color_frame, depth_frame

            except Exception as e:
                print(f"相机 {self.camera_id} 捕获异常: {e}")
                return None, None

    def get_cached_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取缓存的最后一帧"""
        with self.lock:
            return self.latest_color, self.latest_depth

    def get_pointcloud(self) -> Optional[o3d.geometry.PointCloud]:
        """获取当前点云"""
        if self.latest_depth is None:
            return None

        try:
            # 转换为点云
            depth_scale = MultiCameraConfig.CAPTURE_CONFIG['depth_scale']
            max_depth = MultiCameraConfig.CAPTURE_CONFIG['depth_max']

            points, valid_mask = MultiCameraUtils.depth_to_pointcloud(
                self.latest_depth,
                MultiCameraConfig.get_intrinsic_matrix(),
                depth_scale,
                max_depth
            )

            if points is None or len(points) == 0:
                return None

            # 添加颜色
            colors = None
            if self.latest_color is not None and valid_mask is not None:
                colors = self.latest_color.reshape(-1, 3)[valid_mask.flatten()]
                colors = colors.astype(np.float32) / 255.0

            # 创建点云
            pcd = MultiCameraUtils.create_open3d_pointcloud(points, colors)

            # 变换到世界坐标系
            extrinsic = MultiCameraConfig.get_camera_extrinsic(self.camera_id)
            if extrinsic is not None:
                pcd.transform(extrinsic)

            return pcd

        except Exception as e:
            print(f"相机 {self.camera_id}: 生成点云失败: {e}")
            return None

    def cleanup(self):
        """清理资源"""
        with self.lock:
            print(f"相机 {self.camera_id}: 清理资源...")

            self.deactivate()

            # 释放彩色相机
            if self.color_camera:
                try:
                    self.color_camera.release()
                except:
                    pass
                self.color_camera = None

            # 关闭OpenNI2设备
            if self.device:
                try:
                    self.device.close()
                except:
                    pass
                self.device = None

            self.initialized = False
            print(f"相机 {self.camera_id}: 资源已清理")


class VoltageOptimizedSystem:
    """电压优化的多相机系统（两两交替工作）"""

    def __init__(self):
        self.cameras = {}
        self.openni_initialized = False
        self.running = False

        # 电压优化参数
        self.config = MultiCameraConfig.VOLTAGE_OPTIMIZATION
        self.rotation_index = 0
        self.last_rotation_time = 0
        self.active_combo = []

        # 采集线程
        self.capture_thread = None
        self.capture_queue = queue.Queue(maxsize=20)

        # 统计
        self.stats = {
            'frames_captured': [0, 0, 0],
            'activations': [0, 0, 0],
            'rotation_count': 0,
            'start_time': 0,
            'total_frames': 0,
        }

    def initialize(self) -> bool:
        """初始化系统"""
        print("=" * 70)
        print("电压优化的多相机系统")
        print(f"模式: 两两交替 (最大{self.config['max_simultaneous']}个同时激活)")
        print("=" * 70)

        if not OPENNI2_AVAILABLE:
            print("❌ OpenNI2不可用")
            return False

        # 初始化OpenNI2
        try:
            openni2.initialize()
            self.openni_initialized = True
            print("✅ OpenNI2初始化成功")
        except Exception as e:
            print(f"❌ OpenNI2初始化失败: {e}")
            return False

        # 初始化所有相机
        print(f"\n初始化 {MultiCameraConfig.NUM_CAMERAS} 个相机...")

        for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
            camera = VoltageSafeCamera(cam_id)
            if camera.initialize():
                self.cameras[cam_id] = camera
                print(f"  相机 {cam_id}: ✅ 初始化成功")
            else:
                print(f"  相机 {cam_id}: ❌ 初始化失败")

        if len(self.cameras) == 0:
            print("❌ 没有相机初始化成功")
            return False

        # 激活第一个组合
        self.rotate_active_combo()

        self.stats['start_time'] = time.time()
        print(f"\n✅ 系统初始化完成，激活组合: {self.active_combo}")
        return True

    def rotate_active_combo(self) -> bool:
        """轮换激活的组合"""
        old_combo = self.active_combo.copy()

        # 停用旧的组合
        for cam_id in old_combo:
            if cam_id in self.cameras:
                self.cameras[cam_id].deactivate()

        # 选择新的组合
        combo_list = self.config['active_combinations']
        self.active_combo = combo_list[self.rotation_index]

        print(f"\n🔄 切换到组合 {self.active_combo} (轮换 {self.rotation_index + 1}/{len(combo_list)})")

        # 预热延时
        time.sleep(self.config['warmup_time'])

        # 激活新的组合
        success_count = 0
        for cam_id in self.active_combo:
            if cam_id in self.cameras:
                if self.cameras[cam_id].activate():
                    success_count += 1
                    self.stats['activations'][cam_id] += 1

        if success_count == len(self.active_combo):
            self.rotation_index = (self.rotation_index + 1) % len(combo_list)
            self.last_rotation_time = time.time()
            self.stats['rotation_count'] += 1
            return True
        else:
            print(f"⚠️ 组合激活失败，回退到 {old_combo}")
            self.active_combo = old_combo
            return False

    def capture_worker(self):
        """采集工作线程"""
        print("采集线程启动...")

        while self.running:
            try:
                current_time = time.time()

                # 检查是否需要轮换
                if current_time - self.last_rotation_time >= self.config['rotation_interval']:
                    self.rotate_active_combo()

                # 采集激活的相机
                frames = {}

                for cam_id in self.active_combo:
                    if cam_id in self.cameras and self.cameras[cam_id].active:
                        color, depth = self.cameras[cam_id].capture_frame()

                        if color is not None or depth is not None:
                            frames[cam_id] = (color, depth)
                            self.stats['frames_captured'][cam_id] += 1
                            self.stats['total_frames'] += 1

                # 如果有数据，放入队列
                if frames:
                    try:
                        self.capture_queue.put_nowait((frames, current_time))
                    except queue.Full:
                        # 队列满，丢弃旧数据
                        try:
                            self.capture_queue.get_nowait()
                            self.capture_queue.put_nowait((frames, current_time))
                        except:
                            pass

                # 控制帧率
                target_fps = MultiCameraConfig.CAPTURE_CONFIG['target_fps']
                time.sleep(1.0 / target_fps)

            except Exception as e:
                print(f"采集线程错误: {e}")
                time.sleep(0.1)

        print("采集线程停止")

    def start_capture(self):
        """开始采集"""
        if self.running:
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
        self.capture_thread.start()

        print(f"✅ 采集开始，轮换间隔: {self.config['rotation_interval']}秒")

    def stop_capture(self):
        """停止采集"""
        if not self.running:
            return

        print("停止采集...")
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        # 停用所有相机
        for cam_id in self.cameras:
            self.cameras[cam_id].deactivate()

        print("采集停止")

    def get_frames(self, timeout: float = 0.1):
        """获取最新的帧数据"""
        try:
            return self.capture_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def get_all_cached_frames(self) -> Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """获取所有相机的缓存帧"""
        frames = {}

        for cam_id, camera in self.cameras.items():
            color, depth = camera.get_cached_frame()
            if color is not None or depth is not None:
                frames[cam_id] = (color, depth)

        return frames

    def get_pointclouds(self) -> Dict[int, o3d.geometry.PointCloud]:
        """获取所有相机的点云"""
        pointclouds = {}

        for cam_id, camera in self.cameras.items():
            pcd = camera.get_pointcloud()
            if pcd and len(pcd.points) > 0:
                pointclouds[cam_id] = pcd

        return pointclouds

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        current_time = time.time()
        elapsed = current_time - self.stats['start_time']

        stats = {
            'elapsed_time': elapsed,
            'total_frames': self.stats['total_frames'],
            'frames_per_camera': self.stats['frames_captured'].copy(),
            'fps_per_camera': [
                self.stats['frames_captured'][i] / elapsed if elapsed > 0 else 0
                for i in range(len(self.stats['frames_captured']))
            ],
            'average_fps': self.stats['total_frames'] / elapsed if elapsed > 0 else 0,
            'rotation_count': self.stats['rotation_count'],
            'activations': self.stats['activations'].copy(),
            'active_combo': self.active_combo.copy(),
            'next_rotation_in': max(0, self.config['rotation_interval'] - (current_time - self.last_rotation_time)),
        }

        return stats

    def visualize_alternating_view(self, window_name: str = "电压优化相机视图"):
        """可视化交替采集视图"""
        print("\n" + "=" * 60)
        print("电压优化相机视图")
        print("=" * 60)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)

        print("按ESC键退出")

        try:
            self.start_capture()

            while True:
                # 获取统计信息
                stats = self.get_statistics()

                # 获取所有缓存帧
                frames = self.get_all_cached_frames()

                # 创建显示图像
                displays = []

                for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
                    display = np.zeros((300, 400, 3), dtype=np.uint8)

                    # 状态标签
                    if cam_id in self.active_combo:
                        status_color = (0, 0, 255)  # 红色，激活
                        status_text = "ACTIVE"
                    else:
                        status_color = (100, 100, 100)  # 灰色，非激活
                        status_text = "INACTIVE"

                    if cam_id in frames:
                        color, depth = frames[cam_id]

                        if color is not None:
                            color_resized = cv2.resize(color, (400, 300))
                            display = color_resized

                            # 帧数信息
                            frame_count = self.stats['frames_captured'][cam_id]
                            cv2.putText(display, f"Frames: {frame_count}", (10, 280),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # 深度信息
                            if depth is not None:
                                valid_pixels = np.sum(depth > 0)
                                cv2.putText(display, f"Depth: {valid_pixels}", (10, 250),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # 无数据
                        cv2.putText(display, "No Data", (150, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

                    # 相机ID和状态
                    cv2.putText(display, f"Cam{cam_id} [{status_text}]", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    displays.append(display)

                # 组合显示
                if displays:
                    if len(displays) >= 2:
                        row1 = np.hstack(displays[:2])

                        if len(displays) == 3:
                            # 调整第三幅图的大小并居中
                            display3 = displays[2]
                            padding = (row1.shape[1] - display3.shape[1]) // 2

                            if padding > 0:
                                display3_padded = np.zeros((display3.shape[0], row1.shape[1], 3), dtype=np.uint8)
                                display3_padded[:, padding:padding + display3.shape[1]] = display3
                                display3 = display3_padded

                            combined = np.vstack([row1, display3])
                        else:
                            combined = row1

                        # 添加统计信息面板
                        info_height = 60
                        info_panel = np.zeros((info_height, combined.shape[1], 3), dtype=np.uint8)

                        fps_text = f"平均FPS: {stats['average_fps']:.1f} | "
                        fps_text += f"激活组合: {self.active_combo} | "
                        fps_text += f"总帧数: {stats['total_frames']}"

                        cv2.putText(info_panel, fps_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        next_rotation = stats['next_rotation_in']
                        cv2.putText(info_panel, f"下次轮换: {next_rotation:.1f}秒", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        combined_with_info = np.vstack([combined, info_panel])
                        cv2.imshow(window_name, combined_with_info)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n用户退出")
                    break

                # 每30帧显示一次统计
                if stats['total_frames'] % 30 == 0:
                    print(f"已处理 {stats['total_frames']} 帧，平均FPS: {stats['average_fps']:.1f}")

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"可视化错误: {e}")
            traceback.print_exc()
        finally:
            self.stop_capture()
            cv2.destroyAllWindows()

    def cleanup(self):
        """清理系统"""
        print("\n清理系统...")

        self.stop_capture()

        for camera in self.cameras.values():
            camera.cleanup()

        self.cameras.clear()

        if self.openni_initialized:
            try:
                openni2.unload()
                print("OpenNI2已卸载")
            except:
                pass

        print("系统清理完成")