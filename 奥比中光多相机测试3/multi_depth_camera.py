"""
电压优化的多相机深度系统
基于两两交替工作原则，避免同时开启三个相机
"""

import cv2
import numpy as np
import threading
import time
import queue
import os
import sys
import traceback
from datetime import datetime
from collections import deque
import warnings

# 检查OpenNI2
try:
    from primesense import openni2
    OPENNI2_AVAILABLE = True
except ImportError:
    OPENNI2_AVAILABLE = False
    print("警告: OpenNI2不可用，深度功能将受限")

from config import MultiCameraConfig


class VoltageOptimizedCamera:
    """电压优化的单个相机控制器"""

    def __init__(self, camera_id, driver_path=""):
        self.camera_id = camera_id
        self.driver_path = driver_path
        self.device = None
        self.depth_stream = None
        self.color_camera = None
        self.initialized = False
        self.active = False
        self.last_activation_time = 0
        self.frame_count = 0

        # 线程安全
        self.lock = threading.RLock()

        # 缓存
        self.latest_color = None
        self.latest_depth = None
        self.latest_timestamp = 0

    def initialize(self):
        """初始化相机（但不激活流）"""
        if not OPENNI2_AVAILABLE:
            print(f"相机 {self.camera_id}: OpenNI2不可用")
            return False

        try:
            with self.lock:
                print(f"相机 {self.camera_id}: 初始化设备...")

                # 尝试打开设备（使用不同的策略）
                for attempt in range(3):
                    try:
                        # 使用不同的打开方式
                        if attempt == 0:
                            self.device = openni2.Device.open_any()
                        elif attempt == 1:
                            # 尝试通过URI打开
                            devices = openni2.Device.enumerate_uris()
                            if devices and len(devices) > 0:
                                device_idx = self.camera_id % len(devices)
                                self.device = openni2.Device.open(devices[device_idx])
                        else:
                            time.sleep(0.5)
                            self.device = openni2.Device.open_any()

                        if self.device:
                            print(f"相机 {self.camera_id}: 设备打开成功 (尝试{attempt+1})")
                            break
                    except Exception as e:
                        if attempt == 2:
                            print(f"相机 {self.camera_id}: 打开设备失败: {e}")
                            return False

                # 彩色相机
                print(f"相机 {self.camera_id}: 初始化彩色相机...")
                color_indices = [self.camera_id, self.camera_id + 10, 0, 1, 2]

                for idx in color_indices:
                    try:
                        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                        if cap.isOpened():
                            # 测试读取
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                self.color_camera = cap
                                print(f"相机 {self.camera_id}: 彩色相机打开成功 (索引{idx})")
                                break
                            else:
                                cap.release()
                    except:
                        pass

                if not self.color_camera:
                    print(f"相机 {self.camera_id}: 彩色相机初始化失败，仅使用深度")

                self.initialized = True
                print(f"✅ 相机 {self.camera_id} 初始化完成")
                return True

        except Exception as e:
            print(f"❌ 相机 {self.camera_id} 初始化失败: {e}")
            traceback.print_exc()
            return False

    def activate(self):
        """激活相机（开始采集）"""
        if not self.initialized or self.active:
            return False

        with self.lock:
            try:
                print(f"相机 {self.camera_id}: 激活...")

                # 创建深度流
                if self.device and not self.depth_stream:
                    self.depth_stream = self.device.create_depth_stream()
                    self.depth_stream.start()
                    print(f"相机 {self.camera_id}: 深度流启动")

                # 预热
                if self.depth_stream:
                    # 读取几帧丢弃，让传感器稳定
                    for _ in range(5):
                        try:
                            self.depth_stream.read_frame()
                        except:
                            pass
                        time.sleep(0.01)

                self.active = True
                self.last_activation_time = time.time()
                self.frame_count = 0

                print(f"✅ 相机 {self.camera_id} 激活完成")
                return True

            except Exception as e:
                print(f"❌ 相机 {self.camera_id} 激活失败: {e}")
                return False

    def deactivate(self):
        """停用相机（停止采集）"""
        if not self.active:
            return False

        with self.lock:
            try:
                print(f"相机 {self.camera_id}: 停用...")

                # 停止深度流
                if self.depth_stream:
                    self.depth_stream.stop()
                    self.depth_stream = None
                    print(f"相机 {self.camera_id}: 深度流停止")

                self.active = False
                self.latest_color = None
                self.latest_depth = None

                # 小延时，让设备完全释放
                time.sleep(0.05)

                print(f"✅ 相机 {self.camera_id} 停用完成")
                return True

            except Exception as e:
                print(f"相机 {self.camera_id} 停用失败: {e}")
                return False

    def capture_frame(self):
        """捕获一帧"""
        if not self.active:
            return None, None

        with self.lock:
            try:
                # 捕获深度
                depth_frame = None
                if self.depth_stream:
                    try:
                        frame_data = self.depth_stream.read_frame()
                        depth_buffer = frame_data.get_buffer_as_uint16()
                        depth_frame = np.frombuffer(depth_buffer, dtype=np.uint16).reshape(480, 640)
                    except Exception as e:
                        # print(f"深度采集错误: {e}")
                        pass

                # 捕获彩色
                color_frame = None
                if self.color_camera:
                    try:
                        ret, color_frame = self.color_camera.read()
                        if ret and color_frame is not None:
                            if color_frame.shape[:2] != (480, 640):
                                color_frame = cv2.resize(color_frame, (640, 480))
                    except:
                        pass

                # 更新缓存
                if depth_frame is not None or color_frame is not None:
                    self.latest_color = color_frame
                    self.latest_depth = depth_frame
                    self.latest_timestamp = time.time()
                    self.frame_count += 1

                return color_frame, depth_frame

            except Exception as e:
                print(f"相机 {self.camera_id} 捕获错误: {e}")
                return None, None

    def get_cached_frame(self):
        """获取缓存的最后一帧"""
        with self.lock:
            return self.latest_color, self.latest_depth

    def get_pointcloud(self):
        """获取当前点云"""
        if self.latest_depth is None:
            return None

        try:
            # 转换为点云
            import open3d as o3d
            from utils import MultiCameraUtils

            depth_scale = MultiCameraConfig.CAPTURE_CONFIG['depth_scale']
            max_depth = MultiCameraConfig.CAPTURE_CONFIG['depth_max']

            points, valid_mask = MultiCameraUtils.depth_to_pointcloud(
                self.latest_depth, MultiCameraConfig.get_intrinsic_matrix(),
                depth_scale, max_depth
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

            if self.color_camera:
                try:
                    self.color_camera.release()
                except:
                    pass
                self.color_camera = None

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

    def __init__(self, driver_path=""):
        self.driver_path = driver_path
        self.cameras = {}
        self.openni_initialized = False
        self.running = False

        # 电压优化参数
        self.config = MultiCameraConfig.VOLTAGE_OPTIMIZATION
        self.rotation_index = 0
        self.last_rotation_time = 0
        self.active_combo = []

        # 统计信息
        self.stats = {
            'frames_captured': [0, 0, 0],
            'activations': [0, 0, 0],
            'rotation_count': 0,
            'start_time': 0,
        }

        # 线程
        self.capture_thread = None
        self.capture_queue = queue.Queue(maxsize=10)

    def initialize(self):
        """初始化系统"""
        print("=" * 60)
        print("电压优化的多相机系统")
        print(f"模式: 两两交替 (最大{self.config['max_active_cameras']}个同时激活)")
        print("=" * 60)

        if not OPENNI2_AVAILABLE:
            print("❌ OpenNI2不可用")
            return False

        # 初始化OpenNI2
        try:
            if self.driver_path and os.path.exists(self.driver_path):
                openni2.initialize(self.driver_path)
            else:
                openni2.initialize()

            self.openni_initialized = True
            print("✅ OpenNI2初始化成功")
        except Exception as e:
            print(f"❌ OpenNI2初始化失败: {e}")
            return False

        # 初始化所有相机
        print(f"\n初始化 {MultiCameraConfig.NUM_CAMERAS} 个相机...")

        for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
            camera = VoltageOptimizedCamera(cam_id, self.driver_path)
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

    def rotate_active_combo(self):
        """轮换激活的组合"""
        old_combo = self.active_combo.copy()

        # 停用旧的组合
        for cam_id in old_combo:
            if cam_id in self.cameras:
                self.cameras[cam_id].deactivate()

        # 选择新的组合
        combo_list = self.config['active_combinations']
        self.active_combo = combo_list[self.rotation_index]

        # 激活新的组合
        print(f"\n🔄 切换到组合 {self.active_combo} (轮换 {self.rotation_index + 1}/{len(combo_list)})")

        # 预热延时
        time.sleep(self.config['warmup_delay'])

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

                # 如果有数据，放入队列
                if frames:
                    try:
                        self.capture_queue.put_nowait((frames, current_time))
                    except queue.Full:
                        # 队列满，丢弃最旧的数据
                        try:
                            self.capture_queue.get_nowait()
                            self.capture_queue.put_nowait((frames, current_time))
                        except:
                            pass

                # 小延时，控制帧率
                time.sleep(1.0 / MultiCameraConfig.CAPTURE_CONFIG['fps'])

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

        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        # 停用所有相机
        for cam_id in self.cameras:
            self.cameras[cam_id].deactivate()

        print("采集停止")

    def get_frames(self, timeout=0.1):
        """获取最新的帧数据"""
        try:
            return self.capture_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def get_all_cached_frames(self):
        """获取所有相机的缓存帧（包括非激活的）"""
        frames = {}

        for cam_id, camera in self.cameras.items():
            color, depth = camera.get_cached_frame()
            if color is not None or depth is not None:
                frames[cam_id] = (color, depth)

        return frames

    def get_statistics(self):
        """获取统计信息"""
        current_time = time.time()
        elapsed = current_time - self.stats['start_time']

        stats = {
            'elapsed_time': elapsed,
            'total_frames': sum(self.stats['frames_captured']),
            'fps_per_camera': [
                self.stats['frames_captured'][i] / elapsed if elapsed > 0 else 0
                for i in range(len(self.stats['frames_captured']))
            ],
            'average_fps': sum(self.stats['frames_captured']) / elapsed if elapsed > 0 else 0,
            'rotation_count': self.stats['rotation_count'],
            'activations': self.stats['activations'],
            'active_combo': self.active_combo,
            'next_rotation_in': max(0, self.config['rotation_interval'] - (current_time - self.last_rotation_time)),
        }

        return stats

    def visualize_alternating_view(self, window_name="电压优化相机视图"):
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
                    status_color = (0, 100, 255) if cam_id in self.active_combo else (100, 100, 100)
                    status_text = "ACTIVE" if cam_id in self.active_combo else "INACTIVE"

                    cv2.putText(display, f"Cam{cam_id} [{status_text}]", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    if cam_id in frames:
                        color, depth = frames[cam_id]

                        if color is not None:
                            color_resized = cv2.resize(color, (400, 300))
                            display = color_resized

                            # 在图像上叠加帧数
                            frame_count = self.stats['frames_captured'][cam_id]
                            cv2.putText(display, f"Frames: {frame_count}", (10, 280),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # 显示深度信息
                            if depth is not None:
                                valid_pixels = np.sum(depth > 0)
                                cv2.putText(display, f"Depth: {valid_pixels}", (10, 250),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    else:
                        # 无数据
                        cv2.putText(display, "No Data", (150, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

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
                                display3_padded[:, padding:padding+display3.shape[1]] = display3
                                display3 = display3_padded

                            combined = np.vstack([row1, display3])
                        else:
                            combined = row1

                        # 添加统计信息
                        info_height = 60
                        info_panel = np.zeros((info_height, combined.shape[1], 3), dtype=np.uint8)

                        fps_text = f"平均FPS: {stats['average_fps']:.1f} | "
                        fps_text += f"激活组合: {self.active_combo} | "
                        fps_text += f"轮换计数: {stats['rotation_count']}"

                        cv2.putText(info_panel, fps_text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        next_rotation = stats['next_rotation_in']
                        cv2.putText(info_panel, f"下次轮换: {next_rotation:.1f}秒", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        combined_with_info = np.vstack([combined, info_panel])
                        cv2.imshow(window_name, combined_with_info)

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


def main():
    """测试函数"""
    import sys

    print("电压优化的多相机系统测试")
    print("=" * 60)

    # 创建系统
    system = VoltageOptimizedSystem()

    try:
        # 初始化
        if not system.initialize():
            print("初始化失败")
            return

        # 可视化
        system.visualize_alternating_view()

        # 显示最终统计
        stats = system.get_statistics()
        print("\n最终统计:")
        print(f"  总时间: {stats['elapsed_time']:.1f}秒")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  平均FPS: {stats['average_fps']:.1f}")
        print(f"  轮换次数: {stats['rotation_count']}")

        for i in range(MultiCameraConfig.NUM_CAMERAS):
            print(f"  相机{i}: {stats['activations'][i]}次激活, {stats['frames_captured'][i]}帧")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
    finally:
        system.cleanup()

    print("\n程序结束")


if __name__ == "__main__":
    main()