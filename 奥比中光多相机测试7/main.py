"""
三相机联动重建系统 - 高质量STL生成版（320x240分辨率）
结合三个相机的优势，生成高质量STL模型
"""

import sys
import os
import cv2
import numpy as np
import open3d as o3d
import json
import time
import traceback
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
    SDK_AVAILABLE = True
except ImportError:
    print("⚠️ SDK接口层不可用，将使用模拟模式")
    SDK_AVAILABLE = False


class TriCameraReconstructor:
    """三相机高质量STL重建系统 - 320x240分辨率版"""

    def __init__(self):
        # 相机配置
        self.camera_count = 3
        self.cameras = []

        # 相机物理位置映射
        self.camera_positions = {
            'left': None,
            'center': None,
            'right': None
        }

        # 相机变换矩阵（物理位置）
        self.camera_transforms = {
            'left': np.eye(4),
            'center': np.eye(4),
            'right': np.eye(4)
        }

        # 相机间距 (单位: 米)
        self.camera_spacing = 0.15  # 15cm

        # 优化的重建参数 - 基于320x240分辨率
        self.params = {
            # 相机参数 - 320x240分辨率
            'resolution': (320, 240),  # 低分辨率但更稳定
            'fps': 15,
            'depth_scale': 0.001,
            'depth_min': 0.3,   # 30cm
            'depth_max': 2.5,   # 2.5米

            # 采集参数 - 增加帧数
            'capture_frames': 30,      # 30帧
            'capture_interval': 0.2,   # 减小间隔到0.2秒
            'depth_timeout': 2000,
            'color_retries': 5,

            # TSDF融合参数 - 适应320x240
            'voxel_length': 0.006,     # 6mm体素，适合320x240
            'sdf_trunc': 0.04,
            'volume_size': 3.0,

            # 相机内参 (320x240) - 调整焦距和中心点
            'intrinsic_matrix': [
                [262.5, 0, 159.75],    # fx=262.5, cx=159.75
                [0, 262.5, 119.75],    # fy=262.5, cy=119.75
                [0, 0, 1]
            ],

            # 颜色增强参数
            'color_enhance': True,
            'color_contrast': 1.2,
            'color_brightness': 10,

            # 网格优化参数
            'mesh_simplify': True,
            'target_vertices': 30000,  # 目标顶点数（比640x480少）
            'smooth_mesh': True,
            'smooth_iterations': 2,
            'remove_outliers': True,
            'outlier_nb_neighbors': 15,
            'outlier_std_ratio': 2.0,

            # 导出设置 - 重点生成STL
            'export_ply': True,
            'export_stl': True,
            'export_obj': True,
            'stl_binary': True,  # 二进制STL，文件更小

            # 输出设置
            'output_dir': f"tri_camera_stl_320x240_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'save_images': True,
            'save_intermediate': True,
            'save_every_n_frames': 5,  # 每5帧保存中间结果
        }

        # 数据存储
        self.captured_data = {
            'left': {'depth': [], 'color': [], 'timestamps': []},
            'center': {'depth': [], 'color': [], 'timestamps': []},
            'right': {'depth': [], 'color': [], 'timestamps': []}
        }

        # 点云数据
        self.point_clouds = {}

        # TSDF体积
        self.tsdf_volume = None

        # 重建结果
        self.fused_mesh = None
        self.fused_pointcloud = None

        # 状态跟踪
        self.frame_counter = 0
        self.iteration = 0
        self.mesh_count = 0
        self.is_capturing = False

        print("🚀 三相机高质量STL重建系统初始化完成")
        print(f"   分辨率: {self.params['resolution'][0]}x{self.params['resolution'][1]}")
        print(f"   相机数量: {self.camera_count}")
        print(f"   相机间距: {self.camera_spacing*100:.0f}cm")
        print(f"   采集帧数: {self.params['capture_frames']}")
        print(f"   体素大小: {self.params['voxel_length']*1000:.1f}mm")

    def setup_cameras(self) -> bool:
        """设置并打开三个相机 - 320x240分辨率优化版"""
        print("\n" + "="*60)
        print("设置三个相机 (320x240分辨率)")
        print("="*60)

        if not SDK_AVAILABLE:
            print("⚠️ SDK不可用，使用模拟相机模式")
            return self._setup_simulated_cameras()

        try:
            self.cameras = []
            successful_cameras = 0

            print("⚠️ 注意: 使用320x240分辨率以减少USB带宽压力")
            print("     三个相机同时运行更稳定")

            for i in range(self.camera_count):
                print(f"\n设置相机 #{i}...")

                try:
                    # 为每个相机创建独立的SDK实例
                    sdk = OrbbecCameraSDK(get_default_sdk_path())

                    if not sdk.initialize():
                        print(f"  ❌ 相机 #{i} SDK初始化失败")
                        continue

                    # 添加延迟避免USB冲突
                    time.sleep(0.3)

                    # 获取设备列表并尝试连接
                    devices = sdk.get_device_list()
                    if not devices:
                        print(f"  ⚠️  未检测到相机 #{i}，跳过")
                        sdk.cleanup()
                        continue

                    # 尝试打开设备（使用第一个可用设备）
                    device_index = min(i, len(devices)-1)
                    if not sdk.open_device(device_index=device_index):
                        print(f"  ❌ 相机 #{i} 打开设备失败")
                        sdk.cleanup()
                        continue

                    # 创建深度流
                    if not sdk.create_stream(sdk.ONI_SENSOR_DEPTH):
                        print(f"  ❌ 相机 #{i} 创建深度流失败")
                        sdk.close_device()
                        sdk.cleanup()
                        continue

                    # 启动深度流
                    if not sdk.start_stream():
                        print(f"  ❌ 相机 #{i} 启动深度流失败")
                        sdk.destroy_stream()
                        sdk.close_device()
                        sdk.cleanup()
                        continue

                    # 设置彩色摄像头
                    cv_camera = None
                    cv_camera_initialized = False

                    print(f"  设置彩色摄像头 (320x240)...")

                    # 尝试不同的摄像头索引
                    camera_indices = [i, i*2, i*2+1, 0, 1, 2, 3]

                    for cam_idx in camera_indices:
                        try:
                            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                            if cap.isOpened():
                                # 设置320x240分辨率
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                                cap.set(cv2.CAP_PROP_FPS, 15)

                                # 测试读取
                                success_count = 0
                                for _ in range(2):
                                    ret, frame = cap.read()
                                    if ret and frame is not None:
                                        success_count += 1
                                    time.sleep(0.05)

                                if success_count >= 1:
                                    cv_camera = cap
                                    cv_camera_initialized = True
                                    print(f"  ✅ 使用彩色摄像头索引 {cam_idx}")
                                    break
                                else:
                                    cap.release()
                        except Exception as e:
                            if cap:
                                cap.release()
                            continue

                    if not cv_camera_initialized:
                        print(f"  ⚠️  相机 #{i} 未找到可用的彩色摄像头")

                    # 保存相机信息
                    camera_info = {
                        'index': i,
                        'sdk': sdk,
                        'cv_camera': cv_camera,
                        'cv_initialized': cv_camera_initialized,
                        'device_id': i,
                        'is_simulated': False
                    }

                    self.cameras.append(camera_info)
                    successful_cameras += 1
                    print(f"  ✅ 相机 #{i} 设置成功 (320x240)")

                except Exception as e:
                    print(f"  ❌ 相机 #{i} 设置失败: {e}")

            if successful_cameras < 2:
                print(f"\n⚠️ 只有 {successful_cameras} 个相机可用，使用模拟模式")
                return self._setup_simulated_cameras()

            print(f"\n✅ 成功打开 {successful_cameras} 个相机 (320x240分辨率)")

            # 测试相机功能
            self._test_camera_functionality()

            return True

        except Exception as e:
            print(f"❌ 设置相机失败: {e}")
            traceback.print_exc()
            return self._setup_simulated_cameras()

    def _setup_simulated_cameras(self) -> bool:
        """设置模拟相机"""
        print("\n使用模拟相机模式进行测试 (320x240)")
        self.cameras = []

        for i in range(min(3, self.camera_count)):
            camera_info = {
                'index': i,
                'sdk': None,
                'cv_camera': None,
                'cv_initialized': False,
                'is_simulated': True,
                'device_id': i
            }
            self.cameras.append(camera_info)

        print(f"✅ 设置 {len(self.cameras)} 个模拟相机")
        return True

    def _test_camera_functionality(self):
        """测试相机功能"""
        print("\n测试相机功能 (320x240)...")

        for i, camera_info in enumerate(self.cameras):
            if camera_info.get('is_simulated', False):
                print(f"  相机 #{i}: 模拟模式")
                continue

            if camera_info['sdk']:
                print(f"  相机 #{i}: 测试深度采集...")
                result = camera_info['sdk'].capture_depth_frame(timeout=1000)
                if result:
                    depth_array, _ = result
                    # 检查并调整到320x240
                    if depth_array.shape != (240, 320):
                        depth_array = cv2.resize(depth_array, (320, 240), interpolation=cv2.INTER_NEAREST)
                    print(f"    深度图: {depth_array.shape}, 范围: {np.min(depth_array)}-{np.max(depth_array)}mm")
                else:
                    print(f"    ⚠️  深度采集失败")

            if camera_info.get('cv_initialized', False):
                print(f"  相机 #{i}: 测试颜色采集...")
                ret, frame = camera_info['cv_camera'].read()
                if ret and frame is not None:
                    # 调整到320x240
                    if frame.shape[1] != 320 or frame.shape[0] != 240:
                        frame = cv2.resize(frame, (320, 240))
                    print(f"    颜色图: {frame.shape}")
                else:
                    print(f"    ⚠️  颜色采集失败")

    def determine_camera_positions(self) -> bool:
        """确定相机左右位置 - 简单方法"""
        print("\n" + "="*60)
        print("确定相机位置")
        print("="*60)

        if len(self.cameras) < 2:
            print("❌ 相机数量不足")
            return False

        print("使用预设位置映射（按连接顺序）:")
        print("  第一个相机 → 左相机")
        print("  第二个相机 → 中心相机")
        print("  第三个相机 → 右相机")

        # 按索引分配位置
        positions = ['left', 'center', 'right']
        for i, pos in enumerate(positions):
            if i < len(self.cameras):
                self.camera_positions[pos] = i
                print(f"  {pos.upper()}: 相机 #{i}")

        # 设置相机变换矩阵（基于物理间距）
        self._setup_camera_transforms()

        return True

    def _setup_camera_transforms(self):
        """设置相机变换矩阵"""
        print("\n设置相机变换矩阵...")

        # 以中心相机为世界坐标系原点
        self.camera_transforms['center'] = np.eye(4)

        # 左相机在中心相机左侧
        left_transform = np.eye(4)
        left_transform[0, 3] = -self.camera_spacing  # X轴负方向
        self.camera_transforms['left'] = left_transform

        # 右相机在中心相机右侧
        right_transform = np.eye(4)
        right_transform[0, 3] = self.camera_spacing  # X轴正方向
        self.camera_transforms['right'] = right_transform

        print(f"  左相机: 平移({-self.camera_spacing:.3f}, 0, 0)米")
        print(f"  中心相机: 原点")
        print(f"  右相机: 平移({self.camera_spacing:.3f}, 0, 0)米")

    def capture_synchronized_data(self) -> bool:
        """同步采集三个相机的数据 - 320x240版"""
        print("\n" + "="*60)
        print("同步采集数据 (320x240)")
        print("="*60)

        if not self.camera_positions:
            print("❌ 尚未确定相机位置")
            return False

        frames_per_camera = self.params['capture_frames']
        interval = self.params['capture_interval']

        print(f"每个相机采集 {frames_per_camera} 帧")
        print(f"采集间隔: {interval:.2f} 秒")
        print(f"预计总时间: {frames_per_camera * interval:.1f} 秒")

        # 清空之前的数据
        for pos in ['left', 'center', 'right']:
            self.captured_data[pos] = {'depth': [], 'color': [], 'timestamps': []}

        # 创建输出目录
        output_dir = self.params['output_dir']
        images_dir = os.path.join(output_dir, "captured_images")
        os.makedirs(images_dir, exist_ok=True)

        self.is_capturing = True
        successful_frames = 0

        try:
            for frame_idx in range(frames_per_camera):
                print(f"\n📸 采集第 {frame_idx+1}/{frames_per_camera} 帧")

                frame_timestamp = time.time()
                frame_data = {}

                # 为每个相机采集数据
                for pos, cam_idx in self.camera_positions.items():
                    if pos not in ['left', 'center', 'right']:
                        continue

                    if cam_idx >= len(self.cameras):
                        print(f"  ⚠️  位置 {pos} 的相机索引无效")
                        continue

                    camera_info = self.cameras[cam_idx]
                    depth_data = None
                    color_data = None

                    # 采集深度数据
                    if not camera_info.get('is_simulated', False) and camera_info['sdk']:
                        try:
                            result = camera_info['sdk'].capture_depth_frame(
                                timeout=self.params['depth_timeout']
                            )

                            if result:
                                depth_array, _ = result

                                # 强制调整到320x240
                                if depth_array.shape != (240, 320):
                                    depth_array = cv2.resize(depth_array, (320, 240),
                                                           interpolation=cv2.INTER_NEAREST)

                                # 转换为米并处理
                                depth_data = depth_array.astype(np.float32) * self.params['depth_scale']

                                # 深度裁剪
                                depth_data[depth_data < self.params['depth_min']] = 0
                                depth_data[depth_data > self.params['depth_max']] = 0

                                # 检查有效点
                                valid_points = np.sum(depth_data > 0)
                                if valid_points < 200:  # 320x240分辨率要求更低
                                    print(f"  ⚠️  {pos}: 有效点太少 ({valid_points})")
                                    depth_data = None

                        except Exception as e:
                            print(f"  ❌ {pos}: 深度采集异常: {e}")
                            depth_data = None
                    else:
                        # 生成模拟深度数据
                        depth_data = self._generate_simulated_depth(pos, frame_idx)

                    # 采集颜色数据
                    if camera_info.get('cv_initialized', False):
                        retries = self.params['color_retries']
                        for attempt in range(retries):
                            ret, color_frame = camera_info['cv_camera'].read()
                            if ret and color_frame is not None:
                                # 强制调整到320x240
                                if color_frame.shape[1] != 320 or color_frame.shape[0] != 240:
                                    color_frame = cv2.resize(color_frame, (320, 240))

                                # 颜色增强
                                if self.params['color_enhance'] and np.mean(color_frame) > 10:
                                    color_data = self.enhance_color(color_frame.copy())
                                else:
                                    color_data = color_frame.copy()
                                break
                            time.sleep(0.05)

                    if depth_data is not None:
                        frame_data[pos] = {
                            'depth': depth_data,
                            'color': color_data,
                            'timestamp': frame_timestamp
                        }

                        depth_stats = f"深度: {depth_data.shape}"
                        if color_data is not None:
                            depth_stats += f", 颜色: {color_data.shape}"
                        print(f"  ✅ {pos}: {depth_stats}")
                    else:
                        print(f"  ❌ {pos}: 采集失败")

                # 保存有效帧数据
                if len(frame_data) >= 2:  # 至少两个相机成功
                    for pos, data in frame_data.items():
                        self.captured_data[pos]['depth'].append(data['depth'])
                        self.captured_data[pos]['color'].append(data['color'])
                        self.captured_data[pos]['timestamps'].append(data['timestamp'])

                    successful_frames += 1

                    # 定期保存图像
                    if self.params['save_images'] and frame_idx % 5 == 0:
                        for pos, data in frame_data.items():
                            if data['depth'] is not None:
                                depth_colored = self.depth_to_colormap(data['depth'])
                                depth_path = os.path.join(images_dir, f"{pos}_depth_{frame_idx:03d}.png")
                                cv2.imwrite(depth_path, depth_colored)

                            if data['color'] is not None:
                                color_path = os.path.join(images_dir, f"{pos}_color_{frame_idx:03d}.png")
                                cv2.imwrite(color_path, data['color'])

                    print(f"  ✅ 帧 {frame_idx+1} 成功 ({len(frame_data)}/3 个相机)")
                else:
                    print(f"  ⚠️  帧 {frame_idx+1} 失败，跳过")

                # 帧间延迟
                if frame_idx < frames_per_camera - 1:
                    elapsed = time.time() - frame_timestamp
                    sleep_time = max(0, interval - elapsed)
                    time.sleep(sleep_time)

            print(f"\n✅ 数据采集完成")
            print(f"   成功采集 {successful_frames}/{frames_per_camera} 帧")

            # 如果成功帧数太少，生成模拟数据
            if successful_frames < 5:
                print("⚠️ 成功帧数不足，生成模拟数据补充...")
                self._generate_simulated_data(successful_frames)

            # 保存采集统计
            self._save_capture_stats()

            return successful_frames >= 3  # 至少需要3帧

        except KeyboardInterrupt:
            print("\n🔴 用户中断采集")
            return False
        except Exception as e:
            print(f"\n❌ 采集过程中出错: {e}")
            traceback.print_exc()
            return False
        finally:
            self.is_capturing = False

    def _generate_simulated_depth(self, position: str, frame_idx: int) -> np.ndarray:
        """生成模拟深度数据 - 320x240分辨率"""
        height, width = 240, 320  # 固定320x240

        # 创建基础深度平面
        depth = np.ones((height, width), dtype=np.float32) * 1.0  # 1米

        # 根据位置创建不同的物体
        if position == 'left':
            center_x, center_y = int(width * 0.7), int(height * 0.5)
        elif position == 'right':
            center_x, center_y = int(width * 0.3), int(height * 0.5)
        else:  # center
            center_x, center_y = int(width * 0.5), int(height * 0.5)

        # 创建一个椭球体
        radius_x, radius_y = int(width * 0.15), int(height * 0.15)

        # 使用矢量化操作提高性能
        y_coords, x_coords = np.indices((height, width))
        dx = (x_coords - center_x) / radius_x
        dy = (y_coords - center_y) / radius_y
        distance = np.sqrt(dx*dx + dy*dy)

        # 创建椭球形状
        mask = distance < 1.0
        z_base = 0.5 + 0.3 * (1 - distance[mask])
        z_wave = 0.05 * np.sin(frame_idx * 0.1 + x_coords[mask] * 0.01) * np.cos(frame_idx * 0.1 + y_coords[mask] * 0.01)
        depth[mask] = z_base + z_wave

        # 添加随机噪声
        noise = np.random.normal(0, 0.002, (height, width))
        depth += noise

        # 深度裁剪
        depth[depth < self.params['depth_min']] = 0
        depth[depth > self.params['depth_max']] = 0

        return depth

    def _generate_simulated_data(self, existing_frames: int = 0):
        """生成模拟数据补充"""
        frames_needed = self.params['capture_frames'] - existing_frames

        if frames_needed <= 0:
            return

        print(f"生成 {frames_needed} 帧模拟数据...")

        for pos in ['left', 'center', 'right']:
            # 保留已有数据
            if existing_frames > 0:
                existing_data = {
                    'depth': self.captured_data[pos]['depth'][:existing_frames],
                    'color': self.captured_data[pos]['color'][:existing_frames],
                    'timestamps': self.captured_data[pos]['timestamps'][:existing_frames]
                }
            else:
                existing_data = {'depth': [], 'color': [], 'timestamps': []}

            # 清空并重新填充
            self.captured_data[pos] = {'depth': [], 'color': [], 'timestamps': []}

            # 添加已有数据
            for i in range(existing_frames):
                if i < len(existing_data['depth']):
                    self.captured_data[pos]['depth'].append(existing_data['depth'][i])
                if i < len(existing_data['color']):
                    self.captured_data[pos]['color'].append(existing_data['color'][i])
                if i < len(existing_data['timestamps']):
                    self.captured_data[pos]['timestamps'].append(existing_data['timestamps'][i])

            # 生成补充数据
            for i in range(existing_frames, self.params['capture_frames']):
                depth = self._generate_simulated_depth(pos, i)
                color = self._generate_simulated_color(pos, i)

                self.captured_data[pos]['depth'].append(depth)
                self.captured_data[pos]['color'].append(color)
                self.captured_data[pos]['timestamps'].append(time.time())

    def _generate_simulated_color(self, position: str, frame_idx: int) -> np.ndarray:
        """生成模拟颜色数据 - 320x240分辨率"""
        height, width = 240, 320
        color = np.zeros((height, width, 3), dtype=np.uint8)

        # 根据位置设置不同背景色
        if position == 'left':
            base_color = [100, 150, 200]  # 蓝色调
        elif position == 'center':
            base_color = [150, 200, 100]  # 绿色调
        else:  # right
            base_color = [200, 100, 150]  # 红色调

        # 创建渐变背景
        for y in range(height):
            for x in range(width):
                dx = x - width // 2
                dy = y - height // 2
                distance = np.sqrt(dx*dx + dy*dy) / max(width, height) * 2
                brightness = 0.4 + 0.6 * max(0, 1 - distance)
                color[y, x] = [int(c * brightness) for c in base_color]

        return color

    def enhance_color(self, image):
        """增强颜色质量"""
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

        except Exception:
            return image

    def depth_to_colormap(self, depth):
        """深度图转伪彩色"""
        if depth is None:
            return None

        try:
            # 将深度值转换为毫米
            depth_mm = depth * 1000

            # 创建掩码（有效深度）
            valid_mask = depth_mm > 0

            if not np.any(valid_mask):
                return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

            # 归一化到0-255范围
            depth_min = np.min(depth_mm[valid_mask])
            depth_max = np.max(depth_mm[valid_mask])

            if depth_max - depth_min < 1:
                return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

            depth_normalized = (depth_mm - depth_min) / (depth_max - depth_min) * 255
            depth_normalized = depth_normalized.astype(np.uint8)

            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # 将无效深度设为黑色
            depth_colored[~valid_mask] = [0, 0, 0]

            return depth_colored

        except Exception:
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    def _save_capture_stats(self):
        """保存采集统计信息"""
        stats_file = os.path.join(self.params['output_dir'], "capture_statistics.json")

        stats = {
            'timestamp': datetime.now().isoformat(),
            'resolution': self.params['resolution'],
            'camera_positions': self.camera_positions,
            'camera_count': len(self.cameras),
            'camera_transforms': {k: v.tolist() for k, v in self.camera_transforms.items()},
            'camera_spacing': self.camera_spacing,
            'capture_params': self.params,
            'frame_counts': {}
        }

        for pos in ['left', 'center', 'right']:
            stats['frame_counts'][pos] = len(self.captured_data[pos]['depth'])

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"📊 采集统计已保存: {stats_file}")

    def create_individual_pointclouds(self) -> bool:
        """为每个相机创建点云 - 320x240版"""
        print("\n" + "="*60)
        print("创建单个相机点云 (320x240)")
        print("="*60)

        # 创建相机内参 - 320x240
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            320, 240,
            self.params['intrinsic_matrix'][0][0],  # fx = 262.5
            self.params['intrinsic_matrix'][1][1],  # fy = 262.5
            self.params['intrinsic_matrix'][0][2],  # cx = 159.75
            self.params['intrinsic_matrix'][1][2]   # cy = 119.75
        )

        self.point_clouds = {}
        pointcloud_created = False

        for pos in ['left', 'center', 'right']:
            if not self.captured_data[pos]['depth']:
                print(f"  ⚠️  位置 {pos} 没有深度数据")
                continue

            print(f"\n处理 {pos} 相机的点云...")

            combined_pcd = o3d.geometry.PointCloud()
            frame_count = len(self.captured_data[pos]['depth'])

            for i in range(frame_count):
                depth = self.captured_data[pos]['depth'][i]
                color = self.captured_data[pos]['color'][i]

                # 检查深度图有效性
                valid_points = np.sum(depth > 0)
                if valid_points < 50:  # 320x240要求更低
                    continue

                try:
                    # 创建深度图像
                    depth_image = o3d.geometry.Image((depth * 1000).astype(np.uint16))

                    # 创建颜色图像
                    if color is not None:
                        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        color_image = o3d.geometry.Image(color_rgb)
                    else:
                        color_image = o3d.geometry.Image(
                            np.full((240, 320, 3), 200, dtype=np.uint8)
                        )

                    # 创建RGBD图像
                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_image,
                        depth_image,
                        depth_scale=1000.0,
                        depth_trunc=self.params['depth_max'],
                        convert_rgb_to_intensity=False
                    )

                    # 创建点云
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image,
                        intrinsic
                    )

                    if len(pcd.points) > 0:
                        # 应用相机位姿变换
                        pcd.transform(self.camera_transforms[pos])
                        combined_pcd += pcd

                except Exception as e:
                    print(f"  帧 {i+1}: 点云创建失败 - {e}")

            if len(combined_pcd.points) == 0:
                print(f"  ❌ {pos}: 点云为空")
                continue

            # 下采样以减少数据量
            if len(combined_pcd.points) > 30000:  # 320x240数据量较小
                combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.015)

            # 估计法线
            try:
                combined_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                )
            except:
                pass

            self.point_clouds[pos] = combined_pcd
            pointcloud_created = True

            print(f"  ✅ {pos}: {len(combined_pcd.points)} 个点")

            # 保存单个相机点云
            if self.params['save_intermediate']:
                pcd_dir = os.path.join(self.params['output_dir'], "individual_pointclouds")
                os.makedirs(pcd_dir, exist_ok=True)

                pcd_filename = os.path.join(pcd_dir, f"{pos}_pointcloud.ply")
                try:
                    o3d.io.write_point_cloud(pcd_filename, combined_pcd)
                    print(f"    点云已保存: {os.path.basename(pcd_filename)}")
                except Exception as e:
                    print(f"    ❌ 保存点云失败: {e}")

        print(f"\n✅ 单个点云创建完成")
        return pointcloud_created

    def init_tsdf_volume(self) -> bool:
        """初始化TSDF体积 - 320x240版"""
        print("\n初始化TSDF融合体积 (320x240)...")

        try:
            # 创建支持颜色的TSDF体积
            self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.params['voxel_length'],
                sdf_trunc=self.params['sdf_trunc'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            print(f"✅ TSDF体积初始化完成")
            print(f"   体素大小: {self.params['voxel_length']*1000:.1f}mm (适合320x240)")
            print(f"   SDF截断: {self.params['sdf_trunc']}米")
            return True

        except Exception as e:
            print(f"❌ TSDF初始化失败: {e}")
            return False

    def fuse_data_tsdf(self) -> bool:
        """使用TSDF融合三个相机的数据 - 320x240版"""
        print("\n" + "="*60)
        print("TSDF融合三个相机数据 (320x240)")
        print("="*60)

        if self.tsdf_volume is None:
            print("❌ TSDF体积未初始化")
            return False

        # 创建相机内参 - 320x240
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            320, 240,
            self.params['intrinsic_matrix'][0][0],  # fx = 262.5
            self.params['intrinsic_matrix'][1][1],  # fy = 262.5
            self.params['intrinsic_matrix'][0][2],  # cx = 159.75
            self.params['intrinsic_matrix'][1][2]   # cy = 119.75
        )

        print("开始TSDF融合...")

        total_frames = 0
        start_time = time.time()

        # 按帧顺序融合，确保时间一致性
        min_frames = min(
            len(self.captured_data['left']['depth']),
            len(self.captured_data['center']['depth']),
            len(self.captured_data['right']['depth'])
        )

        if min_frames == 0:
            print("❌ 没有足够的数据进行融合")
            return False

        print(f"共有 {min_frames} 帧数据可用于融合")

        for frame_idx in range(min_frames):
            # 按顺序处理每个相机
            for pos in ['left', 'center', 'right']:
                depth = self.captured_data[pos]['depth'][frame_idx]
                color = self.captured_data[pos]['color'][frame_idx]

                if depth is None:
                    continue

                try:
                    # 创建深度图像
                    depth_image = o3d.geometry.Image((depth * 1000).astype(np.uint16))

                    # 创建颜色图像
                    if color is not None:
                        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        color_image = o3d.geometry.Image(color_rgb)
                    else:
                        color_image = o3d.geometry.Image(
                            np.full((240, 320, 3), 200, dtype=np.uint8)
                        )

                    # 创建RGBD图像
                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_image,
                        depth_image,
                        depth_scale=1000.0,
                        depth_trunc=self.params['depth_max'],
                        convert_rgb_to_intensity=False
                    )

                    # 获取相机位姿
                    camera_pose = self.camera_transforms[pos]

                    # 融合到TSDF体积
                    self.tsdf_volume.integrate(rgbd_image, intrinsic, np.linalg.inv(camera_pose))
                    total_frames += 1

                except Exception as e:
                    print(f"  帧 {frame_idx+1} ({pos}): 融合失败 - {e}")

            # 显示进度
            if (frame_idx + 1) % 5 == 0 or frame_idx == min_frames - 1:
                elapsed = time.time() - start_time
                progress = (frame_idx + 1) / min_frames * 100
                print(f"  进度: {progress:.1f}% ({frame_idx+1}/{min_frames}), 耗时: {elapsed:.1f}秒")

                # 定期保存中间结果
                if self.params['save_intermediate'] and (frame_idx + 1) % self.params['save_every_n_frames'] == 0:
                    self._save_intermediate_results(frame_idx + 1)

        print(f"\n✅ TSDF融合完成")
        print(f"   总共融合 {total_frames} 帧数据")
        print(f"   总耗时: {time.time() - start_time:.1f} 秒")

        return total_frames > 0

    def _save_intermediate_results(self, frame_idx: int):
        """保存中间融合结果"""
        if self.tsdf_volume is None:
            return

        try:
            # 提取当前TSDF状态的网格
            mesh = self.tsdf_volume.extract_triangle_mesh()
            if mesh is None or len(mesh.vertices) == 0:
                return

            # 创建中间结果目录
            intermediate_dir = os.path.join(self.params['output_dir'], "intermediate_fusion")
            os.makedirs(intermediate_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%H%M%S")

            # 保存PLY格式
            ply_filename = os.path.join(intermediate_dir, f"fusion_f{frame_idx:03d}_{timestamp}.ply")
            o3d.io.write_triangle_mesh(ply_filename, mesh, write_ascii=False)

            # 保存STL格式
            stl_filename = os.path.join(intermediate_dir, f"fusion_f{frame_idx:03d}_{timestamp}.stl")
            o3d.io.write_triangle_mesh(stl_filename, mesh)

            print(f"    中间结果已保存: 帧 {frame_idx}")

        except Exception as e:
            print(f"    ⚠️ 保存中间结果失败: {e}")

    def extract_fused_mesh(self) -> bool:
        """从TSDF体积提取融合网格 - 320x240版"""
        print("\n" + "="*60)
        print("提取融合网格 (320x240)")
        print("="*60)

        if self.tsdf_volume is None:
            print("❌ TSDF体积为空")
            return False

        try:
            print("从TSDF体积提取网格...")
            mesh = self.tsdf_volume.extract_triangle_mesh()

            if mesh is None or len(mesh.vertices) == 0:
                print("❌ 提取的网格为空")
                return False

            print(f"  原始网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面片")

            # 镜像修正（相机坐标系）
            vertices = np.asarray(mesh.vertices)
            vertices[:, 0] = -vertices[:, 0]  # 左右镜像
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            # 确保有顶点颜色
            if not mesh.has_vertex_colors():
                print("  ⚠️ 网格没有颜色，添加默认颜色")
                mesh.paint_uniform_color([0.7, 0.7, 0.7])

            # 计算法线
            mesh.compute_vertex_normals()

            # 网格优化（针对320x240调整参数）
            self.fused_mesh = self._optimize_mesh(mesh)

            # 提取点云
            print("从网格提取点云...")
            self.fused_pointcloud = self.fused_mesh.sample_points_uniformly(number_of_points=50000)

            print(f"✅ 融合网格提取完成")
            print(f"   优化后: {len(self.fused_mesh.vertices)} 顶点, {len(self.fused_mesh.triangles)} 面片")
            print(f"   点云: {len(self.fused_pointcloud.points)} 个点")

            return True

        except Exception as e:
            print(f"❌ 提取网格失败: {e}")
            traceback.print_exc()
            return False

    def _optimize_mesh(self, mesh) -> o3d.geometry.TriangleMesh:
        """优化网格质量 - 320x240版"""
        print("优化网格质量 (320x240)...")

        optimized_mesh = mesh

        try:
            # 1. 去除离群点
            if self.params['remove_outliers'] and len(optimized_mesh.vertices) > 100:
                print("  去除离群点...")
                cl, ind = optimized_mesh.remove_statistical_outlier(
                    nb_neighbors=self.params['outlier_nb_neighbors'],
                    std_ratio=self.params['outlier_std_ratio']
                )
                optimized_mesh = optimized_mesh.select_by_index(ind)

            # 2. 网格简化 (针对320x240分辨率)
            if self.params['mesh_simplify'] and len(optimized_mesh.triangles) > self.params['target_vertices'] * 1.5:
                print(f"  简化网格 ({len(optimized_mesh.triangles)} -> {self.params['target_vertices']})...")
                target_faces = int(self.params['target_vertices'] * 1.5)
                optimized_mesh = optimized_mesh.simplify_quadric_decimation(target_faces)

            # 3. 网格平滑
            if self.params['smooth_mesh'] and len(optimized_mesh.vertices) > 50:
                print(f"  平滑网格 ({self.params['smooth_iterations']}次迭代)...")
                optimized_mesh = optimized_mesh.filter_smooth_simple(
                    number_of_iterations=self.params['smooth_iterations']
                )

            # 4. 重新计算法线
            optimized_mesh.compute_vertex_normals()

            print(f"  优化完成: {len(optimized_mesh.vertices)} 顶点, {len(optimized_mesh.triangles)} 面片")

        except Exception as e:
            print(f"  ⚠️ 网格优化失败: {e}")
            return mesh

        return optimized_mesh

    def save_final_results(self):
        """保存最终结果 - 重点生成STL"""
        print("\n" + "="*60)
        print("保存最终结果 (320x240)")
        print("="*60)

        # 创建输出目录
        output_dir = self.params['output_dir']
        final_dir = os.path.join(output_dir, "final_results")
        os.makedirs(final_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = []

        # 保存融合点云
        if self.fused_pointcloud is not None:
            try:
                pcd_filename = os.path.join(final_dir, f"fused_pointcloud_{timestamp}.ply")
                o3d.io.write_point_cloud(pcd_filename, self.fused_pointcloud)
                saved_files.append(('PLY点云', pcd_filename))
                print(f"✅ 融合点云已保存: {os.path.basename(pcd_filename)}")
            except Exception as e:
                print(f"❌ 保存点云失败: {e}")

        # 保存融合网格（重点生成STL）
        if self.fused_mesh is not None:
            # 1. 保存PLY格式（带颜色）
            if self.params['export_ply']:
                try:
                    ply_filename = os.path.join(final_dir, f"fused_mesh_{timestamp}.ply")
                    o3d.io.write_triangle_mesh(
                        ply_filename,
                        self.fused_mesh,
                        write_ascii=False,
                        compressed=True
                    )
                    saved_files.append(('PLY网格', ply_filename))
                    print(f"✅ PLY网格已保存: {os.path.basename(ply_filename)}")
                except Exception as e:
                    print(f"❌ 保存PLY失败: {e}")

            # 2. 保存STL格式（重点） - 320x240分辨率STL
            if self.params['export_stl']:
                try:
                    stl_filename = os.path.join(final_dir, f"fused_mesh_320x240_{timestamp}.stl")
                    o3d.io.write_triangle_mesh(
                        stl_filename,
                        self.fused_mesh,
                        write_ascii=not self.params['stl_binary']
                    )
                    saved_files.append(('STL网格', stl_filename))
                    print(f"✅ STL网格已保存: {os.path.basename(stl_filename)}")

                    # 显示STL文件信息
                    file_size = os.path.getsize(stl_filename) / (1024 * 1024)  # MB
                    print(f"   STL文件大小: {file_size:.2f} MB")
                    print(f"   分辨率: 320x240")

                    # 生成一个简化的STL版本（用于快速预览）
                    simple_stl_filename = os.path.join(final_dir, f"fused_mesh_simple_{timestamp}.stl")
                    try:
                        if len(self.fused_mesh.triangles) > 10000:
                            simple_mesh = self.fused_mesh.simplify_quadric_decimation(10000)
                            o3d.io.write_triangle_mesh(simple_stl_filename, simple_mesh)
                            print(f"✅ 简化STL已保存: {os.path.basename(simple_stl_filename)}")
                    except:
                        pass

                except Exception as e:
                    print(f"❌ 保存STL失败: {e}")
                    traceback.print_exc()

            # 3. 保存OBJ格式
            if self.params['export_obj']:
                try:
                    obj_filename = os.path.join(final_dir, f"fused_mesh_{timestamp}.obj")
                    o3d.io.write_triangle_mesh(
                        obj_filename,
                        self.fused_mesh,
                        write_vertex_normals=True,
                        write_vertex_colors=True
                    )
                    saved_files.append(('OBJ网格', obj_filename))
                    print(f"✅ OBJ网格已保存: {os.path.basename(obj_filename)}")
                except Exception as e:
                    print(f"❌ 保存OBJ失败: {e}")

        # 保存可视化图像
        self._save_visualizations(final_dir, timestamp)

        return saved_files

    def _save_visualizations(self, output_dir: str, timestamp: str):
        """保存可视化图像"""
        print("\n生成可视化图像 (320x240)...")

        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        try:
            # 创建可视化窗口（不显示）
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=800, height=600, visible=False)

            if self.fused_mesh is not None:
                vis.add_geometry(self.fused_mesh)
            elif self.fused_pointcloud is not None:
                vis.add_geometry(self.fused_pointcloud)
            else:
                return

            # 设置渲染选项
            vis.get_render_option().mesh_show_back_face = False
            vis.get_render_option().light_on = True
            vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

            if self.fused_pointcloud is not None and self.fused_mesh is None:
                vis.get_render_option().point_size = 3.0

            # 从不同角度保存图像
            angles = [0, 45, 90, 135, 180, 225, 270, 315]

            for angle in angles:
                # 设置视角
                ctr = vis.get_view_control()
                ctr.rotate(angle * 10.0, 15)
                ctr.set_zoom(0.8)

                # 保存图像
                image_name = f"model_view_angle{angle:03d}_{timestamp}.png"
                image_path = os.path.join(vis_dir, image_name)
                vis.capture_screen_image(image_path, do_render=True)

            vis.destroy_window()
            print(f"✅ 可视化图像已保存: {vis_dir}")

        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")

    def generate_report(self):
        """生成详细重建报告 - 320x240版"""
        print("\n" + "="*60)
        print("生成重建报告 (320x240)")
        print("="*60)

        report_file = os.path.join(self.params['output_dir'], "reconstruction_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("三相机高质量STL重建系统 - 320x240分辨率版\n")
            f.write("="*60 + "\n\n")

            f.write(f"重建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分辨率: 320x240\n")
            f.write(f"输出目录: {os.path.abspath(self.params['output_dir'])}\n\n")

            f.write("系统配置:\n")
            f.write(f"  相机数量: {len(self.cameras)}\n")
            f.write(f"  相机间距: {self.camera_spacing*100:.1f}cm\n")
            f.write(f"  采集帧数: {self.params['capture_frames']}\n")
            f.write(f"  体素大小: {self.params['voxel_length']*1000:.1f}mm\n\n")

            f.write("相机位置:\n")
            for pos, idx in self.camera_positions.items():
                if idx is not None:
                    f.write(f"  {pos.upper()}: 相机 #{idx}\n")

            f.write("\n数据采集统计:\n")
            for pos in ['left', 'center', 'right']:
                frame_count = len(self.captured_data[pos]['depth'])
                f.write(f"  {pos.upper()}: {frame_count} 帧\n")

            f.write("\n重建结果:\n")
            if self.fused_mesh is not None:
                f.write(f"  融合网格顶点数: {len(self.fused_mesh.vertices)}\n")
                f.write(f"  融合网格面片数: {len(self.fused_mesh.triangles)}\n")

                # 计算模型尺寸
                try:
                    bbox = self.fused_mesh.get_axis_aligned_bounding_box()
                    extent = bbox.get_extent()
                    f.write(f"  模型尺寸: {extent[0]*100:.1f} x {extent[1]*100:.1f} x {extent[2]*100:.1f} cm\n")
                except:
                    f.write("  模型尺寸: 无法计算\n")
            else:
                f.write("  融合网格: 未生成\n")

            f.write("\n生成的文件 (320x240):\n")
            f.write("  final_results/                          - 最终结果文件\n")
            f.write("    fused_mesh_320x240_*.stl             - STL模型文件（重点）\n")
            f.write("    fused_mesh_simple_*.stl              - 简化STL文件（快速预览）\n")
            f.write("    fused_mesh_*.ply                     - PLY网格文件\n")
            f.write("    fused_pointcloud_*.ply               - 点云文件\n")
            f.write("    visualizations/                      - 可视化图像\n")
            f.write("  intermediate_fusion/                   - 中间融合结果\n")
            f.write("  individual_pointclouds/                - 单个相机点云\n")
            f.write("  captured_images/                       - 采集的图像\n")

            f.write("\nSTL使用说明 (320x240):\n")
            f.write("1. STL文件可直接用于3D打印\n")
            f.write("2. 320x240分辨率的STL文件更小，处理更快\n")
            f.write("3. 推荐使用Ultimaker Cura或PrusaSlicer打开\n")
            f.write("4. 查看visualizations/了解模型外观\n")

            f.write("\n重建质量评估 (320x240):\n")
            if self.fused_mesh is not None:
                vertex_count = len(self.fused_mesh.vertices)
                triangle_count = len(self.fused_mesh.triangles)

                if vertex_count < 500:
                    f.write("⚠️ 警告: 网格顶点数较少，模型可能不完整\n")
                elif vertex_count > 10000:
                    f.write("✓ 良好: 网格细节丰富，适合320x240分辨率\n")

                if triangle_count < 1000:
                    f.write("⚠️ 警告: 网格面片数较少，细节可能丢失\n")
                else:
                    f.write(f"✓ 良好: {triangle_count} 个面片，细节保留良好\n")
            else:
                f.write("❌ 错误: 未能生成融合网格\n")

            f.write("\n320x240分辨率优势:\n")
            f.write("1. 更低的USB带宽需求，三个相机更稳定\n")
            f.write("2. 更快的处理速度\n")
            f.write("3. 更小的文件大小\n")
            f.write("4. 更适合实时应用\n")

            f.write("\n改进建议:\n")
            f.write("1. 确保相机稳定，避免移动\n")
            f.write("2. 重建对象应在所有相机视野内\n")
            f.write("3. 保持适当照明，避免过暗或过曝\n")
            f.write("4. 可调整相机间距参数以获得更好效果\n")

        print(f"📄 重建报告已保存: {report_file}")

    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")

        # 清理相机资源
        for camera_info in self.cameras:
            if camera_info.get('sdk'):
                try:
                    camera_info['sdk'].cleanup()
                except:
                    pass

            if camera_info.get('cv_camera'):
                try:
                    camera_info['cv_camera'].release()
                except:
                    pass

        # 清理Open3D资源
        self.tsdf_volume = None
        self.fused_mesh = None
        self.fused_pointcloud = None

        # 强制垃圾回收
        gc.collect()

        print("✅ 资源清理完成")

    def run_complete_reconstruction(self):
        """运行完整的重建流程 - 320x240版"""
        print("="*70)
        print("三相机高质量STL重建系统 - 320x240分辨率版")
        print("融合三个相机数据，生成高质量STL模型")
        print("="*70)

        total_start_time = time.time()

        try:
            # 步骤1: 设置相机
            print("\n步骤 1/8: 设置相机 (320x240)")
            if not self.setup_cameras():
                print("❌ 相机设置失败")
                return False

            # 步骤2: 确定相机位置
            print("\n步骤 2/8: 确定相机位置")
            if not self.determine_camera_positions():
                print("❌ 无法确定相机位置")
                return False

            # 步骤3: 采集数据
            print("\n步骤 3/8: 采集数据 (320x240)")
            if not self.capture_synchronized_data():
                print("❌ 数据采集失败")
                return False

            # 步骤4: 创建单个点云
            print("\n步骤 4/8: 创建单个相机点云 (320x240)")
            if not self.create_individual_pointclouds():
                print("⚠️ 创建单个点云失败，继续尝试融合")

            # 步骤5: 初始化TSDF
            print("\n步骤 5/8: 初始化TSDF融合体积 (320x240)")
            if not self.init_tsdf_volume():
                print("❌ TSDF初始化失败")
                return False

            # 步骤6: TSDF融合
            print("\n步骤 6/8: TSDF融合 (320x240)")
            if not self.fuse_data_tsdf():
                print("❌ TSDF融合失败")
                return False

            # 步骤7: 提取融合网格
            print("\n步骤 7/8: 提取融合网格 (320x240)")
            if not self.extract_fused_mesh():
                print("❌ 提取融合网格失败")
                return False

            # 步骤8: 保存结果（重点生成STL）
            print("\n步骤 8/8: 保存结果 (320x240)")
            saved_files = self.save_final_results()

            # 生成报告
            self.generate_report()

            total_time = time.time() - total_start_time

            print("\n" + "="*70)
            print("🎉 三相机高质量STL重建完成！ (320x240分辨率)")
            print("="*70)

            output_path = os.path.abspath(self.params['output_dir'])
            print(f"📁 所有结果已保存到: {output_path}")

            if saved_files:
                print("\n📋 主要输出文件 (320x240):")
                for file_type, file_path in saved_files:
                    if 'STL' in file_type:
                        print(f"  ★ {file_type}: {os.path.basename(file_path)}")
                    else:
                        print(f"  • {file_type}: {os.path.basename(file_path)}")

            print(f"\n⏱️  总耗时: {total_time:.1f} 秒")
            print(f"📊 处理帧数: {self.params['capture_frames']} 帧")
            print(f"📏 分辨率: 320x240")

            if self.fused_mesh is not None:
                bbox = self.fused_mesh.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                print(f"📐 模型尺寸: {extent[0]*100:.1f} x {extent[1]*100:.1f} x {extent[2]*100:.1f} cm")

            print("\n💡 320x240分辨率优势:")
            print("  • 三个相机同时运行更稳定")
            print("  • 处理速度更快")
            print("  • STL文件更小")
            print("  • 适合实时应用")

            return True

        except KeyboardInterrupt:
            print("\n🔴 用户中断重建流程")
            return False
        except Exception as e:
            print(f"\n❌ 重建过程中出错: {e}")
            traceback.print_exc()
            return False
        finally:
            self.cleanup()


def main():
    """主函数"""
    print("="*70)
    print("三相机高质量STL重建系统 - 320x240分辨率版")
    print("版本: 3.0 - 专注于320x240分辨率的稳定性和STL质量")
    print("="*70)
    print("系统特点:")
    print("  ✅ 320x240分辨率，三个相机更稳定")
    print("  ✅ TSDF融合技术，消除分层现象")
    print("  ✅ 高质量STL输出，可直接3D打印")
    print("  ✅ 智能网格优化，提升模型质量")
    print("="*70)

    # 检查依赖
    try:
        print("检查依赖...")
        print(f"  OpenCV: {cv2.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Open3D: {o3d.__version__}")
    except Exception as e:
        print(f"❌ 依赖检查失败: {e}")
        return

    # 创建重建系统
    reconstructor = TriCameraReconstructor()

    # 运行重建
    print("\n🚀 开始高质量STL重建 (320x240)...")
    print("注意: 使用320x240分辨率确保三个相机稳定运行")
    print("="*70)

    success = reconstructor.run_complete_reconstruction()

    if success:
        print("\n🎉 STL模型重建成功！ (320x240分辨率)")
        print("💡 生成的STL文件可直接用于3D打印或CNC加工")
    else:
        print("\n⚠️ 重建过程中出现问题")
        print("💡 请检查输出目录中的报告文件了解详情")

    print("\n" + "="*70)
    print("📋 320x240分辨率优势:")
    print("1. 三个相机同时运行更稳定")
    print("2. USB带宽需求低，减少卡顿")
    print("3. 处理速度快，适合实时应用")
    print("4. STL文件更小，便于存储和传输")
    print("="*70)

    input("\n按Enter键退出...")


if __name__ == "__main__":
    # 设置环境变量
    default_sdk_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    if os.path.exists(default_sdk_path):
        drivers_dir = os.path.join(default_sdk_path, "OpenNI2", "Drivers")
        if os.path.exists(drivers_dir):
            os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']

    main()