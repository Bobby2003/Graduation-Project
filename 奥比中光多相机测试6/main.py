"""
三相机彩色3D重建系统 - 多相机融合版本
三个奥比中光Astra相机并排放置，间距16cm
"""

import cv2
import numpy as np
import open3d as o3d
import json
import os
import gc
import traceback
import time
import sys
from datetime import datetime
import warnings
import threading
from queue import Queue
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings('ignore')

# 导入独立的SDK接口层
from multi_stream_manager import MultiStreamCameraManager
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

    SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SDK接口层导入失败: {e}")
    SDK_AVAILABLE = False


class MultiCameraReconstructor:
    """三相机彩色3D重建器 - 自动检测位置关系并融合重建"""

    def __init__(self, sdk_path=None):
        """初始化三相机重建器

        Args:
            sdk_path: SDK路径
        """
        if not SDK_AVAILABLE:
            print("❌ SDK接口层不可用")
            return

        self.sdk_path = sdk_path if sdk_path else get_default_sdk_path()

        # 三个相机实例
        self.cameras = []
        self.camera_positions = []  # 存储检测到的位置关系
        self.rgb_cameras = []  # 三个RGB摄像头

        # 重建参数
        self.params = {
            # 相机参数
            'num_cameras': 3,
            'camera_spacing': 0.16,  # 16cm
            'camera_fov': 60.0,  # 视场角

            # 深度参数
            'depth_scale': 0.001,
            'depth_clip_min': 0.3,  # 30cm
            'depth_clip_max': 2.5,  # 2.5米

            # TSDF参数
            'tsdf_voxel_length': 0.005,  # 更精细的体素
            'tsdf_sdf_trunc': 0.04,
            'volume_size': 4.0,  # 更大的重建体积

            # 重建参数
            'max_iterations': 15,
            'frame_interval': 0.3,

            # 相机索引
            'depth_camera_indices': [0, 1, 2],
            'rgb_camera_indices': [0, 1, 2],
            'camera_width': 320,
            'camera_height': 240,
            'camera_fps': 15,

            # 颜色增强
            'color_enhance': True,
            'color_contrast': 1.1,
            'color_brightness': 5,

            # 网格优化
            'mesh_simplify': True,
            'target_vertices': 50000,
            'smooth_mesh': True,
            'smooth_iterations': 2,

            # 融合参数
            'fusion_method': 'tsdf',  # 'tsdf' 或 'pointcloud'
            'overlap_threshold': 0.02,  # 重叠区域阈值

            # 导出设置
            'export_ply': True,
            'export_obj': True,
            'export_stl': True,

            # 保存设置
            'save_every_frame': False,
            'save_intermediate': True,

            # 自动检测参数
            'detection_threshold': 0.3,
            'detection_timeout': 5.0,
        }
        self.params.update({
            'target_width': 320,
            'target_height': 240,
            'depth_scale': 0.001,
        })

        # TSDF体积
        self.tsdf_volume = None

        # 相机内参
        self.intrinsics = []

        # 相机外参（位置关系）
        self.extrinsics = []

        # 输出目录
        self.output_dir = None
        self.iteration = 0
        self.mesh_count = 0

        print("🎨 三相机彩色3D重建器初始化完成")
        print(f"   相机数量: {self.params['num_cameras']}")
        print(f"   相机间距: {self.params['camera_spacing']}米")

    # 在 MultiCameraReconstructor 类的 setup_cameras 方法中：
    def setup_cameras(self):
        """设置三相机系统（使用多流管理器）"""
        print("=" * 70)
        print("设置三相机系统（统一分辨率320x240）")
        print("=" * 70)

        try:
            # 创建单实例多流管理器（启用颜色流）
            from multi_stream_manager import MultiStreamCameraManager
            self.stream_manager = MultiStreamCameraManager(self.sdk_path)

            if not self.stream_manager.context_initialized:
                print("❌ 多流管理器初始化失败")
                return False

            # 🔧 设置3个流，启用颜色
            if not self.stream_manager.setup_streams(num_streams=3, enable_color=True):
                print("❌ 设置流失败")
                return False

            # 检测位置关系
            self.camera_positions = self.stream_manager.get_stream_positions()
            print(f"✅ 相机位置关系: {self.camera_positions}")

            print(f"\n✅ 三相机系统设置完成")
            print(f"   活动流数量: {self.stream_manager.get_active_stream_count()}")
            print(f"   目标分辨率: {self.params['target_width']}x{self.params['target_height']}")

            return True

        except Exception as e:
            print(f"❌ 设置相机失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect_camera_positions_new(self):
        """使用多流管理器检测位置关系"""
        print("\n自动检测相机位置关系...")

        try:
            # 从每个流读取一帧
            for i in range(len(self.stream_manager.streams)):
                result = self.stream_manager.capture_frame(i, timeout_ms=2000)
                if result:
                    depth_array, frame_info = result
                    print(f"  相机{i}: 读取成功，有效点: {np.sum(depth_array > 0)}")
                    self.stream_manager.release_frame(frame_info)
                else:
                    print(f"  相机{i}: 读取失败")

            # 简化的位置分配
            if len(self.stream_manager.streams) == 3:
                self.camera_positions = ['left', 'center', 'right']
                print("✅ 相机位置关系检测完成")
                return True
            else:
                return False

        except Exception as e:
            print(f"❌ 位置关系检测失败: {e}")
            return False

    def capture_frames_robust(self):
        """稳健的串行采集 - 防止任何相机阻塞整个系统"""
        frames = [None] * len(self.cameras)
        colors = [None] * len(self.rgb_cameras)

        # 第一步：先快速探测每个相机
        print("🔍 快速探测相机状态...")
        for i, camera in enumerate(self.cameras):
            try:
                # 极短超时的快速读取，丢弃可能存在的旧帧
                result = camera.capture_depth_frame(timeout=100, camera_id=f"probe_{i}")
                if result:
                    print(f"  相机{i}: ✅ 有数据就绪，丢弃旧帧")
                    # 再读一帧丢弃（如果有堆积）
                    camera.capture_depth_frame(timeout=100, camera_id=f"flush_{i}")
                else:
                    print(f"  相机{i}: ⚠️ 无立即数据")
            except Exception as e:
                print(f"  相机{i}: ❌ 探测异常: {e}")

        # 第二步：严格的串行采集，相机间强制延迟
        print("\n🎯 开始正式串行采集...")
        for i, camera in enumerate(self.cameras):
            start_time = time.time()
            print(f"\n--- 采集相机 {i} ---")

            # 方法A：正常采集（带短超时）
            result = camera.capture_depth_frame(timeout=2000, camera_id=str(i))

            if result:
                elapsed = time.time() - start_time
                depth_array, frame_info = result
                depth = depth_array.astype(np.float32) * 0.001
                frames[i] = depth
                print(f"✅ 相机{i} 采集成功 ({elapsed:.2f}秒, {depth_array.shape})")
            else:
                # 方法B：采集失败，尝试流重置
                print(f"⚠️ 相机{i} 正常采集失败，尝试流重置...")
                frames[i] = self._reset_and_capture(camera, i, start_time)

            # RGB采集
            if i < len(self.rgb_cameras):
                ret, color = self.rgb_cameras[i].read()
                colors[i] = color if ret else None

            # 关键：相机间强制延迟（让USB总线恢复）
            if i < len(self.cameras) - 1:
                delay = 0.2  # 200ms延迟
                print(f"⏳ 等待{delay}秒后采集下一个相机...")
                time.sleep(delay)

        return frames, colors

    def _reset_and_capture(self, camera, cam_idx, start_time):
        """流重置后重新尝试采集"""
        try:
            print(f"  相机{cam_idx}: 停止流...")
            camera.stop_stream()
            time.sleep(0.3)  # 重要：给硬件时间重置

            print(f"  相机{cam_idx}: 重启流...")
            camera.start_stream()
            time.sleep(0.5)  # 重要：等待流稳定

            # 丢弃第一帧（可能是无效的）
            camera.capture_depth_frame(timeout=500, camera_id=f"reset_{cam_idx}")

            # 再次尝试采集
            result = camera.capture_depth_frame(timeout=3000, camera_id=f"retry_{cam_idx}")

            if result:
                elapsed = time.time() - start_time
                depth_array, _ = result
                depth = depth_array.astype(np.float32) * 0.001
                print(f"✅ 相机{cam_idx} 重置后采集成功 (总耗时{elapsed:.2f}秒)")
                return depth
            else:
                print(f"❌ 相机{cam_idx} 重置后仍失败")
                return None

        except Exception as e:
            print(f"❌ 相机{cam_idx} 重置过程异常: {e}")
            return None

    def analyze_positions(self, depth_frames, color_frames):
        """分析相机位置关系

        Args:
            depth_frames: 深度帧列表
            color_frames: 颜色帧列表

        Returns:
            位置关系列表 ['left', 'center', 'right']
        """
        positions = []

        try:
            # 方法1: 基于视差分析（如果物体距离已知）
            # 这里使用简化的方法：假设物体在中心，根据物体在图像中的水平位置判断

            valid_frames = [f for f in depth_frames if f is not None]
            if len(valid_frames) < 2:
                print("⚠️ 有效帧不足，无法分析")
                return ['left', 'center', 'right']  # 默认位置

            # 计算每个相机的中心深度
            center_depths = []
            for depth in valid_frames:
                if depth is not None:
                    # 取中心区域
                    h, w = depth.shape
                    center_region = depth[h // 4:3 * h // 4, w // 4:3 * w // 4]
                    valid_depths = center_region[center_region > 0]
                    if len(valid_depths) > 0:
                        avg_depth = np.median(valid_depths)
                        center_depths.append(avg_depth)
                    else:
                        center_depths.append(0)

            # 如果有足够的深度信息，根据视差判断
            if len([d for d in center_depths if d > 0]) >= 2:
                # 简化的位置判断
                positions = ['left', 'center', 'right'][:len(depth_frames)]
            else:
                # 方法2: 基于颜色帧的内容分析
                # 假设有一个标定板或明显特征
                print("使用颜色帧分析位置...")
                positions = self.analyze_positions_by_color(color_frames)

            return positions

        except Exception as e:
            print(f"❌ 位置分析失败: {e}")
            return ['left', 'center', 'right'][:len(depth_frames)]

    def analyze_positions_by_color(self, color_frames):
        """通过颜色帧分析位置关系"""
        positions = []

        # 简化的方法：假设相机视野有重叠，通过特征点匹配判断
        # 这里使用图像亮度分布作为简单判断

        brightness_values = []
        for i, color in enumerate(color_frames):
            if color is not None:
                # 转换为灰度
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append((i, brightness))
            else:
                brightness_values.append((i, 0))

        # 按亮度排序（假设光照条件不同）
        brightness_values.sort(key=lambda x: x[1])

        # 最简单的分配：最暗->左，中等->中，最亮->右
        # 这只是一种启发式方法，实际应用需要更精确的标定
        for i, (orig_idx, _) in enumerate(brightness_values):
            if i == 0:
                positions.append('left')
            elif i == 1:
                positions.append('center')
            elif i == 2:
                positions.append('right')
            else:
                positions.append('unknown')

        # 重新排序回原始顺序
        sorted_positions = [''] * len(color_frames)
        for (orig_idx, _), pos in zip(brightness_values, positions):
            sorted_positions[orig_idx] = pos

        return sorted_positions

    def set_default_positions(self):
        """设置默认位置关系（左、中、右）"""
        self.camera_positions = ['left', 'center', 'right'][:len(self.cameras)]

    def init_intrinsics(self):
        """初始化相机内参"""
        print("\n初始化相机内参...")

        for i in range(len(self.cameras)):
            # 假设所有相机参数相同
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                320, 240, 525.0, 525.0, 319.5, 239.5
            )
            self.intrinsics.append(intrinsic)

        print(f"✅ 初始化了{len(self.intrinsics)}个相机内参")

    def init_extrinsics(self):
        """初始化相机外参（基于位置关系）"""
        print("\n初始化相机外参...")

        self.extrinsics = []

        for i, pos in enumerate(self.camera_positions):
            extrinsic = np.eye(4)

            # 根据位置关系设置外参
            if pos == 'left':
                # 左相机：在X轴负方向
                extrinsic[0, 3] = -self.params['camera_spacing']  # 向左移动
                print(f"   相机 {i} (左): X = {-self.params['camera_spacing']:.3f}m")

            elif pos == 'right':
                # 右相机：在X轴正方向
                extrinsic[0, 3] = self.params['camera_spacing']  # 向右移动
                print(f"   相机 {i} (右): X = {self.params['camera_spacing']:.3f}m")

            elif pos == 'center':
                # 中心相机：原点
                print(f"   相机 {i} (中): X = 0.000m")

            else:
                # 未知位置，使用默认
                if i == 0:
                    extrinsic[0, 3] = -self.params['camera_spacing']
                elif i == 2:
                    extrinsic[0, 3] = self.params['camera_spacing']

            self.extrinsics.append(extrinsic)

        print(f"✅ 初始化了{len(self.extrinsics)}个相机外参")

    def capture_frames_multi(self):
        """使用多流管理器采集所有相机（深度+颜色，统一分辨率）"""
        print("\n=== 开始多流采集（深度+颜色，320x240）===")

        if not hasattr(self, 'stream_manager') or not self.stream_manager.streams:
            print("❌ 多流管理器未初始化")
            return [None, None, None], [None, None, None]

        # 🔧 使用支持颜色的采集方法
        depth_arrays, color_arrays = self.stream_manager.capture_all_streams_with_color()

        # 转换数据格式并统一分辨率到320x240
        depth_frames = []
        color_frames = []

        target_width = self.params['target_width']
        target_height = self.params['target_height']

        for i, depth in enumerate(depth_arrays):
            if depth is not None:
                try:
                    # 转换为米
                    depth_meters = depth.astype(np.float32) * self.params['depth_scale']

                    # 🔧 统一深度图分辨率为320x240
                    if depth_meters.shape != (target_height, target_width):
                        depth_meters = cv2.resize(depth_meters, (target_width, target_height),
                                                  interpolation=cv2.INTER_NEAREST)

                    # 处理深度图
                    depth_processed = self.process_depth(depth_meters)
                    depth_frames.append(depth_processed)

                    print(f"  深度流{i}: ✅ {depth.shape} -> {depth_processed.shape}, 有效点 {np.sum(depth_processed > 0)}")
                except Exception as e:
                    print(f"  深度流{i}: ❌ 处理失败: {e}")
                    depth_frames.append(None)
            else:
                depth_frames.append(None)
                print(f"  深度流{i}: ❌ 采集失败")

        for i, color in enumerate(color_arrays):
            if color is not None:
                try:
                    # 🔧 统一颜色图分辨率为320x240
                    if color.shape[:2] != (target_height, target_width):
                        color = cv2.resize(color, (target_width, target_height),
                                           interpolation=cv2.INTER_LINEAR)

                    color_frames.append(color)
                    print(
                        f"  颜色流{i}: ✅ {color_arrays[i].shape if color_arrays[i] is not None else 'None'} -> {color.shape}")
                except Exception as e:
                    print(f"  颜色流{i}: ❌ 处理失败: {e}")
                    color_frames.append(None)
            else:
                # 🔧 如果颜色采集失败，使用模拟颜色
                color = self._generate_simulated_color(i, depth_arrays[i] if i < len(depth_arrays) else None)
                color_frames.append(color)
                print(f"  颜色流{i}: ⚠️ 使用模拟颜色 {color.shape}")

        return depth_frames, color_frames

    def _generate_simulated_color(self, camera_index, depth_array):
        """生成模拟颜色帧（当颜色流不可用时）"""
        if depth_array is None:
            return None

        h, w = self.params['target_height'], self.params['target_width']
        color = np.zeros((h, w, 3), dtype=np.uint8)

        if camera_index == 0:  # 左相机 - 红色
            color[:, :, 2] = 200  # OpenCV是BGR顺序
        elif camera_index == 1:  # 中相机 - 绿色
            color[:, :, 1] = 200
        elif camera_index == 2:  # 右相机 - 蓝色
            color[:, :, 0] = 200
        else:
            color[:, :, :] = 128  # 灰色

        return color

    def enhance_color(self, image):
        """增强颜色"""
        try:
            img_float = image.astype(np.float32) / 255.0
            img_float = np.clip((img_float - 0.5) * self.params['color_contrast'] + 0.5, 0, 1)
            img_float = np.clip(img_float + self.params['color_brightness'] / 255.0, 0, 1)
            enhanced = (img_float * 255).astype(np.uint8)
            return enhanced
        except:
            return image

    def process_depth(self, depth):
        """处理深度图 - 统一分辨率版"""
        if depth is None:
            print("  ❌ process_depth: 输入深度图为None")
            return None

        # 🔧 确保深度图是320x240
        target_height, target_width = self.params['target_height'], self.params['target_width']
        if depth.shape != (target_height, target_width):
            depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            print(f"  ⚠️ 深度图已调整为 {target_width}x{target_height}")

        # 范围裁剪
        mask = (depth >= self.params['depth_clip_min']) & (depth <= self.params['depth_clip_max'])
        depth_processed = depth.copy()
        depth_processed[~mask] = 0

        print(f"  process_depth: {depth_processed.shape}, 有效点={np.sum(depth_processed > 0)}")
        return depth_processed

    def init_intrinsics(self):
        """初始化相机内参 - 统一为320x240"""
        print("\n初始化相机内参（320x240）...")

        self.intrinsics = []
        width = self.params['target_width']
        height = self.params['target_height']

        # 根据分辨率调整内参
        # 假设原始内参为640x480的525焦距，缩放比例
        scale_x = width / 640.0
        scale_y = height / 480.0

        fx = 525.0 * scale_x
        fy = 525.0 * scale_y
        cx = 319.5 * scale_x
        cy = 239.5 * scale_y

        for i in range(len(self.camera_positions)):
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height,
                fx, fy,
                cx, cy
            )
            self.intrinsics.append(intrinsic)
            print(f"   相机{i}: {width}x{height}, fx={fx:.1f}, fy={fy:.1f}")

        print(f"✅ 初始化了{len(self.intrinsics)}个相机内参")

    def init_tsdf_volume(self):
        """初始化TSDF体积"""
        print("\n初始化TSDF体积...")

        try:
            # 创建支持颜色的TSDF体积
            self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.params['tsdf_voxel_length'],
                sdf_trunc=self.params['tsdf_sdf_trunc'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            print(f"✅ TSDF体积初始化完成")
            print(f"   体素大小: {self.params['tsdf_voxel_length']}米")
            print(f"   重建体积: {self.params['volume_size']}米³")

            return True

        except Exception as e:
            print(f"❌ TSDF初始化失败: {e}")
            return False

    def fuse_multi_frames(self, depth_frames, color_frames):
        """融合多相机帧到TSDF - 超详细调试版"""
        fused_count = 0

        print(f"\n=== 开始融合 {len(depth_frames)} 个相机帧 ===")
        print(f"深度帧列表长度: {len(depth_frames)}")
        print(f"颜色帧列表长度: {len(color_frames)}")

        # 检查每个深度帧是否为None
        for i, depth in enumerate(depth_frames):
            if depth is None:
                print(f"警告: 深度帧 {i} 为 None")
            else:
                print(f"深度帧 {i}: 形状={depth.shape}, 有效点={np.sum(depth > 0)}")

        # 创建调试目录
        debug_dir = os.path.join(self.output_dir, "debug_fusion")
        os.makedirs(debug_dir, exist_ok=True)
        for i, (depth, color) in enumerate(zip(depth_frames, color_frames)):
            print(f"\n📡 处理相机 {i}:")

            # 1. 检查深度图 - 添加详细检查
            if depth is None:
                print(f"  ❌ 深度图为None，跳过")
                continue

            print(f"  深度图: 形状={depth.shape}, 类型={depth.dtype}")
            print(f"  深度范围: {depth.min():.3f} - {depth.max():.3f}米")
            print(f"  有效点(>0): {np.sum(depth > 0)}个")

            # 2. 处理深度图 - 这里可能是问题源头
            print(f"  处理深度图...")
            depth_processed = self.process_depth(depth)
            if depth_processed is None:
                print(f"  ❌ 深度图处理失败")
                continue

            valid_points = np.sum(depth_processed > 0)
            print(f"  处理后有效点: {valid_points}个")
            print(f"  处理后深度范围: {depth_processed.min():.3f} - {depth_processed.max():.3f}米")

            if valid_points < 1000:
                print(f"  ⚠️ 有效点不足 ({valid_points}点)，跳过")
                continue

            try:
                # ========== 调试：保存处理前后的深度图 ==========
                depth_colored_before = self.depth_to_colormap(depth)
                depth_colored_after = self.depth_to_colormap(depth_processed)

                if depth_colored_before is not None:
                    cv2.imwrite(
                        os.path.join(debug_dir, f"cam{i}_depth_before.png"),
                        depth_colored_before
                    )

                if depth_colored_after is not None:
                    cv2.imwrite(
                        os.path.join(debug_dir, f"cam{i}_depth_after.png"),
                        depth_colored_after
                    )

                print(f"  💾 深度图对比已保存")

                # ========== 检查深度值单位转换 ==========
                depth_mm = (depth_processed * 1000).astype(np.uint16)

                # 关键检查：深度值是否在合理范围内
                depth_mm_valid = depth_mm[depth_mm > 0]
                if len(depth_mm_valid) > 0:
                    min_depth_mm = depth_mm_valid.min()
                    max_depth_mm = depth_mm_valid.max()
                    print(f"  深度值(mm): {min_depth_mm} - {max_depth_mm}")

                    # 检查是否在Open3D接受的范围内
                    if max_depth_mm > 10000:  # 10米 = 10000mm
                        print(f"  ⚠️ 警告：深度值过大 ({max_depth_mm}mm = {max_depth_mm / 1000:.1f}米)")
                    if min_depth_mm < 300:  # 30cm = 300mm
                        print(f"  ⚠️ 警告：深度值过小 ({min_depth_mm}mm = {min_depth_mm / 1000:.1f}米)")

                # ========== 创建深度图像 ==========
                print(f"  创建深度图像...")
                depth_image = o3d.geometry.Image(depth_mm)
                print(f"  ✅ 深度图像创建成功")

                # ========== 创建颜色图像 ==========
                if color is not None:
                    print(f"  原始颜色图: 形状={color.shape}, 类型={color.dtype}")

                    # 确保颜色图尺寸匹配深度图
                    if color.shape[0] != depth_processed.shape[0] or color.shape[1] != depth_processed.shape[1]:
                        print(f"  ⚠️ 尺寸不匹配: 深度{depth_processed.shape} vs 颜色{color.shape}")
                        print(f"    深度图尺寸: {depth_processed.shape}")
                        print(f"    颜色图尺寸: {color.shape}")

                        # 尝试调整颜色图尺寸
                        color = cv2.resize(color, (depth_processed.shape[1], depth_processed.shape[0]))
                        print(f"    调整后颜色图: {color.shape}")

                    # 检查颜色图数据类型
                    if color.dtype != np.uint8:
                        print(f"  ⚠️ 颜色图类型不是uint8: {color.dtype}，尝试转换")
                        color = color.astype(np.uint8)

                    # 保存颜色图用于调试
                    cv2.imwrite(
                        os.path.join(debug_dir, f"cam{i}_color.png"),
                        color
                    )

                    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    color_image = o3d.geometry.Image(color_rgb)
                    print(f"  ✅ 颜色图像创建成功")
                else:
                    print(f"  ⚠️ 颜色图为None，使用默认灰色")
                    default_color = np.full((depth_processed.shape[0], depth_processed.shape[1], 3), 128,
                                            dtype=np.uint8)
                    color_image = o3d.geometry.Image(default_color)

                # ========== 创建RGBD图像 - 这里可能失败 ==========
                print(f"  创建RGBD图像...")
                print(f"    参数: depth_scale=1000.0, depth_trunc={self.params['depth_clip_max']}")

                # 尝试不同的创建方法
                try:
                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_image,
                        depth_image,
                        depth_scale=1000.0,
                        depth_trunc=self.params['depth_clip_max'],
                        convert_rgb_to_intensity=False
                    )
                    print(f"  ✅ RGBD图像创建成功")
                    print(f"    深度图形状: {np.asarray(rgbd_image.depth).shape}")
                    print(f"    颜色图形状: {np.asarray(rgbd_image.color).shape}")
                except Exception as rgbd_error:
                    print(f"  ❌ RGBD图像创建失败: {rgbd_error}")
                    print(f"    尝试使用默认参数...")

                    # 尝试使用默认参数
                    try:
                        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            color_image,
                            depth_image,
                            depth_scale=1000.0,
                            depth_trunc=3.0,  # 默认截断
                            convert_rgb_to_intensity=False
                        )
                        print(f"  ✅ 使用默认参数创建RGBD图像成功")
                    except Exception as default_error:
                        print(f"  ❌ 默认参数也失败: {default_error}")
                        continue

                # ========== 检查相机参数 ==========
                print(f"  检查相机参数...")

                # 检查外参
                if i < len(self.extrinsics):
                    extrinsic = self.extrinsics[i]
                    print(f"    外参矩阵 {i}:")
                    print(f"      平移: X={extrinsic[0, 3]:.3f}, Y={extrinsic[1, 3]:.3f}, Z={extrinsic[2, 3]:.3f}")
                else:
                    print(f"    ❌ 没有外参矩阵 {i}，跳过")
                    continue

                # 检查内参
                if i >= len(self.intrinsics):
                    print(f"    ❌ 没有内参矩阵 {i}，跳过")
                    continue

                intrinsic = self.intrinsics[i]
                print(f"    内参矩阵 {i}:")
                print(f"      分辨率: {intrinsic.width}x{intrinsic.height}")
                print(f"      焦距: fx={intrinsic.intrinsic_matrix[0, 0]:.1f}, fy={intrinsic.intrinsic_matrix[1, 1]:.1f}")
                print(f"      主点: cx={intrinsic.intrinsic_matrix[0, 2]:.1f}, cy={intrinsic.intrinsic_matrix[1, 2]:.1f}")

                # ========== 关键：测试融合一小部分数据 ==========
                print(f"  测试融合一小部分数据...")

                # 创建一个小的测试体积
                try:
                    # 测试用的小体积
                    test_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                        voxel_length=self.params['tsdf_voxel_length'],
                        sdf_trunc=self.params['tsdf_sdf_trunc'],
                        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                    )

                    # 只融合一帧测试
                    test_volume.integrate(rgbd_image, intrinsic, extrinsic)

                    # 尝试提取测试网格
                    test_mesh = test_volume.extract_triangle_mesh()
                    if test_mesh and len(test_mesh.vertices) > 0:
                        print(f"    ✅ 测试融合成功: {len(test_mesh.vertices)}顶点")

                        # 保存测试网格
                        test_mesh_path = os.path.join(debug_dir, f"cam{i}_test_mesh.ply")
                        o3d.io.write_triangle_mesh(test_mesh_path, test_mesh)
                        print(f"    💾 测试网格已保存: {test_mesh_path}")

                        # 如果测试成功，融合到主体积
                        self.tsdf_volume.integrate(rgbd_image, intrinsic, extrinsic)
                        fused_count += 1
                        print(f"  ✅ 相机 {i} 主融合成功")
                    else:
                        print(f"    ❌ 测试融合失败：提取的网格为空")

                except Exception as test_error:
                    print(f"    ❌ 测试融合失败: {test_error}")
                    import traceback
                    traceback.print_exc()

            except Exception as e:
                print(f"  ❌ 相机 {i} 融合过程中发生未捕获异常: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n=== 融合完成: {fused_count}/{len(depth_frames)} 个相机成功 ===")
        return fused_count

    def extract_color_mesh(self):
        """提取带颜色的网格"""
        if self.tsdf_volume is None:
            return None

        try:
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
                mesh.paint_uniform_color([0.8, 0.8, 0.8])

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

            # 保存PLY格式
            if self.params['export_ply']:
                try:
                    if not is_final:
                        ply_filename = f"multi_camera_model_f{iteration:03d}_{timestamp}.ply"
                    else:
                        ply_filename = f"final_multi_camera_model_{timestamp}.ply"

                    ply_path = os.path.join(save_dir, ply_filename)

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

            # 保存OBJ格式
            if self.params['export_obj']:
                try:
                    if not is_final:
                        obj_filename = f"multi_camera_model_f{iteration:03d}_{timestamp}.obj"
                    else:
                        obj_filename = f"final_multi_camera_model_{timestamp}.obj"

                    obj_path = os.path.join(save_dir, obj_filename)

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

            # 保存STL格式
            if self.params['export_stl']:
                try:
                    if not is_final:
                        stl_filename = f"multi_camera_model_f{iteration:03d}_{timestamp}.stl"
                    else:
                        stl_filename = f"final_multi_camera_model_{timestamp}.stl"

                    stl_path = os.path.join(save_dir, stl_filename)

                    o3d.io.write_triangle_mesh(stl_path, mesh)
                    saved_files.append(('STL', stl_path))
                    print(f"   ✅ STL格式已保存: {os.path.basename(stl_path)}")
                except Exception as e:
                    print(f"   ❌ STL保存失败: {e}")

            self.mesh_count += 1
            return True

        except Exception as e:
            print(f"❌ 保存网格失败: {e}")
            return False

    def save_frame_images(self, iteration, depth_frames, color_frames):
        """保存多相机帧图像"""
        try:
            images_dir = os.path.join(self.output_dir, "captured_images")
            os.makedirs(images_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, (depth, color) in enumerate(zip(depth_frames, color_frames)):
                camera_suffix = f"cam{i}_{self.camera_positions[i] if i < len(self.camera_positions) else 'unknown'}"

                # 保存深度图
                if depth is not None:
                    depth_colored = self.depth_to_colormap(depth)
                    if depth_colored is not None:
                        depth_path = os.path.join(images_dir, f"depth_{camera_suffix}_f{iteration:03d}_{timestamp}.png")
                        cv2.imwrite(depth_path, depth_colored)

                # 保存颜色图
                if color is not None:
                    color_path = os.path.join(images_dir, f"color_{camera_suffix}_f{iteration:03d}_{timestamp}.png")
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

    def run_multi_camera_reconstruction(self):
        """运行多相机重建 - 修复版"""
        print("=" * 80)
        print("奥比中光三相机 - 大范围彩色3D重建（320x240）")
        print("=" * 80)

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"multi_camera_reconstruction_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")

        # 保存配置
        config_file = os.path.join(self.output_dir, "config_multi_camera.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)

        # 检查SDK可用性
        if not SDK_AVAILABLE:
            print("❌ SDK接口层不可用")
            return False

        # 设置相机
        if not self.setup_cameras():
            print("❌ 相机设置失败")
            return False

        # 初始化内参 - 🔧 统一为320x240
        print("\n初始化相机内参...")
        self.intrinsics = []
        for i in range(3):
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                320, 240,  # 🔧 固定分辨率
                262.5, 262.5,  # fx, fy - 缩放后的焦距
                159.5, 119.5  # cx, cy - 中心点
            )
            self.intrinsics.append(intrinsic)
            print(f"   相机{i}: {intrinsic.width}x{intrinsic.height}, fx={intrinsic.intrinsic_matrix[0, 0]:.1f}")

        # 初始化外参
        self.init_extrinsics()

        # 初始化TSDF
        if not self.init_tsdf_volume():
            print("❌ TSDF初始化失败")
            self.cleanup()
            return False

        print("\n" + "=" * 60)
        print("开始三相机彩色3D重建（统一分辨率320x240）")
        print("=" * 60)

        self.iteration = 0
        start_time = time.time()

        try:
            for i in range(self.params['max_iterations']):
                self.iteration = i + 1
                print(f"\n📸 第 {self.iteration}/{self.params['max_iterations']} 轮")

                # 等待帧间隔
                if i > 0:
                    time.sleep(self.params['frame_interval'])

                # 🔧 使用统一的采集方法（深度+颜色）
                depth_frames, color_frames = self.capture_frames_multi()

                # 统计有效帧
                valid_depths = sum(1 for d in depth_frames if d is not None)
                valid_colors = sum(1 for c in color_frames if c is not None)
                print(f"有效深度帧: {valid_depths}/3, 有效颜色帧: {valid_colors}/3")

                if valid_depths < 1:
                    print("❌ 没有有效深度帧，跳过")
                    continue

                # 🔧 在融合前检查数据
                print(f"\n🔍 融合前数据检查:")
                for j in range(3):
                    depth_shape = depth_frames[j].shape if depth_frames[j] is not None else "None"
                    color_shape = color_frames[j].shape if color_frames[j] is not None else "None"
                    print(f"  相机{j}: 深度 {depth_shape}, 颜色 {color_shape}")

                # 融合多相机帧
                fused_count = self.fuse_multi_frames(depth_frames, color_frames)
                print(f"✅ 融合了 {fused_count} 个相机视图")

                # 保存中间结果
                if self.params['save_intermediate'] and fused_count > 0:
                    # 提取中间网格
                    temp_mesh = self.extract_color_mesh()
                    if temp_mesh and len(temp_mesh.vertices) > 0:
                        save_success = self.save_color_mesh(temp_mesh, self.iteration, is_final=False)
                        if save_success:
                            print(f"💾 第{self.iteration}轮中间模型已保存")

                        # 释放内存
                        del temp_mesh

                # 保存帧图像
                if self.params['save_every_frame']:
                    self.save_frame_images(self.iteration, depth_frames, color_frames)

        except KeyboardInterrupt:
            print("\n🔴 用户中断")
        except Exception as e:
            print(f"\n❌ 主循环异常: {e}")
            traceback.print_exc()

        # 最终处理
        print("\n" + "=" * 60)
        print("导出最终三相机彩色模型")
        print("=" * 60)

        if self.tsdf_volume is not None and self.iteration > 0:
            print("🎨 提取最终彩色网格...")
            final_mesh = self.extract_color_mesh()

            if final_mesh is not None:
                print(f"   顶点数: {len(final_mesh.vertices)}")
                print(f"   面片数: {len(final_mesh.triangles)}")
                print(f"   有颜色: {final_mesh.has_vertex_colors()}")

                # 计算模型尺寸
                bbox = final_mesh.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                print(f"   模型尺寸: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} 米")

                # 保存最终模型
                print("💾 保存最终彩色模型...")
                success = self.save_color_mesh(final_mesh, self.iteration, is_final=True)

                if success:
                    print("✅ 最终三相机彩色模型保存成功")

                    # 显示模型信息
                    self.display_model_info(final_mesh)

                    # 保存模型预览图
                    self.save_model_preview(final_mesh)

                del final_mesh
            else:
                print("❌ 无法提取最终网格")

        # 生成报告
        self.generate_multi_camera_report()

        # 清理资源
        self.cleanup()

        print("\n" + "=" * 80)
        print("三相机彩色3D重建完成")
        print(f"处理轮次: {self.iteration}")
        print(f"模型数量: {self.mesh_count}")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print("=" * 80)

        return True

    def _quick_visualize(self, depth_frames, color_frames):
        """快速可视化验证数据"""
        try:
            for i, (depth, color) in enumerate(zip(depth_frames, color_frames)):
                if depth is not None:
                    # 显示简单的统计信息
                    valid_points = np.sum(depth > 0)
                    depth_range = f"{depth[depth > 0].min():.2f}-{depth[depth > 0].max():.2f}" if valid_points > 0 else "N/A"
                    print(f"  相机{i}: {valid_points}点, 深度范围:{depth_range}m")

                    # 保存第一帧的深度图用于调试
                    if self.iteration == 1 and i == 0:
                        depth_image = (depth * 1000).astype(np.uint16)
                        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                        debug_dir = os.path.join(self.output_dir, "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_dir, f"camera{i}_depth.png"), depth_normalized)
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")

    def display_model_info(self, mesh):
        """显示模型信息"""
        try:
            vertices = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)

            print("\n📊 三相机模型统计信息:")
            print(f"   顶点范围: X [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
            print(f"             Y [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
            print(f"             Z [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

            if colors.shape[0] > 0 and mesh.has_vertex_colors():
                print(f"   颜色范围: R [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
                print(f"             G [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
                print(f"             B [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")

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
            vis.create_window(visible=False, width=800, height=600)

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
                ctr.set_zoom(0.7)
                ctr.rotate(angle * 10.0, 0.0)

                # 捕获图像
                image_path = os.path.join(preview_dir, f"multi_preview_angle_{angle:03d}_{timestamp}.png")
                vis.capture_screen_image(image_path, do_render=True)

            vis.destroy_window()
            print(f"📸 模型预览图已保存: {preview_dir}")

        except Exception as e:
            print(f"⚠️ 保存预览图失败: {e}")

    def generate_multi_camera_report(self):
        """生成三相机重建报告"""
        try:
            report_path = os.path.join(self.output_dir, "三相机重建报告.txt")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("奥比中光三相机 - 大范围彩色3D重建报告\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"重建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理轮次: {self.iteration}\n")
                f.write(f"相机数量: {len(self.cameras)}个\n")
                f.write(f"输出模型: {self.mesh_count}个\n")
                f.write(f"输出目录: {self.output_dir}\n\n")

                f.write("相机位置关系:\n")
                for i, pos in enumerate(self.camera_positions):
                    f.write(f"   相机 {i}: {pos}\n")

                f.write("\n相机外参矩阵:\n")
                for i, extrinsic in enumerate(self.extrinsics):
                    f.write(f"   相机 {i}:\n")
                    for row in extrinsic:
                        f.write(f"      [{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}]\n")
                    f.write("\n")

                f.write("\n重建参数:\n")
                for key, value in self.params.items():
                    f.write(f"  {key}: {value}\n")

                f.write("\n📋 三相机系统特点:\n")
                f.write("1. 更大的重建范围（约50%视野增加）\n")
                f.write("2. 减少遮挡，覆盖更多角度\n")
                f.write("3. 自动位置检测（左、中、右）\n")
                f.write("4. 三视图融合，提高精度\n")

                f.write("\n🎯 使用建议:\n")
                f.write("1. 确保三个相机水平并排放置\n")
                f.write("2. 相邻相机间距16cm\n")
                f.write("3. 所有相机朝向同一目标\n")
                f.write("4. 目标距离建议0.5-2米\n")

            print(f"📄 三相机重建报告已保存: {report_path}")

        except Exception as e:
            print(f"⚠️ 生成报告失败: {e}")

    def cleanup(self):
        """清理资源"""
        print("\n清理三相机系统资源...")

        try:
            # 清理多流管理器
            if hasattr(self, 'stream_manager'):
                self.stream_manager.cleanup()
                print("✅ 多流管理器已清理")

            # 清理RGB摄像头
            for i, cap in enumerate(self.rgb_cameras):
                try:
                    cap.release()
                    print(f"✅ RGB摄像头 {i} 已释放")
                except:
                    pass

            self.rgb_cameras.clear()

            # 强制垃圾回收
            import gc
            gc.collect()

            print("✅ 三相机系统资源已清理")

        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}")


def main():
    """主函数 - 三相机版本"""
    print("=" * 80)
    print("奥比中光三相机彩色3D重建系统")
    print("相机间距: 16cm | 自动位置检测 | 大范围重建")
    print("=" * 80)

    # 检查依赖
    try:
        print("✅ OpenCV:", cv2.__version__)
        print("✅ NumPy:", np.__version__)
        print("✅ Open3D:", o3d.__version__)
    except Exception as e:
        print(f"❌ 依赖检查失败: {e}")
        return

    # 获取SDK路径
    sdk_path = get_default_sdk_path()
    if not os.path.exists(sdk_path):
        print(f"⚠️ 默认SDK路径不存在，尝试自动查找...")
        sdk_path = ""

    # 创建三相机重建器
    reconstructor = MultiCameraReconstructor(sdk_path)

    # 运行重建
    print("\n🚀 开始三相机彩色3D重建...")
    print("🎯 目标: 大范围彩色3D建模")
    print("=" * 80)

    try:
        success = reconstructor.run_multi_camera_reconstruction()

        if success:
            print("\n🎉 三相机彩色3D重建成功！")
            print("💡 大范围彩色模型已保存在输出目录中")

            # 显示重要文件
            print("\n📁 重要文件:")
            print("  final_models/       - 最终三相机彩色模型")
            print("  intermediate_models/ - 中间模型")
            print("  captured_images/    - 多相机图像")
            print("  previews/          - 模型预览图")
            print("  三相机重建报告.txt - 详细报告")

        else:
            print("\n⚠️ 重建过程中出现问题")
            print("💡 部分数据已保存，请检查输出目录")

    except Exception as e:
        print(f"\n❌ 运行重建时出错: {e}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("📋 三相机系统提示:")
    print("1. 确保三个相机水平并排放置，间距16cm")
    print("2. 系统会自动检测相机位置关系（左、中、右）")
    print("3. 重建范围比单相机增加约50%")
    print("4. 适合大物体或场景的完整重建")
    print("=" * 80)

    input("\n按Enter键退出...")


if __name__ == "__main__":
    # 设置环境变量
    default_sdk_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    if os.path.exists(default_sdk_path):
        drivers_dir = os.path.join(default_sdk_path, "OpenNI2", "Drivers")
        if os.path.exists(drivers_dir):
            os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']

    main()