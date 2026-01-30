"""
多相机交替实时建模系统 - 使用primesense获取真实深度
三个奥比中光相机交替工作，电压安全，两两交替
"""

import sys
import os
import time
import threading
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import traceback
import queue

# ============================================
# 配置日志
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_camera_primesense.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# 检查primesense是否可用
# ============================================

try:
    import primesense
    from primesense import openni2
    PRIMESENSE_AVAILABLE = True
    logger.info("primesense库可用，可以使用OpenNI2获取深度数据")
except ImportError:
    PRIMESENSE_AVAILABLE = False
    logger.warning("primesense库不可用，将使用模拟深度数据")

# ============================================
# 配置参数
# ============================================

class MultiCameraConfig:
    """多相机配置"""

    # 相机数量
    NUM_CAMERAS = 3

    # 相机布局（16cm间距）
    CAMERA_POSITIONS = {
        0: (-0.16, 0.0, 0.0),  # 左相机
        1: (0.0, 0.0, 0.0),    # 中相机
        2: (0.16, 0.0, 0.0)    # 右相机
    }

    # 采集配置
    CAPTURE_CONFIG = {
        'color_resolution': (640, 480),  # 彩色图分辨率
        'depth_resolution': (640, 480),  # 深度图分辨率
        'target_fps': 30,                # 目标帧率
        'depth_scale': 0.001,            # 深度缩放因子（毫米转米）
        'depth_max': 8.0,                # 最大深度（米）
        'depth_min': 0.3                 # 最小深度（米）
    }

    # 电压优化配置
    VOLTAGE_OPTIMIZATION = {
        'max_simultaneous': 2,           # 同时激活的最大相机数
        'rotation_interval': 3.0,        # 切换间隔（秒）
        'warmup_time': 0.2,              # 相机预热时间（秒）
        'cooldown_time': 0.1,            # 相机冷却时间（秒）
        'active_combinations': [         # 激活组合
            [0, 1],  # 相机0和1
            [1, 2],  # 相机1和2
            [2, 0]   # 相机2和0
        ]
    }

    # 工作模式
    WORKING_MODES = {
        'MODE_01_12_20': [0, 1],        # 模式一：01 → 12 → 20 → 01
        'MODE_01_12': [0, 1]            # 模式二：01 → 12 → 01 → 12
    }

    # 相机内参（假设三个相机相同）
    INTRINSIC_MATRIX = np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0]
    ])

    @staticmethod
    def get_camera_extrinsic(camera_id: int) -> np.ndarray:
        """获取相机外参矩阵"""
        if camera_id not in MultiCameraConfig.CAMERA_POSITIONS:
            return np.eye(4)

        position = MultiCameraConfig.CAMERA_POSITIONS[camera_id]
        extrinsic = np.eye(4)
        extrinsic[0:3, 3] = position
        return extrinsic

    @staticmethod
    def get_intrinsic_matrix():
        """获取相机内参矩阵"""
        return MultiCameraConfig.INTRINSIC_MATRIX

# ============================================
# 工具函数
# ============================================

class MultiCameraUtils:
    """多相机工具类"""

    @staticmethod
    def depth_to_pointcloud(depth_image: np.ndarray, intrinsic_matrix: np.ndarray,
                           depth_scale: float = 0.001, max_depth: float = 8.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """深度图转换为点云"""
        if depth_image is None:
            return None, None

        try:
            height, width = depth_image.shape

            # 创建网格
            u = np.arange(width)
            v = np.arange(height)
            uu, vv = np.meshgrid(u, v)

            # 深度值
            z = depth_image.astype(np.float32) * depth_scale

            # 有效深度掩码
            valid_mask = (z > 0) & (z < max_depth)

            # 应用有效掩码
            u_valid = uu[valid_mask]
            v_valid = vv[valid_mask]
            z_valid = z[valid_mask]

            if len(z_valid) == 0:
                return None, None

            # 相机坐标系下的3D点
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]

            x_valid = (u_valid - cx) * z_valid / fx
            y_valid = (v_valid - cy) * z_valid / fy

            points = np.stack([x_valid, y_valid, z_valid], axis=-1)

            return points, valid_mask

        except Exception as e:
            logger.error(f"深度图转点云失败: {e}")
            return None, None

    @staticmethod
    def create_open3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """创建Open3D点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def apply_motion_compensation(extrinsic: np.ndarray, velocity: np.ndarray,
                                 time_diff: float) -> np.ndarray:
        """应用运动补偿（物体运动，相机静止）"""
        # 计算物体的位移
        displacement = velocity * time_diff

        # 创建补偿矩阵
        # 物体运动相当于相机坐标系中的点反向运动
        compensation = np.eye(4)
        compensation[0:3, 3] = -displacement

        # 应用补偿：修正后的外参 = 原始外参 × 补偿矩阵
        compensated_extrinsic = np.dot(extrinsic, compensation)

        return compensated_extrinsic

    @staticmethod
    def save_depth_image(depth_data: np.ndarray, filepath: str):
        """保存深度图像（伪彩色）"""
        try:
            # 转换为毫米并归一化
            depth_mm = (depth_data * 1000).astype(np.uint16)
            depth_normalized = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(filepath, depth_colored)

        except Exception as e:
            logger.error(f"保存深度图像失败: {e}")

    @staticmethod
    def save_color_image(color_data: np.ndarray, filepath: str):
        """保存彩色图像"""
        try:
            cv2.imwrite(filepath, color_data)
        except Exception as e:
            logger.error(f"保存彩色图像失败: {e}")

# ============================================
# 电压安全的相机控制器
# ============================================

class VoltageSafeCamera:
    """电压安全的单个相机控制器"""

    def __init__(self, camera_id: int, use_real_depth: bool = True):
        """
        初始化相机控制器

        Args:
            camera_id: 相机索引 (0,1,2)
            use_real_depth: 是否使用真实深度数据
        """
        self.camera_id = camera_id
        self.use_real_depth = use_real_depth and PRIMESENSE_AVAILABLE

        # 相机位置
        self.position = MultiCameraConfig.CAMERA_POSITIONS.get(camera_id, (0.0, 0.0, 0.0))

        # 外参矩阵
        self.extrinsic = np.eye(4)
        self.extrinsic[0:3, 3] = self.position

        # OpenNI2资源
        self.device = None
        self.depth_stream = None

        # OpenCV彩色相机
        self.color_camera = None

        # 状态
        self.initialized = False
        self.active = False

        # 线程安全锁
        self.lock = threading.RLock()

        # 缓存
        self.latest_color = None
        self.latest_depth = None
        self.last_update = 0

        # 统计
        self.frame_count = 0
        self.error_count = 0

        logger.info(f"初始化相机 {camera_id}，使用真实深度: {self.use_real_depth}")

    def initialize(self) -> bool:
        """初始化相机"""
        with self.lock:
            try:
                logger.info(f"相机 {self.camera_id}: 初始化...")

                if self.use_real_depth:
                    # 初始化OpenNI2
                    if not self._initialize_openni2():
                        logger.warning(f"相机 {self.camera_id}: OpenNI2初始化失败，将使用模拟深度")
                        self.use_real_depth = False

                # 初始化彩色相机
                if not self._initialize_color_camera():
                    logger.warning(f"相机 {self.camera_id}: 彩色相机初始化失败")

                self.initialized = True
                logger.info(f"相机 {self.camera_id} 初始化成功")
                return True

            except Exception as e:
                logger.error(f"相机 {self.camera_id} 初始化失败: {e}")
                return False

    def _initialize_openni2(self) -> bool:
        """初始化OpenNI2"""
        try:
            # 尝试初始化OpenNI2
            # 注意：需要设置OpenNI2的库路径
            openni2_initialized = False

            # 尝试不同的OpenNI2库路径
            possible_paths = [
                r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs",
                r"C:\Program Files\OpenNI2\Redist",
                r"C:\OpenNI2\Redist",
                ""
            ]

            for path in possible_paths:
                try:
                    if path and os.path.exists(path):
                        openni2.initialize(path)
                    else:
                        openni2.initialize()

                    openni2_initialized = True
                    logger.info(f"相机 {self.camera_id}: OpenNI2初始化成功，路径: {path}")
                    break
                except Exception as e:
                    logger.debug(f"相机 {self.camera_id}: OpenNI2初始化尝试失败，路径: {path}, 错误: {e}")
                    continue

            if not openni2_initialized:
                logger.error(f"相机 {self.camera_id}: 所有OpenNI2路径尝试都失败")
                return False

            # 尝试打开设备
            for attempt in range(3):
                try:
                    # 尝试打开指定索引的设备
                    # 注意：open_any()打开第一个可用设备，我们需要更精确的控制
                    devices = []
                    try:
                        # 获取设备列表
                        from primesense.utils import DeviceInfo
                        devices = DeviceInfo.get_connected_devices()
                        logger.info(f"相机 {self.camera_id}: 找到 {len(devices)} 个OpenNI2设备")
                    except:
                        pass

                    if devices and self.camera_id < len(devices):
                        # 使用设备URI打开特定设备
                        device_uri = devices[self.camera_id]['uri']
                        self.device = openni2.Device.open(device_uri)
                    else:
                        # 回退到打开任意设备
                        self.device = openni2.Device.open_any()

                    if self.device:
                        logger.info(f"相机 {self.camera_id}: OpenNI2设备打开成功")
                        break

                except Exception as e:
                    if attempt == 2:
                        logger.error(f"相机 {self.camera_id}: OpenNI2设备打开失败: {e}")
                        return False
                    time.sleep(0.5)

            return True

        except Exception as e:
            logger.error(f"相机 {self.camera_id}: OpenNI2初始化异常: {e}")
            return False

    def _initialize_color_camera(self) -> bool:
        """初始化彩色相机"""
        # 尝试不同的索引
        color_indices = [
            self.camera_id,        # 直接索引
            self.camera_id + 1,    # 偏移索引
            0, 1, 2, 3             # 通用索引
        ]

        for idx in color_indices:
            try:
                # 尝试使用DSHOW后端（Windows）
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

                if not cap.isOpened():
                    # 尝试其他后端
                    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

                if cap.isOpened():
                    # 测试读取
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # 设置分辨率
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                        # 设置帧率
                        cap.set(cv2.CAP_PROP_FPS, 30)

                        # 减少缓冲区
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except:
                            pass

                        self.color_camera = cap
                        logger.info(f"相机 {self.camera_id}: 彩色相机打开成功 (索引{idx})")

                        # 清空缓冲区
                        for _ in range(5):
                            cap.read()

                        return True
                    else:
                        cap.release()

            except Exception as e:
                logger.debug(f"相机 {self.camera_id}: 尝试索引{idx}失败: {e}")
                continue

        logger.info(f"相机 {self.camera_id}: 彩色相机初始化失败，仅使用深度数据")
        return False

    def activate(self) -> bool:
        """激活相机"""
        if not self.initialized or self.active:
            return False

        with self.lock:
            try:
                logger.info(f"相机 {self.camera_id}: 激活...")

                # 创建并启动深度流
                if self.use_real_depth and self.device and not self.depth_stream:
                    try:
                        self.depth_stream = self.device.create_depth_stream()
                        self.depth_stream.start()

                        # 预热：读取并丢弃前几帧
                        for _ in range(3):
                            try:
                                self.depth_stream.read_frame()
                            except:
                                pass
                            time.sleep(0.01)

                        logger.info(f"相机 {self.camera_id}: 深度流启动")
                    except Exception as e:
                        logger.error(f"相机 {self.camera_id}: 启动深度流失败: {e}")
                        self.use_real_depth = False

                self.active = True
                self.frame_count = 0
                self.error_count = 0

                # 激活后延时
                time.sleep(0.05)

                logger.info(f"相机 {self.camera_id} 激活完成")
                return True

            except Exception as e:
                logger.error(f"相机 {self.camera_id} 激活失败: {e}")
                return False

    def deactivate(self) -> bool:
        """停用相机"""
        if not self.active:
            return False

        with self.lock:
            try:
                logger.info(f"相机 {self.camera_id}: 停用...")

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

                logger.info(f"相机 {self.camera_id} 停用完成")
                return True

            except Exception as e:
                logger.error(f"相机 {self.camera_id} 停用失败: {e}")
                return False

    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """捕获一帧"""
        if not self.active:
            return None, None

        with self.lock:
            try:
                color_frame = None
                depth_frame = None

                # 捕获彩色帧
                if self.color_camera:
                    try:
                        ret, color_frame = self.color_camera.read()
                        if ret and color_frame is not None:
                            # 调整尺寸
                            if color_frame.shape[:2] != (480, 640):
                                color_frame = cv2.resize(color_frame, (640, 480))
                            # 转换为RGB
                            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        self.error_count += 1
                        if self.error_count % 10 == 0:
                            logger.debug(f"相机 {self.camera_id} 彩色采集错误: {e}")

                # 捕获深度帧
                if self.use_real_depth and self.depth_stream:
                    try:
                        frame_data = self.depth_stream.read_frame()
                        depth_buffer = frame_data.get_buffer_as_uint16()
                        depth_frame = np.frombuffer(depth_buffer, dtype=np.uint16).reshape(480, 640)
                        # 转换为米
                        depth_frame = depth_frame.astype(np.float32) * 0.001
                    except Exception as e:
                        self.error_count += 1
                        if self.error_count % 10 == 0:
                            logger.debug(f"相机 {self.camera_id} 深度采集错误: {e}")
                else:
                    # 生成模拟深度数据
                    depth_frame = self._generate_simulated_depth()

                # 更新缓存
                if color_frame is not None or depth_frame is not None:
                    self.latest_color = color_frame
                    self.latest_depth = depth_frame
                    self.last_update = time.time()
                    self.frame_count += 1

                return color_frame, depth_frame

            except Exception as e:
                logger.error(f"相机 {self.camera_id} 捕获异常: {e}")
                return None, None

    def _generate_simulated_depth(self) -> np.ndarray:
        """生成模拟深度数据"""
        height, width = 480, 640
        depth = np.ones((height, width), dtype=np.float32) * 2.0

        # 根据相机索引生成不同的场景
        center_x, center_y = width // 2, height // 2

        if self.camera_id == 0:  # 左相机
            for i in range(center_y-100, center_y+100):
                for j in range(center_x-150, center_x):
                    if 0 <= i < height and 0 <= j < width:
                        depth[i, j] = 1.0 + 0.2 * np.sin(i/50 + j/50)

        elif self.camera_id == 1:  # 中相机
            for i in range(center_y-100, center_y+100):
                for j in range(center_x-100, center_x+100):
                    if 0 <= i < height and 0 <= j < width:
                        depth[i, j] = 1.2 + 0.1 * np.cos(i/40 + j/40)

        else:  # 右相机
            for i in range(center_y-100, center_y+100):
                for j in range(center_x, center_x+150):
                    if 0 <= i < height and 0 <= j < width:
                        depth[i, j] = 0.8 + 0.15 * np.sin(i/60 - j/60)

        # 添加地面
        for i in range(height//2, height):
            for j in range(width):
                ground_depth = 1.8 + 0.005 * (i - height//2)
                depth[i, j] = min(depth[i, j], ground_depth)

        # 添加噪声
        noise = np.random.normal(0, 0.02, (height, width)).astype(np.float32)
        depth = np.clip(depth + noise, 0.3, 8.0)

        return depth

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
            depth_scale = 0.001
            max_depth = 8.0

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
            if self.extrinsic is not None:
                pcd.transform(self.extrinsic)

            return pcd

        except Exception as e:
            logger.error(f"相机 {self.camera_id}: 生成点云失败: {e}")
            return None

    def cleanup(self):
        """清理资源"""
        with self.lock:
            logger.info(f"相机 {self.camera_id}: 清理资源...")

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
            logger.info(f"相机 {self.camera_id}: 资源已清理")

# ============================================
# 电压优化的多相机系统
# ============================================

class VoltageOptimizedSystem:
    """电压优化的多相机系统（两两交替工作）"""

    def __init__(self, mode: str = "MODE_01_12_20", use_real_depth: bool = True):
        """
        初始化电压优化系统

        Args:
            mode: 工作模式 ("MODE_01_12_20" 或 "MODE_01_12")
            use_real_depth: 是否使用真实深度数据
        """
        self.mode = mode
        self.use_real_depth = use_real_depth

        # 相机控制器
        self.cameras = {}

        # 电压优化参数
        self.config = MultiCameraConfig.VOLTAGE_OPTIMIZATION

        # 根据模式调整激活组合
        if mode == "MODE_01_12":
            # 模式二：01 → 12 → 01 → 12
            self.config['active_combinations'] = [
                [0, 1],  # 相机0和1
                [1, 2],  # 相机1和2
                [0, 1],  # 相机0和1
                [1, 2]   # 相机1和2
            ]

        self.rotation_index = 0
        self.last_rotation_time = 0
        self.active_combo = []

        # 运动参数
        self.motion_params = {
            'velocity_x': 0.0,  # 物体横向平移速度 (m/s)
            'velocity_y': 0.0,
            'velocity_z': 0.0,
            'update_time': time.time()
        }

        # 采集线程
        self.running = False
        self.capture_thread = None
        self.capture_queue = queue.Queue(maxsize=20)

        # 统计数据
        self.stats = {
            'frames_captured': [0, 0, 0],
            'activations': [0, 0, 0],
            'rotation_count': 0,
            'start_time': 0,
            'total_frames': 0,
            'saved_images': 0
        }

        logger.info(f"电压优化系统初始化，模式: {mode}，使用真实深度: {use_real_depth}")

    def initialize(self) -> bool:
        """初始化系统"""
        print("=" * 70)
        print("电压优化的多相机系统")
        print(f"模式: {self.mode}")
        print(f"同时激活: {self.config['max_simultaneous']}个相机")
        print(f"切换间隔: {self.config['rotation_interval']}秒")
        print("=" * 70)

        if self.use_real_depth and not PRIMESENSE_AVAILABLE:
            logger.warning("primesense不可用，将使用模拟深度数据")
            self.use_real_depth = False

        # 初始化所有相机
        print(f"\n初始化 {MultiCameraConfig.NUM_CAMERAS} 个相机...")

        for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
            camera = VoltageSafeCamera(cam_id, self.use_real_depth)
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

        # 等待冷却时间
        time.sleep(self.config['cooldown_time'])

        # 选择新的组合
        combo_list = self.config['active_combinations']
        self.active_combo = combo_list[self.rotation_index % len(combo_list)]

        print(f"\n🔄 切换到组合 {self.active_combo} (轮换 {self.rotation_index + 1})")

        # 预热延时
        time.sleep(self.config['warmup_time'])

        # 激活新的组合
        success_count = 0
        for cam_id in self.active_combo:
            if cam_id in self.cameras:
                if self.cameras[cam_id].activate():
                    success_count += 1
                    self.stats['activations'][cam_id] += 1

        if success_count > 0:
            self.rotation_index += 1
            self.last_rotation_time = time.time()
            self.stats['rotation_count'] += 1
            return True
        else:
            print(f"⚠️ 组合激活失败，回退到 {old_combo}")
            self.active_combo = old_combo
            return False

    def capture_worker(self):
        """采集工作线程"""
        logger.info("采集线程启动...")

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
                            # 应用运动补偿
                            if depth is not None:
                                # 计算时间差
                                time_diff = current_time - self.motion_params['update_time']

                                # 获取运动速度
                                velocity = np.array([
                                    self.motion_params['velocity_x'],
                                    self.motion_params['velocity_y'],
                                    self.motion_params['velocity_z']
                                ])

                                # 应用运动补偿到外参
                                camera = self.cameras[cam_id]
                                compensated_extrinsic = MultiCameraUtils.apply_motion_compensation(
                                    camera.extrinsic, velocity, time_diff
                                )

                                # 存储补偿后的外参
                                camera.extrinsic = compensated_extrinsic

                            frames[cam_id] = {
                                'color': color,
                                'depth': depth,
                                'timestamp': current_time,
                                'camera_index': cam_id,
                                'frame_index': self.cameras[cam_id].frame_count
                            }

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
                logger.error(f"采集线程错误: {e}")
                time.sleep(0.1)

        logger.info("采集线程停止")

    def start_capture(self):
        """开始采集"""
        if self.running:
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
        self.capture_thread.start()

        logger.info(f"采集开始，轮换间隔: {self.config['rotation_interval']}秒")

    def stop_capture(self):
        """停止采集"""
        if not self.running:
            return

        logger.info("停止采集...")
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        # 停用所有相机
        for cam_id in self.cameras:
            self.cameras[cam_id].deactivate()

        logger.info("采集停止")

    def get_frames(self, timeout: float = 0.1):
        """获取最新的帧数据"""
        try:
            return self.capture_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def update_motion_params(self, velocity_x: float = 0.0,
                            velocity_y: float = 0.0,
                            velocity_z: float = 0.0):
        """更新运动参数（物体运动速度）"""
        self.motion_params.update({
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
            'update_time': time.time()
        })
        logger.info(f"运动参数更新: 物体速度 vx={velocity_x:.3f} m/s")

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
            'motion_params': self.motion_params.copy(),
            'saved_images': self.stats['saved_images']
        }

        return stats

    def save_frames(self, frames: Dict, output_dir: str, save_interval: int = 5):
        """保存帧数据"""
        try:
            for cam_id, frame_data in frames.items():
                frame_index = frame_data['frame_index']

                # 每5帧保存一次
                if frame_index % save_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    # 保存彩色图
                    if frame_data['color'] is not None:
                        color_path = os.path.join(
                            output_dir, "color",
                            f"color_cam{cam_id}_f{frame_index:04d}_{timestamp}.jpg"
                        )
                        MultiCameraUtils.save_color_image(frame_data['color'], color_path)

                    # 保存深度图
                    if frame_data['depth'] is not None:
                        depth_path = os.path.join(
                            output_dir, "depth",
                            f"depth_cam{cam_id}_f{frame_index:04d}_{timestamp}.png"
                        )
                        MultiCameraUtils.save_depth_image(frame_data['depth'], depth_path)

                    self.stats['saved_images'] += 1

        except Exception as e:
            logger.error(f"保存帧数据失败: {e}")

    def cleanup(self):
        """清理系统"""
        logger.info("\n清理系统...")

        self.stop_capture()

        for camera in self.cameras.values():
            camera.cleanup()

        self.cameras.clear()

        # 卸载OpenNI2
        if PRIMESENSE_AVAILABLE:
            try:
                openni2.unload()
                logger.info("OpenNI2已卸载")
            except:
                pass

        logger.info("系统清理完成")

# ============================================
# 主程序
# ============================================

def main():
    """主程序"""
    print("="*70)
    print("多相机交替实时建模系统 - primesense版")
    print("使用真实奥比中光相机深度数据")
    print("="*70)

    # 选择工作模式
    print("\n请选择工作模式:")
    print("1. 模式一: 01 → 12 → 20 → 01 (循环)")
    print("2. 模式二: 01 → 12 → 01 → 12 (循环)")

    choice = input("请输入选择 (1 或 2, 默认为1): ").strip()
    if choice == "2":
        mode = "MODE_01_12"
    else:
        mode = "MODE_01_12_20"

    # 设置运行时间
    try:
        duration = int(input("请输入运行时间(秒, 默认为30): ").strip() or "30")
    except ValueError:
        duration = 30

    # 是否显示图像
    show_images = input("显示实时图像吗? (y/n, 默认为n): ").strip().lower()
    show_images = show_images == 'y'

    # 是否使用真实深度
    use_real_depth = input("使用真实深度数据吗? (y/n, 默认为y): ").strip().lower()
    use_real_depth = use_real_depth != 'n'

    # 图像保存频率
    try:
        save_interval = int(input("图像保存间隔(帧数, 默认为5): ").strip() or "5")
    except ValueError:
        save_interval = 5

    print(f"\n系统配置:")
    print(f"  工作模式: {mode}")
    print(f"  运行时间: {duration} 秒")
    print(f"  相机切换间隔: 3 秒")
    print(f"  图像保存频率: 每{save_interval}帧保存一次")
    print(f"  使用真实深度: {'是' if use_real_depth else '否'}")
    print(f"  实时显示: {'启用' if show_images else '禁用'}")
    print("-" * 40)

    # 创建输出目录
    output_dir = f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 创建电压优化系统
    print("\n初始化多相机系统...")
    system = VoltageOptimizedSystem(mode=mode, use_real_depth=use_real_depth)

    if not system.initialize():
        print("系统初始化失败")
        return

    # 启动系统
    print("启动多相机系统...")
    system.start_capture()

    try:
        # 运行指定时间
        start_time = time.time()
        last_status_time = start_time
        last_motion_update = start_time
        last_save_time = start_time

        print(f"系统运行中 ({duration}秒)...")
        print("按 Ctrl+C 可提前终止\n")

        # 创建显示窗口
        if show_images:
            cv2.namedWindow("多相机交替采集", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("多相机交替采集", 1200, 800)

        while time.time() - start_time < duration:
            # 获取帧数据
            frames, timestamp = system.get_frames(timeout=0.01)

            if frames:
                # 保存图像
                current_time = time.time()
                if current_time - last_save_time > 1.0:  # 每秒保存一次
                    system.save_frames(frames, output_dir, save_interval)
                    last_save_time = current_time

                # 显示图像
                if show_images and frames:
                    displays = []

                    for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
                        display = np.zeros((300, 400, 3), dtype=np.uint8)

                        # 状态标签
                        if cam_id in system.active_combo:
                            status_color = (0, 0, 255)  # 红色，激活
                            status_text = "ACTIVE"
                        else:
                            status_color = (100, 100, 100)  # 灰色，非激活
                            status_text = "INACTIVE"

                        if cam_id in frames:
                            frame_data = frames[cam_id]

                            # 显示彩色图
                            if frame_data['color'] is not None:
                                color_resized = cv2.resize(frame_data['color'], (400, 300))
                                display = cv2.cvtColor(color_resized, cv2.COLOR_RGB2BGR)

                            # 帧数信息
                            frame_count = system.stats['frames_captured'][cam_id]
                            cv2.putText(display, f"Frames: {frame_count}", (10, 280),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
                                    display3_padded[:, padding:padding+display3.shape[1]] = display3
                                    display3 = display3_padded

                                combined = np.vstack([row1, display3])
                            else:
                                combined = row1

                            cv2.imshow("多相机交替采集", combined)

            # 定期更新运动参数（模拟物体横向平移）
            current_time = time.time()
            if current_time - last_motion_update > 5.0:
                # 物体以0.05 m/s的速度横向平移
                system.update_motion_params(velocity_x=0.05)
                last_motion_update = current_time

            # 定期显示状态
            if current_time - last_status_time > 2.0:
                stats = system.get_statistics()
                print(f"[{current_time - start_time:6.1f}s] ", end="")
                print(f"激活相机: {stats['active_combo']}, ", end="")
                print(f"下次切换: {stats['next_rotation_in']:4.1f}s, ", end="")
                print(f"总帧数: {stats['total_frames']:4d}, ", end="")
                print(f"帧率: {stats['average_fps']:5.1f} FPS, ", end="")
                print(f"物体速度: {stats['motion_params']['velocity_x']:5.3f} m/s, ", end="")
                print(f"保存图像: {stats['saved_images']}")

                last_status_time = current_time

            # 检查ESC键
            if show_images:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n用户退出")
                    break

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n运行异常: {e}")
        traceback.print_exc()
    finally:
        # 停止系统
        print("\n停止系统...")
        system.stop_capture()

        if show_images:
            cv2.destroyAllWindows()

        # 显示最终状态
        stats = system.get_statistics()
        print("\n" + "="*70)
        print("系统运行完成!")
        print("="*70)
        print(f"总运行时间: {stats['elapsed_time']:.1f}秒")
        print(f"总帧数: {stats['total_frames']}")
        print(f"切换次数: {stats['rotation_count']}")
        print(f"平均帧率: {stats['average_fps']:.1f} FPS")
        print(f"保存图像: {stats['saved_images']}")
        print(f"输出目录: {output_dir}")

        # 相机统计
        print(f"\n相机统计:")
        for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
            if cam_id < len(stats['frames_per_camera']):
                print(f"  相机 {cam_id}: {stats['frames_per_camera'][cam_id]} 帧, "
                      f"{stats['fps_per_camera'][cam_id]:.1f} FPS")

        print("="*70)

        # 清理系统
        system.cleanup()

if __name__ == "__main__":
    main()