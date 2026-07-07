"""
三相机系统配置 - 针对奥比中光Astra相机优化
"""

import numpy as np
from datetime import datetime
import os

class MultiCameraConfig:
    # ==================== 系统配置 ====================
    NUM_CAMERAS = 3
    CAMERA_POSITIONS = ['left', 'center', 'right']

    # 相机布局参数（单位：米）
    CAMERA_SPACING = 0.16  # 相机中心点间距16cm
    CAMERA_BASELINE = 0.16  # 基线长度

    # ==================== 电压优化配置 ====================
    VOLTAGE_OPTIMIZATION = {
        'enabled': True,               # 启用电压优化
        'mode': 'alternating',         # 交替模式
        'max_simultaneous': 2,         # 最大同时激活数
        'rotation_interval': 5.0,      # 轮换间隔（秒）
        'warmup_time': 0.2,            # 预热时间（秒）
        'cooldown_time': 0.1,          # 冷却时间（秒）
        'active_combinations': [
            [0, 1],  # 左+中
            [1, 2],  # 中+右
            [0, 2],  # 左+右（对角线）
        ],
    }

    # ==================== 采集配置 ====================
    CAPTURE_CONFIG = {
        # 分辨率设置
        'color_resolution': (640, 480),
        'depth_resolution': (640, 480),

        # 帧率设置
        'target_fps': 15,              # 目标帧率
        'depth_fps': 15,
        'color_fps': 15,

        # 深度参数
        'depth_scale': 0.001,          # 深度缩放因子（米/单位）
        'depth_min': 0.3,              # 最小深度（米）
        'depth_max': 2.0,              # 最大深度（米）

        # 采集控制
        'auto_exposure': True,
        'auto_white_balance': True,
        'exposure_compensation': 0,

        # 数据保存
        'save_interval': 5,            # 保存间隔（帧）
        'capture_duration': 30,        # 采集时长（秒）
        'max_frames': 300,             # 最大帧数
    }

    # ==================== 相机内参 ====================
    # Astra相机默认内参（640x480）
    INTRINSIC_MATRIX = np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0]
    ])

    # 相机畸变系数（假设无畸变）
    DISTORTION_COEFFS = np.zeros(5)

    # ==================== 相机外参（标定后更新） ====================
    # 初始外参矩阵（基于16cm间距的并排摆放）
    EXTRINSIC_MATRICES = {
        # 左侧相机（X方向偏移-0.16m）
        0: np.array([
            [1.0, 0.0, 0.0, -0.16],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),

        # 中心相机（原点）
        1: np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),

        # 右侧相机（X方向偏移+0.16m）
        2: np.array([
            [1.0, 0.0, 0.0, 0.16],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }

    # ==================== 3D重建配置 ====================
    RECONSTRUCTION_CONFIG = {
        'voxel_size': 0.01,            # 体素大小（米）
        'tsdf_trunc': 0.04,            # TSDF截断距离
        'icp_threshold': 0.02,         # ICP配准阈值
        'integration_length': 100,     # 积分长度
        'enable_color': True,          # 启用颜色
        'mesh_resolution': 0.005,      # 网格分辨率
        'enable_smoothing': True,      # 启用平滑
        'smoothing_iterations': 3,     # 平滑迭代次数
    }

    # ==================== 标定配置 ====================
    CALIBRATION_CONFIG = {
        'checkerboard_size': (9, 6),   # 棋盘格角点数
        'checkerboard_square_size': 0.025,  # 棋盘格方格大小（米）
        'min_calibration_images': 20,  # 最少标定图像数
        'max_calibration_images': 50,  # 最多标定图像数
        'capture_interval': 1.0,       # 采集间隔（秒）
    }

    # ==================== 输出配置 ====================
    OUTPUT_CONFIG = {
        'base_dir': 'outputs',
        'save_pointcloud': True,
        'save_mesh': True,
        'save_depth_maps': False,
        'save_color_images': True,
        'compression_level': 9,        # PLY压缩级别
    }

    @staticmethod
    def get_intrinsic_matrix():
        """获取内参矩阵"""
        return MultiCameraConfig.INTRINSIC_MATRIX.copy()

    @staticmethod
    def get_camera_extrinsic(camera_id):
        """获取相机外参矩阵"""
        if camera_id in MultiCameraConfig.EXTRINSIC_MATRICES:
            return MultiCameraConfig.EXTRINSIC_MATRICES[camera_id].copy()
        return np.eye(4)

    @staticmethod
    def get_output_dir(category='default'):
        """获取输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = MultiCameraConfig.OUTPUT_CONFIG['base_dir']
        output_dir = os.path.join(base_dir, f"{category}_{timestamp}")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def get_calibration_dir():
        """获取标定目录"""
        return os.path.join(MultiCameraConfig.OUTPUT_CONFIG['base_dir'], 'calibration')

    @staticmethod
    def update_extrinsics(camera_id, extrinsic_matrix):
        """更新外参矩阵"""
        if camera_id in MultiCameraConfig.EXTRINSIC_MATRICES:
            MultiCameraConfig.EXTRINSIC_MATRICES[camera_id] = extrinsic_matrix.copy()
            return True
        return False