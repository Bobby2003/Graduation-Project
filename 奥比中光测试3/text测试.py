"""
基于奥比中光Astra相机的3D重建原理演示系统 - 修复版
从基础数据采集到完整3D重建的完整流程展示
作者：Bobby2003
日期：2024
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime
from primesense import openni2
from primesense import _openni2 as c_api
from scipy import ndimage
from sklearn.cluster import DBSCAN
import trimesh


class Astra3DReconstructor:
    """Astra相机3D重建器 - 修复版"""

    def __init__(self, driver_path=""):
        self.driver_path = driver_path
        self.device = None
        self.depth_stream = None
        self.ir_stream = None

        # 原始数据存储
        self.raw_depth = None
        self.raw_ir = None
        self.processed_depth = None
        self.processed_ir = None

        # 处理步骤记录
        self.processing_steps = []
        self.step_times = []

        # 相机参数 (Astra相机内参)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525.0, 525.0, 319.5, 239.5
        )

        # 处理参数 - 修复双边滤波参数
        self.params = {
            'depth_scale': 0.001,
            'depth_clip_min': 0.6,  # 确保楼梯在最小深度以内
            'depth_clip_max': 8.0,  # 确保楼梯在最大深度以内
            'median_filter_size': 1,  # 不使用中值滤波（核大小为1相当于不滤波）
            'bilateral_d': 0,  # 如果不使用双边滤波，可以设置为0，并在代码中跳过双边滤波
            'bilateral_sigma_color': 50,
            'bilateral_sigma_space': 50,
            'outlier_std': 2.0,  # 增加离群点去除的标准差倍数，避免去除有效点
            'voxel_size': 0.005,  # 减小体素大小，保留更多点
            'normal_radius': 0.01,
            'poisson_depth': 10,  # 增加Poisson重建深度，以获取更多细节
            'mesh_simplify': 0.8,  # 减少简化比例，保留更多面片
        }

        print("🚀 Astra 3D重建器初始化完成")

    # ==================== 1. 相机初始化与数据采集 ====================

    def initialize_camera(self):
        """初始化相机"""
        print("=" * 60)
        print("步骤1: 初始化Astra相机")
        print("=" * 60)

        start_time = time.time()

        try:
            # 初始化OpenNI2
            if self.driver_path and os.path.exists(self.driver_path):
                openni2.initialize(self.driver_path)
                print(f"✅ 使用驱动路径: {self.driver_path}")
            else:
                openni2.initialize()
                print("✅ 使用默认驱动路径")

            # 打开设备
            self.device = openni2.Device.open_any()
            dev_info = self.device.get_device_info()
            device_name = dev_info.name.decode('utf-8', errors='ignore')
            print(f"✅ 设备已连接: {device_name}")

            # 创建并启动深度流
            self.depth_stream = self.device.create_depth_stream()
            self.depth_stream.start()
            print("✅ 深度流已启动")

            # 创建并启动红外流
            self.ir_stream = self.device.create_ir_stream()
            self.ir_stream.start()
            print("✅ 红外流已启动")

            elapsed = time.time() - start_time
            self._record_step("相机初始化", elapsed, "初始化相机并启动数据流")
            return True

        except Exception as e:
            print(f"❌ 相机初始化失败: {e}")
            return False

    def capture_single_frame(self):
        """采集单帧数据"""
        print("\n" + "=" * 60)
        print("步骤2: 采集单帧数据")
        print("=" * 60)

        start_time = time.time()

        try:
            # 采集深度帧
            depth_frame = self.depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            self.raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

            # 采集红外帧
            ir_frame = self.ir_stream.read_frame()
            ir_data = ir_frame.get_buffer_as_uint16()
            self.raw_ir = np.frombuffer(ir_data, dtype=np.uint16).reshape(480, 640)

            # 转换为米单位
            self.raw_depth = self.raw_depth.astype(np.float32) * self.params['depth_scale']

            print(f"✅ 数据采集成功")
            print(f"   深度图: {self.raw_depth.shape}, 范围: {self.raw_depth.min():.3f}-{self.raw_depth.max():.3f}米")
            print(f"   红外图: {self.raw_ir.shape}, 范围: {self.raw_ir.min()}-{self.raw_ir.max()}")

            # 显示原始数据
            self._display_raw_data()

            elapsed = time.time() - start_time
            self._record_step("数据采集", elapsed, f"采集深度和红外图像")

            return True

        except Exception as e:
            print(f"❌ 数据采集失败: {e}")
            return False

    def _display_raw_data(self):
        """显示原始数据"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 显示深度图
        depth_display = self._depth_to_colormap(self.raw_depth)
        axes[0].imshow(cv2.cvtColor(depth_display, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Raw Depth\nRange: {self.raw_depth.min():.3f}-{self.raw_depth.max():.3f}m")
        axes[0].axis('off')

        # 显示红外图
        ir_normalized = self._normalize_ir_image(self.raw_ir)
        axes[1].imshow(ir_normalized, cmap='gray')
        axes[1].set_title(f"Raw IR\nRange: {self.raw_ir.min()}-{self.raw_ir.max()}")
        axes[1].axis('off')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'raw_data_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📸 Raw data saved: raw_data_{timestamp}.png")
        plt.close(fig)

    def save_raw_data(self, filename_prefix="raw_frame"):
        """保存原始数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存深度图
        depth_filename = f"{filename_prefix}_{timestamp}_depth.npy"
        np.save(depth_filename, self.raw_depth)

        # 保存红外图
        ir_filename = f"{filename_prefix}_{timestamp}_ir.npy"
        np.save(ir_filename, self.raw_ir)

        # 保存为图像文件
        depth_img = self._depth_to_colormap(self.raw_depth)
        ir_img = self._normalize_ir_image(self.raw_ir)

        cv2.imwrite(depth_filename.replace('.npy', '.png'), depth_img)
        cv2.imwrite(ir_filename.replace('.npy', '.png'), ir_img)

        # 保存参数
        params_filename = f"{filename_prefix}_{timestamp}_params.json"
        with open(params_filename, 'w') as f:
            json.dump(self.params, f, indent=2)

        print(f"📁 原始数据已保存:")
        print(f"   深度图: {depth_filename}")
        print(f"   红外图: {ir_filename}")
        print(f"   参数文件: {params_filename}")

        return True

    # ==================== 2. 数据预处理与清洗 ====================

    def preprocess_data(self):
        """数据预处理"""
        print("\n" + "=" * 60)
        print("步骤3: 数据预处理与清洗")
        print("=" * 60)

        start_time = time.time()

        if self.raw_depth is None or self.raw_ir is None:
            print("❌ 无原始数据可处理")
            return False

        # 深度图处理
        print("\n📊 深度图处理流程:")
        self._process_depth_image()

        # 红外图处理
        print("\n📊 红外图处理流程:")
        self._process_ir_image()

        # 数据分析
        self._analyze_data()

        elapsed = time.time() - start_time
        self._record_step("数据预处理", elapsed, "深度图和红外图的预处理与清洗")

        return True

    def _process_depth_image(self):
        """处理深度图 - 修复双边滤波问题"""
        depth = self.raw_depth.copy()

        print("  1. 深度范围裁剪")
        depth[depth < self.params['depth_clip_min']] = 0
        depth[depth > self.params['depth_clip_max']] = 0

        valid_count = np.sum(depth > 0)
        total_count = depth.size
        valid_percent = valid_count / total_count * 100
        print(f"    有效深度点: {valid_count}/{total_count} ({valid_percent:.1f}%)")

        print("  2. 中值滤波去噪")
        depth_filtered = ndimage.median_filter(depth, size=self.params['median_filter_size'])

        print("  3. 改进的双边滤波")
        depth_filled = self._improved_bilateral_filter(depth_filtered)

        self.processed_depth = depth_filled

        # 显示处理效果
        self._display_depth_processing_steps()

        return True

    def _improved_bilateral_filter(self, depth):
        """改进的双边滤波 - 处理格式问题"""
        # 创建深度图的副本
        depth_filled = depth.copy()

        # 如果双边滤波参数d<=0，则跳过双边滤波，直接进行空洞填充
        if self.params['bilateral_d'] <= 0:
            print("    ⚠️ 跳过双边滤波")
            depth_filled = self._fill_depth_holes(depth_filled)
            return depth_filled

        # 创建掩码（有效点）
        mask = (depth > 0).astype(np.uint8)

        if np.sum(mask) > 0:
            try:
                print("    a. 将深度图转换为8位用于滤波")
                # 将有效深度值缩放到0-255范围
                valid_depth = depth_filled[depth_filled > 0]
                if len(valid_depth) > 0:
                    min_depth, max_depth = valid_depth.min(), valid_depth.max()
                    if max_depth > min_depth:
                        # 归一化到0-255
                        depth_normalized = np.zeros_like(depth_filled, dtype=np.uint8)
                        depth_normalized[mask > 0] = ((depth_filled[mask > 0] - min_depth) /
                                                      (max_depth - min_depth) * 255).astype(np.uint8)

                        print("    b. 应用双边滤波")
                        # 应用双边滤波到8位图像
                        depth_filtered = cv2.bilateralFilter(
                            depth_normalized,
                            d=self.params['bilateral_d'],
                            sigmaColor=self.params['bilateral_sigma_color'],
                            sigmaSpace=self.params['bilateral_sigma_space']
                        )

                        print("    c. 还原深度值")
                        # 将滤波后的图像转换回深度值
                        depth_filled[mask > 0] = depth_filtered[mask > 0].astype(np.float32) / 255 * (max_depth - min_depth) + min_depth

            except Exception as e:
                print(f"    ⚠️ 双边滤波失败，跳过: {e}")
                # 如果双边滤波失败，使用中值滤波结果
                pass

        print("    d. 空洞填充")
        # 使用形态学操作和插值填充空洞
        depth_filled = self._fill_depth_holes(depth_filled)

        return depth_filled

    def _process_ir_image(self):
        """处理红外图"""
        ir = self.raw_ir.copy()

        print("  1. 去除极端值")
        # 计算百分位数，只处理非零值
        ir_nonzero = ir[ir > 0]
        if len(ir_nonzero) > 0:
            p1, p99 = np.percentile(ir_nonzero, [1, 99])
            ir_clipped = np.clip(ir, p1, p99)
        else:
            ir_clipped = ir
            print("    ⚠️ 无有效红外数据")

        print("  2. 对比度增强")
        # 直方图均衡化
        if ir_clipped.max() > ir_clipped.min():
            ir_normalized = cv2.normalize(ir_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            ir_equalized = cv2.equalizeHist(ir_normalized)
        else:
            ir_equalized = ir_clipped.astype(np.uint8)
            print("    ⚠️ 对比度增强跳过，动态范围太小")

        print("  3. 高斯平滑")
        ir_smoothed = cv2.GaussianBlur(ir_equalized, (3, 3), 1.0)

        self.processed_ir = ir_smoothed

        # 显示处理效果
        self._display_ir_processing_steps()

        return True

    def _fill_depth_holes(self, depth):
        """填充深度图中的空洞"""
        depth_filled = depth.copy()
        mask = (depth > 0).astype(np.uint8)

        if np.sum(mask) > 0:
            try:
                # 使用形态学操作扩大有效区域
                kernel = np.ones((3, 3), np.uint8)
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)

                # 将深度转换为uint16用于inpaint
                depth_uint16 = (depth_filled * 1000).astype(np.uint16)

                # 使用inpaint填充小空洞
                inpaint_mask = 255 - mask_dilated * 255

                # 只对确实有空洞的区域进行处理
                if np.sum(inpaint_mask > 0) > 0:
                    depth_filled_uint16 = cv2.inpaint(depth_uint16, inpaint_mask, 3, cv2.INPAINT_TELEA)
                    depth_filled = depth_filled_uint16.astype(np.float32) / 1000
            except Exception as e:
                print(f"    ⚠️ 空洞填充失败: {e}")

        return depth_filled

    def _analyze_data(self):
        """数据分析"""
        print("\n📈 数据分析:")

        # 深度图分析
        valid_depth = self.processed_depth[self.processed_depth > 0]

        if len(valid_depth) > 0:
            print(f"  深度统计:")
            print(f"    有效点数: {len(valid_depth)}")
            print(f"    均值: {valid_depth.mean():.3f}米")
            print(f"    标准差: {valid_depth.std():.3f}米")
            print(f"    中位数: {np.median(valid_depth):.3f}米")
            print(f"    最小值: {valid_depth.min():.3f}米")
            print(f"    最大值: {valid_depth.max():.3f}米")

            # 计算直方图
            hist, bins = np.histogram(valid_depth, bins=50, range=(valid_depth.min(), valid_depth.max()))
            if len(hist) > 0:
                max_bin_idx = np.argmax(hist)
                max_bin = (bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2
                print(f"    最频繁深度: {max_bin:.3f}米")
        else:
            print(f"  深度统计: 无有效深度数据")

        # 红外图分析
        if self.processed_ir is not None:
            ir_flat = self.processed_ir.flatten()
            print(f"\n  红外统计:")
            print(f"    均值: {ir_flat.mean():.1f}")
            print(f"    标准差: {ir_flat.std():.1f}")
            print(f"    动态范围: {ir_flat.min()}-{ir_flat.max()}")
        else:
            print(f"\n  红外统计: 无红外数据")

    def _display_depth_processing_steps(self):
        """显示深度图处理步骤"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 准备要显示的图像
        depth_clipped = self._apply_depth_clip(self.raw_depth)
        depth_median = ndimage.median_filter(depth_clipped, size=self.params['median_filter_size'])

        images = [
            ("Raw Depth", self.raw_depth),
            ("Range Clipped", depth_clipped),
            ("Median Filter", depth_median),
            ("Final Processed", self.processed_depth)
        ]

        for i, (title, img) in enumerate(images):
            if i < len(axes):
                ax = axes[i]
                if img is not None:
                    # 显示深度图（伪彩色）
                    depth_display = self._depth_to_colormap(img)
                    ax.imshow(cv2.cvtColor(depth_display, cv2.COLOR_BGR2RGB))
                    ax.set_title(title)
                    ax.axis('off')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'depth_processing_steps_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📸 Depth processing steps saved: depth_processing_steps_{timestamp}.png")
        plt.close(fig)

    def _display_ir_processing_steps(self):
        """显示红外图处理步骤"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 准备要显示的图像
        ir_clipped = self._clip_ir_extremes(self.raw_ir) if self.raw_ir is not None else None

        images = [
            ("Raw IR", self.raw_ir),
            ("Range Clipped", ir_clipped),
            ("Contrast Enhanced",
             cv2.equalizeHist(self._normalize_ir_image(ir_clipped)) if ir_clipped is not None else None),
            ("Final Processed", self.processed_ir)
        ]

        for i, (title, img) in enumerate(images):
            if i < len(axes):
                ax = axes[i]
                if img is not None:
                    ax.imshow(img, cmap='gray')
                    ax.set_title(title)
                    ax.axis('off')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'ir_processing_steps_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📸 IR processing steps saved: ir_processing_steps_{timestamp}.png")
        plt.close(fig)

    def _apply_depth_clip(self, depth):
        """应用深度裁剪"""
        if depth is None:
            return None
        depth_clipped = depth.copy()
        depth_clipped[depth_clipped < self.params['depth_clip_min']] = 0
        depth_clipped[depth_clipped > self.params['depth_clip_max']] = 0
        return depth_clipped

    def _clip_ir_extremes(self, ir):
        """裁剪红外图极端值"""
        if ir is None:
            return None

        ir_nonzero = ir[ir > 0]
        if len(ir_nonzero) > 0:
            p1, p99 = np.percentile(ir_nonzero, [1, 99])
            return np.clip(ir, p1, p99)
        return ir

    # ==================== 3. 点云生成与处理 ====================

    def generate_pointcloud(self):
        """生成点云"""
        print("\n" + "=" * 60)
        print("步骤4: 点云生成与处理")
        print("=" * 60)

        start_time = time.time()

        if self.processed_depth is None:
            print("❌ 无处理后的深度数据")
            return None

        print("🔄 从深度图生成点云...")

        # 创建Open3D深度图像
        depth_image = o3d.geometry.Image((self.processed_depth * 1000).astype(np.uint16))

        # 生成点云
        try:
            pcd_raw = o3d.geometry.PointCloud.create_from_depth_image(
                depth_image,
                self.intrinsic,
                depth_scale=1000.0,
                depth_trunc=self.params['depth_clip_max']
            )
        except Exception as e:
            print(f"❌ 点云生成失败: {e}")
            return None

        print(f"✅ 原始点云生成: {len(pcd_raw.points)} 个点")

        # 点云处理步骤
        print("\n📊 点云处理流程:")

        # 1. 移除无效点
        print("  1. 移除无效点")
        points = np.asarray(pcd_raw.points)
        valid_mask = np.all(np.isfinite(points), axis=1)
        pcd_filtered = pcd_raw.select_by_index(np.where(valid_mask)[0])
        removed_invalid = len(points) - len(pcd_filtered.points)
        print(f"    移除 {removed_invalid} 个无效点")

        # 2. 移除离群点
        print("  2. 统计离群点移除")
        if len(pcd_filtered.points) > 20:  # 需要有足够多的点
            pcd_filtered, inliers = pcd_filtered.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=self.params['outlier_std']
            )
            print(f"    移除 {len(valid_mask) - len(inliers) - removed_invalid} 个离群点")
        else:
            print("    ⚠️ 点数不足，跳过离群点移除")

        # 3. 体素降采样
        print("  3. 体素降采样")
        if len(pcd_filtered.points) > 100:  # 需要有足够多的点
            pcd_down = pcd_filtered.voxel_down_sample(voxel_size=self.params['voxel_size'])
            print(f"    降采样后: {len(pcd_down.points)} 个点")
        else:
            pcd_down = pcd_filtered
            print("    ⚠️ 点数不足，跳过降采样")

        # 4. 估计法向量
        print("  4. 估计法向量")
        if len(pcd_down.points) > 30:  # 需要有足够多的点
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.params['normal_radius'], max_nn=30
                )
            )
        else:
            print("    ⚠️ 点数不足，跳过法向量估计")

        # 5. 为点云着色
        print("  5. 为点云着色")
        self._color_pointcloud_with_ir(pcd_down)

        # 保存中间点云
        self._save_pointcloud_stages(pcd_raw, pcd_filtered, pcd_down)

        # 点云分析
        self._analyze_pointcloud(pcd_down)

        elapsed = time.time() - start_time
        self._record_step("点云生成", elapsed, f"生成并处理点云，最终点数: {len(pcd_down.points)}")

        return pcd_down

    def _color_pointcloud_with_ir(self, pcd):
        """使用红外图像为点云着色"""
        if self.processed_ir is None or len(pcd.points) == 0:
            # 使用统一颜色
            colors = np.ones((len(pcd.points), 3)) * 0.7
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return

        points = np.asarray(pcd.points)
        colors = np.zeros((len(points), 3))

        # 将3D点投影回2D图像坐标
        fx, fy = self.intrinsic.intrinsic_matrix[0, 0], self.intrinsic.intrinsic_matrix[1, 1]
        cx, cy = self.intrinsic.intrinsic_matrix[0, 2], self.intrinsic.intrinsic_matrix[1, 2]

        colored_count = 0
        for i, point in enumerate(points):
            if point[2] > 0:  # 深度大于0
                u = int((point[0] * fx / point[2]) + cx)
                v = int((point[1] * fy / point[2]) + cy)

                if 0 <= u < 640 and 0 <= v < 480:
                    ir_value = self.processed_ir[v, u] / 255.0
                    colors[i] = [ir_value, ir_value, ir_value]  # 灰度
                    colored_count += 1
                else:
                    colors[i] = [0.7, 0.7, 0.7]  # 默认灰色

        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"    成功为 {colored_count}/{len(points)} 个点着色")

    def _save_pointcloud_stages(self, pcd_raw, pcd_filtered, pcd_final):
        """保存点云处理的不同阶段"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存原始点云
        if len(pcd_raw.points) > 0:
            o3d.io.write_point_cloud(f"pointcloud_raw_{timestamp}.ply", pcd_raw)

        # 保存过滤后点云
        if len(pcd_filtered.points) > 0:
            o3d.io.write_point_cloud(f"pointcloud_filtered_{timestamp}.ply", pcd_filtered)

        # 保存最终点云
        if len(pcd_final.points) > 0:
            o3d.io.write_point_cloud(f"pointcloud_final_{timestamp}.ply", pcd_final)

        print(f"📁 点云已保存:")
        if len(pcd_raw.points) > 0:
            print(f"   原始点云: pointcloud_raw_{timestamp}.ply")
        if len(pcd_filtered.points) > 0:
            print(f"   过滤后点云: pointcloud_filtered_{timestamp}.ply")
        if len(pcd_final.points) > 0:
            print(f"   最终点云: pointcloud_final_{timestamp}.ply")

    def _analyze_pointcloud(self, pcd):
        """点云分析"""
        print("\n📈 点云分析:")

        if pcd is None or len(pcd.points) == 0:
            print("  无点云数据")
            return

        points = np.asarray(pcd.points)

        print(f"  点云统计:")
        print(f"    总点数: {len(points):,}")

        # 空间分布
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        size = max_coords - min_coords
        print(f"    空间范围: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
        print(f"    尺寸: {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} 米")

        # 密度分析
        volume = np.prod(size)
        if volume > 0:
            density = len(points) / volume
            print(f"    点密度: {density:.0f} 点/立方米")

        # 聚类分析
        if len(points) > 100:
            print("  聚类分析:")
            try:
                # 使用DBSCAN进行聚类
                dbscan = DBSCAN(eps=0.05, min_samples=10)
                labels = dbscan.fit_predict(points)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                print(f"    聚类数量: {n_clusters}")
                print(f"    噪声点: {n_noise} ({n_noise/len(points)*100:.1f}%)")
            except Exception as e:
                print(f"    ⚠️ 聚类分析失败: {e}")

    # ==================== 4. 网格重建 ====================

    def reconstruct_mesh(self, pointcloud):
        """重建网格"""
        print("\n" + "=" * 60)
        print("步骤5: 网格重建")
        print("=" * 60)

        if pointcloud is None or len(pointcloud.points) < 500:
            print(f"❌ 点云点数不足 ({len(pointcloud.points) if pointcloud else 0})，无法重建网格")
            return None

        start_time = time.time()

        print("🔄 开始网格重建...")
        print(f"  输入点云: {len(pointcloud.points):,} 个点")

        # 方法1: Poisson表面重建
        print("\n1. Poisson表面重建:")
        mesh_poisson = self._poisson_reconstruction(pointcloud)

        # 方法2: Ball Pivoting算法
        print("\n2. Ball Pivoting算法:")
        mesh_bpa = self._ball_pivoting_reconstruction(pointcloud)

        # 方法3: Alpha Shape算法
        print("\n3. Alpha Shape算法:")
        mesh_alpha = self._alpha_shape_reconstruction(pointcloud)

        # 选择最佳网格
        mesh_final = None
        if mesh_poisson is not None and len(mesh_poisson.vertices) > 0:
            mesh_final = mesh_poisson
        elif mesh_bpa is not None and len(mesh_bpa.vertices) > 0:
            mesh_final = mesh_bpa
        elif mesh_alpha is not None and len(mesh_alpha.vertices) > 0:
            mesh_final = mesh_alpha

        if mesh_final is not None:
            # 网格后处理
            print("\n4. 网格后处理:")
            mesh_processed = self._post_process_mesh(mesh_final)

            # 网格简化
            print("\n5. 网格简化:")
            mesh_simplified = self._simplify_mesh(mesh_processed)

            # 保存网格
            self._save_mesh_results(mesh_poisson, mesh_bpa, mesh_alpha, mesh_processed, mesh_simplified)

            # 网格分析
            self._analyze_mesh(mesh_simplified)

            elapsed = time.time() - start_time
            self._record_step("网格重建", elapsed, f"重建网格，顶点: {len(mesh_simplified.vertices):,}, 面片: {len(mesh_simplified.triangles):,}")

            return mesh_simplified
        else:
            print("❌ 所有网格重建方法都失败")
            return None

    def _poisson_reconstruction(self, pointcloud):
        """Poisson表面重建"""
        try:
            print("  执行Poisson重建...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pointcloud, depth=self.params['poisson_depth']
            )

            if len(mesh.vertices) > 0:
                # 移除低密度区域
                if len(densities) > 0:
                    vertices_to_remove = densities < np.quantile(densities, 0.01)
                    mesh.remove_vertices_by_mask(vertices_to_remove)

                print(f"    顶点: {len(mesh.vertices):,}, 面片: {len(mesh.triangles):,}")
                return mesh
            else:
                print("    ⚠️ Poisson重建无结果")
                return None

        except Exception as e:
            print(f"    ⚠️ Poisson重建失败: {e}")
            return None

    def _ball_pivoting_reconstruction(self, pointcloud):
        """Ball Pivoting算法"""
        try:
            print("  执行Ball Pivoting重建...")

            # 估计点云半径
            distances = pointcloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances) if len(distances) > 0 else 0.01

            # 设置半径
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            radii = o3d.utility.DoubleVector(radii)

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pointcloud, radii
            )

            if len(mesh.vertices) > 0:
                print(f"    顶点: {len(mesh.vertices):,}, 面片: {len(mesh.triangles):,}")
                return mesh
            else:
                print("    ⚠️ Ball Pivoting重建无结果")
                return None

        except Exception as e:
            print(f"    ⚠️ Ball Pivoting重建失败: {e}")
            return None

    def _alpha_shape_reconstruction(self, pointcloud):
        """Alpha Shape算法"""
        try:
            print("  执行Alpha Shape重建...")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pointcloud, alpha=0.03
            )

            if len(mesh.vertices) > 0:
                print(f"    顶点: {len(mesh.vertices):,}, 面片: {len(mesh.triangles):,}")
                return mesh
            else:
                print("    ⚠️ Alpha Shape重建无结果")
                return None

        except Exception as e:
            print(f"    ⚠️ Alpha Shape重建失败: {e}")
            return None

    def _post_process_mesh(self, mesh):
        """网格后处理"""
        print("  网格后处理...")

        mesh_processed = mesh

        try:
            # 1. 移除重复顶点
            mesh_processed.remove_duplicated_vertices()
            mesh_processed.remove_duplicated_triangles()

            # 2. 移除非流形边缘
            mesh_processed.remove_non_manifold_edges()

            # 3. 计算法向量
            mesh_processed.compute_vertex_normals()

            # 4. 平滑网格（可选）
            if len(mesh_processed.vertices) > 100:
                try:
                    mesh_processed = mesh_processed.filter_smooth_simple(number_of_iterations=1)
                    mesh_processed.compute_vertex_normals()
                except:
                    print("    ⚠️ 网格平滑失败，跳过")

            print(f"    后处理后顶点: {len(mesh_processed.vertices):,}, 面片: {len(mesh_processed.triangles):,}")

            return mesh_processed

        except Exception as e:
            print(f"    ⚠️ 后处理失败: {e}")
            return mesh

    def _simplify_mesh(self, mesh):
        """网格简化"""
        if mesh is None or len(mesh.triangles) < 1000:
            return mesh

        print("  网格简化...")

        try:
            # 使用quadric decimation进行简化
            target_number_of_triangles = int(len(mesh.triangles) * self.params['mesh_simplify'])

            mesh_simplified = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_number_of_triangles
            )

            # 重新计算法向量
            mesh_simplified.compute_vertex_normals()

            print(f"    简化后: {len(mesh_simplified.vertices):,}顶点, {len(mesh_simplified.triangles):,}面片")
            print(f"    简化比例: {len(mesh_simplified.triangles)/len(mesh.triangles)*100:.1f}%")

            return mesh_simplified

        except Exception as e:
            print(f"    ⚠️ 网格简化失败: {e}")
            return mesh

    def _save_mesh_results(self, *meshes):
        """保存网格结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        mesh_names = ["poisson", "bpa", "alpha", "processed", "simplified"]

        for i, (name, mesh) in enumerate(zip(mesh_names, meshes)):
            if mesh is not None and len(mesh.vertices) > 0:
                try:
                    # 保存为OBJ
                    obj_filename = f"mesh_{name}_{timestamp}.obj"
                    o3d.io.write_triangle_mesh(obj_filename, mesh, write_vertex_normals=True)

                    # 保存为STL
                    stl_filename = f"mesh_{name}_{timestamp}.stl"
                    o3d.io.write_triangle_mesh(stl_filename, mesh, write_vertex_normals=True)

                    print(f"    {name}: OBJ, STL格式已保存")
                except Exception as e:
                    print(f"    ⚠️ 保存{name}网格失败: {e}")

    def _analyze_mesh(self, mesh):
        """网格分析"""
        if mesh is None:
            return

        print("\n📈 网格分析:")

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        print(f"  网格统计:")
        print(f"    顶点数: {len(vertices):,}")
        print(f"    面片数: {len(triangles):,}")

        if len(vertices) > 0:
            # 顶点分布
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            size = max_coords - min_coords
            print(f"    空间范围: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
            print(f"    尺寸: {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} 米")

            # 面片质量分析
            if len(triangles) > 0:
                print("  面片质量分析:")

                # 计算面片面积
                areas = []
                for tri in triangles:
                    if tri[0] < len(vertices) and tri[1] < len(vertices) and tri[2] < len(vertices):
                        v0, v1, v2 = vertices[tri]
                        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                        areas.append(area)

                if len(areas) > 0:
                    areas = np.array(areas)
                    print(f"    平均面片面积: {areas.mean():.6f} 平方米")
                    print(f"    面片面积标准差: {areas.std():.6f}")
                    print(f"    最小面片面积: {areas.min():.6f}")
                    print(f"    最大面片面积: {areas.max():.6f}")

    # ==================== 5. 辅助方法 ====================

    def _record_step(self, step_name, time_taken, description=""):
        """记录处理步骤"""
        self.processing_steps.append({
            'step': step_name,
            'time': time_taken,
            'description': description
        })

        print(f"⏱️  步骤 '{step_name}' 完成，耗时: {time_taken:.2f}秒")

    def _depth_to_colormap(self, depth):
        """深度图转伪彩色"""
        if depth is None or depth.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # 创建深度图的副本
        depth_copy = depth.copy()

        # 将深度转换为毫米
        depth_mm = depth_copy * 1000

        # 找出有效深度值
        valid_depth = depth_mm[depth_mm > 0]
        if len(valid_depth) == 0:
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

        # 归一化到0-255，排除0值
        min_val, max_val = valid_depth.min(), valid_depth.max()
        if max_val > min_val:
            depth_normalized = np.zeros_like(depth_mm, dtype=np.uint8)
            mask = depth_mm > 0
            depth_normalized[mask] = ((depth_mm[mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_mm, dtype=np.uint8)

        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 将无效值标记为黑色
        depth_colored[depth_mm == 0] = [0, 0, 0]

        return depth_colored

    def _normalize_ir_image(self, ir):
        """归一化红外图像"""
        if ir is None or ir.size == 0:
            return np.zeros((480, 640), dtype=np.uint8)

        ir_copy = ir.copy()
        ir_nonzero = ir_copy[ir_copy > 0]

        if len(ir_nonzero) == 0:
            return np.zeros_like(ir_copy, dtype=np.uint8)

        min_val, max_val = ir_nonzero.min(), ir_nonzero.max()
        if max_val > min_val:
            ir_normalized = np.zeros_like(ir_copy, dtype=np.uint8)
            mask = ir_copy > 0
            ir_normalized[mask] = ((ir_copy[mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            ir_normalized = np.zeros_like(ir_copy, dtype=np.uint8)

        return ir_normalized

    def generate_report(self):
        """生成处理报告"""
        print("\n" + "=" * 60)
        print("处理报告")
        print("=" * 60)

        if not self.processing_steps:
            print("无处理步骤记录")
            return

        total_time = sum(step['time'] for step in self.processing_steps)

        print(f"📊 处理统计:")
        print(f"  总处理步骤: {len(self.processing_steps)}")
        print(f"  总处理时间: {total_time:.2f}秒")

        print(f"\n📋 处理步骤详情:")
        for i, step in enumerate(self.processing_steps, 1):
            print(f"  {i}. {step['step']}: {step['time']:.2f}秒 - {step['description']}")

        # 保存报告到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reconstruction_report_{timestamp}.txt"

        with open(report_filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Astra相机3D重建处理报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("📊 处理统计:\n")
            f.write(f"  总处理步骤: {len(self.processing_steps)}\n")
            f.write(f"  总处理时间: {total_time:.2f}秒\n\n")

            f.write("📋 处理步骤详情:\n")
            for i, step in enumerate(self.processing_steps, 1):
                f.write(f"  {i}. {step['step']}: {step['time']:.2f}秒 - {step['description']}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("处理参数:\n")
            f.write("=" * 60 + "\n")

            for key, value in self.params.items():
                f.write(f"  {key}: {value}\n")

        print(f"\n📄 详细报告已保存: {report_filename}")

        # 生成处理时间图表
        self._generate_time_chart()

    def _generate_time_chart(self):
        """生成处理时间图表"""
        if not self.processing_steps:
            return

        step_names = [step['step'] for step in self.processing_steps]
        step_times = [step['time'] for step in self.processing_steps]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(step_names, step_times, color='skyblue')
        ax.set_xlabel('Processing Step')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Processing Time for Each Step')

        # 在柱状图上添加数值标签
        for bar, time_val in zip(bars, step_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{time_val:.2f}s', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"processing_time_chart_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Processing time chart saved: processing_time_chart_{timestamp}.png")

    def cleanup(self):
        """清理资源"""
        print("\n" + "=" * 60)
        print("清理资源")
        print("=" * 60)

        try:
            if self.depth_stream:
                try:
                    self.depth_stream.stop()
                    print("✅ 深度流已停止")
                except:
                    print("⚠️ 深度流停止失败")

            if self.ir_stream:
                try:
                    self.ir_stream.stop()
                    print("✅ 红外流已停止")
                except:
                    print("⚠️ 红外流停止失败")

            if self.device:
                try:
                    self.device.close()
                    print("✅ 设备已关闭")
                except:
                    print("⚠️ 设备关闭失败")

            try:
                openni2.unload()
                print("✅ OpenNI2已卸载")
            except:
                print("⚠️ OpenNI2卸载失败")

            print("🎯 资源清理完成")

        except Exception as e:
            print(f"⚠️ 清理过程中出现错误: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("奥比中光Astra相机3D重建原理演示系统 - 修复版")
    print("从基础数据采集到完整3D重建的完整流程")
    print("=" * 70)

    # 创建带时间戳的输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"reconstruction_output_{timestamp}"

    # 如果目录已存在（几乎不可能，因为时间戳精确到秒），则添加毫秒
    import time
    if os.path.exists(output_dir):
        timestamp = f"{timestamp}_{int(time.time() * 1000) % 1000:03d}"
        output_dir = f"reconstruction_output_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")

    # 初始化重建器
    driver_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    reconstructor = Astra3DReconstructor(driver_path)

    try:
        # 步骤1: 初始化相机
        if not reconstructor.initialize_camera():
            print("❌ 相机初始化失败，程序退出")
            return

        # 步骤2: 采集单帧数据
        if not reconstructor.capture_single_frame():
            print("❌ 数据采集失败，程序退出")
            reconstructor.cleanup()
            return

        # 保存原始数据
        reconstructor.save_raw_data()

        # 步骤3: 数据预处理
        if not reconstructor.preprocess_data():
            print("⚠️ 数据预处理出现问题，继续尝试后续步骤")

        # 步骤4: 点云生成
        pointcloud = reconstructor.generate_pointcloud()

        # 步骤5: 网格重建
        if pointcloud is not None:
            mesh = reconstructor.reconstruct_mesh(pointcloud)
        else:
            print("❌ 点云生成失败，跳过网格重建")
            mesh = None

        # 生成报告
        reconstructor.generate_report()

        print("\n" + "=" * 70)
        if mesh is not None:
            print("🎉 3D重建流程完成!")
            print(f"   成功生成3D网格:")
            print(f"     顶点数: {len(mesh.vertices):,}")
            print(f"     面片数: {len(mesh.triangles):,}")

            # 显示最终结果
            try:
                o3d.visualization.draw_geometries([mesh],
                                                 window_name="最终3D重建结果",
                                                 width=1024,
                                                 height=768)
            except Exception as e:
                print(f"⚠️ 3D显示失败: {e}")
        else:
            print("⚠️ 3D重建流程完成，但未生成网格")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n🔴 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        reconstructor.cleanup()

        print("\n" + "=" * 70)
        print("程序结束")
        print("=" * 70)


if __name__ == "__main__":
    # 检查必要库
    try:
        import cv2
        import numpy as np
        import open3d as o3d
        import matplotlib.pyplot as plt
        print("✅ 所有依赖库已加载")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install opencv-python numpy open3d matplotlib scipy scikit-learn trimesh")
        exit(1)

    main()