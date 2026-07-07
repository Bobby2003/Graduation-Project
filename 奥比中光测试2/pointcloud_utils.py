"""
点云处理工具模块
"""

import numpy as np
import open3d as o3d
import copy
from collections import deque


class PointCloudProcessor:
    def __init__(self, config=None):
        """初始化点云处理器"""
        self.config = config or {}

        # 处理参数
        self.voxel_size = self.config.get('voxel_size', 0.01)
        self.max_points = self.config.get('max_points', 100000)
        self.downsample_rate = self.config.get('downsample_rate', 2)

        # 相机参数
        self.fx = 475.0
        self.fy = 475.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 0.001

        # 深度范围
        self.min_depth = 0.2
        self.max_depth = 2.5

        # 点云存储
        self.global_pointcloud = o3d.geometry.PointCloud()
        self.frame_buffer = deque(maxlen=50)  # 最近50帧的点云
        self.transformed_pointclouds = []  # 经过位姿变换的点云

        # 统计
        self.total_points = 0
        self.processed_frames = 0

        print("☁️ 点云处理器初始化完成")

    def set_camera_params(self, fx, fy, cx, cy, depth_scale=0.001):
        """设置相机参数"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def depth_to_pointcloud(self, depth_frame, pose=None):
        """深度图转点云"""
        if depth_frame is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

        height, width = depth_frame.shape

        # 创建像素网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 计算3D坐标
        z = depth_frame.astype(np.float32) * self.depth_scale
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # 修正坐标系
        x_corrected = x
        y_corrected = -y  # 翻转Y轴
        z_corrected = -z  # 翻转Z轴

        # 创建点云
        points = np.stack([x_corrected, y_corrected, z_corrected], axis=-1).reshape(-1, 3)

        # 过滤无效点
        valid_mask = (z.reshape(-1) > self.min_depth) & (z.reshape(-1) < self.max_depth)
        points = points[valid_mask]

        # 应用位姿变换
        if pose is not None and len(points) > 0:
            points = self._apply_pose(points, pose)

        # 生成颜色（基于深度）
        if len(points) > 0:
            z_vals = points[:, 2]
            depth_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
            depth_norm = np.clip(depth_norm, 0, 1)

            colors = np.zeros((len(points), 3))
            colors[:, 0] = depth_norm  # 红色分量
            colors[:, 1] = 0.3  # 绿色分量
            colors[:, 2] = 1.0 - depth_norm  # 蓝色分量
        else:
            colors = np.zeros((0, 3))

        return points, colors

    def _apply_pose(self, points, pose):
        """应用位姿变换"""
        # pose: [x, y, z, qx, qy, qz, qw]
        if len(points) == 0:
            return points

        # 从四元数构建旋转矩阵
        x, y, z, qx, qy, qz, qw = pose

        # 四元数转旋转矩阵
        R = np.array([
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
        ])

        # 平移向量
        t = np.array([x, y, z])

        # 应用变换
        points_transformed = (R @ points.T).T + t

        return points_transformed

    def process_frame(self, depth_frame, pose=None):
        """处理一帧深度图"""
        if depth_frame is None:
            return False

        try:
            # 下采样（提高处理速度）
            if self.downsample_rate > 1:
                depth_frame = depth_frame[::self.downsample_rate, ::self.downsample_rate]

            # 转换为点云
            points, colors = self.depth_to_pointcloud(depth_frame, pose)

            if len(points) < 100:  # 点太少，跳过
                return False

            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 添加到缓冲区
            self.frame_buffer.append(pcd)

            # 定期合并点云（每秒一次）
            if self.processed_frames % 10 == 0:
                self._merge_pointclouds()

            self.processed_frames += 1

            return True

        except Exception as e:
            print(f"[点云] 处理帧失败: {e}")
            return False

    def _merge_pointclouds(self):
        """合并缓冲区中的点云"""
        if not self.frame_buffer:
            return

        # 合并所有点云
        all_points = []
        all_colors = []

        for pcd in self.frame_buffer:
            if len(pcd.points) > 0:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                all_points.append(points)
                all_colors.append(colors)

        if not all_points:
            return

        # 合并
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        # 创建点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        # 点云处理
        merged_pcd = self._process_pointcloud(merged_pcd)

        # 更新全局点云
        self.global_pointcloud = merged_pcd
        self.total_points = len(merged_points)

    def _process_pointcloud(self, pointcloud):
        """点云后处理"""
        if len(pointcloud.points) < 100:
            return pointcloud

        try:
            # 1. 离群点移除
            cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pointcloud = pointcloud.select_by_index(ind)
        except:
            pass

        try:
            # 2. 半径离群点移除
            cl, ind = pointcloud.remove_radius_outlier(nb_points=16, radius=0.05)
            pointcloud = pointcloud.select_by_index(ind)
        except:
            pass

        # 3. 体素下采样（控制点云密度）
        if len(pointcloud.points) > self.max_points:
            pointcloud = pointcloud.voxel_down_sample(voxel_size=self.voxel_size)

        return pointcloud

    def get_pointcloud(self):
        """获取当前点云"""
        return copy.deepcopy(self.global_pointcloud)

    def save_pointcloud(self, filename):
        """保存点云到文件"""
        if len(self.global_pointcloud.points) > 0:
            o3d.io.write_point_cloud(filename, self.global_pointcloud)
            return True
        return False

    def reconstruct_mesh(self, filename, method='poisson'):
        """重建网格"""
        if len(self.global_pointcloud.points) < 1000:
            print("❌ 点云点数不足，无法重建网格")
            return False

        try:
            print(f"🔄 开始网格重建 (点数: {self.total_points})...")

            # 深度拷贝点云
            pcd = copy.deepcopy(self.global_pointcloud)

            # 估计法向量
            print("  正在估计法向量...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

            # 泊松重建
            print("  使用泊松重建...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )

            # 根据密度移除低密度顶点
            if densities is not None and len(densities) > 0:
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)

            # 网格后处理
            print("  网格后处理...")

            # 简化（如果面数太多）
            if len(mesh.triangles) > 100000:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)

            # 清理网格
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_non_manifold_edges()

            # 计算法线
            mesh.compute_vertex_normals()

            # 保存网格
            o3d.io.write_triangle_mesh(filename, mesh)

            print(f"✅ 网格已保存: {filename}")
            print(f"  顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
            return True

        except Exception as e:
            print(f"❌ 网格重建失败: {e}")
            return False

    def clear(self):
        """清除所有数据"""
        self.global_pointcloud.clear()
        self.frame_buffer.clear()
        self.transformed_pointclouds.clear()
        self.total_points = 0
        self.processed_frames = 0

        print("✅ 点云数据已清除")