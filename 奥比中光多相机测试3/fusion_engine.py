"""
多视角融合引擎 - 融合三个相机的点云并进行3D重建
"""

import numpy as np
import open3d as o3d
import time
import os
from datetime import datetime
from config import MultiCameraConfig
from utils import MultiCameraUtils


class MultiViewFusionEngine:
    """多视角融合引擎"""

    def __init__(self, config=None):
        self.config = config or MultiCameraConfig.RECONSTRUCTION_CONFIG
        self.output_dir = None

        # 重建状态
        self.pointclouds = {}
        self.registered_pointclouds = {}
        self.fused_pointcloud = None
        self.mesh = None

        # 配准变换
        self.registration_transforms = {}

    def process_pointclouds(self, pointclouds):
        """处理点云（去噪、下采样等）"""
        processed_pcds = {}

        for cam_id, pcd in pointclouds.items():
            if pcd is None or len(pcd.points) == 0:
                continue

            # 统计原始点云
            print(f"相机 {cam_id} 原始点云: {len(pcd.points)} 个点")

            # 1. 移除离群点
            if self.config['outlier_removal']:
                pcd, outlier_indices = pcd.remove_statistical_outlier(
                    nb_neighbors=self.config['outlier_neighbors'],
                    std_ratio=self.config['outlier_std_ratio']
                )
                print(f"  移除离群点: {len(outlier_indices)} 个")

            # 2. 体素下采样
            if self.config['voxel_size'] > 0:
                pcd = pcd.voxel_down_sample(self.config['voxel_size'])
                print(f"  下采样后: {len(pcd.points)} 个点")

            # 3. 估计法线（如果需要）
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )

            processed_pcds[cam_id] = pcd

        return processed_pcds

    def register_pointclouds(self, pointclouds):
        """配准点云到统一坐标系"""
        if len(pointclouds) < 2:
            print("点云数量不足，跳过配准")
            return pointclouds, {}

        print("\n开始点云配准...")

        # 使用已知的相机位姿进行初始配准
        camera_poses = MultiCameraConfig.get_camera_poses()
        registered_pcds = {}

        for cam_id, pcd in pointclouds.items():
            if cam_id in camera_poses:
                # 应用已知的相机位姿变换
                pcd.transform(camera_poses[cam_id])
                registered_pcds[cam_id] = pcd
                print(f"相机 {cam_id}: 使用已知位姿")

        # 如果需要进一步精细配准
        if self.config['registration_method'] == 'icp':
            self._fine_registration(registered_pcds)

        return registered_pcds, self.registration_transforms

    def _fine_registration(self, pointclouds):
        """精细配准（ICP）"""
        if len(pointclouds) < 2:
            return

        # 以第一个相机为参考
        ref_id = list(pointclouds.keys())[0]
        ref_pcd = pointclouds[ref_id]

        for cam_id, pcd in pointclouds.items():
            if cam_id == ref_id:
                continue

            print(f"使用ICP配准相机 {cam_id} 到相机 {ref_id}...")

            # 执行ICP
            transform, fitness = MultiCameraUtils.align_pointclouds_icp(
                pcd, ref_pcd,
                self.config['icp_threshold'],
                self.config['icp_max_iteration']
            )

            if fitness > 0:
                # 应用变换
                pcd.transform(transform)
                self.registration_transforms[(cam_id, ref_id)] = {
                    'transform': transform.tolist(),
                    'fitness': fitness
                }
                print(f"  配准质量: {fitness:.4f}")

    def fuse_pointclouds(self, pointclouds):
        """融合多个点云"""
        if not pointclouds:
            return None

        print("\n融合点云...")

        # 合并所有点云
        combined_pcd = o3d.geometry.PointCloud()

        for i, (cam_id, pcd) in enumerate(pointclouds.items()):
            if i == 0:
                combined_pcd = pcd
            else:
                combined_pcd += pcd

        print(f"融合后点云: {len(combined_pcd.points)} 个点")

        # 进一步下采样
        if self.config['voxel_size'] > 0:
            combined_pcd = combined_pcd.voxel_down_sample(self.config['voxel_size'])
            print(f"最终点云: {len(combined_pcd.points)} 个点")

        self.fused_pointcloud = combined_pcd
        return combined_pcd

    def create_tsdf_volume(self, pointclouds):
        """使用TSDF体积融合"""
        if not pointclouds:
            return None

        print("\n创建TSDF体积...")

        # 创建TSDF体积
        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config['tsdf_voxel_length'],
            sdf_trunc=self.config['tsdf_sdf_trunc'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # 获取相机内参
        intrinsic = MultiCameraConfig.get_intrinsic_matrix()
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            MultiCameraConfig.CAMERA_INTRINSICS['width'],
            MultiCameraConfig.CAMERA_INTRINSICS['height'],
            intrinsic[0, 0], intrinsic[1, 1],
            intrinsic[0, 2], intrinsic[1, 2]
        )

        # 获取相机位姿
        camera_poses = MultiCameraConfig.get_camera_poses()

        # 对于每个点云，创建虚拟深度图和彩色图
        for cam_id, pcd in pointclouds.items():
            if cam_id not in camera_poses:
                continue

            # 注意：这里需要将点云投影回深度图和彩色图
            # 这是一个简化的实现，实际应用中可能需要更复杂的方法
            print(f"处理相机 {cam_id} 的TSDF融合...")

            # 暂时跳过，直接使用点云融合
            pass

        return tsdf_volume

    def reconstruct_mesh_poisson(self, pointcloud):
        """使用泊松重建生成网格"""
        if pointcloud is None or len(pointcloud.points) == 0:
            print("点云为空，无法进行泊松重建")
            return None

        print("\n进行泊松重建...")

        # 确保有法线
        if not pointcloud.has_normals():
            pointcloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        # 执行泊松重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pointcloud, depth=self.config['poisson_depth']
        )

        # 移除低密度区域
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # 计算顶点颜色
        if pointcloud.has_colors():
            # 将点云颜色转移到网格
            mesh.compute_vertex_normals()

            # 创建KD树用于最近邻颜色查询
            from scipy.spatial import KDTree
            points = np.asarray(pointcloud.points)
            colors = np.asarray(pointcloud.colors)

            if len(points) > 0 and len(colors) > 0:
                tree = KDTree(points)
                mesh_vertices = np.asarray(mesh.vertices)

                if len(mesh_vertices) > 0:
                    distances, indices = tree.query(mesh_vertices, k=1)
                    mesh_colors = colors[indices]
                    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

        print(f"泊松重建网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面片")

        return mesh

    def reconstruct_mesh_ball_pivot(self, pointcloud):
        """使用Ball Pivoting算法生成网格"""
        if pointcloud is None or len(pointcloud.points) == 0:
            return None

        print("\n进行Ball Pivoting重建...")

        # 估计法线
        if not pointcloud.has_normals():
            pointcloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        # 计算半径（基于点云密度）
        distances = pointcloud.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        radii = [avg_distance * 1.5, avg_distance * 3.0, avg_distance * 6.0]

        # 执行Ball Pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pointcloud, o3d.utility.DoubleVector(radii)
        )

        print(f"Ball Pivoting网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面片")

        return mesh

    def optimize_mesh(self, mesh):
        """优化网格"""
        if mesh is None or len(mesh.vertices) == 0:
            return mesh

        print("\n优化网格...")

        # 1. 计算法线
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # 2. 网格简化
        if self.config['mesh_simplify'] and len(mesh.triangles) > self.config['target_faces']:
            target_faces = self.config['target_faces']
            mesh = mesh.simplify_quadric_decimation(target_faces)
            print(f"  简化后: {len(mesh.triangles)} 面片")

        # 3. 网格平滑
        if self.config['smooth_mesh']:
            mesh = mesh.filter_smooth_simple(
                number_of_iterations=self.config['smooth_iterations']
            )
            mesh.compute_vertex_normals()
            print(f"  平滑完成")

        # 4. 移除重复顶点
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        print(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面片")

        return mesh

    def save_results(self, output_dir):
        """保存所有结果"""
        if not output_dir:
            return

        print(f"\n保存结果到: {output_dir}")

        # 创建目录
        subdirs = ['pointclouds', 'registered_pcd', 'meshes', 'previews']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存原始点云
        for cam_id, pcd in self.pointclouds.items():
            if pcd and len(pcd.points) > 0:
                filepath = os.path.join(output_dir, 'pointclouds',
                                        f'camera_{cam_id}_{timestamp}.ply')
                MultiCameraUtils.save_pointcloud(pcd, filepath)

        # 保存配准后点云
        for cam_id, pcd in self.registered_pointclouds.items():
            if pcd and len(pcd.points) > 0:
                filepath = os.path.join(output_dir, 'registered_pcd',
                                        f'registered_{cam_id}_{timestamp}.ply')
                MultiCameraUtils.save_pointcloud(pcd, filepath)

        # 保存融合点云
        if self.fused_pointcloud and len(self.fused_pointcloud.points) > 0:
            filepath = os.path.join(output_dir, 'pointclouds',
                                    f'fused_{timestamp}.ply')
            MultiCameraUtils.save_pointcloud(self.fused_pointcloud, filepath)

        # 保存网格
        if self.mesh and len(self.mesh.vertices) > 0:
            # 保存不同格式
            formats = self.config.get('export_formats', ['ply'])

            for fmt in formats:
                if fmt == 'ply':
                    filepath = os.path.join(output_dir, 'meshes',
                                            f'mesh_{timestamp}.ply')
                    MultiCameraUtils.save_mesh(self.mesh, filepath)
                elif fmt == 'obj':
                    filepath = os.path.join(output_dir, 'meshes',
                                            f'mesh_{timestamp}.obj')
                    o3d.io.write_triangle_mesh(filepath, self.mesh,
                                               write_vertex_normals=True,
                                               write_vertex_colors=True)
                elif fmt == 'stl':
                    filepath = os.path.join(output_dir, 'meshes',
                                            f'mesh_{timestamp}.stl')
                    o3d.io.write_triangle_mesh(filepath, self.mesh)

        print("✅ 所有结果已保存")

    def run_reconstruction(self, pointclouds, output_dir):
        """运行完整重建流程"""
        self.output_dir = output_dir

        print("=" * 70)
        print("多视角3D重建开始")
        print("=" * 70)

        start_time = time.time()

        # 1. 处理点云
        self.pointclouds = self.process_pointclouds(pointclouds)
        if not self.pointclouds:
            print("❌ 没有有效的点云数据")
            return False

        # 2. 配准点云
        self.registered_pointclouds, transforms = self.register_pointclouds(self.pointclouds)

        # 3. 融合点云
        self.fused_pointcloud = self.fuse_pointclouds(self.registered_pointclouds)
        if self.fused_pointcloud is None:
            print("❌ 点云融合失败")
            return False

        # 4. 重建网格
        if len(self.fused_pointcloud.points) > 1000:
            # 使用泊松重建
            self.mesh = self.reconstruct_mesh_poisson(self.fused_pointcloud)

            # 如果泊松重建失败，尝试Ball Pivoting
            if self.mesh is None or len(self.mesh.vertices) == 0:
                print("泊松重建失败，尝试Ball Pivoting...")
                self.mesh = self.reconstruct_mesh_ball_pivot(self.fused_pointcloud)
        else:
            print("❌ 点云数量不足，无法重建网格")
            return False

        # 5. 优化网格
        if self.mesh is not None:
            self.mesh = self.optimize_mesh(self.mesh)

        # 6. 保存结果
        self.save_results(output_dir)

        # 7. 生成统计信息
        elapsed_time = time.time() - start_time
        stats = self._collect_statistics(elapsed_time)

        # 8. 生成报告
        MultiCameraUtils.generate_report(stats, output_dir)

        print("\n" + "=" * 70)
        print("多视角3D重建完成")
        print(f"总时间: {elapsed_time:.1f} 秒")
        print(f"输出目录: {output_dir}")
        print("=" * 70)

        return True

    def _collect_statistics(self, elapsed_time):
        """收集统计信息"""
        stats = {
            'processing_time': elapsed_time,
            'pointcloud_stats': {},
            'registration_stats': {},
            'mesh_stats': {}
        }

        # 点云统计
        for cam_id, pcd in self.pointclouds.items():
            if pcd:
                stats['pointcloud_stats'][cam_id] = {
                    'points': len(pcd.points),
                    'has_colors': pcd.has_colors()
                }

        # 配准统计
        for pair, transform_info in self.registration_transforms.items():
            stats['registration_stats'][f'{pair[0]}-{pair[1]}'] = {
                'fitness': transform_info['fitness']
            }

        # 网格统计
        if self.mesh:
            stats['mesh_stats'] = {
                'vertices': len(self.mesh.vertices),
                'faces': len(self.mesh.triangles),
                'has_colors': self.mesh.has_vertex_colors()
            }

        return stats