"""
基于Python视觉里程计的实时3D扫描系统 - Astra相机专用版本
支持实时网格重建、OBJ/STL显示和保存
作者：Bobby2003
日期：2024
"""

import cv2
import numpy as np
import open3d as o3d
import threading
import queue
import time
import os
import json
import trimesh
from datetime import datetime
from primesense import openni2
from primesense import _openni2 as c_api


# ==================== 1. Astra相机管理器 ====================
class AstraCameraManager:
    """Astra相机专用管理器 - 深度+红外模式"""

    def __init__(self, driver_path=""):
        self.driver_path = driver_path
        self.device = None
        self.depth_stream = None
        self.ir_stream = None
        self.running = False

        # 队列
        self.depth_queue = queue.Queue(maxsize=20)
        self.ir_queue = queue.Queue(maxsize=20)

        # 相机参数（Astra相机默认值）
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5
        self.depth_scale = 0.001  # 毫米转米

        self.frame_count = 0
        self.last_frame_time = time.time()

    def initialize(self):
        """初始化Astra相机"""
        try:
            print("🚀 初始化Astra相机...")

            # 设置驱动路径
            if self.driver_path and os.path.exists(self.driver_path):
                print(f"使用驱动路径: {self.driver_path}")
            else:
                self.driver_path = ""
                print("使用默认驱动路径")

            # 初始化OpenNI2
            try:
                openni2.initialize(self.driver_path)
                print("✅ OpenNI2初始化成功")
            except Exception as e:
                print(f"⚠️ OpenNI2初始化警告: {e}")
                try:
                    openni2.unload()
                    openni2.initialize(self.driver_path)
                except:
                    print("❌ 无法初始化OpenNI2")
                    return False

            # 打开设备
            try:
                self.device = openni2.Device.open_any()
                print("✅ 设备打开成功")

                # 获取设备信息
                dev_info = self.device.get_device_info()
                device_name = dev_info.name.decode('utf-8', errors='ignore')
                print(f"📱 设备名称: {device_name}")

            except Exception as e:
                print(f"❌ 无法打开设备: {e}")
                return False

            return True

        except Exception as e:
            print(f"❌ 相机初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_streams(self):
        """启动深度和红外流"""
        try:
            print("🔄 启动深度流...")
            self.depth_stream = self.device.create_depth_stream()
            self.depth_stream.start()
            print("✅ 深度流启动成功")

            print("🔄 启动红外流...")
            self.ir_stream = self.device.create_ir_stream()
            self.ir_stream.start()
            print("✅ 红外流启动成功")

            # 启动采集线程
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            print("🎯 相机采集已启动")
            return True

        except Exception as e:
            print(f"❌ 启动流失败: {e}")
            return False

    def _capture_loop(self):
        """采集循环 - 简化稳定版本"""
        print("[采集线程] 开始运行")

        while self.running:
            try:
                # 采集深度帧
                if self.depth_stream:
                    depth_frame = self.depth_stream.read_frame()
                    if depth_frame:
                        depth_data = depth_frame.get_buffer_as_uint16()
                        depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

                        # 转换为米并移除过远的点
                        depth_array = depth_array.astype(np.float32) * self.depth_scale
                        depth_array[depth_array > 2.0] = 0  # 移除2米以外的点

                        if not self.depth_queue.full():
                            self.depth_queue.put(depth_array)

                # 采集红外帧
                if self.ir_stream:
                    ir_frame = self.ir_stream.read_frame()
                    if ir_frame:
                        ir_data = ir_frame.get_buffer_as_uint16()
                        ir_array = np.frombuffer(ir_data, dtype=np.uint16).reshape(480, 640)

                        # 转换为8位用于显示
                        if ir_array.max() > ir_array.min():
                            ir_8bit = cv2.normalize(ir_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        else:
                            ir_8bit = ir_array.astype(np.uint8)

                        if not self.ir_queue.full():
                            self.ir_queue.put(ir_8bit)

                self.frame_count += 1

                # 控制采集频率
                elapsed = time.time() - self.last_frame_time
                if elapsed < 0.033:  # 约30FPS
                    time.sleep(0.033 - elapsed)
                self.last_frame_time = time.time()

            except Exception as e:
                print(f"[采集线程] 错误: {e}")
                time.sleep(0.01)

        print("[采集线程] 停止运行")

    def get_frames(self):
        """获取最新的深度和红外帧（非阻塞）"""
        depth_frame = None
        ir_frame = None

        try:
            if not self.depth_queue.empty():
                depth_frame = self.depth_queue.get_nowait()

            if not self.ir_queue.empty():
                ir_frame = self.ir_queue.get_nowait()

        except queue.Empty:
            pass

        return depth_frame, ir_frame

    def get_stats(self):
        """获取统计信息"""
        return {
            'frames_captured': self.frame_count,
            'depth_queue_size': self.depth_queue.qsize(),
            'ir_queue_size': self.ir_queue.qsize(),
            'running': self.running
        }

    def stop(self):
        """停止相机"""
        print("🛑 停止相机...")
        self.running = False

        # 等待采集线程结束
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)

        # 停止流
        if self.depth_stream:
            try:
                self.depth_stream.stop()
                print("✅ 深度流已停止")
            except:
                pass

        if self.ir_stream:
            try:
                self.ir_stream.stop()
                print("✅ 红外流已停止")
            except:
                pass

        # 关闭设备
        if self.device:
            try:
                self.device.close()
                print("✅ 设备已关闭")
            except:
                pass

        # 卸载OpenNI2
        try:
            openni2.unload()
            print("✅ OpenNI2已卸载")
        except:
            pass

        print("🎯 相机已完全停止")


# ==================== 2. 增强型3D处理器（支持网格重建）====================
class Enhanced3DProcessor:
    """增强型3D处理器 - 支持点云和网格重建"""

    def __init__(self):
        # 点云
        self.pointcloud = None
        self.total_points = 0
        self.max_points = 500000

        # 网格
        self.mesh = None
        self.last_mesh_update = 0
        self.mesh_update_interval = 5.0  # 网格更新间隔（秒）

        # 相机内参（Astra相机）
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5

        # 网格重建参数
        self.voxel_size = 0.02  # 体素大小
        self.depth_trunc = 2.0  # 最大深度
        self.poisson_depth = 8  # Poisson重建深度

        # 处理统计
        self.stats = {
            'frames_processed': 0,
            'points_processed': 0,
            'meshes_generated': 0,
            'last_mesh_vertices': 0,
            'last_mesh_faces': 0
        }

    def depth_to_pointcloud(self, depth_frame):
        """深度图转点云"""
        if depth_frame is None or depth_frame.size == 0:
            return None

        height, width = depth_frame.shape

        # 生成像素坐标网格
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # 计算3D坐标
        z = depth_frame
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # 展平并组合
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

        # 移除无效点
        valid_mask = (z.flatten() > 0.1) & (z.flatten() < self.depth_trunc)
        points = points[valid_mask]

        if len(points) == 0:
            return None

        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 使用统一颜色（浅灰色）
        colors = np.ones((len(points), 3)) * 0.8
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def merge_pointclouds(self, pcd1, pcd2):
        """合并两个点云"""
        if pcd1 is None:
            return pcd2
        if pcd2 is None:
            return pcd1

        # 合并点
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        colors1 = np.asarray(pcd1.colors)
        colors2 = np.asarray(pcd2.colors)

        points = np.vstack([points1, points2])
        colors = np.vstack([colors1, colors2])

        # 限制最大点数
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(points)
        merged_pcd.colors = o3d.utility.Vector3dVector(colors)

        return merged_pcd

    def process_frame(self, depth_frame):
        """处理一帧深度图"""
        current_pcd = self.depth_to_pointcloud(depth_frame)
        if current_pcd is None:
            return False

        # 合并到全局点云
        self.pointcloud = self.merge_pointclouds(self.pointcloud, current_pcd)
        self.total_points = len(self.pointcloud.points) if self.pointcloud else 0

        self.stats['frames_processed'] += 1
        self.stats['points_processed'] += len(current_pcd.points)

        return True

    def reconstruct_mesh(self, method='poisson'):
        """重建网格 - 支持多种方法"""
        if self.pointcloud is None or len(self.pointcloud.points) < 1000:
            print("❌ 点云点数不足，无法重建网格")
            return None

        try:
            print(f"🔄 开始重建网格 (方法: {method})...")

            # 预处理点云
            pcd = self.pointcloud.voxel_down_sample(voxel_size=self.voxel_size)

            # 估计法向量
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2, max_nn=30
                )
            )

            if method == 'poisson':
                # Poisson表面重建
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=self.poisson_depth, width=0, scale=1.1, linear_fit=False
                )

                # 移除低密度顶点
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)

            elif method == 'ball_pivoting':
                # Ball Pivoting算法
                radii = [self.voxel_size * 2, self.voxel_size * 4]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            elif method == 'alpha_shape':
                # Alpha Shape算法
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )

            # 网格后处理
            mesh = self._post_process_mesh(mesh)

            # 保存网格
            self.mesh = mesh
            self.last_mesh_update = time.time()
            self.stats['meshes_generated'] += 1
            self.stats['last_mesh_vertices'] = len(mesh.vertices)
            self.stats['last_mesh_faces'] = len(mesh.triangles)

            print(f"✅ 网格重建完成 - 顶点: {len(mesh.vertices)}, 面片: {len(mesh.triangles)}")

            return mesh

        except Exception as e:
            print(f"❌ 网格重建失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _post_process_mesh(self, mesh):
        """网格后处理"""
        try:
            # 移除重复顶点
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()

            # 移除非流形边缘
            mesh.remove_non_manifold_edges()

            # 计算顶点法向量（用于平滑着色）
            mesh.compute_vertex_normals()

            # 平滑网格（可选）
            # mesh = mesh.filter_smooth_simple(number_of_iterations=1)

            return mesh

        except Exception as e:
            print(f"⚠️ 网格后处理失败: {e}")
            return mesh

    def save_pointcloud(self, filename):
        """保存点云到文件"""
        if self.pointcloud and len(self.pointcloud.points) > 0:
            try:
                # 支持多种格式
                if filename.endswith('.ply'):
                    o3d.io.write_point_cloud(filename, self.pointcloud)
                elif filename.endswith('.pcd'):
                    o3d.io.write_point_cloud(filename, self.pointcloud)
                elif filename.endswith('.xyz'):
                    # 保存为文本格式
                    points = np.asarray(self.pointcloud.points)
                    colors = np.asarray(self.pointcloud.colors) * 255
                    data = np.hstack([points, colors.astype(np.uint8)])
                    np.savetxt(filename, data, fmt='%.6f %.6f %.6f %d %d %d')
                else:
                    # 默认保存为PLY
                    filename = filename.replace('.', '_') + '.ply'
                    o3d.io.write_point_cloud(filename, self.pointcloud)

                print(f"✅ 点云已保存: {filename}")
                return True

            except Exception as e:
                print(f"❌ 保存点云失败: {e}")
        return False

    def save_mesh(self, filename, format='auto'):
        """保存网格到文件"""
        if self.mesh is None or len(self.mesh.vertices) == 0:
            print("❌ 无网格数据可保存")
            return False

        try:
            # 根据扩展名确定格式
            if format == 'auto':
                if filename.endswith('.obj'):
                    format = 'obj'
                elif filename.endswith('.stl'):
                    format = 'stl'
                elif filename.endswith('.ply'):
                    format = 'ply'
                else:
                    format = 'obj'
                    filename += '.obj'

            # 保存网格
            if format == 'obj':
                o3d.io.write_triangle_mesh(filename, self.mesh, write_vertex_normals=True)
            elif format == 'stl':
                o3d.io.write_triangle_mesh(filename, self.mesh, write_vertex_normals=True)
            elif format == 'ply':
                o3d.io.write_triangle_mesh(filename, self.mesh, write_vertex_normals=True)
            elif format == 'gltf':
                # 使用trimesh转换为glTF
                mesh_trimesh = trimesh.Trimesh(
                    vertices=np.asarray(self.mesh.vertices),
                    faces=np.asarray(self.mesh.triangles),
                    vertex_normals=np.asarray(self.mesh.vertex_normals)
                )
                mesh_trimesh.export(filename)
            else:
                print(f"❌ 不支持的格式: {format}")
                return False

            print(f"✅ 网格已保存 ({format.upper()}): {filename}")
            return True

        except Exception as e:
            print(f"❌ 保存网格失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_pointcloud(self):
        """获取当前点云"""
        return self.pointcloud

    def get_mesh(self):
        """获取当前网格"""
        return self.mesh

    def clear(self):
        """清空所有数据"""
        self.pointcloud = None
        self.mesh = None
        self.total_points = 0
        self.stats['last_mesh_vertices'] = 0
        self.stats['last_mesh_faces'] = 0
        print("✅ 所有3D数据已清空")

    def get_stats(self):
        """获取统计信息"""
        stats = self.stats.copy()
        stats['total_points'] = self.total_points
        stats['has_mesh'] = self.mesh is not None
        return stats


# ==================== 3. 增强型3D可视化器（支持网格显示）====================
class Enhanced3DViewer:
    """增强型3D可视化器 - 支持网格和点云切换显示"""

    def __init__(self):
        self.vis = None
        self.running = False

        # 显示模式
        self.display_mode = 'pointcloud'  # 'pointcloud', 'mesh', 'both'
        self.current_geometry = None

        # 显示选项
        self.point_size = 2.0
        self.mesh_color = [0.8, 0.8, 0.8]  # 浅灰色
        self.wireframe = False
        self.show_axes = True

        # 相机视角
        self.camera_params = None

    def initialize(self, window_name="3D扫描与重建"):
        """初始化可视化窗口"""
        try:
            print("🔄 初始化3D可视化...")

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=window_name,
                width=1280,
                height=720,
                visible=True
            )

            # 设置渲染选项
            render_opt = self.vis.get_render_option()
            render_opt.background_color = np.array([0.1, 0.1, 0.2])
            render_opt.point_size = self.point_size
            render_opt.light_on = True
            render_opt.mesh_show_wireframe = self.wireframe

            # 添加坐标系
            if self.show_axes:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                self.vis.add_geometry(coordinate_frame)

            self.running = True
            print("✅ 3D可视化已启动")

            # 设置初始视角
            self._set_default_view()

            return True

        except Exception as e:
            print(f"❌ 初始化可视化失败: {e}")
            return False

    def _set_default_view(self):
        """设置默认视角"""
        try:
            ctr = self.vis.get_view_control()
            ctr.set_front([0, -0.5, -1])
            ctr.set_up([0, -1, 0.5])
            ctr.set_zoom(0.8)
            ctr.set_lookat([0, 0, 0])
        except:
            pass

    def update_display(self, pointcloud=None, mesh=None):
        """更新显示内容"""
        if not self.running:
            return False

        try:
            # 清除旧几何体
            self.vis.clear_geometries()

            # 添加坐标系
            if self.show_axes:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                self.vis.add_geometry(coordinate_frame)

            # 根据显示模式添加几何体
            if self.display_mode == 'pointcloud' and pointcloud is not None:
                self.vis.add_geometry(pointcloud)
                self.current_geometry = pointcloud

            elif self.display_mode == 'mesh' and mesh is not None:
                # 设置网格颜色
                mesh.paint_uniform_color(self.mesh_color)
                self.vis.add_geometry(mesh)
                self.current_geometry = mesh

            elif self.display_mode == 'both':
                if pointcloud is not None:
                    self.vis.add_geometry(pointcloud)

                if mesh is not None:
                    mesh.paint_uniform_color(self.mesh_color)
                    self.vis.add_geometry(mesh)

                if pointcloud is not None and mesh is not None:
                    self.current_geometry = (pointcloud, mesh)
                elif pointcloud is not None:
                    self.current_geometry = pointcloud
                elif mesh is not None:
                    self.current_geometry = mesh

            return True

        except Exception as e:
            print(f"[可视化] 更新显示失败: {e}")
            return False

    def set_display_mode(self, mode):
        """设置显示模式"""
        valid_modes = ['pointcloud', 'mesh', 'both']
        if mode in valid_modes:
            self.display_mode = mode
            print(f"✅ 显示模式切换为: {mode}")
            return True
        return False

    def toggle_wireframe(self):
        """切换线框显示"""
        self.wireframe = not self.wireframe
        if self.vis:
            render_opt = self.vis.get_render_option()
            render_opt.mesh_show_wireframe = self.wireframe
        print(f"✅ 线框显示: {'开启' if self.wireframe else '关闭'}")
        return self.wireframe

    def save_viewpoint(self):
        """保存当前视角"""
        if self.vis:
            try:
                param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                self.camera_params = param
                print("✅ 视角已保存")
                return True
            except:
                pass
        return False

    def load_viewpoint(self):
        """加载保存的视角"""
        if self.vis and self.camera_params is not None:
            try:
                self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.camera_params)
                print("✅ 视角已加载")
                return True
            except:
                pass
        return False

    def render(self):
        """渲染一帧"""
        if self.running:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                return True
            except:
                return False
        return False

    def save_screenshot(self, filename):
        """保存截图"""
        if self.running:
            try:
                self.vis.capture_screen_image(filename)
                return True
            except:
                return False
        return False

    def stop(self):
        """停止可视化"""
        if self.vis:
            try:
                self.vis.destroy_window()
                self.running = False
                print("✅ 3D可视化已关闭")
            except:
                pass


# ==================== 4. 主应用程序 ====================
class Astra3DScannerPro:
    """Astra相机3D扫描器增强版 - 支持网格重建"""

    def __init__(self):
        # 驱动路径
        self.driver_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"

        # 初始化模块
        self.camera = AstraCameraManager(self.driver_path)
        self.processor = Enhanced3DProcessor()
        self.viewer = Enhanced3DViewer()

        # 状态变量
        self.running = False
        self.scanning = False
        self.auto_reconstruct = False
        self.paused = False

        # 性能监控
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_mesh_reconstruct = 0

        # 数据存储
        self.scan_data = []

        # 网格重建参数
        self.mesh_reconstruct_interval = 10.0  # 自动重建间隔（秒）

        print("🚀 Astra 3D扫描器增强版初始化完成")
        print("📦 支持功能: 点云采集、网格重建、OBJ/STL保存")

    def run(self):
        """运行主程序"""
        print("=" * 60)
        print("Astra相机3D扫描系统 - 增强版")
        print("支持实时网格重建和OBJ/STL导出")
        print("=" * 60)

        # 1. 初始化相机
        print("\n1. 初始化相机...")
        if not self.camera.initialize():
            print("❌ 相机初始化失败")
            return

        # 2. 启动流
        print("\n2. 启动采集流...")
        if not self.camera.start_streams():
            print("❌ 流启动失败")
            self.camera.stop()
            return

        # 3. 初始化可视化
        print("\n3. 初始化3D可视化...")
        if not self.viewer.initialize():
            print("❌ 可视化初始化失败")
            self.camera.stop()
            return

        # 4. 创建OpenCV窗口
        cv2.namedWindow("深度图", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("深度图", 640, 480)

        cv2.namedWindow("红外图", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("红外图", 640, 480)

        cv2.namedWindow("控制面板", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("控制面板", 500, 300)

        # 5. 打印控制说明
        self._print_controls()

        # 6. 主循环
        print("\n" + "=" * 60)
        print("系统运行中... 按Q退出")
        print("=" * 60)

        self.running = True
        self._main_loop()

        # 7. 清理
        self._cleanup()

    def _main_loop(self):
        """主循环"""
        last_update_time = time.time()
        update_interval = 0.1  # 10Hz更新

        try:
            while self.running:
                current_time = time.time()

                # 获取帧数据
                depth_frame, ir_frame = self.camera.get_frames()

                # 显示图像
                if depth_frame is not None:
                    depth_display = self._depth_to_colormap(depth_frame)
                    cv2.imshow("深度图", depth_display)

                if ir_frame is not None:
                    if len(ir_frame.shape) == 2:
                        ir_display = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                    else:
                        ir_display = ir_frame
                    cv2.imshow("红外图", ir_display)

                # 处理点云和网格（控制频率）
                if not self.paused and depth_frame is not None:
                    if self.scanning and current_time - last_update_time > update_interval:
                        # 处理深度帧
                        if self.processor.process_frame(depth_frame):
                            # 自动网格重建
                            if (self.auto_reconstruct and
                                current_time - self.last_mesh_reconstruct > self.mesh_reconstruct_interval):
                                self._reconstruct_mesh_async()
                                self.last_mesh_reconstruct = current_time

                        last_update_time = current_time

                # 更新3D显示
                self._update_3d_view()

                # 显示控制面板
                control_panel = self._create_control_panel()
                cv2.imshow("控制面板", control_panel)

                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard(key)

                # 更新FPS
                self._update_fps()

        except KeyboardInterrupt:
            print("\n🔴 用户中断")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()

    def _update_3d_view(self):
        """更新3D视图"""
        if not self.viewer.running:
            return

        # 获取当前几何体
        pointcloud = self.processor.get_pointcloud()
        mesh = self.processor.get_mesh()

        # 更新显示
        self.viewer.update_display(pointcloud, mesh)

        # 渲染
        self.viewer.render()

    def _reconstruct_mesh_async(self):
        """异步重建网格（避免阻塞主线程）"""
        if self.processor.total_points < 1000:
            return

        def reconstruct():
            try:
                mesh = self.processor.reconstruct_mesh(method='poisson')
                if mesh is not None:
                    print(f"🔄 自动网格重建完成")
            except Exception as e:
                print(f"⚠️ 异步网格重建失败: {e}")

        # 在后台线程中重建
        thread = threading.Thread(target=reconstruct, daemon=True)
        thread.start()

    def _depth_to_colormap(self, depth_frame):
        """深度图转伪彩色"""
        # 转换为毫米用于显示
        depth_mm = depth_frame * 1000

        # 归一化到0-255
        depth_normalized = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 添加文本信息
        stats = self.camera.get_stats()
        processor_stats = self.processor.get_stats()

        # 第一行：基本状态
        text = f"深度图 | 队列: {stats['depth_queue_size']} | 帧数: {stats['frames_captured']}"
        cv2.putText(depth_colored, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 第二行：点云信息
        if self.scanning:
            text2 = f"扫描中 | 点数: {processor_stats['total_points']:,}"
            cv2.putText(depth_colored, text2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 第三行：网格信息
        if processor_stats['has_mesh']:
            text3 = f"网格: {processor_stats['last_mesh_vertices']:,}顶点 {processor_stats['last_mesh_faces']:,}面片"
            cv2.putText(depth_colored, text3, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

        return depth_colored

    def _create_control_panel(self):
        """创建控制面板"""
        panel = np.zeros((300, 500, 3), dtype=np.uint8)

        # 标题
        title = "Astra 3D扫描控制器 - 增强版"
        cv2.putText(panel, title, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 系统状态
        stats = self.processor.get_stats()
        status_lines = [
            f"系统状态: {'运行中' if self.running else '停止'}",
            f"采集状态: {'🟢 扫描中' if self.scanning else '⚪ 待机'}",
            f"FPS: {self.fps:.1f}",
            f"点云点数: {stats['total_points']:,}",
            f"已处理帧: {stats['frames_processed']}",
            f"显示模式: {self.viewer.display_mode}",
            f"网格重建: {'🟢 自动' if self.auto_reconstruct else '⚪ 手动'}",
        ]

        if stats['has_mesh']:
            status_lines.extend([
                f"网格顶点: {stats['last_mesh_vertices']:,}",
                f"网格面片: {stats['last_mesh_faces']:,}",
                f"重建次数: {stats['meshes_generated']}",
            ])

        y_offset = 60
        for line in status_lines:
            color = self._get_status_color(line)
            cv2.putText(panel, line, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        # 控制提示
        hints = [
            "控制键:",
            "  Q: 退出  SPACE: 开始/停止扫描  P: 暂停",
            "  S: 保存点云  M: 重建网格  G: 保存网格",
            "  1: 显示点云  2: 显示网格  3: 同时显示",
            "  W: 切换线框  A: 自动重建  C: 清除数据",
            "  V: 保存截图  R: 重置视角  I: 保存视角",
        ]

        y_offset += 10
        for hint in hints:
            cv2.putText(panel, hint, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
            y_offset += 20

        return panel

    def _get_status_color(self, text):
        """根据文本内容获取颜色"""
        if '🟢' in text:
            return (0, 255, 0)  # 绿色
        elif '⚪' in text:
            return (200, 200, 200)  # 灰色
        elif '扫描中' in text:
            return (0, 255, 0)  # 绿色
        elif '网格' in text:
            return (255, 200, 0)  # 橙色
        else:
            return (200, 200, 200)  # 灰色

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
            else:
                print("⏹️ 停止扫描")
                print(f"  扫描了 {self.processor.stats['frames_processed']} 帧数据")

        # 暂停/继续
        elif key == ord('p'):
            self.paused = not self.paused
            print(f"⏸️  {'暂停' if self.paused else '继续'}")

        # 保存点云
        elif key == ord('s'):
            if self.processor.total_points > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 提供多种格式选择
                formats = [
                    ('PLY格式', f"pointcloud_{timestamp}.ply"),
                    ('PCD格式', f"pointcloud_{timestamp}.pcd"),
                    ('XYZ格式', f"pointcloud_{timestamp}.xyz"),
                ]

                print("\n💾 选择点云保存格式:")
                for i, (name, filename) in enumerate(formats, 1):
                    print(f"  {i}. {name}")

                # 简单实现：保存为PLY
                filename = formats[0][1]
                if self.processor.save_pointcloud(filename):
                    print(f"✅ 点云已保存: {filename}")
                else:
                    print("❌ 保存点云失败")
            else:
                print("❌ 当前无点云数据")

        # 重建网格
        elif key == ord('m'):
            if self.processor.total_points > 1000:
                print("🔄 开始重建网格...")
                mesh = self.processor.reconstruct_mesh(method='poisson')
                if mesh is not None:
                    print("✅ 网格重建完成")
                    # 切换到网格显示模式
                    self.viewer.set_display_mode('mesh')
                else:
                    print("❌ 网格重建失败")
            else:
                print("❌ 点云点数不足，无法重建网格")

        # 保存网格
        elif key == ord('g'):
            mesh = self.processor.get_mesh()
            if mesh is not None and len(mesh.vertices) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 多种网格格式
                formats = [
                    ('OBJ格式', f"mesh_{timestamp}.obj"),
                    ('STL格式', f"mesh_{timestamp}.stl"),
                    ('PLY格式', f"mesh_{timestamp}.ply"),
                    ('GLTF格式', f"mesh_{timestamp}.gltf"),
                ]

                print("\n💾 选择网格保存格式:")
                for i, (name, filename) in enumerate(formats, 1):
                    print(f"  {i}. {name}")

                # 简单实现：保存为OBJ和STL
                for name, filename in [formats[0], formats[1]]:
                    if self.processor.save_mesh(filename):
                        print(f"✅ 网格已保存 ({name}): {filename}")
            else:
                print("❌ 当前无网格数据")

        # 显示模式切换
        elif key == ord('1'):
            self.viewer.set_display_mode('pointcloud')
        elif key == ord('2'):
            self.viewer.set_display_mode('mesh')
        elif key == ord('3'):
            self.viewer.set_display_mode('both')

        # 切换线框显示
        elif key == ord('w'):
            self.viewer.toggle_wireframe()

        # 切换自动重建
        elif key == ord('a'):
            self.auto_reconstruct = not self.auto_reconstruct
            print(f"🔄 自动网格重建: {'开启' if self.auto_reconstruct else '关闭'}")

        # 清除数据
        elif key == ord('c'):
            self.processor.clear()
            self.scan_data = []
            print("✅ 所有3D数据已清除")

        # 保存截图
        elif key == ord('v'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            if self.viewer.save_screenshot(filename):
                print(f"✅ 3D视图截图已保存: {filename}")

        # 保存视角
        elif key == ord('i'):
            if self.viewer.save_viewpoint():
                print("✅ 当前3D视角已保存")

        # 加载视角
        elif key == ord('r'):
            if self.viewer.load_viewpoint():
                print("✅ 3D视角已加载")
            else:
                print("❌ 无可用的视角配置")

    def _update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def _print_controls(self):
        """打印控制说明"""
        controls = """
        ==================== 控制说明 ====================
        控制键:
          Q / ESC: 退出程序
          SPACE: 开始/停止扫描
          P: 暂停/继续
        
        点云操作:
          S: 保存点云 (PLY/PCD/XYZ格式)
          C: 清除所有数据
        
        网格操作:
          M: 手动重建网格
          G: 保存网格 (OBJ/STL/PLY/GLTF格式)
          A: 切换自动重建模式
          W: 切换线框显示
        
        显示控制:
          1: 显示点云模式
          2: 显示网格模式  
          3: 同时显示点云和网格
          V: 保存3D视图截图
          I: 保存当前视角
          R: 加载保存的视角
        
        操作流程:
        1. 按SPACE开始扫描，缓慢移动相机
        2. 按M重建网格（或开启A自动重建）
        3. 按G保存网格为OBJ/STL格式
        4. 按2切换网格显示模式查看效果
        ===============================================
        """
        print(controls)

    def _cleanup(self):
        """清理资源"""
        print("\n正在关闭系统...")

        # 自动保存未保存的数据
        if self.processor.total_points > 0:
            print("💾 自动保存扫描数据...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存点云
            ply_filename = f"autosave_pointcloud_{timestamp}.ply"
            self.processor.save_pointcloud(ply_filename)

            # 如果有网格，也保存
            mesh = self.processor.get_mesh()
            if mesh is not None and len(mesh.vertices) > 0:
                obj_filename = f"autosave_mesh_{timestamp}.obj"
                self.processor.save_mesh(obj_filename)

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()

        # 停止可视化
        self.viewer.stop()

        # 停止相机
        self.camera.stop()

        print("✅ 系统已安全关闭")


# ==================== 5. 批处理导出功能 ====================
class BatchExporter:
    """批量导出工具"""

    @staticmethod
    def convert_pointcloud_to_mesh(input_file, output_file, method='poisson'):
        """将点云文件转换为网格文件"""
        try:
            print(f"🔄 转换 {input_file} -> {output_file}")

            # 加载点云
            pcd = o3d.io.read_point_cloud(input_file)
            if len(pcd.points) == 0:
                print("❌ 无法加载点云文件")
                return False

            print(f"  点云点数: {len(pcd.points):,}")

            # 下采样
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

            # 估计法向量
            pcd.estimate_normals()

            # 重建网格
            if method == 'poisson':
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9
                )
            elif method == 'ball_pivoting':
                radii = [0.02, 0.04, 0.08]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            # 保存网格
            if output_file.endswith('.obj'):
                o3d.io.write_triangle_mesh(output_file, mesh)
            elif output_file.endswith('.stl'):
                o3d.io.write_triangle_mesh(output_file, mesh)
            elif output_file.endswith('.ply'):
                o3d.io.write_triangle_mesh(output_file, mesh)

            print(f"✅ 转换完成: {output_file}")
            print(f"  网格顶点: {len(mesh.vertices):,}")
            print(f"  网格面片: {len(mesh.triangles):,}")

            return True

        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return False

    @staticmethod
    def batch_convert_folder(input_folder, output_folder, format='obj'):
        """批量转换文件夹中的所有点云文件"""
        try:
            os.makedirs(output_folder, exist_ok=True)

            # 查找所有点云文件
            pointcloud_files = []
            for ext in ['.ply', '.pcd', '.xyz']:
                pointcloud_files.extend(
                    [f for f in os.listdir(input_folder) if f.endswith(ext)]
                )

            print(f"📁 找到 {len(pointcloud_files)} 个点云文件")

            for i, filename in enumerate(pointcloud_files, 1):
                input_path = os.path.join(input_folder, filename)
                output_name = os.path.splitext(filename)[0] + f'.{format}'
                output_path = os.path.join(output_folder, output_name)

                print(f"\n[{i}/{len(pointcloud_files)}] 处理: {filename}")
                BatchExporter.convert_pointcloud_to_mesh(input_path, output_path)

            print(f"\n🎉 批量转换完成!")
            return True

        except Exception as e:
            print(f"❌ 批量转换失败: {e}")
            return False


# ==================== 6. 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("Astra相机3D扫描系统 - 专业版")
    print("支持实时网格重建和OBJ/STL导出")
    print("=" * 60)

    # 检查必要依赖
    try:
        import cv2
        import numpy as np
        import open3d as o3d
        import primesense
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install opencv-python numpy open3d primesense")
        return

    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/pointclouds", exist_ok=True)
    os.makedirs("output/meshes", exist_ok=True)

    # 运行扫描器
    try:
        scanner = Astra3DScannerPro()
        scanner.run()
    except KeyboardInterrupt:
        print("\n\n🔴 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()