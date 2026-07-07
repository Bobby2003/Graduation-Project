import cv2
import numpy as np
import open3d as o3d
import threading
import queue
import time
import copy
import os
import warnings
from collections import deque
from primesense import openni2
from primesense import _openni2 as c_api

# 忽略特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ==================== 1. 修复的深度流管理器 ====================
class DepthStreamManager:
    def __init__(self, device_id=None):
        self.device = None
        self.depth_stream = None
        self.running = False
        self.depth_queue = queue.Queue(maxsize=20)
        self.frame_count = 0
        self.fx = 475.0  # 默认内参
        self.fy = 475.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 0.001
        self.device_id = device_id
        self.depth_width = 640
        self.depth_height = 480

    def initialize(self):
        try:
            # 初始化OpenNI2
            print("🔄 初始化OpenNI2...")
            try:
                openni2.initialize()
            except Exception as e:
                print(f"⚠️ OpenNI2初始化警告: {e}")

            if not openni2.is_initialized():
                print("❌ OpenNI2 未初始化")
                return False

            # 打开设备
            try:
                if self.device_id:
                    self.device = openni2.Device.open(self.device_id)
                else:
                    self.device = openni2.Device.open_any()
            except Exception as e:
                print(f"❌ 无法打开设备: {e}")
                return False

            # 获取设备信息
            try:
                dev_info = self.device.get_device_info()
                device_name = dev_info.name.decode('utf-8', errors='ignore')
                print(f"✅ 设备已连接: {device_name}")
                print(f"  供应商: {dev_info.vendor.decode('utf-8', errors='ignore')}")
                print(f"  设备ID: {dev_info.usbProductId}")
            except Exception as e:
                print(f"⚠️ 获取设备信息失败: {e}")
                print("✅ 设备已连接 (使用默认参数)")

            return True

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False

    def start_stream(self):
        try:
            # 创建深度流
            print("🔄 创建深度流...")
            self.depth_stream = self.device.create_depth_stream()

            # 设置视频模式
            video_mode = c_api.OniVideoMode(
                pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=self.depth_width,
                resolutionY=self.depth_height,
                fps=30
            )
            self.depth_stream.set_video_mode(video_mode)

            # 启动深度流
            self.depth_stream.start()

            # 尝试获取传感器信息（可选）
            try:
                sensor_info = self.depth_stream.get_sensor_info()
                print(f"✅ 深度传感器信息获取成功")
            except Exception as e:
                print(f"⚠️ 无法获取传感器信息，使用默认参数: {e}")

            print(f"✅ 深度流已启动 ({self.depth_width}x{self.depth_height} @ 30fps)")

            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            return True

        except Exception as e:
            print(f"❌ 启动深度流失败: {e}")
            return False

    def _capture_loop(self):
        print("📷 深度采集线程启动...")
        while self.running:
            try:
                # 读取深度帧
                frame = self.depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()

                # 转换为numpy数组
                depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape(
                    self.depth_height, self.depth_width
                )

                # 简单的去噪处理
                if np.any(depth_array > 0):
                    depth_array = cv2.medianBlur(depth_array, 3)

                # 添加时间戳
                timestamp = time.time()

                # 放入队列（非阻塞）
                if not self.depth_queue.full():
                    self.depth_queue.put((depth_array, timestamp))

                self.frame_count += 1

                # 每100帧打印一次状态
                if self.frame_count % 100 == 0:
                    print(f"[采集线程] 已采集 {self.frame_count} 帧")

            except Exception as e:
                if self.running:
                    print(f"[采集线程] 读取帧失败: {e}")
                break

        print("📷 深度采集线程结束")

    def get_depth_frame(self, timeout=0.1):
        try:
            return self.depth_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self):
        print("🛑 正在停止深度流...")
        self.running = False

        # 等待采集线程结束
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)

        # 停止深度流
        if self.depth_stream:
            try:
                self.depth_stream.stop()
                print("✅ 深度流已停止")
            except Exception as e:
                print(f"⚠️ 停止深度流时出错: {e}")

        # 关闭设备
        if self.device:
            try:
                self.device.close()
                print("✅ 设备已关闭")
            except Exception as e:
                print(f"⚠️ 关闭设备时出错: {e}")

        # 卸载OpenNI2
        try:
            openni2.unload()
            print("✅ OpenNI2已卸载")
        except Exception as e:
            print(f"⚠️ 卸载OpenNI2时出错: {e}")


# ==================== 2. 简化的点云处理器 ====================
class SimplePointCloudProcessor:
    def __init__(self, fx=475.0, fy=475.0, cx=320.0, cy=240.0, depth_scale=0.001):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

        # 深度范围
        self.min_depth = 0.3
        self.max_depth = 2.5

        # 多帧存储
        self.pointcloud_buffer = deque(maxlen=30)  # 存储最近30帧的点云
        self.current_pointcloud = o3d.geometry.PointCloud()

        # 统计
        self.total_points = 0
        self.processed_frames = 0

    def reconstruct_and_save_mesh(self, filename, method='poisson'):
        """
        将当前点云重建为网格并保存 (支持 OBJ, STL, PLY 等格式)
        参数:
            filename: 保存的文件路径
            method: 重建方法 ('poisson' 或 'ball_pivoting')
        """
        if len(self.current_pointcloud.points) < 1000:
            print("❌ 点云点数不足，无法重建网格")
            return False

        try:
            print(f"🔄 开始网格重建 (点数: {len(self.current_pointcloud.points)})...")

            # 深度拷贝点云
            pcd = copy.deepcopy(self.current_pointcloud)

            # 估计法向量
            print("  正在估计法向量...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=30)

            # 重建网格
            if method == 'ball_pivoting':
                print("  使用滚球法重建...")
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = avg_dist * 2.5
                radii = [radius, radius * 2]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))

            else:  # poisson reconstruction
                print("  使用泊松重建...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9, width=0, scale=1.1, linear_fit=False)

                # 根据密度移除低密度顶点（可选）
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)

            # 网格后处理
            print("  网格后处理...")

            # 1. 网格简化（如果面数太多）
            if len(mesh.triangles) > 100000:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)

            # 2. 移除重复顶点和无效面
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_non_manifold_edges()

            # 3. 计算顶点法线（用于渲染）
            mesh.compute_vertex_normals()

            # 保存网格
            o3d.io.write_triangle_mesh(filename, mesh)

            print(f"✅ 网格已保存至: {filename}")
            print(f"  顶点数: {len(mesh.vertices)}, 三角面数: {len(mesh.triangles)}")
            return True

        except Exception as e:
            print(f"❌ 网格重建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def depth_to_pointcloud(self, depth_frame):
        """将深度图转换为点云"""
        if depth_frame is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # 获取有效深度点
        height, width = depth_frame.shape

        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 计算3D坐标（核心修改在这里）
        z = depth_frame.astype(np.float32) * self.depth_scale
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # 【修改点1：修正坐标系】通常需要翻转Y轴和Z轴
        # 假设原始坐标系为：X向右，Y向下，Z向前
        # 目标坐标系为：X向右，Y向上，Z向后（OpenGL/大多数3D软件）
        x_corrected = x
        y_corrected = -y  # 翻转Y轴，解决上下颠倒
        z_corrected = -z  # 翻转Z轴，使正方向向后，更符合直觉（可选，但建议）

        # 创建点云
        points = np.stack([x_corrected, y_corrected, z_corrected], axis=-1).reshape(-1, 3)

        # 过滤无效点 (保持不变)
        valid_mask = (z.reshape(-1) > self.min_depth) & (z.reshape(-1) < self.max_depth)
        points = points[valid_mask]

        # 生成颜色（基于深度，可选修改颜色以匹配新坐标系）
        if len(points) > 0:
            # 使用校正后的Z值计算颜色
            z_vals_corrected = -points[:, 2]  # 因为z_corrected = -z
            depth_norm = (z_vals_corrected - self.min_depth) / (self.max_depth - self.min_depth)
            depth_norm = np.clip(depth_norm, 0, 1)

            colors = np.zeros((len(points), 3))
            colors[:, 0] = depth_norm  # 红色分量
            colors[:, 1] = 0.2  # 固定绿色分量
            colors[:, 2] = 1.0 - depth_norm  # 蓝色分量
        else:
            colors = np.zeros((0, 3))

        return points, colors

    def add_depth_frame(self, depth_frame):
        """添加深度帧并更新点云"""
        if depth_frame is None:
            return False

        try:
            # 转换为点云
            points, colors = self.depth_to_pointcloud(depth_frame)

            if len(points) < 100:  # 点太少，跳过
                return False

            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 添加到缓冲区
            self.pointcloud_buffer.append(pcd)

            # 合并点云
            self._merge_pointclouds()

            self.processed_frames += 1
            self.total_points = len(self.current_pointcloud.points)

            return True

        except Exception as e:
            print(f"点云处理失败: {e}")
            return False

    def _merge_pointclouds(self):
        """合并缓冲区中的点云"""
        if not self.pointcloud_buffer:
            return

        # 合并所有点云
        all_points = []
        all_colors = []

        for pcd in self.pointcloud_buffer:
            if len(pcd.points) > 0:
                all_points.append(np.asarray(pcd.points))
                all_colors.append(np.asarray(pcd.colors))

        if not all_points:
            return

        # 合并
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        # 创建新点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        # 下采样（如果点太多）
        if len(merged_points) > 50000:
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)

        # 更新当前点云
        self.current_pointcloud = merged_pcd

    def get_pointcloud(self):
        """获取当前点云"""
        return copy.deepcopy(self.current_pointcloud)

    def clear(self):
        """清除所有数据"""
        self.pointcloud_buffer.clear()
        self.current_pointcloud.clear()
        self.total_points = 0
        self.processed_frames = 0
        print("✅ 已清除所有点云数据")

    def save_pointcloud(self, filename):
        """保存点云到文件"""
        if len(self.current_pointcloud.points) > 0:
            o3d.io.write_point_cloud(filename, self.current_pointcloud)
            return True
        return False


# ==================== 3. 可视化管理器 ====================
class VisualizerManager:
    def __init__(self):
        self.vis = None
        self.running = False
        self.current_pcd = None
        self.window_width = 1200
        self.window_height = 800

    def initialize(self):
        """初始化可视化窗口"""
        try:
            print("🔄 初始化3D可视化...")
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="实时3D建模",
                width=self.window_width,
                height=self.window_height,
                visible=True
            )

            # 设置渲染选项
            render_opt = self.vis.get_render_option()
            render_opt.background_color = np.array([0.1, 0.1, 0.2])
            render_opt.point_size = 1.5
            render_opt.light_on = True
            render_opt.mesh_show_wireframe = False

            # 添加坐标轴
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(coordinate_frame)

            # 添加网格地面
            grid = self._create_grid()
            self.vis.add_geometry(grid)

            self.running = True
            print("✅ 3D可视化窗口已启动")
            return True

        except Exception as e:
            print(f"❌ 初始化可视化失败: {e}")
            return False

    def _create_grid(self):
        """创建网格地面"""
        grid_size = 1.0
        grid_step = 0.1
        grid_lines = []
        points = []

        # 创建网格线
        for i in range(int(-grid_size / grid_step), int(grid_size / grid_step) + 1):
            x = i * grid_step
            points.append([x, -grid_size, 0])
            points.append([x, grid_size, 0])
            points.append([-grid_size, x, 0])
            points.append([grid_size, x, 0])

        # 创建线集
        lines = []
        for i in range(0, len(points), 2):
            lines.append([i, i + 1])

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(len(lines))])

        return grid

    def update_pointcloud(self, pointcloud):
        """更新点云显示"""
        if not self.running or pointcloud is None:
            return

        try:
            # 清除旧点云
            self.vis.clear_geometries()

            # 重新添加坐标系和网格
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(coordinate_frame)
            grid = self._create_grid()
            self.vis.add_geometry(grid)

            # 添加新点云
            if len(pointcloud.points) > 0:
                self.vis.add_geometry(pointcloud)
                self.current_pcd = pointcloud

        except Exception as e:
            print(f"[可视化] 更新点云失败: {e}")

    def render(self):
        """渲染当前帧"""
        if self.running:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                return True
            except Exception as e:
                print(f"[可视化] 渲染失败: {e}")
                return False
        return False

    def stop(self):
        """停止可视化"""
        if self.vis:
            self.vis.destroy_window()
            self.running = False
            print("✅ 3D可视化窗口已关闭")


# ==================== 4. 主应用程序 ====================
class MainApplication:
    def __init__(self):
        self.depth_manager = DepthStreamManager()
        self.pointcloud_processor = None
        self.visualizer = VisualizerManager()
        self.running = False

        # 性能监控
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # 创建输出目录
        os.makedirs("output", exist_ok=True)

    def run(self):
        """主运行函数"""
        print("=" * 60)
        print("奥比中光 MSC-1 3D建模系统 (简化版)")
        print("=" * 60)

        # 1. 初始化深度设备
        print("\n1. 初始化深度设备...")
        if not self.depth_manager.initialize():
            print("❌ 深度设备初始化失败")
            return

        # 2. 启动深度流
        print("\n2. 启动深度流...")
        if not self.depth_manager.start_stream():
            print("❌ 深度流启动失败")
            self.depth_manager.stop()
            return

        # 3. 初始化点云处理器
        print("\n3. 初始化点云处理器...")
        self.pointcloud_processor = SimplePointCloudProcessor(
            fx=self.depth_manager.fx,
            fy=self.depth_manager.fy,
            cx=self.depth_manager.cx,
            cy=self.depth_manager.cy,
            depth_scale=self.depth_manager.depth_scale
        )

        # 4. 初始化可视化
        print("\n4. 初始化可视化...")
        if not self.visualizer.initialize():
            print("❌ 可视化初始化失败")
            self.depth_manager.stop()
            return

        # 5. 创建OpenCV窗口
        cv2.namedWindow("深度图", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("深度图", 640, 480)

        # 6. 打印控制说明
        self._print_controls()

        # 7. 主循环
        print("\n" + "=" * 60)
        print("系统运行中...")
        print("按 'Q' 退出，按 'S' 保存点云，按 'C' 清除点云")
        print("=" * 60)

        self.running = True
        self._main_loop()

        # 8. 清理
        self._cleanup()

    def _main_loop(self):
        """主循环"""
        last_update_time = time.time()
        update_interval = 0.1  # 10 FPS更新

        try:
            while self.running:
                # 1. 获取深度帧
                depth_frame, timestamp = self.depth_manager.get_depth_frame(timeout=0.001)
                if depth_frame is None:
                    time.sleep(0.001)
                    continue

                # 2. 显示深度图
                depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_display.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imshow("深度图", depth_colored)

                # 3. 处理点云（控制频率）
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    if self.pointcloud_processor.add_depth_frame(depth_frame):
                        # 获取并显示点云
                        current_pcd = self.pointcloud_processor.get_pointcloud()
                        self.visualizer.update_pointcloud(current_pcd)

                    last_update_time = current_time

                # 4. 渲染3D视图
                self.visualizer.render()

                # 5. 处理键盘输入
                self._handle_keyboard()

                # 6. 更新FPS
                self._update_fps()

        except KeyboardInterrupt:
            print("\n🔴 用户中断 (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()

    def _handle_keyboard(self):
        """处理键盘输入"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q 或 ESC
            print("\n收到退出指令...")
            self.running = False

        elif key == ord('s'):  # 保存点云为PLY
            if self.pointcloud_processor and self.pointcloud_processor.total_points > 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"output/pointcloud_{timestamp}.ply"

                if self.pointcloud_processor.save_pointcloud(filename):
                    print(f"✅ 点云已保存: {filename}")
                    print(f"   点数: {self.pointcloud_processor.total_points}")
                else:
                    print("❌ 保存点云失败")
            else:
                print("❌ 当前无点云数据")

        elif key == ord('o'):  # 新增：保存为OBJ网格文件
            if self.pointcloud_processor and self.pointcloud_processor.total_points > 1000:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"output/mesh_{timestamp}.obj"  # 扩展名改为 .obj

                # 使用泊松重建方法
                if self.pointcloud_processor.reconstruct_and_save_mesh(filename, method='poisson'):
                    print(f"✅ 网格已保存为OBJ格式: {filename}")
                else:
                    print("❌ 保存网格失败")
            else:
                print("❌ 点云点数不足，需要至少1000个点")

        elif key == ord('c'):  # 清除点云
            if self.pointcloud_processor:
                self.pointcloud_processor.clear()
                print("✅ 已清除点云数据")

        elif key == ord('i'):  # 显示信息
            if self.pointcloud_processor:
                print("\n" + "=" * 40)
                print("系统信息:")
                print(f"  当前FPS: {self.fps:.1f}")
                print(f"  总点数: {self.pointcloud_processor.total_points}")
                print(f"  处理帧数: {self.pointcloud_processor.processed_frames}")
                print("=" * 40)

    def _update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

            # 在控制台显示状态
            total_points = self.pointcloud_processor.total_points if self.pointcloud_processor else 0
            print(f"\r状态: FPS={self.fps:.1f}, 点数={total_points}, 帧数={self.pointcloud_processor.processed_frames}",
                  end="")


    def _print_controls(self):
        """打印控制说明"""
        controls = """
            ==================== 控制说明 ====================
            Q / ESC: 退出程序
            S: 保存当前点云为PLY文件
            O: 保存当前点云为OBJ网格文件
            C: 清除所有点云数据
            I: 显示系统信息
    
            操作建议:
            1. 将设备对准要扫描的物体
            2. 缓慢移动设备以获取不同角度
            3. 累积足够点数后按S保存点云，或按O保存网格
            ===============================================
            """
        print(controls)

    def _cleanup(self):
        """清理资源"""
        print("\n正在关闭系统...")

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()

        # 停止可视化
        if self.visualizer:
            self.visualizer.stop()

        # 停止深度流
        if self.depth_manager:
            self.depth_manager.stop()

        print("✅ 系统已安全关闭")


# ==================== 5. 主函数 ====================
def main():
    # 检查必要的库
    try:
        import open3d as o3d
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        print("请安装: pip install open3d opencv-python numpy")
        return

    # 运行应用程序
    app = MainApplication()
    app.run()


if __name__ == "__main__":
    main()