"""
R200深度图3D建模系统
使用深度图创建3D点云和网格模型
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import os


class R200_3D_Modeler:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.depth_scale = 0.001  # 深度单位转换为米（假设深度单位是毫米）

        # 相机内参（需要根据R200标定，这是估计值）
        self.fx = 525.0  # 焦距x
        self.fy = 525.0  # 焦距y
        self.cx = 319.5  # 主点x
        self.cy = 239.5  # 主点y

        # 点云和模型存储
        self.point_clouds = []
        self.meshes = []

    def capture_depth_frame(self, capture_time=5):
        """捕获深度帧"""
        print("=" * 60)
        print("捕获深度帧")
        print("=" * 60)

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print("无法打开摄像头")
            return None

        print(f"开始捕获 {capture_time} 秒...")

        depth_frames = []
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < capture_time:
            ret, frame = self.cap.read()

            if not ret:
                print("读取帧失败")
                continue

            frame_count += 1

            # 如果是彩色图像，尝试提取深度信息
            if len(frame.shape) == 3:
                # 检查是否是伪装的深度图（三个通道相同）
                if np.array_equal(frame[:, :, 0], frame[:, :, 1]) and np.array_equal(frame[:, :, 1], frame[:, :, 2]):
                    depth_frame = frame[:, :, 0].astype(np.float32)
                else:
                    # 转换为灰度作为深度估计
                    depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                depth_frame = frame.astype(np.float32)

            depth_frames.append(depth_frame)

            # 显示进度
            elapsed = time.time() - start_time
            if frame_count % 10 == 0:
                print(f"\r已捕获 {frame_count} 帧，时间: {elapsed:.1f}/{capture_time}秒", end="")

        print(f"\n捕获完成: {frame_count} 帧")

        # 返回平均深度图以减少噪声
        if depth_frames:
            avg_depth = np.mean(depth_frames, axis=0)
            return avg_depth
        else:
            return None

    def depth_to_pointcloud(self, depth_frame):
        """将深度图转换为点云"""
        print("\n转换深度图为点云...")

        h, w = depth_frame.shape

        # 创建坐标网格
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xv, yv = np.meshgrid(x, y)

        # 转换为3D坐标
        # Z = 深度值（转换为米）
        Z = depth_frame * self.depth_scale

        # X = (u - cx) * Z / fx
        X = (xv - self.cx) * Z / self.fx

        # Y = (v - cy) * Z / fy
        Y = (yv - self.cy) * Z / self.fy

        # 展平数组
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

        # 移除无效点（深度为0的点）
        valid_mask = Z.flatten() > 0
        points = points[valid_mask]

        print(f"点云点数: {len(points):,}")

        return points

    def create_color_pointcloud(self, depth_frame, color_frame=None):
        """创建带颜色的点云"""
        print("\n创建带颜色的点云...")

        points = self.depth_to_pointcloud(depth_frame)

        if color_frame is not None:
            h, w = depth_frame.shape

            # 获取颜色值
            if len(color_frame.shape) == 3:
                # BGR转RGB
                colors = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                colors = colors.reshape(-1, 3) / 255.0  # 归一化到0-1
            else:
                # 灰度图
                colors = color_frame.reshape(-1, 1)
                colors = np.repeat(colors, 3, axis=1) / 255.0

            # 展平坐标网格用于索引
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.stack([yv.flatten(), xv.flatten()], axis=-1)

            # 移除无效点
            valid_mask = depth_frame.flatten() > 0
            colors = colors[valid_mask]
        else:
            # 使用深度值作为颜色
            colors = cm.viridis(depth_frame.flatten()[depth_frame.flatten() > 0] / depth_frame.max())
            colors = colors[:, :3]  # 取RGB，去掉alpha

        return points, colors

    def create_open3d_pointcloud(self, points, colors=None):
        """创建Open3D点云对象"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def visualize_pointcloud_matplotlib(self, points, colors=None, title="3D Point Cloud"):
        """使用Matplotlib可视化点云"""
        print("\n使用Matplotlib可视化点云...")

        fig = plt.figure(figsize=(15, 10))

        # 3D散点图
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                 c=colors if colors.shape[1] == 1 else colors,
                                 s=1, alpha=0.6, cmap='viridis')
        else:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                 c=points[:, 2], s=1, alpha=0.6, cmap='viridis')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        # 添加颜色条
        if colors is None or colors.shape[1] == 1:
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Depth (m)')

        # 设置相等的纵横比
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def create_mesh_from_pointcloud(self, points, method='ball_pivoting'):
        """从点云创建网格"""
        print(f"\n使用 {method} 方法创建网格...")

        pcd = self.create_open3d_pointcloud(points)

        # 下采样点云（可选，加速处理）
        print("下采样点云...")
        pcd_down = pcd.voxel_down_sample(voxel_size=0.01)

        # 估计法线
        print("估计法线...")
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

        if method == 'poisson':
            # 泊松重建
            print("泊松表面重建...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_down, depth=8)

            # 移除低密度顶点
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif method == 'ball_pivoting':
            # Ball Pivoting算法
            print("Ball Pivoting表面重建...")
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd_down, o3d.utility.DoubleVector(radii))

        # 计算顶点颜色
        print("计算顶点颜色...")
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

        print(f"网格信息: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")

        return mesh

    def save_3d_model(self, mesh, filename="3d_model.ply"):
        """保存3D模型"""
        print(f"\n保存3D模型到 {filename}...")

        # 确保目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        # 保存为PLY格式
        o3d.io.write_triangle_mesh(filename, mesh)

        # 同时保存为OBJ格式（更通用）
        obj_filename = filename.replace('.ply', '.obj')
        o3d.io.write_triangle_mesh(obj_filename, mesh)

        print(f"✅ 模型已保存:")
        print(f"  - {filename}")
        print(f"  - {obj_filename}")

        return filename, obj_filename

    def scan_multiple_views(self, num_views=4, capture_time=3):
        """多视角扫描"""
        print("=" * 60)
        print(f"多视角扫描 ({num_views} 个视角)")
        print("=" * 60)

        print("请按以下顺序放置物体:")
        print("1. 正面")
        print("2. 右侧")
        print("3. 背面（可选）")
        print("4. 左侧")
        print("\n按提示移动摄像头或物体")

        all_points = []
        all_colors = []

        for view in range(num_views):
            input(f"\n准备捕获视角 {view + 1}/{num_views}... 按回车键开始")

            print(f"捕获视角 {view + 1}...")
            depth_frame = self.capture_depth_frame(capture_time)

            if depth_frame is None:
                print(f"视角 {view + 1} 捕获失败")
                continue

            # 转换为点云
            points = self.depth_to_pointcloud(depth_frame)

            # 根据视角旋转点云
            angle = view * (360 / num_views)
            if angle > 0:
                # 绕Y轴旋转
                theta = np.radians(angle)
                rotation_matrix = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
                points = np.dot(points, rotation_matrix.T)

            all_points.append(points)

            # 可视化当前视角
            self.visualize_pointcloud_matplotlib(points, title=f"View {view + 1}")

            print(f"视角 {view + 1} 完成")

        # 合并所有点云
        if all_points:
            merged_points = np.vstack(all_points)
            print(f"\n合并后的点云: {len(merged_points):,} 个点")

            return merged_points
        else:
            print("没有成功捕获任何视角")
            return None

    def realtime_3d_scanning(self, duration=30):
        """实时3D扫描"""
        print("=" * 60)
        print("实时3D扫描模式")
        print("=" * 60)
        print("按 'q' 退出，按 's' 保存当前点云")

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print("无法打开摄像头")
            return

        # 用于存储所有点云
        all_points = []

        start_time = time.time()
        frame_count = 0

        # 创建实时可视化窗口
        plt.ion()
        fig = plt.figure(figsize=(15, 5))

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()

            if not ret:
                print("读取帧失败")
                continue

            frame_count += 1

            # 处理深度帧
            if len(frame.shape) == 3:
                depth_frame = frame[:, :, 0].astype(np.float32)
            else:
                depth_frame = frame.astype(np.float32)

            # 转换为点云
            points = self.depth_to_pointcloud(depth_frame)
            all_points.append(points)

            # 每10帧更新一次显示
            if frame_count % 10 == 0:
                # 清空图形
                plt.clf()

                # 显示原始深度图
                ax1 = fig.add_subplot(131)
                ax1.imshow(depth_frame, cmap='viridis')
                ax1.set_title(f'Depth Map - Frame {frame_count}')
                ax1.axis('off')

                # 显示点云俯视图
                if len(points) > 0:
                    ax2 = fig.add_subplot(132)
                    ax2.scatter(points[:, 0], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
                    ax2.set_xlabel('X (m)')
                    ax2.set_ylabel('Z (m)')
                    ax2.set_title('Top View')
                    ax2.axis('equal')

                # 显示点云侧视图
                ax3 = fig.add_subplot(133, projection='3d')
                if len(points) > 0:
                    scatter = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                                          c=points[:, 2], s=1, cmap='viridis')
                    ax3.set_xlabel('X (m)')
                    ax3.set_ylabel('Y (m)')
                    ax3.set_zlabel('Z (m)')
                    ax3.set_title('3D Point Cloud')

                plt.suptitle(f'Realtime 3D Scanning - FPS: {frame_count / (time.time() - start_time):.1f}')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

            # 检查按键
            if plt.waitforbuttonpress(0.001):
                break

        plt.ioff()
        plt.close()

        # 合并所有点云
        if all_points:
            merged_points = np.vstack(all_points)
            print(f"\n实时扫描完成:")
            print(f"  总帧数: {frame_count}")
            print(f"  总点数: {len(merged_points):,}")
            print(f"  平均FPS: {frame_count / duration:.1f}")

            return merged_points
        else:
            print("没有捕获到点云数据")
            return None


def main_menu():
    """主菜单"""
    print("=" * 60)
    print("R200 3D建模系统")
    print("=" * 60)

    # 创建3D建模器
    modeler = R200_3D_Modeler(camera_index=0)

    while True:
        print("\n选择功能:")
        print("1. 单帧深度图捕获")
        print("2. 创建3D点云")
        print("3. 创建3D网格模型")
        print("4. 多视角扫描")
        print("5. 实时3D扫描")
        print("6. 批量处理保存的深度图")
        print("7. 退出")

        choice = input("\n请选择 (1-7): ").strip()

        if choice == '1':
            # 单帧捕获
            duration = input("输入捕获时间(秒，默认3): ").strip()
            duration = int(duration) if duration.isdigit() else 3

            depth_frame = modeler.capture_depth_frame(duration)
            if depth_frame is not None:
                # 显示深度图
                plt.figure(figsize=(10, 8))
                plt.imshow(depth_frame, cmap='viridis')
                plt.colorbar(label='Depth Value')
                plt.title(f'Depth Map - {depth_frame.shape[1]}x{depth_frame.shape[0]}')
                plt.show()

                # 保存深度图
                save = input("是否保存深度图？(y/n): ").strip().lower()
                if save == 'y':
                    filename = f"depth_map_{time.strftime('%Y%m%d_%H%M%S')}.npy"
                    np.save(filename, depth_frame)
                    print(f"✅ 深度图已保存: {filename}")

        elif choice == '2':
            # 创建3D点云
            depth_frame = modeler.capture_depth_frame(3)
            if depth_frame is not None:
                # 创建点云
                points = modeler.depth_to_pointcloud(depth_frame)

                # 可视化
                modeler.visualize_pointcloud_matplotlib(points, title="3D Point Cloud")

                # 保存点云
                save = input("是否保存点云？(y/n): ").strip().lower()
                if save == 'y':
                    filename = f"point_cloud_{time.strftime('%Y%m%d_%H%M%S')}.npy"
                    np.save(filename, points)
                    print(f"✅ 点云已保存: {filename}")

        elif choice == '3':
            # 创建3D网格模型
            print("\n选择网格创建方法:")
            print("1. Poisson重建 (适用于复杂表面)")
            print("2. Ball Pivoting (适用于简单物体)")

            method_choice = input("请选择 (1-2): ").strip()
            method = 'poisson' if method_choice == '1' else 'ball_pivoting'

            depth_frame = modeler.capture_depth_frame(5)
            if depth_frame is not None:
                points = modeler.depth_to_pointcloud(depth_frame)
                mesh = modeler.create_mesh_from_pointcloud(points, method=method)

                # 可视化网格
                print("\n可视化3D网格...")
                o3d.visualization.draw_geometries([mesh], window_name="3D Mesh Model")

                # 保存模型
                save = input("是否保存3D模型？(y/n): ").strip().lower()
                if save == 'y':
                    filename = f"3d_model_{method}_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                    modeler.save_3d_model(mesh, filename)

        elif choice == '4':
            # 多视角扫描
            num_views = input("输入视角数量 (默认4): ").strip()
            num_views = int(num_views) if num_views.isdigit() else 4

            capture_time = input("输入每个视角的捕获时间(秒，默认3): ").strip()
            capture_time = int(capture_time) if capture_time.isdigit() else 3

            points = modeler.scan_multiple_views(num_views, capture_time)

            if points is not None:
                # 创建并保存完整模型
                mesh = modeler.create_mesh_from_pointcloud(points, method='poisson')

                # 可视化
                o3d.visualization.draw_geometries([mesh], window_name="Multi-view 3D Model")

                # 保存
                save = input("是否保存多视角模型？(y/n): ").strip().lower()
                if save == 'y':
                    filename = f"multiview_model_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                    modeler.save_3d_model(mesh, filename)

        elif choice == '5':
            # 实时3D扫描
            duration = input("输入扫描时间(秒，默认30): ").strip()
            duration = int(duration) if duration.isdigit() else 30

            points = modeler.realtime_3d_scanning(duration)

            if points is not None:
                # 从实时扫描创建模型
                mesh = modeler.create_mesh_from_pointcloud(points, method='poisson')

                # 可视化
                o3d.visualization.draw_geometries([mesh], window_name="Realtime 3D Scan")

                # 保存
                save = input("是否保存实时扫描模型？(y/n): ").strip().lower()
                if save == 'y':
                    filename = f"realtime_scan_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                    modeler.save_3d_model(mesh, filename)

        elif choice == '6':
            # 批量处理
            print("\n批量处理已保存的深度图")
            print("请确保深度图文件在当前目录下")

            import glob
            depth_files = glob.glob("depth_*.npy") + glob.glob("*.npy")

            if not depth_files:
                print("没有找到深度图文件")
                continue

            print(f"找到 {len(depth_files)} 个深度图文件")

            all_points = []
            for i, file in enumerate(depth_files, 1):
                print(f"\n处理文件 {i}/{len(depth_files)}: {file}")

                try:
                    depth_frame = np.load(file)
                    points = modeler.depth_to_pointcloud(depth_frame)
                    all_points.append(points)
                    print(f"  提取 {len(points):,} 个点")
                except Exception as e:
                    print(f"  处理失败: {e}")

            if all_points:
                merged_points = np.vstack(all_points)
                print(f"\n合并所有点云: {len(merged_points):,} 个点")

                # 创建模型
                mesh = modeler.create_mesh_from_pointcloud(merged_points, method='poisson')

                # 可视化
                o3d.visualization.draw_geometries([mesh], window_name="Batch Processed Model")

                # 保存
                save = input("是否保存批量处理模型？(y/n): ").strip().lower()
                if save == 'y':
                    filename = f"batch_model_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                    modeler.save_3d_model(mesh, filename)

        elif choice == '7':
            print("退出3D建模系统")
            if modeler.cap:
                modeler.cap.release()
            break

        else:
            print("无效选择")


if __name__ == "__main__":
    # 检查依赖
    try:
        import open3d as o3d

        print("✅ Open3D已安装")
    except ImportError:
        print("❌ Open3D未安装")
        print("安装命令: pip install open3d")
        print("或者使用: conda install -c open3d-admin open3d")
        exit(1)

    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\n3D建模系统结束")