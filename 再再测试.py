"""
R200深度相机实时3D建模 - 技术文档实现版
基于Intel R200技术文档：Z16_2_1 16位深度格式
"""

import cv2
import numpy as np
import open3d as o3d
import threading
import queue
import time
import os
import sys
from datetime import datetime
import copy

class R200DepthModeler:
    def __init__(self):
        self.caps = {}  # 多个摄像头句柄
        self.running = False

        # R200参数（根据技术文档）
        self.depth_config = {
            "index": 0,          # 深度流索引
            "width": 628,        # 深度图宽度
            "height": 468,       # 深度图高度
            "format": "Z16_2_1", # 16位深度格式
            "fps": 30,
            "scale": 0.001       # 毫米转米
        }

        self.ir_config = {
            "index": 1,          # IR流索引
            "width": 640,        # IR图宽度
            "height": 480,       # IR图高度
            "format": "LY12_2_1",# 12位IR格式
            "fps": 30
        }

        self.color_config = {
            "index": 2,          # 彩色流索引（可能需要尝试）
            "width": 640,        # 彩色图宽度
            "height": 480,       # 彩色图高度
            "format": "YUY2",    # YUY2格式
            "fps": 30
        }

        # 相机内参（R200典型值）
        self.fx_depth = 475.0    # 深度相机焦距x
        self.fy_depth = 475.0    # 深度相机焦距y
        self.cx_depth = self.depth_config["width"] / 2
        self.cy_depth = self.depth_config["height"] / 2

        self.fx_color = 525.0    # 彩色相机焦距x
        self.fy_color = 525.0    # 彩色相机焦距y
        self.cx_color = self.color_config["width"] / 2
        self.cy_color = self.color_config["height"] / 2

        # 队列
        self.depth_queue = queue.Queue(maxsize=5)
        self.ir_queue = queue.Queue(maxsize=5)
        self.color_queue = queue.Queue(maxsize=5)
        self.pointcloud_queue = queue.Queue(maxsize=3)

        # 3D数据
        self.global_pointcloud = None
        self.current_mesh = None
        self.accumulated_points = []
        self.accumulated_colors = []
        self.total_points = 0

        # 显示窗口
        self.window_depth = "R200深度图 (16位)"
        self.window_ir = "R200红外图"
        self.window_color = "R200彩色图"
        self.window_3d = "实时3D建模"

        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()

        # 处理参数
        self.min_depth = 0.3     # 最小深度 (米)
        self.max_depth = 3.0     # 最大深度 (米)
        self.voxel_size = 0.01   # 体素大小

        # 输出目录
        self.output_dir = "r200_depth_3d"
        os.makedirs(self.output_dir, exist_ok=True)

        print("基于Intel R200技术文档的3D建模系统")
        print("=" * 60)
        print("深度流: Z16_2_1格式, 628x468, 16位")
        print("红外流: LY12_2_1格式, 640x480, 12位")
        print("彩色流: YUY2格式, 640x480")
        print("=" * 60)

    def _detect_r200_streams(self):
        """检测R200的所有流"""
        print("检测R200流...")

        # 检测深度流（索引0，628x468）
        print(f"尝试打开深度流 (索引{self.depth_config['index']})...")
        depth_cap = cv2.VideoCapture(self.depth_config["index"], cv2.CAP_DSHOW)

        if depth_cap.isOpened():
            # 尝试设置16位格式
            try:
                # 某些情况下可以设置格式
                depth_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ' '))
            except:
                pass

            # 获取实际参数
            width = int(depth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(depth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"深度流: {width}x{height}")

            if width == 628 or width == 640:
                self.depth_config["width"] = width
                self.depth_config["height"] = height
                self.caps["depth"] = depth_cap
                print(f"✅ 深度流已打开: {width}x{height}")
            else:
                print(f"⚠️ 深度流分辨率不匹配: {width}x{height}")
                depth_cap.release()
        else:
            print("❌ 无法打开深度流")

        # 检测IR流（索引1，640x480）
        print(f"尝试打开IR流 (索引{self.ir_config['index']})...")
        ir_cap = cv2.VideoCapture(self.ir_config["index"], cv2.CAP_DSHOW)

        if ir_cap.isOpened():
            width = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"IR流: {width}x{height}")

            if width == 640:
                self.ir_config["width"] = width
                self.ir_config["height"] = height
                self.caps["ir"] = ir_cap
                print(f"✅ IR流已打开: {width}x{height}")
            else:
                print(f"⚠️ IR流分辨率不匹配: {width}x{height}")
                ir_cap.release()
        else:
            print("❌ 无法打开IR流")

        # 检测彩色流（可能需要尝试多个索引）
        print("尝试打开彩色流...")
        for idx in [2, 3, 700, 701]:
            color_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if color_cap.isOpened():
                width = int(color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"索引{idx}: {width}x{height}")

                if width == 640 or width == 1920:
                    self.color_config["index"] = idx
                    self.color_config["width"] = width
                    self.color_config["height"] = height
                    self.caps["color"] = color_cap
                    print(f"✅ 彩色流已打开: {width}x{height}")
                    break
                else:
                    color_cap.release()

        if "color" not in self.caps and "depth" in self.caps:
            # 如果没有彩色流，使用深度流作为伪彩色
            print("⚠️ 未找到彩色流，将使用深度图伪彩色")

        return len(self.caps) > 0

    def start_capture(self):
        """启动所有流"""
        if self.running:
            return True

        if not self._detect_r200_streams():
            print("无法检测到R200流，使用备用方案...")
            return self._start_fallback()

        self.running = True

        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_all_streams)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_depth_stream)
        self.process_thread.daemon = True
        self.process_thread.start()

        # 初始化点云
        self.global_pointcloud = o3d.geometry.PointCloud()

        print(f"✅ R200 3D建模系统已启动，找到 {len(self.caps)} 个流")
        return True

    def _start_fallback(self):
        """备用方案：使用单个摄像头"""
        print("使用备用方案...")

        # 尝试打开第一个可用的摄像头
        for idx in range(0, 5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                print(f"找到摄像头 {idx}: {width}x{height}")

                if height == 468 or height == 480:
                    # 可能是深度或IR流
                    self.depth_config["index"] = idx
                    self.depth_config["width"] = width
                    self.depth_config["height"] = height
                    self.caps["depth"] = cap
                    self.running = True

                    # 启动线程
                    self.capture_thread = threading.Thread(target=self._capture_single_stream)
                    self.capture_thread.daemon = True
                    self.capture_thread.start()

                    self.process_thread = threading.Thread(target=self._process_single_stream)
                    self.process_thread.daemon = True
                    self.process_thread.start()

                    self.global_pointcloud = o3d.geometry.PointCloud()

                    print(f"✅ 使用单流模式: {width}x{height}")
                    return True
                else:
                    cap.release()

        print("❌ 未找到合适的摄像头")
        return False

    def _capture_all_streams(self):
        """捕获所有流"""
        while self.running:
            try:
                frames = {}

                # 读取深度流
                if "depth" in self.caps:
                    ret, frame = self.caps["depth"].read()
                    if ret:
                        # 检查是否是16位
                        if frame.dtype == np.uint16:
                            frames["depth"] = frame
                        elif frame.dtype == np.uint8 and len(frame.shape) == 2:
                            # 8位深度图，转换为16位
                            frames["depth"] = frame.astype(np.uint16) * 256
                        else:
                            # 其他格式
                            frames["depth"] = frame

                # 读取IR流
                if "ir" in self.caps:
                    ret, frame = self.caps["ir"].read()
                    if ret:
                        frames["ir"] = frame

                # 读取彩色流
                if "color" in self.caps:
                    ret, frame = self.caps["color"].read()
                    if ret:
                        frames["color"] = frame

                # 放入队列
                if "depth" in frames:
                    if self.depth_queue.full():
                        try:
                            self.depth_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.depth_queue.put(frames.get("depth"))

                if "ir" in frames:
                    if self.ir_queue.full():
                        try:
                            self.ir_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.ir_queue.put(frames.get("ir"))

                if "color" in frames:
                    if self.color_queue.full():
                        try:
                            self.color_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.color_queue.put(frames.get("color"))

                self.frame_count += 1

                # 计算FPS
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()

            except Exception as e:
                print(f"捕获流时出错: {e}")

    def _capture_single_stream(self):
        """捕获单流"""
        while self.running:
            try:
                ret, frame = self.caps["depth"].read()
                if ret:
                    if self.depth_queue.full():
                        try:
                            self.depth_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.depth_queue.put(frame)

                    self.frame_count += 1

                    elapsed = time.time() - self.start_time
                    if elapsed > 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.start_time = time.time()

            except Exception as e:
                print(f"捕获帧时出错: {e}")

    def _process_depth_stream(self):
        """处理深度流"""
        frame_index = 0

        while self.running:
            try:
                # 获取深度帧
                depth_frame = self.depth_queue.get(timeout=1)

                # 获取彩色帧（如果有）
                color_frame = None
                try:
                    color_frame = self.color_queue.get_nowait()
                except queue.Empty:
                    # 如果没有彩色帧，使用IR帧或创建伪彩色
                    try:
                        ir_frame = self.ir_queue.get_nowait()
                        if ir_frame is not None:
                            # IR转伪彩色
                            if len(ir_frame.shape) == 2:
                                color_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                    except queue.Empty:
                        pass

                if color_frame is None:
                    # 创建深度伪彩色
                    if depth_frame.dtype == np.uint16:
                        depth_8bit = (depth_frame / 256).astype(np.uint8)
                    else:
                        depth_8bit = depth_frame
                    color_frame = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                # 处理深度数据
                points_3d, colors = self._process_depth_data(depth_frame, color_frame)

                if len(points_3d) > 50:
                    # 累积点云
                    self.accumulated_points.append(points_3d)
                    self.accumulated_colors.append(colors)

                    # 定期更新全局点云
                    if frame_index % 3 == 0:
                        self._update_global_pointcloud()

                    # 创建显示点云
                    temp_pcd = o3d.geometry.PointCloud()
                    temp_pcd.points = o3d.utility.Vector3dVector(points_3d)
                    temp_pcd.colors = o3d.utility.Vector3dVector(colors)

                    # 放入队列
                    if self.pointcloud_queue.full():
                        try:
                            self.pointcloud_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.pointcloud_queue.put((depth_frame, color_frame, temp_pcd))

                frame_index += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理深度流时出错: {e}")

    def _process_single_stream(self):
        """处理单流"""
        frame_index = 0

        while self.running:
            try:
                frame = self.depth_queue.get(timeout=1)

                # 判断帧类型
                if len(frame.shape) == 2:
                    # 单通道，可能是深度或IR
                    depth_frame = frame
                    # 创建伪彩色
                    color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                else:
                    # 彩色图，从中提取深度
                    color_frame = frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    depth_frame = self._estimate_depth(gray)

                # 处理深度数据
                points_3d, colors = self._process_depth_data(depth_frame, color_frame)

                if len(points_3d) > 50:
                    self.accumulated_points.append(points_3d)
                    self.accumulated_colors.append(colors)

                    if frame_index % 5 == 0:
                        self._update_global_pointcloud()

                    temp_pcd = o3d.geometry.PointCloud()
                    temp_pcd.points = o3d.utility.Vector3dVector(points_3d)
                    temp_pcd.colors = o3d.utility.Vector3dVector(colors)

                    if self.pointcloud_queue.full():
                        try:
                            self.pointcloud_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.pointcloud_queue.put((depth_frame, color_frame, temp_pcd))

                frame_index += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理单流时出错: {e}")

    def _estimate_depth(self, gray_frame):
        """从灰度图估计深度"""
        # 使用边缘检测和距离变换
        edges = cv2.Canny(gray_frame, 50, 150)

        # 距离变换
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

        # 归一化
        depth = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)

        # 模糊
        depth = cv2.GaussianBlur(depth, (5, 5), 0)

        return depth.astype(np.float32)

    def _process_depth_data(self, depth_frame, color_frame):
        """处理深度数据生成点云"""
        # 确保深度图是单通道
        if len(depth_frame.shape) == 3:
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)

        # 转换深度单位
        if depth_frame.dtype == np.uint16:
            # 16位深度数据，毫米转米
            depth_meters = depth_frame.astype(np.float32) * self.depth_config["scale"]
        else:
            # 8位或其他，归一化到0-3米
            depth_meters = depth_frame.astype(np.float32) / 255.0 * 3.0

        # 滤波
        depth_filtered = cv2.medianBlur(depth_meters, 3)
        depth_filtered = cv2.bilateralFilter(depth_filtered, 5, 50, 50)

        # 深度范围限制
        depth_filtered = np.clip(depth_filtered, self.min_depth, self.max_depth)

        # 生成点云
        h, w = depth_filtered.shape

        # 调整彩色图大小
        color_resized = cv2.resize(color_frame, (w, h))

        # 创建坐标网格
        y, x = np.mgrid[0:h, 0:w]

        # 展平
        x_flat = x.flatten().astype(np.float32)
        y_flat = y.flatten().astype(np.float32)
        z_flat = depth_filtered.flatten()

        # 3D坐标计算
        X = (x_flat - self.cx_depth) * z_flat / self.fx_depth
        Y = (y_flat - self.cy_depth) * z_flat / self.fy_depth
        Z = z_flat

        points_3d = np.stack([X, Y, Z], axis=-1)

        # 过滤无效点
        valid_mask = (
            np.isfinite(points_3d).all(axis=1) &
            (Z > self.min_depth) &
            (Z < self.max_depth) &
            (z_flat > 0)
        )

        points_3d = points_3d[valid_mask]

        if len(points_3d) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # 获取颜色
        colors_flat = color_resized.reshape(-1, 3) / 255.0
        colors_flat = colors_flat[valid_mask]

        # BGR转RGB
        colors_rgb = colors_flat[:, [2, 1, 0]]

        return points_3d, colors_rgb

    def _update_global_pointcloud(self):
        """更新全局点云"""
        if not self.accumulated_points:
            return

        try:
            # 合并最近的点
            recent_frames = min(5, len(self.accumulated_points))
            recent_points = self.accumulated_points[-recent_frames:]
            recent_colors = self.accumulated_colors[-recent_frames:]

            all_points = np.vstack(recent_points)
            all_colors = np.vstack(recent_colors)

            if len(all_points) == 0:
                return

            # 创建新点云
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(all_points)
            new_pcd.colors = o3d.utility.Vector3dVector(all_colors)

            # 体素下采样
            new_pcd = new_pcd.voxel_down_sample(voxel_size=self.voxel_size)

            # 与全局点云合并
            if self.global_pointcloud is not None and len(self.global_pointcloud.points) > 0:
                combined_points = np.vstack([
                    np.asarray(self.global_pointcloud.points),
                    np.asarray(new_pcd.points)
                ])
                combined_colors = np.vstack([
                    np.asarray(self.global_pointcloud.colors),
                    np.asarray(new_pcd.colors)
                ])

                self.global_pointcloud.points = o3d.utility.Vector3dVector(combined_points)
                self.global_pointcloud.colors = o3d.utility.Vector3dVector(combined_colors)
            else:
                self.global_pointcloud = new_pcd

            # 全局下采样
            if len(self.global_pointcloud.points) > 50000:
                self.global_pointcloud = self.global_pointcloud.voxel_down_sample(
                    voxel_size=self.voxel_size * 2)

            # 统计
            self.total_points = len(self.global_pointcloud.points)

            # 定期重建网格
            if self.total_points > 2000 and time.time() - getattr(self, 'last_mesh_time', 0) > 5:
                self._reconstruct_mesh()
                self.last_mesh_time = time.time()

        except Exception as e:
            print(f"更新全局点云时出错: {e}")

    def _reconstruct_mesh(self):
        """重建网格"""
        if self.total_points < 1000:
            return

        try:
            print(f"重建网格，点云数量: {self.total_points}")

            pcd = copy.deepcopy(self.global_pointcloud)

            # 下采样
            if self.total_points > 10000:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size * 2)

            # 估计法线
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            # Ball Pivoting算法
            radii = [0.02, 0.04, 0.06]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))

            # 简化
            if len(mesh.triangles) > 5000:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

            # 平滑
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh.compute_vertex_normals()

            self.current_mesh = mesh
            print(f"网格重建完成: {len(mesh.vertices)}顶点, {len(mesh.triangles)}三角形")

        except Exception as e:
            print(f"重建网格时出错: {e}")

    def update_display(self):
        """更新显示"""
        try:
            depth_frame, color_frame, pointcloud = self.pointcloud_queue.get_nowait()

            # 保存用于显示
            self.last_depth = depth_frame
            self.last_color = color_frame
            self.last_pcd = pointcloud

        except queue.Empty:
            pass

    def create_display_frames(self):
        """创建显示帧"""
        frames = {}

        # 深度帧显示
        if hasattr(self, 'last_depth'):
            depth_frame = self.last_depth

            if depth_frame.dtype == np.uint16:
                # 16位深度图
                depth_8bit = (depth_frame / 256).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                # 添加信息
                info = f"16位深度图: {depth_frame.shape[1]}x{depth_frame.shape[0]}"
                cv2.putText(depth_colored, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                depth_range = f"范围: {depth_frame.min()}-{depth_frame.max()} mm"
                cv2.putText(depth_colored, depth_range, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                frames["depth"] = depth_colored
            else:
                # 8位深度图
                depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                frames["depth"] = depth_colored

        # 彩色帧显示
        if hasattr(self, 'last_color'):
            color_frame = self.last_color.copy()

            # 添加信息
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(color_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            points_text = f"点云: {self.total_points:,}"
            cv2.putText(color_frame, points_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            frames["color"] = color_frame

        return frames

    def save_model(self):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # 保存点云
        if self.global_pointcloud and len(self.global_pointcloud.points) > 0:
            pcd_file = os.path.join(save_dir, "pointcloud.ply")
            o3d.io.write_point_cloud(pcd_file, self.global_pointcloud)
            print(f"✅ 保存点云: {pcd_file}")

        # 保存网格
        if self.current_mesh:
            mesh_file = os.path.join(save_dir, "mesh.ply")
            o3d.io.write_triangle_mesh(mesh_file, self.current_mesh)
            print(f"✅ 保存网格: {mesh_file}")

        # 保存深度图
        if hasattr(self, 'last_depth'):
            if self.last_depth.dtype == np.uint16:
                depth_file = os.path.join(save_dir, "depth_16bit.npy")
                np.save(depth_file, self.last_depth)
                print(f"✅ 保存16位深度图: {depth_file}")

            depth_img = os.path.join(save_dir, "depth.png")
            if self.last_depth.dtype == np.uint16:
                depth_8bit = (self.last_depth / 256).astype(np.uint8)
            else:
                depth_8bit = self.last_depth.astype(np.uint8)
            cv2.imwrite(depth_img, depth_8bit)
            print(f"✅ 保存深度图: {depth_img}")

        # 保存彩色图
        if hasattr(self, 'last_color'):
            color_file = os.path.join(save_dir, "color.jpg")
            cv2.imwrite(color_file, self.last_color)
            print(f"✅ 保存彩色图: {color_file}")

        print(f"所有文件保存到: {save_dir}")

    def clear(self):
        """清除数据"""
        self.accumulated_points = []
        self.accumulated_colors = []
        self.global_pointcloud = o3d.geometry.PointCloud()
        self.current_mesh = None
        self.total_points = 0
        print("已清除所有数据")

    def stop(self):
        """停止"""
        self.running = False
        time.sleep(0.5)

        for name, cap in self.caps.items():
            cap.release()
            print(f"关闭 {name} 流")

        cv2.destroyAllWindows()
        print("系统已停止")

def main():
    # 检查OpenCV和Open3D
    try:
        import cv2
        import open3d as o3d
    except ImportError as e:
        print(f"❌ 缺少库: {e}")
        print("请安装: pip install opencv-python open3d numpy")
        return

    # 创建模型器
    modeler = R200DepthModeler()

    # 启动
    if not modeler.start_capture():
        print("启动失败")
        return

    print("\n" + "="*60)
    print("R200实时3D建模系统")
    print("操作说明:")
    print("  Q: 退出")
    print("  S: 保存模型")
    print("  C: 清除数据")
    print("  R: 重建网格")
    print("="*60)

    # 创建Open3D窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=modeler.window_3d, width=1024, height=768)

    # 初始点云
    init_pcd = o3d.geometry.PointCloud()
    init_pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
    vis.add_geometry(init_pcd)

    # 渲染选项
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    render_opt.point_size = 2.0

    display_mode = 0  # 0:点云, 1:网格
    last_3d_update = time.time()

    try:
        while modeler.running:
            # 更新显示
            modeler.update_display()

            # 显示2D窗口
            frames = modeler.create_display_frames()
            for name, frame in frames.items():
                if name == "depth":
                    cv2.imshow(modeler.window_depth, frame)
                elif name == "color":
                    cv2.imshow(modeler.window_color, frame)

            # 更新3D显示
            vis.poll_events()

            current_time = time.time()
            if current_time - last_3d_update > 0.3:
                last_3d_update = current_time

                if display_mode == 0 and modeler.global_pointcloud and len(modeler.global_pointcloud.points) > 0:
                    vis.clear_geometries()

                    # 显示点云
                    display_pcd = copy.deepcopy(modeler.global_pointcloud)
                    if len(display_pcd.points) > 50000:
                        display_pcd = display_pcd.voxel_down_sample(voxel_size=0.02)

                    vis.add_geometry(display_pcd)

                elif display_mode == 1 and modeler.current_mesh:
                    vis.clear_geometries()

                    # 显示网格
                    mesh_display = copy.deepcopy(modeler.current_mesh)
                    if not mesh_display.has_vertex_colors():
                        mesh_display.paint_uniform_color([0.7, 0.7, 0.7])

                    vis.add_geometry(mesh_display)

            vis.update_renderer()

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                modeler.save_model()
            elif key == ord('c'):
                modeler.clear()
                vis.clear_geometries()
                init_pcd = o3d.geometry.PointCloud()
                init_pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
                vis.add_geometry(init_pcd)
            elif key == ord('r'):
                modeler._reconstruct_mesh()
            elif key == ord('m'):
                display_mode = (display_mode + 1) % 2
                mode_name = "点云" if display_mode == 0 else "网格"
                print(f"切换显示模式: {mode_name}")

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        modeler.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()