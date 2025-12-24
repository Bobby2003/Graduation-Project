"""
奥比中光MSC-1 实时3D建模系统 (基于已验证稳定版)
架构：
1. 主线程：稳定运行你提供的深度流采集与2D显示 (生产者)
2. 点云线程：处理深度帧，生成3D点云 (消费者)
3. 可视化线程：独立运行Open3D窗口，显示点云
"""

import cv2
import numpy as np
import open3d as o3d
import threading
import queue
import time
import copy

# 导入已验证稳定的primesense库
from primesense import openni2
from primesense import _openni2 as c_api


class DepthStreamManager:
    """管理深度流的稳定采集 (你已验证的代码)"""

    def __init__(self):
        self.device = None
        self.depth_stream = None
        self.running = False
        self.depth_queue = queue.Queue(maxsize=10)  # 深度图队列
        self.frame_count = 0

        # 相机内参 (Astra系列典型值，可根据需要调整)
        self.fx = 475.0  # 焦距x
        self.fy = 475.0  # 焦距y
        self.cx = 320  # 主点x (640/2)
        self.cy = 240  # 主点y (480/2)
        self.depth_scale = 0.001  # 毫米转米

    def initialize(self):
        """初始化OpenNI2和深度流"""
        try:
            openni2.initialize()
            if not openni2.is_initialized():
                print("❌ OpenNI2 未初始化")
                return False
            print("✅ OpenNI2 初始化成功")

            self.device = openni2.Device.open_any()
            dev_info = self.device.get_device_info()
            print(f"✅ 设备已连接: {dev_info.name.decode('utf-8')}")

            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False

    def start_stream(self):
        """启动深度流采集线程"""
        try:
            self.depth_stream = self.device.create_depth_stream()
            self.depth_stream.set_video_mode(c_api.OniVideoMode(
                pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640, resolutionY=480, fps=30
            ))
            self.depth_stream.start()
            print("✅ 深度流已启动")

            self.running = True
            # 启动采集线程
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            return True
        except Exception as e:
            print(f"❌ 启动深度流失败: {e}")
            return False

    def _capture_loop(self):
        """深度流采集循环 (生产者)"""
        while self.running:
            try:
                frame = self.depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape(frame.height, frame.width)

                # 放入队列供其他线程使用
                if not self.depth_queue.full():
                    self.depth_queue.put(depth_array)

                self.frame_count += 1

            except Exception as e:
                if self.running:  # 只在运行状态下打印错误
                    print(f"[采集线程] 读取帧失败: {e}")
                break

    def get_depth_frame(self, timeout=1.0):
        """从队列获取深度帧 (消费者调用)"""
        try:
            return self.depth_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """停止采集"""
        self.running = False
        time.sleep(0.1)  # 给线程一点时间结束

        if self.depth_stream:
            self.depth_stream.stop()
        if self.device:
            self.device.close()

        openni2.unload()
        print("✅ 深度流已停止")


class PointCloudProcessor:
    """处理深度图，生成3D点云"""

    def __init__(self, fx, fy, cx, cy, depth_scale=0.001):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

        # 处理参数
        self.min_depth = 0.3  # 最小深度 (米)
        self.max_depth = 3.0  # 最大深度 (米)

        # 点云缓存
        self.global_pointcloud = o3d.geometry.PointCloud()
        self.accumulated_points = []
        self.accumulated_colors = []

    def depth_to_pointcloud(self, depth_frame, color_frame=None):
        """
        将深度图转换为3D点云
        depth_frame: 16位深度图 (单位: 毫米)
        color_frame: 彩色图 (可选，用于点云着色)
        返回: (points_3d, colors) 或 (points_3d, None)
        """
        if depth_frame is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # 1. 预处理深度图
        h, w = depth_frame.shape

        # 毫米转米，并应用深度范围过滤
        depth_meters = depth_frame.astype(np.float32) * self.depth_scale
        depth_meters = np.clip(depth_meters, self.min_depth, self.max_depth)

        # 简单的有效深度掩码 (去除0值)
        valid_mask = (depth_meters > self.min_depth) & (depth_meters < self.max_depth)

        # 2. 生成像素坐标网格
        y, x = np.mgrid[0:h, 0:w]
        x_valid = x[valid_mask].astype(np.float32)
        y_valid = y[valid_mask].astype(np.float32)
        z_valid = depth_meters[valid_mask]

        if len(z_valid) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # 3. 计算3D坐标 (针孔相机模型)
        X = (x_valid - self.cx) * z_valid / self.fx
        Y = (y_valid - self.cy) * z_valid / self.fy
        Z = z_valid

        points_3d = np.stack([X, Y, Z], axis=-1)

        # 4. 处理颜色
        colors = None
        if color_frame is not None:
            # 调整颜色图尺寸以匹配深度图
            color_resized = cv2.resize(color_frame, (w, h))
            # 转换为RGB并归一化
            colors_bgr = color_resized.reshape(-1, 3)[valid_mask.flatten()]
            colors_rgb = colors_bgr[:, [2, 1, 0]] / 255.0  # BGR转RGB
            colors = colors_rgb
        else:
            # 使用深度值生成伪彩色 (蓝到红)
            depth_normalized = (z_valid - self.min_depth) / (self.max_depth - self.min_depth)
            colors = np.zeros((len(depth_normalized), 3))
            colors[:, 0] = depth_normalized  # R
            colors[:, 2] = 1.0 - depth_normalized  # B

        return points_3d, colors

    def accumulate_pointcloud(self, points_3d, colors, max_frames=10):
        """累积多帧点云"""
        if len(points_3d) > 0:
            self.accumulated_points.append(points_3d)
            self.accumulated_colors.append(colors)

            # 限制累积的帧数
            if len(self.accumulated_points) > max_frames:
                self.accumulated_points.pop(0)
                self.accumulated_colors.pop(0)

            # 合并累积的点云
            if self.accumulated_points:
                all_points = np.vstack(self.accumulated_points)
                all_colors = np.vstack(self.accumulated_colors)

                # 创建点云对象
                self.global_pointcloud.clear()
                self.global_pointcloud.points = o3d.utility.Vector3dVector(all_points)
                self.global_pointcloud.colors = o3d.utility.Vector3dVector(all_colors)

                # 体素下采样 (控制点云密度)
                if len(all_points) > 5000:
                    self.global_pointcloud = self.global_pointcloud.voxel_down_sample(voxel_size=0.01)

    def get_pointcloud(self):
        """获取当前累积的点云"""
        return copy.deepcopy(self.global_pointcloud)

    def clear(self):
        """清除累积的点云"""
        self.accumulated_points = []
        self.accumulated_colors = []
        self.global_pointcloud.clear()


class Open3DVisualizer:
    """独立管理Open3D可视化窗口"""

    def __init__(self, window_name="3D点云"):
        self.window_name = window_name
        self.vis = None
        self.pointcloud = None
        self.running = False

    def start(self):
        """启动Open3D可视化窗口"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=1024, height=768)

        # 设置渲染选项
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        render_opt.point_size = 2.0
        render_opt.light_on = True

        # 添加初始坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)

        self.running = True
        print("✅ Open3D可视化窗口已启动")

    def update_pointcloud(self, pointcloud):
        """更新显示的点云"""
        if not self.running or pointcloud is None or len(pointcloud.points) == 0:
            return

        try:
            # 清除之前的点云几何体
            self.vis.clear_geometries()

            # 重新添加坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            self.vis.add_geometry(coordinate_frame)

            # 添加新点云
            self.vis.add_geometry(pointcloud)

        except Exception as e:
            print(f"[Open3D] 更新点云失败: {e}")

    def render(self):
        """渲染一帧"""
        if self.running:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                return True
            except Exception as e:
                print(f"[Open3D] 渲染失败: {e}")
                return False
        return False

    def stop(self):
        """停止可视化"""
        if self.vis:
            self.vis.destroy_window()
            self.running = False
            print("✅ Open3D可视化窗口已关闭")


def main():
    """主函数"""
    print("=" * 60)
    print("奥比中光MSC-1 实时3D建模系统")
    print("=" * 60)

    # 1. 初始化深度流管理器
    depth_manager = DepthStreamManager()
    if not depth_manager.initialize():
        print("❌ 深度流初始化失败")
        return

    # 2. 启动深度流
    if not depth_manager.start_stream():
        print("❌ 深度流启动失败")
        return

    # 3. 初始化点云处理器
    pointcloud_processor = PointCloudProcessor(
        fx=depth_manager.fx,
        fy=depth_manager.fy,
        cx=depth_manager.cx,
        cy=depth_manager.cy,
        depth_scale=depth_manager.depth_scale
    )

    # 4. 启动Open3D可视化器
    visualizer = Open3DVisualizer("实时3D点云")
    visualizer.start()

    # 5. 创建2D显示窗口
    cv2.namedWindow("深度图", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("深度图", 640, 480)

    print("\n" + "=" * 60)
    print("操作说明:")
    print("  Q: 退出程序")
    print("  C: 清除累积点云")
    print("  S: 保存当前点云")
    print("=" * 60)
    print("系统运行中...\n")

    # 主循环
    last_fps_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    try:
        while depth_manager.running:
            start_time = time.time()

            # A. 获取深度帧
            depth_frame = depth_manager.get_depth_frame(timeout=0.1)
            if depth_frame is None:
                continue

            # B. 为深度图创建伪彩色显示
            depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # 在深度图上叠加信息
            info_text = f"FPS: {current_fps:.1f}"
            cv2.putText(depth_colored, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            points_text = f"点云数量: {len(pointcloud_processor.global_pointcloud.points)}"
            cv2.putText(depth_colored, points_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示深度图
            cv2.imshow("深度图", depth_colored)

            # C. 处理深度帧，生成点云
            points_3d, colors = pointcloud_processor.depth_to_pointcloud(
                depth_frame, depth_colored)

            # 累积点云
            pointcloud_processor.accumulate_pointcloud(points_3d, colors, max_frames=5)

            # D. 获取当前累积的点云并更新3D显示
            current_pcd = pointcloud_processor.get_pointcloud()
            if len(current_pcd.points) > 0:
                visualizer.update_pointcloud(current_pcd)

            # E. 更新Open3D渲染
            visualizer.render()

            # F. 处理键盘输入
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' 或 ESC
                print("\n收到退出指令...")
                break
            elif key == ord('c'):
                print("清除累积点云")
                pointcloud_processor.clear()
            elif key == ord('s'):
                # 保存点云
                if len(pointcloud_processor.global_pointcloud.points) > 0:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"pointcloud_{timestamp}.ply"
                    o3d.io.write_point_cloud(filename, pointcloud_processor.global_pointcloud)
                    print(f"✅ 点云已保存: {filename}")

            # 计算FPS
            fps_frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_frame_count / (current_time - last_fps_time)
                fps_frame_count = 0
                last_fps_time = current_time

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ 主循环出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在停止系统...")
        visualizer.stop()
        cv2.destroyAllWindows()
        depth_manager.stop()
        print("✅ 系统已安全停止")


if __name__ == "__main__":
    # 检查Open3D是否可用
    try:
        import open3d as o3d
    except ImportError:
        print("❌ 未找到Open3D库，请安装: pip install open3d")
        exit(1)

    # 运行主程序
    main()