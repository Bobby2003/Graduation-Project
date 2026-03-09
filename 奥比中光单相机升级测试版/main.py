import cv2
import numpy as np
import open3d as o3d
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path


class Depth3DVisualizer:
    def __init__(self):
        self.sdk_path = get_default_sdk_path()
        self.camera_sdk = OrbbecCameraSDK(self.sdk_path)

        self.width, self.height = 640, 480
        # 预设内参 (根据典型 Astra 传感器)
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0

        # 预计算网格
        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        self.u, self.v = np.meshgrid(x, y)
        self.first_frame = True  # 用于自动定位物体

    def init_hardware(self):
        if not self.camera_sdk.initialize(): return False
        if not self.camera_sdk.open_device() or not self.camera_sdk.create_stream(self.camera_sdk.ONI_SENSOR_DEPTH):
            return False
        self.camera_sdk.start_stream()
        return True

    def run(self):
        if not self.init_hardware(): return

        # 1. 初始化可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LazyMan 3D Scanner", width=1024, height=768)

        pcd = o3d.geometry.PointCloud()
        # 初始点防止报错
        pcd.points = o3d.utility.Vector3dVector(np.random.uniform(0, 0.1, (10, 3)))
        vis.add_geometry(pcd)

        # 2. 渲染选项设置
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰背景
        opt.point_size = 2.0  # 调大点的大小，看得更清楚
        opt.show_coordinate_frame = True  # 显示坐标轴

        print("正在捕获深度并生成 3D 模型...")

        try:
            while True:
                depth_res = self.camera_sdk.capture_depth_frame()
                if depth_res is None: continue
                depth_raw, _ = depth_res

                # --- 核心逻辑：深度转 3D ---
                # 过滤无效像素 (0) 和 离太远的背景 (3000+)
                mask = (depth_raw > 100) & (depth_raw < 3000)
                z = depth_raw[mask].astype(np.float32)

                # 单位转换：将毫米(mm)转换为米(m)
                # 这是解决“全黑”最关键的一步
                z_m = z / 1000.0

                u_v = self.u[mask]
                v_v = self.v[mask]

                # 公式计算 (注意：Y 轴取负号，否则上下颠倒)
                x_3d = (u_v - self.cx) * z_m / self.fx
                y_3d = -(v_v - self.cy) * z_m / self.fy
                z_3d = z_m

                points = np.stack((x_3d, y_3d, z_3d), axis=-1)

                # --- 更新点云 ---
                pcd.points = o3d.utility.Vector3dVector(points)

                # 根据深度自动着色（从蓝到红）
                colors = np.zeros_like(points)
                colors[:, 0] = np.clip(z_m / 2.0, 0, 1)  # R
                colors[:, 2] = 1 - colors[:, 0]  # B
                pcd.colors = o3d.utility.Vector3dVector(colors)

                vis.update_geometry(pcd)

                # --- 首次捕获时将视角对准物体 ---
                if self.first_frame and len(points) > 0:
                    vis.reset_view_point(True)
                    self.first_frame = False

                if not vis.poll_events(): break
                vis.update_renderer()

                # OpenCV 预览 (用于调试)
                depth_show = np.clip(depth_raw / 3000 * 255, 0, 255).astype(np.uint8)
                cv2.imshow("Debug Depth Map", cv2.applyColorMap(depth_show, cv2.COLORMAP_JET))
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        finally:
            self.camera_sdk.cleanup()
            vis.destroy_window()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    scanner = Depth3DVisualizer()
    scanner.run()