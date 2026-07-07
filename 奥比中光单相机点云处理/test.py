# raw_pointcloud_view.py
import time
import cv2
import numpy as np
import open3d as o3d
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

class RawPointCloudViewer:
    # 这里只保留最基础的可视化参数
    MIN_DEPTH = 0       # mm
    MAX_DEPTH = 3000    # mm
    POINT_SIZE = 2.0

    def __init__(self):
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())
        self.width, self.height = 640, 480

        # 这里沿用你原来的内参
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0

        xx = np.arange(self.width)
        yy = np.arange(self.height)
        self.u, self.v = np.meshgrid(xx, yy)

        self.first_frame = True

    # ── 硬件初始化 ─────────────────────────────
    def init_device(self):
        if not self.sdk.initialize():
            print("[Error] SDK init failed")
            return False
        if not self.sdk.open_device():
            print("[Error] open_device failed")
            return False
        if not self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH):
            print("[Error] create depth stream failed")
            return False
        if not self.sdk.start_stream():
            print("[Error] start stream failed")
            return False
        return True

    def close(self):
        try:
            self.sdk.cleanup()
        except Exception:
            pass

    # ── 不做任何处理：直接使用原始深度 ───────────
    def process_raw(self, raw):
        """
        只做最基本的数据类型转换。
        不做：
        - 去噪
        - 补洞
        - EMA
        - 引导滤波
        - 去飞点
        - 统计滤波
        """
        depth = raw.astype(np.float32)

        # 仅保留有效范围，方便显示
        # 如果你想看“绝对原始”，这两行也可以删掉
        depth[(depth < self.MIN_DEPTH) | (depth > self.MAX_DEPTH)] = 0.0

        mask = (depth > 0).astype(np.uint8)
        return depth, mask

    # ── 深度转点云 ─────────────────────────────
    def to_pointcloud(self, depth_mm, mask):
        sel = (depth_mm > 0) & (mask > 0)
        if not sel.any():
            return None, None

        z = depth_mm[sel] / 1000.0
        u = self.u[sel]
        v = self.v[sel]

        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy

        pts = np.stack([x, y, z], axis=-1)

        # 颜色：近红远蓝
        t = np.clip(z / (self.MAX_DEPTH / 1000.0), 0.0, 1.0)
        col = np.zeros_like(pts)
        col[:, 0] = 1.0 - t
        col[:, 2] = t

        return pts, col

    # ── 主循环 ────────────────────────────────
    def run(self):
        if not self.init_device():
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window("Raw PointCloud Viewer", width=1280, height=800)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(pcd)

        ro = vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.08])
        ro.point_size = self.POINT_SIZE
        ro.show_coordinate_frame = True

        print("运行中 | OpenCV窗口聚焦后：'q'退出  's'保存点云")

        try:
            while True:
                res = self.sdk.capture_depth_frame()
                if res is None:
                    continue

                raw, _ = res

                # 不处理，直接显示原始深度生成的点云
                depth, mask = self.process_raw(raw)
                pts, col = self.to_pointcloud(depth, mask)

                if pts is None:
                    pts = np.zeros((1, 3))
                    col = np.zeros((1, 3))

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(col)
                vis.update_geometry(pcd)

                if self.first_frame and len(pts) > 1:
                    vis.reset_view_point(True)
                    self.first_frame = False

                if not vis.poll_events():
                    break
                vis.update_renderer()

                # 深度预览
                disp = np.clip(depth / self.MAX_DEPTH * 255, 0, 255).astype(np.uint8)
                cv2.imshow(
                    "Raw Depth Preview",
                    cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET)
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("退出")
                    break
                elif key == ord('s'):
                    fname = "raw_pointcloud_{}.ply".format(int(time.time()))
                    o3d.io.write_point_cloud(fname, pcd)
                    print("[已保存] {}".format(fname))

        finally:
            self.close()
            vis.destroy_window()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = RawPointCloudViewer()
    viewer.run()