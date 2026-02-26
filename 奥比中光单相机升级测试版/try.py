import numpy as np
import open3d as o3d
import math
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path


class LazyManSuperTracker:
    def __init__(self):
        self.sdk_path = get_default_sdk_path()
        self.camera_sdk = OrbbecCameraSDK(self.sdk_path)

        # 相机内参
        self.width, self.height = 640, 480
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.u, self.v = x, y

        # 核心位姿矩阵
        self.camera_pose = np.eye(4)
        self.prev_pcd = None
        self.is_first_frame = True

    def rotation_matrix_to_euler(self, R):
        """旋转矩阵转角度 (单位：度)"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy > 1e-6:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.rad2deg([x, y, z])

    def prepare_pcd(self, depth_raw):
        """预处理：生成点云 + 下采样 + 法向量估计"""
        mask = (depth_raw > 400) & (depth_raw < 1500)
        z = depth_raw[mask].astype(np.float32) / 1000.0
        if len(z) < 2000: return None

        u_v, v_v = self.u[mask], self.v[mask]
        x = (u_v - self.cx) * z / self.fx
        y = -(v_v - self.cy) * z / self.fy
        points = np.stack((x, y, z), axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 1. 下采样（保持计算速度）
        pcd = pcd.voxel_down_sample(0.02)

        # 2. 升级核心：计算法向量 (Point-to-Plane 必须步骤)
        # 搜索半径 0.1m 范围内的邻居点来确定平面的朝向
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return pcd

    def run(self):
        if not self.camera_sdk.initialize() or not self.camera_sdk.open_device(): return
        self.camera_sdk.create_stream(3)  # 开启深度流
        self.camera_sdk.start_stream()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LazyMan Point-to-Plane Tracker", width=1280, height=720)

        live_pcd = o3d.geometry.PointCloud()
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(live_pcd)
        vis.add_geometry(camera_frame)

        print(">>> 升级版算法已启动：Point-to-Plane 模式 (无陀螺仪补偿)")

        try:
            while True:
                depth_res = self.camera_sdk.capture_depth_frame()
                if depth_res is None: continue

                curr_pcd = self.prepare_pcd(depth_res[0])
                if curr_pcd is None: continue

                if self.is_first_frame:
                    self.prev_pcd = curr_pcd
                    self.is_first_frame = False
                else:
                    # --- 升级版配准逻辑 ---
                    # 使用 Point-to-Plane 估计器，这比之前的 Point-to-Point 稳得多
                    reg = o3d.pipelines.registration.registration_icp(
                        curr_pcd, self.prev_pcd, 0.05, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    )

                    # 更新总位姿
                    delta_t = reg.transformation
                    self.camera_pose = self.camera_pose @ delta_t

                    # 更新可视化
                    camera_frame.transform(delta_t)
                    live_pcd.points = curr_pcd.points
                    live_pcd.transform(self.camera_pose)
                    live_pcd.paint_uniform_color([1, 0.7, 0])  # 亮橙色点云

                    self.prev_pcd = curr_pcd

                    # 信息输出
                    t = self.camera_pose[:3, 3]
                    angles = self.rotation_matrix_to_euler(self.camera_pose[:3, :3])
                    print(
                        f"\r[位置] X:{t[0]:.2f} Y:{t[1]:.2f} Z:{t[2]:.2f} | [角度] P:{angles[0]:.1f} Y:{angles[1]:.1f} R:{angles[2]:.1f}",
                        end="")

                vis.update_geometry(live_pcd)
                vis.update_geometry(camera_frame)
                if not vis.poll_events(): break
                vis.update_renderer()

        finally:
            self.camera_sdk.cleanup()
            vis.destroy_window()


if __name__ == "__main__":
    LazyManSuperTracker().run()