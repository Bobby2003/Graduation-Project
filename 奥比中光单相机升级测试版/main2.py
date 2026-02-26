import cv2
import numpy as np
import open3d as o3d
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
import copy

class Depth3DVisualizer:
    def __init__(self):
        self.sdk_path = get_default_sdk_path()
        self.camera_sdk = OrbbecCameraSDK(self.sdk_path)
        self.width, self.height = 640, 480
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0
        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        self.u, self.v = np.meshgrid(x, y)

    def init_hardware(self):
        print("初始化相机硬件...")
        if not self.camera_sdk.initialize():
            print("相机SDK初始化失败")
            return False
        if not self.camera_sdk.open_device() or not self.camera_sdk.create_stream(self.camera_sdk.ONI_SENSOR_DEPTH):
            print("打开设备或创建深度流失败")
            return False
        self.camera_sdk.start_stream()
        print("设备打开，深度流启动成功")
        return True

    def run(self):
        if not self.init_hardware():
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LazyMan 3D Scanner", width=1024, height=768)

        global_pcd = o3d.geometry.PointCloud()
        vis.add_geometry(global_pcd)  # 先加个空点云占位，避免Open3D警告

        previous_pcd = None
        camera_pose_global = np.eye(4)

        try:
            while True:
                depth_res = self.camera_sdk.capture_depth_frame()
                if depth_res is None:
                    print("未获取到深度帧")
                    continue
                depth_raw, _ = depth_res

                mask = (depth_raw > 100) & (depth_raw < 8000)
                if not np.any(mask):
                    print("无有效的深度像素，跳过当前帧")
                    continue

                z = depth_raw[mask].astype(np.float32)
                z_m = z / 1000.0
                u_v = self.u[mask]
                v_v = self.v[mask]

                x_3d = (u_v - self.cx) * z_m / self.fx
                y_3d = -(v_v - self.cy) * z_m / self.fy
                z_3d = z_m
                points = np.stack((x_3d, y_3d, z_3d), axis=-1)

                if points.shape[0] == 0:
                    print("当前帧点云为空，跳过")
                    continue

                current_pcd = o3d.geometry.PointCloud()
                current_pcd.points = o3d.utility.Vector3dVector(points)

                print(f"当前帧获得点云数量: {len(current_pcd.points)}")

                if previous_pcd is not None and len(previous_pcd.points) > 0 and len(current_pcd.points) > 0:
                    threshold = 0.05
                    trans_init = np.eye(4)
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        current_pcd, previous_pcd, threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())

                    T_delta = reg_p2p.transformation
                    camera_pose_global = np.dot(camera_pose_global, np.linalg.inv(T_delta))
                else:
                    camera_pose_global = np.eye(4)

                if len(current_pcd.points) == 0:
                    # 保护：空点云不加入
                    print("空点云，停止合并")
                else:
                    current_pcd_global = copy.deepcopy(current_pcd).transform(camera_pose_global)
                    # 使用Combine而非+=，避免赋值弊端
                    global_pcd = global_pcd + current_pcd_global
                    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.005)

                previous_pcd = copy.deepcopy(current_pcd)

                # 只在global_pcd非空时更新视图
                if len(global_pcd.points) > 0:
                    vis.update_geometry(global_pcd)

                vis.poll_events()
                vis.update_renderer()

                depth_show = np.clip(depth_raw / 3000 * 255, 0, 255).astype(np.uint8)
                cv2.imshow("Depth Map", cv2.applyColorMap(depth_show, cv2.COLORMAP_JET))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera_sdk.cleanup()
            vis.destroy_window()
            cv2.destroyAllWindows()

        if len(global_pcd.points) > 0:
            o3d.io.write_point_cloud("full_scene.ply", global_pcd)
            print("扫描完成，点云模型已保存为 full_scene.ply")
        else:
            print("没有有效点云数据，未保存点云文件。")

if __name__ == "__main__":
    visualizer = Depth3DVisualizer()
    visualizer.run()