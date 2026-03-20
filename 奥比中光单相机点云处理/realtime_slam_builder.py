"""
实时 SLAM 系统 (Real-time SLAM Builder)
功能：融合数据流采集与 ICP 全局配准
"""

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import sys
import os
import copy

try:
    from orbbec_sdk import OrbbecCameraSDK
except ImportError:
    print("❌ 找不到 orbbec_sdk.py，请确保它在当前文件夹下。")
    sys.exit(1)


class RealTimeSLAMGPU:
    def __init__(self):
        print("=" * 60)
        # 1. GPU 硬件接管检测
        if o3c.cuda.is_available():
            self.device = o3c.Device("CUDA:0")
            print("✅ 成功接管 NVIDIA GPU")
        else:
            self.device = o3c.Device("CPU:0")
            print("⚠️ 未检测到 CUDA，将使用 CPU 模拟 Tensor API。")
        print("=" * 60)

        # 2. 硬件初始化
        self.sdk_path = r"C:\毕设\奥比中光Win64-Release\奥比中光Win64-Release\sdk\libs"
        if os.path.exists(self.sdk_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(self.sdk_path)

        self.sdk = OrbbecCameraSDK(self.sdk_path)
        if not self.sdk.initialize() or not self.sdk.open_device(0):
            raise RuntimeError("❌ 深度相机初始化失败！")
        self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH)
        self.sdk.start_stream()

        self.color_cam = cv2.VideoCapture(0)
        self.color_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.color_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 3. Tensor 专属内参矩阵 (直接存入 GPU 显存)
        self.intrinsic_tensor = o3c.Tensor(
            [[525.0, 0, 319.5],
             [0, 525.0, 239.5],
             [0, 0, 1]], dtype=o3c.float64, device=self.device)

        self.icp_voxel_size = 0.02
        self.global_voxel_size = 0.01

        # 4. 混合架构状态变量：GPU 算位移，CPU 存地图
        self.global_pcd_legacy = o3d.geometry.PointCloud() # 存在普通内存，给可视化器用
        self.global_transformation = o3c.Tensor(np.eye(4), dtype=o3c.float64, device=self.device)

        self.prev_sparse_t = None
        self.success_count = 0

        # 5. 可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Real-time SLAM (CUDA GPU Engine)", width=1024, height=768)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.15])
        opt.point_size = 2.5
        self.is_first_frame = True

    def grab_and_preprocess_gpu(self):
        """直接将数据送入 GPU 完成预处理"""
        depth_res = self.sdk.capture_depth_frame(timeout=1000)
        ret, color_img = self.color_cam.read()

        if not depth_res or not ret:
            return None, None, None

        depth_data = depth_res[0]
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        depth_t = o3c.Tensor(depth_data, dtype=o3c.uint16, device=self.device)
        color_t = o3c.Tensor(color_rgb, dtype=o3c.uint8, device=self.device)

        depth_im_t = o3d.t.geometry.Image(depth_t)
        color_im_t = o3d.t.geometry.Image(color_t)
        rgbd_im_t = o3d.t.geometry.RGBDImage(color_im_t, depth_im_t)

        pcd_t = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_im_t, self.intrinsic_tensor, depth_scale=1000.0, depth_max=5.0)

        # 废帧拦截
        if pcd_t.point.positions.shape[0] < 500:
            return None, None, None

        # 修复 Astra 相机镜像 (Tensor 矩阵乘法，瞬间完成)
        positions = pcd_t.point.positions
        positions[:, 0] = positions[:, 0] * -1.0
        pcd_t.point.positions = positions

        # GPU 极速降采样与法线计算
        pcd_sparse_t = pcd_t.voxel_down_sample(self.icp_voxel_size)
        pcd_sparse_t.estimate_normals(max_nn=30, radius=self.icp_voxel_size * 2)

        return pcd_t, pcd_sparse_t, color_img

    def run(self):
        print("👉 请拿起相机移动。")

        try:
            while True:
                data = self.grab_and_preprocess_gpu()
                if data[0] is None:
                    continue

                curr_dense_t, curr_sparse_t, color_preview = data

                cv2.imshow("GPU Tracking View", color_preview)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break

                # --- 初始化第一帧 ---
                if self.is_first_frame:
                    self.global_pcd_legacy += curr_dense_t.to_legacy()
                    self.vis.add_geometry(self.global_pcd_legacy)

                    view_ctrl = self.vis.get_view_control()

                    view_ctrl.set_up([0.0, -1.0, 0.0])

                    view_ctrl.set_front([0.0, 0.0, -1.0])

                    view_ctrl.set_lookat([0.0, 0.0, 1.0])

                    view_ctrl.set_zoom(0.8)
                    # ==========================================

                    self.prev_sparse_t = curr_sparse_t.clone()
                    self.is_first_frame = False
                    continue

                # --- GPU Tensor ICP 配准 ---
                estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
                criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)

                icp_result = o3d.t.pipelines.registration.icp(
                    source=curr_sparse_t,
                    target=self.prev_sparse_t,
                    max_correspondence_distance=0.08,
                    init_source_to_target=o3c.Tensor(np.eye(4), dtype=o3c.float64, device=self.device),
                    estimation_method=estimation,
                    criteria=criteria
                )

                # --- 状态更新与渲染 ---
                if icp_result.fitness > 0.30:
                    # 矩阵乘法全在 GPU 里算
                    self.global_transformation = self.global_transformation.matmul(icp_result.transformation)

                    # 在 GPU 里把当前帧移动到全景坐标系中
                    curr_dense_t.transform(self.global_transformation)

                    curr_dense_t.point.colors = o3c.Tensor(np.ones((curr_dense_t.point.positions.shape[0], 3)) * 0.7, dtype=o3c.float32, device=self.device)

                    # 🔥 数据回传：把算好的这一块点云“拉回” CPU 内存，组装进全局地图
                    self.global_pcd_legacy += curr_dense_t.to_legacy()
                    self.success_count += 1

                    self.success_count += 1

                    # 🌟 修复渲染引擎指针丢失的问题
                    if self.success_count % 4 == 0:
                        # 1. 生成降采样后的新点云
                        new_pcd = self.global_pcd_legacy.voxel_down_sample(voxel_size=self.global_voxel_size)

                        # 2. 把旧对象从屏幕上强行卸载
                        self.vis.remove_geometry(self.global_pcd_legacy, reset_bounding_box=False)

                        # 3. 变量指向新对象，并重新挂载到屏幕上
                        self.global_pcd_legacy = new_pcd
                        self.vis.add_geometry(self.global_pcd_legacy, reset_bounding_box=False)

                    self.vis.poll_events()
                    self.vis.update_renderer()

                    # 更新 GPU 里的“上一帧”
                    self.prev_sparse_t = curr_sparse_t.clone()

        except KeyboardInterrupt:
            pass
        finally:
            print("\n🛑 停止采集，正在输出最终的全局点云...")
            self.sdk.cleanup()
            self.color_cam.release()
            cv2.destroyAllWindows()
            self.vis.destroy_window()

            if len(self.global_pcd_legacy.points) > 0:
                print("🧽 正在执行最终 CPU 降噪与去色...")
                self.global_pcd_legacy, _ = self.global_pcd_legacy.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
                self.global_pcd_legacy.colors = o3d.utility.Vector3dVector()

                output_path = r"C:\毕设\Test1\realtime_slam_result.ply"
                o3d.io.write_point_cloud(output_path, self.global_pcd_legacy)
                print(f"🎉 建图成果已保存至: {output_path}")

if __name__ == "__main__":
    slam = RealTimeSLAMGPU()
    slam.run()