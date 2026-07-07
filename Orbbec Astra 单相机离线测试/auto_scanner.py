"""
Astra RGB-D 自动扫描与 3D 重建流水线
核心功能：
1. Data Recorder: 调用 Orbbec Astra 深度相机，实时录制对齐的 RGB 与 16-bit 深度图。
2. ICP Stitcher: 录制结束后，自动跳帧读取数据，利用视觉里程计与点到面 ICP 算法极速生成全局全景点云。
"""

import cv2
import numpy as np
import open3d as o3d
import os
import glob
import copy
import time
import sys
from datetime import datetime

# ==========================================
# 🛠️ 核心一：数据录制模块 (Data Recorder)
# ==========================================
class AstraDataRecorder:
    def __init__(self, sdk_path):
        self.sdk_path = sdk_path
        self.dataset_dir = ""
        self.depth_dir = ""
        self.color_dir = ""

    def _init_folders(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = f"astra_dataset_{timestamp}"
        self.depth_dir = os.path.join(self.dataset_dir, "depth")
        self.color_dir = os.path.join(self.dataset_dir, "color")
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        print(f"\n📁 创建临时数据集目录: {os.path.abspath(self.dataset_dir)}")

    def run_recording(self):
        # 尝试导入底层 SDK
        try:
            from orbbec_sdk import OrbbecCameraSDK
        except ImportError:
            print("❌ 找不到 orbbec_sdk.py，请确保它在当前文件夹下。")
            return None

        # 破解 Windows DLL 加载限制
        if os.path.exists(self.sdk_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(self.sdk_path)

        sdk = OrbbecCameraSDK(self.sdk_path)
        if not sdk.initialize() or not sdk.open_device(0):
            print("❌ 深度相机初始化失败！请检查 USB 连接。")
            return None

        sdk.create_stream(sdk.ONI_SENSOR_DEPTH)
        sdk.start_stream()

        color_cam = cv2.VideoCapture(0)
        color_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        color_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._init_folders()
        frame_count = 0

        print("\n" + "=" * 50)
        print("📷 [录制模式] 已启动！")
        print("👉 请平滑、缓慢地移动相机扫描房间。")
        print("👉 扫描完成后，按下 'Q' 键结束录制，系统将自动开始 3D 建图。")
        print("=" * 50)

        try:
            while True:
                depth_res = sdk.capture_depth_frame(timeout=1000)
                ret, color_frame = color_cam.read()

                if depth_res and ret:
                    depth_data = depth_res[0]
                    timestamp_str = f"{time.time():.4f}.png"

                    # 极速落盘保存数据
                    cv2.imwrite(os.path.join(self.depth_dir, timestamp_str), depth_data.astype(np.uint16))
                    cv2.imwrite(os.path.join(self.color_dir, timestamp_str), color_frame)

                    frame_count += 1

                    # 实时 2D 预览
                    depth_preview = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow("Astra Color Stream (Press 'Q' to Stitch)", color_frame)
                    cv2.imshow("Astra Depth Stream", cv2.applyColorMap(depth_preview, cv2.COLORMAP_JET))

                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    print(f"\n🛑 录制结束。共采集 {frame_count} 帧原始数据。")
                    break
        finally:
            sdk.cleanup()
            color_cam.release()
            cv2.destroyAllWindows()

        return self.dataset_dir if frame_count > 0 else None


# ==========================================
# 🛠️ 核心二：离线拼接模块 (ICP Stitcher)
# ==========================================
class OfflinePointCloudStitcher:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.color_files = sorted(glob.glob(os.path.join(dataset_dir, "color", "*.png")))
        self.depth_files = sorted(glob.glob(os.path.join(dataset_dir, "depth", "*.png")))

        # Astra Pro A 相机内参
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 319.5, 239.5)
        self.icp_voxel_size = 0.02
        self.global_voxel_size = 0.01

    def _process_single_frame(self, frame_idx):
        """读取单帧，去噪，并计算法线"""
        color = o3d.io.read_image(self.color_files[frame_idx])
        depth = o3d.io.read_image(self.depth_files[frame_idx])

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

        # 废帧拦截
        if len(pcd.points) < 500: return None, None, None

        # 修复 Astra 相机的物理镜像问题
        points = np.asarray(pcd.points)
        points[:, 0] = -points[:, 0]
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd_clean.points) < 100: return None, None, None

        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd_clean.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        pcd_sparse = pcd_clean.voxel_down_sample(self.icp_voxel_size)
        return rgbd, pcd_clean, pcd_sparse

    def run_stitching(self):
        print("\n" + "=" * 50)
        print("🚀 [建图模式] 开始离线全局 ICP 拼接...")
        print("=" * 50)

        start_time = time.time()
        global_pcd = o3d.geometry.PointCloud()

        print("📥 正在读取第 1 帧作为原点...")
        frame0_data = self._process_single_frame(0)
        if not frame0_data[0]:
            print("❌ 第 1 帧数据无效，拼接终止。请确保录制开头没有遮挡镜头。")
            return None

        prev_rgbd, prev_dense, prev_sparse = frame0_data
        global_pcd += prev_dense
        global_transformation = np.eye(4)
        success_count = 1
        odo_option = o3d.pipelines.odometry.OdometryOption()

        total_frames = len(self.color_files)
        # ==========================================
        # 💡 核心提速优化：加入 step=5 的抽帧机制
        # 抛弃 80% 的冗余计算，让 CPU 算力集中在有效的位移上
        # ==========================================
        for i in range(1, total_frames, 5):
            print(f"⏳ 正在处理进度: {i}/{total_frames} 帧...", end="\r")

            data = self._process_single_frame(i)
            if not data[0]: continue
            curr_rgbd, curr_dense, curr_sparse = data

            # 1. 视觉里程计粗配准
            odo_success, odo_trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                curr_rgbd, prev_rgbd, self.intrinsic, np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), odo_option)

            # 2. Point-to-Plane ICP 精配准
            icp_result = o3d.pipelines.registration.registration_icp(
                source=curr_sparse, target=prev_sparse, max_correspondence_distance=0.05,
                init=odo_trans if odo_success else np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )

            # 3. 防重影与空间融合
            trans_matrix = icp_result.transformation
            translation_dist = np.linalg.norm(trans_matrix[:3, 3])

            if icp_result.fitness > 0.30 and translation_dist > 0.005:
                global_transformation = np.dot(global_transformation, trans_matrix)
                curr_dense.transform(global_transformation)
                global_pcd += curr_dense
                success_count += 1

                # 每成功拼合 5 个关键帧，进行一次全局压平
                if success_count % 5 == 0:
                    global_pcd = global_pcd.voxel_down_sample(voxel_size=self.global_voxel_size)

                # 更新状态
                prev_rgbd = curr_rgbd
                prev_sparse = copy.deepcopy(curr_sparse)

        # ==========================================
        # 最终输出处理：除噪、去色、保存
        # ==========================================
        print(f"\n✅ 拼接完毕！提取并融合了 {success_count} 个有效关键帧。耗时: {time.time() - start_time:.2f} 秒。")
        print("🧽 正在执行最终全局降噪与去色 ...")

        global_pcd = global_pcd.voxel_down_sample(voxel_size=self.global_voxel_size)
        global_pcd, _ = global_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        global_pcd.colors = o3d.utility.Vector3dVector() # 剥离颜色，展现纯粹几何质感

        output_ply = os.path.join(self.dataset_dir, "final_reconstruction.ply")
        o3d.io.write_point_cloud(output_ply, global_pcd)
        print(f"🎉 端到端建图完成！完美模型已保存至: {os.path.abspath(output_ply)}")

        # ==========================================
        # 自动弹出 3D 可视化窗口 (带视角矫正与质感优化)
        # ==========================================
        print("👀 正在打开 3D 查看器...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Final Reconstruction (End-to-End)", width=1024, height=768)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.15]) # 高级深灰背景
        opt.point_size = 2.5 # 放大点距，连点成面

        vis.add_geometry(global_pcd)

        # 强制修正虚拟摄像机视角，保持与现实人眼一致
        ctrl = vis.get_view_control()
        ctrl.set_up([0.0, -1.0, 0.0])
        ctrl.set_front([0.0, 0.0, -1.0])
        ctrl.set_lookat([0.0, 0.0, 1.0])
        ctrl.set_zoom(0.8)

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    # 🔴 注意：这里已配置为你本地的 SDK 路径，GitHub 开源时请提醒用户修改
    SDK_DLL_PATH = r"C:\毕设\奥比中光Win64-Release\奥比中光Win64-Release\sdk\libs"

    # 步骤 1：触发录制
    recorder = AstraDataRecorder(SDK_DLL_PATH)
    recorded_dataset_path = recorder.run_recording()

    # 步骤 2：自动获取刚刚录制的文件夹路径，触发离线极速拼接q
    if recorded_dataset_path:
        stitcher = OfflinePointCloudStitcher(recorded_dataset_path)
        stitcher.run_stitching()
    else:
        print("⚠️ 录制失败或未采集到有效数据，已取消建图。")