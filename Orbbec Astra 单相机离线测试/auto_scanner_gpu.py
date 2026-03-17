"""
Astra RGB-D 自动扫描与 3D 重建流水线
核心功能：
1. Data Recorder: 调用 Orbbec Astra 相机实时录制数据集。
2. Tensor ICP Stitcher: 录制结束后，唤醒 NVIDIA GPU (CUDA)完成点云提取、法线计算与点到面 ICP 全局拼接。
"""

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import os
import glob
import time
import sys
from datetime import datetime


# ==========================================
# 🛠️ 模块一：数据录制 (Data Recorder)
# ==========================================
class AstraDataRecorder:
    def __init__(self, sdk_path):
        self.sdk_path = sdk_path
        self.dataset_dir = ""
        self.depth_dir = ""
        self.color_dir = ""

    def _init_folders(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = f"astra_dataset_gpu_{timestamp}"
        self.depth_dir = os.path.join(self.dataset_dir, "depth")
        self.color_dir = os.path.join(self.dataset_dir, "color")
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        print(f"\n📁 创建临时数据集目录: {os.path.abspath(self.dataset_dir)}")

    def run_recording(self):
        try:
            from orbbec_sdk import OrbbecCameraSDK
        except ImportError:
            print("❌ 找不到 orbbec_sdk.py，请确保它在当前文件夹下。")
            return None

        if os.path.exists(self.sdk_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(self.sdk_path)

        sdk = OrbbecCameraSDK(self.sdk_path)
        if not sdk.initialize() or not sdk.open_device(0):
            print("❌ 深度相机初始化失败！")
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
        print("👉 请平缓移动相机扫描房间。按下 'Q' 键结束录制并移交 GPU 建图。")
        print("=" * 50)

        try:
            while True:
                depth_res = sdk.capture_depth_frame(timeout=1000)
                ret, color_frame = color_cam.read()

                if depth_res and ret:
                    depth_data = depth_res[0]
                    timestamp_str = f"{time.time():.4f}.png"

                    cv2.imwrite(os.path.join(self.depth_dir, timestamp_str), depth_data.astype(np.uint16))
                    cv2.imwrite(os.path.join(self.color_dir, timestamp_str), color_frame)
                    frame_count += 1

                    depth_preview = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow("Astra Color Stream (Press 'Q' to Stitch)", color_frame)
                    cv2.imshow("Astra Depth Stream", cv2.applyColorMap(depth_preview, cv2.COLORMAP_JET))

                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    print(f"\n🛑 录制结束。共采集 {frame_count} 帧。")
                    break
        finally:
            sdk.cleanup()
            color_cam.release()
            cv2.destroyAllWindows()

        return self.dataset_dir if frame_count > 0 else None


# ==========================================
# 🛠️ 模块二：离线 GPU 拼接 (Tensor ICP Stitcher)
# ==========================================
class GPUPointCloudStitcher:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.color_files = sorted(glob.glob(os.path.join(dataset_dir, "color", "*.png")))
        self.depth_files = sorted(glob.glob(os.path.join(dataset_dir, "depth", "*.png")))

        # 💡 GPU 引擎检测
        if o3c.cuda.is_available():
            self.device = o3c.Device("CUDA:0")
            print("\n✅ 检测到 NVIDIA 显卡！CUDA 狂暴引擎已接管流水线！")
        else:
            self.device = o3c.Device("CPU:0")
            print("\n⚠️ 未检测到 CUDA，将降级为 CPU 张量运算。")

        self.intrinsic_tensor = o3c.Tensor(
            [[525.0, 0, 319.5],
             [0, 525.0, 239.5],
             [0, 0, 1]], dtype=o3c.float64, device=self.device)

        self.icp_voxel_size = 0.02
        self.global_voxel_size = 0.002 # 2毫米精度压平

    def _process_single_frame_gpu(self, frame_idx):
        """完全在 GPU 内部完成图片读取、点云生成与法线计算"""
        # 使用 Tensor I/O 直接读取进显存
        color_t = o3d.t.io.read_image(self.color_files[frame_idx]).to(self.device)
        depth_t = o3d.t.io.read_image(self.depth_files[frame_idx]).to(self.device)

        rgbd_t = o3d.t.geometry.RGBDImage(color_t, depth_t)

        # GPU 反投影
        pcd_t = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_t, self.intrinsic_tensor, depth_scale=1000.0, depth_max=5.0)

        if pcd_t.point.positions.shape[0] < 500:
            return None, None

        # GPU 极速修复 Astra 镜像反转
        positions = pcd_t.point.positions
        positions[:, 0] = positions[:, 0] * -1.0
        pcd_t.point.positions = positions

        # GPU 降采样与法线计算
        pcd_sparse_t = pcd_t.voxel_down_sample(self.icp_voxel_size)
        pcd_sparse_t.estimate_normals(max_nn=30, radius=self.icp_voxel_size * 2)

        return pcd_t, pcd_sparse_t

    def run_stitching(self):
        print("=" * 50)
        print("🚀 [GPU 建图模式] 开始离线 ICP 拼接...")
        print("=" * 50)

        start_time = time.time()

        global_pcd_legacy = o3d.geometry.PointCloud()

        data0 = self._process_single_frame_gpu(0)
        if not data0[0]:
            print("❌ 第 1 帧数据无效，拼接终止。")
            return None

        prev_dense_t, prev_sparse_t = data0
        global_pcd_legacy += prev_dense_t.to_legacy()

        # 在 GPU 里维护全局位置矩阵
        global_transformation_t = o3c.Tensor(np.eye(4), dtype=o3c.float64, device=self.device)
        success_count = 1

        # 💡 使用步长 5 稍微过滤冗余帧
        for i in range(1, len(self.color_files), 5):
            data = self._process_single_frame_gpu(i)
            if not data[0]: continue
            curr_dense_t, curr_sparse_t = data

            # 🔥 核心：调用 GPU 硬件加速的 Point-to-Plane ICP
            estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)

            icp_result = o3d.t.pipelines.registration.icp(
                source=curr_sparse_t,
                target=prev_sparse_t,
                max_correspondence_distance=0.08,
                init_source_to_target=o3c.Tensor(np.eye(4), dtype=o3c.float64, device=self.device),
                estimation_method=estimation,
                criteria=criteria
            )

            # 如果匹配成功且产生有效移动
            if icp_result.fitness > 0.30:
                # 矩阵乘法全在显卡里算
                global_transformation_t = global_transformation_t.matmul(icp_result.transformation)
                curr_dense_t.transform(global_transformation_t)

                # 数据回传到主板内存组装
                global_pcd_legacy += curr_dense_t.to_legacy()
                success_count += 1

                if success_count % 10 == 0:
                    global_pcd_legacy = global_pcd_legacy.voxel_down_sample(voxel_size=self.global_voxel_size)
                    sys.stdout.write(f"\r⚡ GPU 融合中... 已处理 {success_count} 个关键帧")
                    sys.stdout.flush()

                prev_sparse_t = curr_sparse_t.clone()

        print(f"\n\n⏱️ GPU 拼接完成！耗时: {time.time() - start_time:.2f} 秒")

        # 最终输出处理：降噪与去色
        print("🧽 正在执行最终降噪与去色 ...")
        global_pcd_legacy = global_pcd_legacy.voxel_down_sample(voxel_size=self.global_voxel_size)
        global_pcd_legacy, _ = global_pcd_legacy.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

        # 剥离颜色，逼近 Blender 质感
        global_pcd_legacy.colors = o3d.utility.Vector3dVector()

        output_ply = os.path.join(self.dataset_dir, "gpu_final_reconstruction.ply")
        o3d.io.write_point_cloud(output_ply, global_pcd_legacy)
        print(f"🎉 端到端建图完成！模型已保存至: {os.path.abspath(output_ply)}")

        # ==========================================
        # 👀 视觉呈现引擎
        # ==========================================
        print("👀 正在打开高级 3D 查看器...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="GPU Final Reconstruction (High-Res)", width=1280, height=720)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.15])  # 高级深灰背景
        opt.point_size = 2.0  # 连点成面，消除马赛克

        global_pcd_legacy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        global_pcd_legacy.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        vis.add_geometry(global_pcd_legacy)

        # 强制修正为正前导演视角
        ctrl = vis.get_view_control()
        ctrl.set_up([0.0, -1.0, 0.0])
        ctrl.set_front([0.0, 0.0, -1.0])
        ctrl.set_lookat([0.0, 0.0, 1.0])
        ctrl.set_zoom(0.8)

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    # 🔴 注意：确认这是你实际的 Orbbec SDK 路径
    SDK_DLL_PATH = r"C:\毕设\奥比中光Win64-Release\奥比中光Win64-Release\sdk\libs"

    # 第一步：开始录制
    recorder = AstraDataRecorder(SDK_DLL_PATH)
    recorded_dataset_path = recorder.run_recording()

    # 第二步：录制成功后，瞬间移交 GPU 拼接
    if recorded_dataset_path:
        stitcher = GPUPointCloudStitcher(recorded_dataset_path)
        stitcher.run_stitching()
    else:
        print("⚠️ 录制失败或未采集到有效数据，已取消建图。")