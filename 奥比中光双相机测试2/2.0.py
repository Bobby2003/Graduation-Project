"""
extreme_multi_cam_with_depth_bin_calib.py

工程级多相机重建（集成按深度分箱的深度仿射校正）
依赖: open3d, opencv-python, numpy, scipy

说明:
- 代码假设存在 OrbbecCameraSDK 类（initialize/get_devices/open/get_depth/close）。
  若你的 SDK 接口不同，请在 initialize_cameras() 中按实际调整。
- 深度输入 SDK 若为 mm（uint16），本代码内部做单位转换（mm <-> m）。
- 运行前请安装依赖：pip install open3d opencv-python numpy scipy
"""

import os
import cv2
import json
import time
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import copy

# 请替换为你实际的 Orbbec SDK 导入
from orbbec_sdk import OrbbecCameraSDK

class MultiCamDepthBinReconstructor:
    def __init__(self,
                 cam_count=2,
                 video_indices=None,
                 depth_scale=1000.0,  # depth image scale for open3d (mm->m)
                 width=640, height=480,
                 voxel_down=0.003,
                 stable_frames=40,
                 tsdf_voxel=0.0025,
                 tsdf_trunc=0.03,
                 drift_thresh_m=0.015,
                 cache_dir="calib_cache",
                 verbose=True,
                 # New options:
                 use_rgb_matches=False,
                 mesh_export_stl=False,
                 mesh_stl_path="reconstruction_results/model.stl",
                 mesh_smooth_method="taubin",   # 'taubin' or 'simple'
                 mesh_smooth_iters=50,
                 mesh_visualize=False):
        self.cam_count = cam_count
        self.video_indices = video_indices or list(range(cam_count))
        self.depth_scale = depth_scale
        self.width = width
        self.height = height
        self.voxel_down = voxel_down
        self.stable_frames = stable_frames
        self.verbose = verbose

        self.use_rgb_matches = use_rgb_matches
        self.mesh_export_stl = mesh_export_stl
        self.mesh_stl_path = mesh_stl_path
        self.mesh_smooth_method = mesh_smooth_method
        self.mesh_smooth_iters = mesh_smooth_iters
        self.mesh_visualize = mesh_visualize

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_voxel,
            sdf_trunc=tsdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # 默认内参（可改）
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, 525.0, 525.0, 319.5, 239.5
        )

        self.transforms = {}      # cam_idx -> 4x4 transform to cam0 frame
        self.depth_affines = {}   # cam_idx -> (bin_edges, a_bins, b_bins) for correction
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.drift_thresh = drift_thresh_m

    # -------------------------
    # 设备初始化
    # -------------------------
    def initialize_cameras(self):
        if self.verbose: print("🔌 初始化 Orbbec 相机接口...")
        sdk = OrbbecCameraSDK()
        sdk.initialize()
        devs = sdk.get_devices()
        if len(devs) < self.cam_count:
            raise RuntimeError(f"未检测到足够 Orbbec 设备: 需要 {self.cam_count}，检测到 {len(devs)}")

        self.cam_sdks = []
        for i in range(self.cam_count):
            cam = OrbbecCameraSDK()
            cam.initialize()
            cam.open(devs[i]['uri'])
            self.cam_sdks.append(cam)

        self.caps = []
        for vid_idx in self.video_indices:
            cap = cv2.VideoCapture(vid_idx)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频索引 {vid_idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.caps.append(cap)

        if self.verbose: print("✅ 相机初始化完成")

    def close_cameras(self):
        for cam in getattr(self, 'cam_sdks', []):
            try: cam.close()
            except: pass
        for cap in getattr(self, 'caps', []):
            try: cap.release()
            except: pass

    # -------------------------
    # 从 depth 图像构建点云并返回 median depth/color（稳态）
    # 返回: pcd, median_depth_map(m, HxW), median_color(BGR HxW)
    # -------------------------
    def build_stable_pcd_with_median_images(self, cam_sdk, cap, frames=None):
        frames = frames or self.stable_frames
        if self.verbose: print(f"📸 相机累积 {frames} 帧以构建稳态点云和中值图...")
        pcd_acc = o3d.geometry.PointCloud()
        depth_stack = []
        color_stack = []
        got = 0
        attempts = 0
        while got < frames and attempts < frames * 3:
            attempts += 1
            d = cam_sdk.get_depth()  # 假设返回 numpy array 单位 mm 或者按你的 SDK
            ret, c = cap.read()
            if d is None or (not ret):
                continue
            # 保证尺寸
            c_resized = cv2.resize(c, (self.width, self.height))
            # 转为米（假设 SDK 单位 mm）
            depth_m = d.astype(np.float32) / 1000.0
            depth_stack.append(depth_m)
            color_stack.append(c_resized)
            # 构建点云并累加
            pcd = self.rgbd_to_pcd(depth_m, c_resized)
            pcd_acc += pcd
            got += 1
            if self.verbose and got % 10 == 0:
                print(f"  已采集 {got}/{frames}")

        if len(depth_stack) == 0:
            return None, None, None

        # median depth & color
        depth_stack = np.stack(depth_stack, axis=0)  # F x H x W
        median_depth = np.median(depth_stack, axis=0)
        color_stack = np.stack(color_stack, axis=0)  # F x H x W x 3
        median_color = np.median(color_stack.astype(np.float32), axis=0).astype(np.uint8)

        pcd_acc = pcd_acc.voxel_down_sample(self.voxel_down)
        pcd_acc, _ = pcd_acc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        pcd_acc.estimate_normals()
        return pcd_acc, median_depth, median_color

    # -------------------------
    # RGBD -> PCD（输入 depth 单位: m）
    # -------------------------
    def rgbd_to_pcd(self, depth_m, color_bgr):
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth_mm),
            depth_scale=self.depth_scale,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd

    # -------------------------
    # 下采样 + FPFH 预处理
    # -------------------------
    def preprocess_downsample_and_fpfh(self, pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2.0
        radius_feature = voxel_size * 5.0
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, fpfh

    # -------------------------
    # 全局配准（FGR 快速路径 + fallback RANSAC）
    # -------------------------
    def global_registration_fast(self, src, tgt, voxel_size=0.01):
        src_down, src_f = self.preprocess_downsample_and_fpfh(src, voxel_size)
        tgt_down, tgt_f = self.preprocess_downsample_and_fpfh(tgt, voxel_size)

        if self.verbose: print(f"🔍 使用 FGR (voxel={voxel_size:.3f} m) 进行快速配准...")
        fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption()
        fgr_option.maximum_correspondence_distance = voxel_size * 1.5
        fgr_option.iteration_number = 128
        try:
            result_fgr = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                src_down, tgt_down, src_f, tgt_f, fgr_option)
            init_trans = result_fgr.transformation
        except Exception as e:
            if self.verbose: print("⚠️ FGR 失败，回退 RANSAC:", e)
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down, tgt_down, src_f, tgt_f, mutual_filter=True,
                max_correspondence_distance=voxel_size * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 200)
            )
            init_trans = result_ransac.transformation

        # 多尺度 ICP 精修
        if self.verbose: print("✨ 多尺度 ICP 精修...")
        trans = init_trans
        for dist in [voxel_size * 5.0, voxel_size * 2.0, voxel_size * 0.8]:
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, dist, trans,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            trans = reg.transformation
        return trans

    # -------------------------
    # ORB 匹配：对 median 彩色图做关键点匹配并将像素对应投成 3D 对
    # 返回 src_pts3d (N,3), ref_pts3d (N,3)
    # -------------------------
    def orb_match_medians_to_3d(self, ref_median_depth, ref_median_color, src_median_depth, src_median_color, max_keypoints=1500, ratio_test=0.75):
        """
        ref_median_color, src_median_color: HxWx3 (BGR)
        ref_median_depth, src_median_depth: HxW depth in meters
        """
        try:
            gray_ref = cv2.cvtColor(ref_median_color, cv2.COLOR_BGR2GRAY)
            gray_src = cv2.cvtColor(src_median_color, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray_ref = ref_median_color if ref_median_color.ndim == 2 else cv2.cvtColor(ref_median_color, cv2.COLOR_BGR2GRAY)
            gray_src = src_median_color if src_median_color.ndim == 2 else cv2.cvtColor(src_median_color, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=max_keypoints)
        kps_ref, des_ref = orb.detectAndCompute(gray_ref, None)
        kps_src, des_src = orb.detectAndCompute(gray_src, None)
        if des_ref is None or des_src is None or len(kps_ref) < 6 or len(kps_src) < 6:
            return np.zeros((0,3)), np.zeros((0,3))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_src, des_ref, k=2)
        good = []
        for m_n in matches:
            if len(m_n) < 2: continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good.append(m)
        if len(good) == 0:
            return np.zeros((0,3)), np.zeros((0,3))

        fx = self.intrinsic.intrinsic_matrix[0,0]
        fy = self.intrinsic.intrinsic_matrix[1,1]
        cx = self.intrinsic.intrinsic_matrix[0,2]
        cy = self.intrinsic.intrinsic_matrix[1,2]

        src_pts3d = []
        ref_pts3d = []
        H, W = ref_median_depth.shape
        for m in good:
            q = kps_src[m.queryIdx].pt
            t = kps_ref[m.trainIdx].pt
            u1, v1 = int(round(q[0])), int(round(q[1]))
            u2, v2 = int(round(t[0])), int(round(t[1]))
            # bounds check
            if not (0 <= u1 < W and 0 <= v1 < H and 0 <= u2 < W and 0 <= v2 < H):
                continue
            z1 = float(src_median_depth[v1, u1])
            z2 = float(ref_median_depth[v2, u2])
            if z1 <= 0.001 or z2 <= 0.001 or np.isnan(z1) or np.isnan(z2):
                continue
            x1 = (u1 - cx) * z1 / fx
            y1 = (v1 - cy) * z1 / fy
            x2 = (u2 - cx) * z2 / fx
            y2 = (v2 - cy) * z2 / fy
            src_pts3d.append([x1, y1, z1])
            ref_pts3d.append([x2, y2, z2])

        if len(src_pts3d) == 0:
            return np.zeros((0,3)), np.zeros((0,3))
        return np.array(src_pts3d), np.array(ref_pts3d)

    # -------------------------
    # 用 3D-3D 对做 RANSAC 估计变换（Open3D）
    # 返回 trans (4x4), correspondence_set
    # -------------------------
    def estimate_transform_ransac_3d(self, src_pts, ref_pts, max_correspondence_distance=0.05, ransac_n=3, iterations=1000):
        if len(src_pts) < 3:
            return None, []
        src_pcd = o3d.geometry.PointCloud()
        ref_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_pts)
        ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
        corr = np.vstack((np.arange(len(src_pts)), np.arange(len(src_pts)))).T
        corr_o3d = o3d.utility.Vector2iVector(corr.astype(int))
        try:
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                src_pcd, ref_pcd, corr_o3d,
                max_correspondence_distance,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, 100)
            )
            return result.transformation, result.correspondence_set
        except Exception:
            return None, []

    # -------------------------
    # 收集深度对应对：使用稳态点云 + 稳态 depth map
    # 返回 (z_ref_list, z_src_list, u_src_list, v_src_list)
    # 均以 米 为单位
    # -------------------------
    def collect_depth_correspondences_from_medians(self, ref_pcd, src_median_depth, src_median_color,
                                                   src_intrinsic, src_to_ref_trans, max_nn_dist=0.03):
        # 从 src median depth map 生成 src_pts (in src cam coords) 和 pixel uvs
        H, W = src_median_depth.shape
        fx = src_intrinsic.intrinsic_matrix[0, 0]
        fy = src_intrinsic.intrinsic_matrix[1, 1]
        cx = src_intrinsic.intrinsic_matrix[0, 2]
        cy = src_intrinsic.intrinsic_matrix[1, 2]

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        zs = src_median_depth.flatten()
        mask = zs > 0
        if np.sum(mask) == 0:
            return [], [], [], []

        zs = zs[mask]
        us_flat = us.flatten()[mask].astype(np.int32)
        vs_flat = vs.flatten()[mask].astype(np.int32)
        xs = (us_flat - cx) * zs / fx
        ys = (vs_flat - cy) * zs / fy
        src_pts = np.stack([xs, ys, zs], axis=1)  # N x 3

        # 把 src_pts 变换到 ref frame
        ones = np.ones((src_pts.shape[0], 1))
        homo = np.concatenate([src_pts, ones], axis=1).T  # 4 x N
        transformed = (src_to_ref_trans @ homo).T[:, :3]  # N x 3

        # 在 ref_pcd 上做最近邻查找
        ref_tree = o3d.geometry.KDTreeFlann(ref_pcd)
        z_ref_list = []
        z_src_list = []
        u_list = []
        v_list = []
        for i, p in enumerate(transformed):
            [k, idx, d2] = ref_tree.search_knn_vector_3d(p, 1)
            if k > 0:
                dist = np.sqrt(d2[0])
                if dist <= max_nn_dist:
                    ref_pt = np.asarray(ref_pcd.points)[idx[0]]
                    z_ref_list.append(ref_pt[2])
                    z_src_list.append(src_pts[i, 2])  # 注意使用 src 原始 z（相机坐标系）
                    u_list.append(int(us_flat[i]))
                    v_list.append(int(vs_flat[i]))
        return np.array(z_ref_list), np.array(z_src_list), np.array(u_list), np.array(v_list)

    # -------------------------
    # 按 depth bin 拟合仿射模型 z_ref = a * z_src + b
    # -------------------------
    def fit_affine_per_depth_bin(self, z_ref_arr, z_src_arr, n_bins=30, smooth_sigma=1.0):
        if len(z_ref_arr) < 10:
            # 数据太少，返回恒等（不校正）
            return None, None, None, None

        z_src = z_src_arr
        z_ref = z_ref_arr
        zmin, zmax = max(0.0, z_src.min()), z_src.max()
        if zmax <= zmin:
            zmax = zmin + 1e-3
        bin_edges = np.linspace(zmin, zmax, n_bins + 1)
        a_bins = np.full(n_bins, np.nan, dtype=np.float64)
        b_bins = np.full(n_bins, np.nan, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int32)

        for ib in range(n_bins):
            l = bin_edges[ib]
            r = bin_edges[ib + 1]
            idx = np.where((z_src >= l) & (z_src < r))[0]
            counts[ib] = len(idx)
            if len(idx) >= 2:
                A = np.vstack([z_src[idx], np.ones(len(idx))]).T
                x, _, _, _ = np.linalg.lstsq(A, z_ref[idx], rcond=None)
                a_bins[ib] = x[0]
                b_bins[ib] = x[1]

        # 插值填充 NaN
        def fill_nan(arr):
            inds = np.where(~np.isnan(arr))[0]
            if len(inds) == 0:
                return arr
            arr2 = arr.copy()
            for i in range(len(arr2)):
                if np.isnan(arr2[i]):
                    left = inds[inds < i]
                    right = inds[inds > i]
                    if len(left) == 0:
                        arr2[i] = arr2[right[0]]
                    elif len(right) == 0:
                        arr2[i] = arr2[left[-1]]
                    else:
                        lidx = left[-1]; ridx = right[0]
                        t = (i - lidx) / (ridx - lidx)
                        arr2[i] = (1 - t) * arr2[lidx] + t * arr2[ridx]
            return arr2

        a_bins = fill_nan(a_bins)
        b_bins = fill_nan(b_bins)

        # 高斯平滑
        if smooth_sigma > 0:
            a_bins = gaussian_filter1d(a_bins, sigma=smooth_sigma, mode='nearest')
            b_bins = gaussian_filter1d(b_bins, sigma=smooth_sigma, mode='nearest')

        return bin_edges, a_bins, b_bins, counts

    # -------------------------
    # 将 depth map (m) 按 bin 校正
    # -------------------------
    def apply_depth_bins_to_map(self, depth_map_m, bin_edges, a_bins, b_bins):
        if bin_edges is None:
            return depth_map_m
        corrected = depth_map_m.copy().astype(np.float32)
        mask = corrected > 0
        if not np.any(mask):
            return corrected
        zs = corrected[mask]
        bins = np.searchsorted(bin_edges, zs, side='right') - 1
        bins = np.clip(bins, 0, len(a_bins) - 1)
        a_sel = a_bins[bins]; b_sel = b_bins[bins]
        zs_corr = a_sel * zs + b_sel
        corrected[mask] = zs_corr
        return corrected

    # -------------------------
    # 自动标定（集成 depth-bin 校正）
    # -------------------------
    def auto_calibrate_all(self):
        if self.verbose: print("🔧 开始全自动稳态标定（含深度 bin 校正）...")
        stable_pcds = []
        median_depths = []
        median_colors = []
        for i in range(self.cam_count):
            pcd, med_depth, med_color = self.build_stable_pcd_with_median_images(self.cam_sdks[i], self.caps[i], frames=self.stable_frames)
            if pcd is None:
                raise RuntimeError(f"Camera {i} 无法构建稳态点云")
            stable_pcds.append(pcd)
            median_depths.append(med_depth)
            median_colors.append(med_color)

        ref = stable_pcds[0]
        self.transforms[0] = np.eye(4)
        # 第一台相机不需要 depth-bin 校正，保持恒等
        self.depth_affines[0] = None

        # 对后续每台相机进行粗配准 -> 收集对应 -> 拟合 depth-bin -> 保存
        for i in range(1, self.cam_count):
            if self.verbose: print(f"\n=== 标定相机 {i} 到 0 ===")
            src = stable_pcds[i]
            # 快速全局配准（基于几何）
            init_trans = self.global_registration_fast(src, ref, voxel_size=0.01)

            # 可选：使用 RGB median 匹配作为替代或补充初始化
            if self.use_rgb_matches:
                try:
                    src_pts3d, ref_pts3d = self.orb_match_medians_to_3d(
                        ref_median_depth=median_depths[0], ref_median_color=median_colors[0],
                        src_median_depth=median_depths[i], src_median_color=median_colors[i]
                    )
                    if len(src_pts3d) >= 8:
                        trans_rgb, corr = self.estimate_transform_ransac_3d(src_pts3d, ref_pts3d, max_correspondence_distance=0.05, ransac_n=3, iterations=1000)
                        if trans_rgb is not None:
                            # 检验 RGB 变换与几何 init_trans 的一致性：将一些 src_pts3d 用 trans_rgb 转换并与 ref_pts3d 比较
                            src_h = np.hstack((src_pts3d, np.ones((len(src_pts3d),1))))
                            src_trans_rgb = (trans_rgb @ src_h.T).T[:, :3]
                            dists_rgb = np.linalg.norm(src_trans_rgb - ref_pts3d, axis=1)
                            mean_rgb = float(np.mean(dists_rgb))
                            # 若平均误差低于阈值（经验值），则采用 RGB init（或用作候选）
                            if self.verbose:
                                print(f"  RGB-RANSAC mean residual: {mean_rgb:.4f} m (n={len(dists_rgb)})")
                            if mean_rgb < 0.03:
                                if self.verbose: print("  使用 RGB 匹配结果作为初始变换")
                                init_trans = trans_rgb
                            else:
                                if self.verbose: print("  RGB 匹配残差较大，保留几何匹配结果")
                except Exception as e:
                    if self.verbose: print("  RGB 匹配阶段出错，跳过 RGB 初始化:", e)

            # 基于稳态 median depth 收集对应
            z_ref, z_src, us, vs = self.collect_depth_correspondences_from_medians(
                ref, median_depths[i], median_colors[i], self.intrinsic, init_trans, max_nn_dist=0.03
            )
            if len(z_ref) < 20:
                if self.verbose: print("⚠️ 对应点太少，尝试扩大搜索距离或使用更多帧")
                # 仍然保留 init_trans 以便 ICP 精修
            # 拟合 depth-bin（按深度拟合仿射）
            bin_edges, a_bins, b_bins, counts = self.fit_affine_per_depth_bin(z_ref, z_src, n_bins=30, smooth_sigma=1.0)
            self.depth_affines[i] = (bin_edges, a_bins, b_bins)
            if self.verbose:
                print(f"  深度 bin 拟合完成，bins filled counts (示例前5): {counts[:5]}")

            # 将 src 点云的原始深度按 bin 修正再做 ICP 精修（提高精度）
            # 我们先把 median_depths[i] 校正并重建修正点云用于 ICP
            if bin_edges is not None:
                corrected_depth = self.apply_depth_bins_to_map(median_depths[i], bin_edges, a_bins, b_bins)
                src_corr = self.rgbd_to_pcd(corrected_depth, median_colors[i])
                # 精修
                final_trans = init_trans
                for dist in [0.05, 0.02, 0.008]:
                    reg = o3d.pipelines.registration.registration_icp(
                        src_corr, ref, dist, final_trans,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    )
                    final_trans = reg.transformation
            else:
                final_trans = init_trans

            self.transforms[i] = final_trans
            # 质量评估（简单 RMSE 估计）
            try:
                rmse, overlap = self.compute_rmse_and_overlap(src, ref, final_trans)
            except Exception:
                rmse, overlap = float("inf"), 0.0
            if self.verbose:
                print(f"  标定完成: RMSE={rmse:.4f} m, overlap={overlap:.3f}")

        # 保存缓存
        meta = {"stable_frames": self.stable_frames, "use_rgb_matches": self.use_rgb_matches}
        self.save_calibration(meta=meta)

    # -------------------------
    # 质量度量（小心 transform 不要原地修改）
    # -------------------------
    def compute_rmse_and_overlap(self, src, tgt, transform, match_thresh=0.02):
        src_copy = copy.deepcopy(src)
        src_copy.transform(transform)
        tree = o3d.geometry.KDTreeFlann(tgt)
        src_pts = np.asarray(src_copy.points)
        if src_pts.shape[0] == 0 or len(tgt.points) == 0:
            return float("inf"), 0.0
        dists = []
        inliers = 0
        for p in src_pts:
            [k, idx, d2] = tree.search_knn_vector_3d(p, 1)
            if k > 0:
                dist = np.sqrt(d2[0])
                if dist < match_thresh:
                    dists.append(dist)
                    inliers += 1
        if len(dists) == 0:
            return float("inf"), 0.0
        rmse = float(np.sqrt(np.mean(np.square(dists))))
        overlap = float(inliers) / float(min(len(src_pts), len(tgt.points)))
        return rmse, overlap

    # -------------------------
    # 保存/加载 标定
    # -------------------------
    def save_calibration(self, filename=None, meta=None):
        filename = filename or os.path.join(self.cache_dir, f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "transforms": {str(k): v.tolist() for k, v in self.transforms.items()},
            "depth_affines": {}
        }
        for k, v in self.depth_affines.items():
            if v is None:
                payload["depth_affines"][str(k)] = None
            else:
                bin_edges, a_bins, b_bins = v
                payload["depth_affines"][str(k)] = {
                    "bin_edges": bin_edges.tolist(),
                    "b_bins": b_bins.tolist()
                }
        payload["meta"] = meta or {}
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        if self.verbose: print(f"📦 标定缓存已保存: {filename}")
        return filename

    def load_calibration(self, filename):
        with open(filename, "r") as f:
            payload = json.load(f)
        self.transforms = {int(k): np.array(v) for k, v in payload.get("transforms", {}).items()}
        self.depth_affines = {}
        for k, v in payload.get("depth_affines", {}).items():
            ik = int(k)
            if v is None:
                self.depth_affines[ik] = None
            else:
                self.depth_affines[ik] = (np.array(v["bin_edges"]), np.array(v["a_bins"]), np.array(v["b_bins"]))
        if self.verbose: print(f"📂 从缓存加载标定: {filename}")

    def find_latest_cache(self):
        files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith(".json")]
        if not files:
            return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    # -------------------------
    # 在线融合（对每帧先做 depth-bin 校正再 integrate）
    # -------------------------
    def online_fuse(self, max_frames=200, vis=False, save_mesh=True):
        if self.verbose: print("🔁 开始在线融合（对每帧应用 depth-bin 校正）...")
        for frame_idx in range(max_frames):
            for i in range(self.cam_count):
                d = self.cam_sdks[i].get_depth()
                ret, c = self.caps[i].read()
                if d is None or (not ret):
                    continue
                # 转为米
                depth_m = d.astype(np.float32) / 1000.0
                # 读取 depth-affine（bin）
                db = self.depth_affines.get(i, None)
                if db is not None:
                    bin_edges, a_bins, b_bins = db
                    depth_m_corrected = self.apply_depth_bins_to_map(depth_m, bin_edges, a_bins, b_bins)
                else:
                    depth_m_corrected = depth_m

                # 构造 rgbd 并 integrate，Open3D expects depth as uint16 (mm) with depth_scale
                depth_mm = (depth_m_corrected * 1000.0).astype(np.uint16)
                rgb = cv2.cvtColor(cv2.resize(c, (self.width, self.height)), cv2.COLOR_BGR2RGB)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb),
                    o3d.geometry.Image(depth_mm),
                    depth_scale=self.depth_scale,
                    depth_trunc=2.0,
                    convert_rgb_to_intensity=False
                )
                pose = self.transforms.get(i, np.eye(4))
                # 注意：TSDF.integrate 要求 pose 传入的是相机到体素坐标的变换，这里保持与原逻辑一致（直接使用 pose）
                self.volume.integrate(rgbd, self.intrinsic, pose)

            if self.verbose and (frame_idx + 1) % 10 == 0:
                print(f"  已融合 {frame_idx+1}/{max_frames} 帧")

        if save_mesh:
            if self.verbose: print("⏬ 从 TSDF 提取网格...")
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            # 平滑处理
            if self.mesh_smooth_method == "taubin":
                try:
                    mesh = mesh.filter_smooth_taubin(number_of_iterations=self.mesh_smooth_iters)
                except Exception:
                    # 兼容性回退
                    mesh = mesh.filter_smooth_simple(number_of_iterations=self.mesh_smooth_iters)
            else:
                mesh = mesh.filter_smooth_simple(number_of_iterations=self.mesh_smooth_iters)
            mesh.compute_vertex_normals()
            # 修剪与优化基本操作
            try:
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_unreferenced_vertices()
                mesh.remove_duplicated_vertices()
            except Exception:
                pass

            out_dir = "reconstruction_results"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"mesh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply")
            try:
                o3d.io.write_triangle_mesh(out_path, mesh)
                if self.verbose: print(f"✅ 导出 mesh (PLY): {out_path}")
            except Exception as e:
                if self.verbose: print("⚠️ 导出 PLY 失败:", e)

            # 导出 STL（如果需要）
            if self.mesh_export_stl:
                stl_path = self.mesh_stl_path
                os.makedirs(os.path.dirname(stl_path), exist_ok=True)
                try:
                    o3d.io.write_triangle_mesh(stl_path, mesh)
                    if self.verbose: print(f"✅ 导出 STL: {stl_path}")
                except Exception as e:
                    if self.verbose: print("⚠️ 导出 STL 失败:", e)

            # 可视化
            if self.mesh_visualize or vis:
                try:
                    mesh_for_vis = mesh.clone()
                    mesh_for_vis.paint_uniform_color([0.8, 0.6, 0.2])
                    mesh_for_vis.compute_vertex_normals()
                    o3d.visualization.draw_geometries([mesh_for_vis])
                except Exception as e:
                    if self.verbose: print("可视化失败:", e)

            return out_path
        return None

    # -------------------------
    # 一键流水线
    # -------------------------
    def run_full_pipeline(self, max_frames=200, vis=False):
        try:
            self.initialize_cameras()
            time.sleep(1.5)
            latest = self.find_latest_cache()
            if latest:
                try:
                    self.load_calibration(latest)
                except Exception:
                    if self.verbose: print("加载缓存失败，执行自动标定")
                    self.auto_calibrate_all()
            else:
                self.auto_calibrate_all()

            mesh = self.online_fuse(max_frames=max_frames, vis=vis, save_mesh=True)
            return mesh
        finally:
            self.close_cameras()

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", type=int, default=2)
    parser.add_argument("--vids", nargs='+', type=int, help="VideoCapture 索引列表")
    parser.add_argument("--stable", type=int, default=40)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--vis", action="store_true")
    # 新增 CLI 参数
    parser.add_argument("--use_rgb_matches", action="store_true", help="启用 RGB ORB 辅助匹配（用于标定初始变换）")
    parser.add_argument("--export_stl", action="store_true", help="导出 STL 文件（路径由 --stl_path 指定）")
    parser.add_argument("--stl_path", type=str, default="reconstruction_results/model.stl", help="STL 输出路径")
    parser.add_argument("--mesh_smooth_method", type=str, default="taubin", choices=["taubin", "simple"], help="网格平滑方法")
    parser.add_argument("--mesh_smooth_iters", type=int, default=50, help="网格平滑迭代次数")
    args = parser.parse_args()

    if args.vids is None:
        vids = list(range(args.cams))
    else:
        vids = args.vids

    recon = MultiCamDepthBinReconstructor(
        cam_count=args.cams,
        video_indices=vids,
        stable_frames=args.stable,
        use_rgb_matches=args.use_rgb_matches,
        mesh_export_stl=args.export_stl,
        mesh_stl_path=args.stl_path,
        mesh_smooth_method=args.mesh_smooth_method,
        mesh_smooth_iters=args.mesh_smooth_iters,
        mesh_visualize=args.vis
    )
    mesh = recon.run_full_pipeline(max_frames=args.frames, vis=args.vis)
    print("流程完成，mesh:", mesh)

if __name__ == "__main__":
    main()