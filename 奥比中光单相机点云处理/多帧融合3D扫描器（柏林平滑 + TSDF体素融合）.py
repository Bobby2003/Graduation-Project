import time
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path


# ═══════════════════════════════════════════════
#  深度图预处理工具
# ═══════════════════════════════════════════════

def guided_filter(guide_u8, src, r=4, eps=50.0):
    I = guide_u8.astype(np.float32)
    p = src.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    return mean_a * I + mean_b


def fill_small_holes(depth, max_hole_px=200):
    mask_invalid = (depth == 0).astype(np.uint8)
    if mask_invalid.sum() == 0:
        return depth
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_invalid, connectivity=8
    )
    filled = depth.copy()
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_hole_px:
            continue
        hole_mask = (labels == label_id).astype(np.uint8)
        x, y, w, h = (stats[label_id, cv2.CC_STAT_LEFT],
                      stats[label_id, cv2.CC_STAT_TOP],
                      stats[label_id, cv2.CC_STAT_WIDTH],
                      stats[label_id, cv2.CC_STAT_HEIGHT])
        pad = 4
        x0 = max(x - pad, 0);
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, depth.shape[1])
        y1 = min(y + h + pad, depth.shape[0])
        roi_depth = filled[y0:y1, x0:x1].copy()
        roi_hole = hole_mask[y0:y1, x0:x1].astype(bool)
        roi_filled = _nearest_neighbor_fill(roi_depth, roi_hole)
        roi_depth[roi_hole] = roi_filled[roi_hole]
        filled[y0:y1, x0:x1] = roi_depth
    return filled


def _nearest_neighbor_fill(roi, hole_mask):
    result = roi.copy()
    if not hole_mask.any():
        return result
    valid_mask = (roi > 0) & (~hole_mask)
    if not valid_mask.any():
        return result
    valid_coords = np.argwhere(valid_mask)
    hole_coords = np.argwhere(hole_mask)
    tree = cKDTree(valid_coords)
    _, idx = tree.query(hole_coords)
    nearest_vals = roi[valid_coords[idx, 0], valid_coords[idx, 1]]
    result[hole_coords[:, 0], hole_coords[:, 1]] = nearest_vals
    return result


def _fill_all_holes_for_filter(d):
    if (d > 0).all():
        return d
    result = d.copy()
    invalid = result == 0
    _, nearest_idx = distance_transform_edt(
        invalid, return_distances=True, return_indices=True
    )
    result[invalid] = d[nearest_idx[0][invalid], nearest_idx[1][invalid]]
    return result


# ═══════════════════════════════════════════════
#  柏林平滑面片（单帧实时预览用）
# ═══════════════════════════════════════════════

def perlin_smoothstep(t):
    """改进柏林平滑阶跃：6t^5 - 15t^4 + 10t^3（C2连续）"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def depth_to_perlin_mesh(depth_mm, vmask, fx, fy, cx, cy,
                         downsample=3, subdivisions=2,
                         depth_jump_thresh=50.0):
    """单帧柏林平滑面片（用于实时预览）"""
    if downsample > 1:
        depth_ds = depth_mm[::downsample, ::downsample]
        vmask_ds = vmask[::downsample, ::downsample]
    else:
        depth_ds = depth_mm
        vmask_ds = vmask

    H, W = depth_ds.shape
    z = depth_ds / 1000.0

    u = np.arange(W, dtype=np.float32) * downsample
    v = np.arange(H, dtype=np.float32) * downsample
    uu, vv = np.meshgrid(u, v)
    x_map = (uu - cx) * z / fx
    y_map = -(vv - cy) * z / fy
    pts_3d = np.stack([x_map, y_map, z], axis=2)

    v00 = vmask_ds[:-1, :-1];
    v01 = vmask_ds[:-1, 1:]
    v10 = vmask_ds[1:, :-1];
    v11 = vmask_ds[1:, 1:]
    all_valid = (v00 > 0) & (v01 > 0) & (v10 > 0) & (v11 > 0)

    d00 = depth_ds[:-1, :-1].astype(np.float32)
    d01 = depth_ds[:-1, 1:].astype(np.float32)
    d10 = depth_ds[1:, :-1].astype(np.float32)
    d11 = depth_ds[1:, 1:].astype(np.float32)
    max_d = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
    min_d = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11))
    valid_quads = all_valid & ((max_d - min_d) < depth_jump_thresh)

    qy, qx = np.where(valid_quads)
    n_quads = len(qy)
    if n_quads == 0:
        return None

    c00 = pts_3d[qy, qx];
    c01 = pts_3d[qy, qx + 1]
    c10 = pts_3d[qy + 1, qx];
    c11 = pts_3d[qy + 1, qx + 1]

    S = subdivisions
    t_vals = np.linspace(0, 1, S + 1, dtype=np.float32)
    st = perlin_smoothstep(t_vals)

    all_vertices = np.zeros((n_quads, (S + 1) * (S + 1), 3), dtype=np.float32)
    for j, ty in enumerate(st):
        for i, tx in enumerate(st):
            top = c00 * (1 - tx) + c01 * tx
            bot = c10 * (1 - tx) + c11 * tx
            all_vertices[:, j * (S + 1) + i, :] = top * (1 - ty) + bot * ty

    # Laplacian 平滑内部点
    if S > 1:
        for j_idx in range(1, S):
            for i_idx in range(1, S):
                vid = j_idx * (S + 1) + i_idx
                neighbors = (all_vertices[:, j_idx * (S + 1) + (i_idx - 1), :] +
                             all_vertices[:, j_idx * (S + 1) + (i_idx + 1), :] +
                             all_vertices[:, (j_idx - 1) * (S + 1) + i_idx, :] +
                             all_vertices[:, (j_idx + 1) * (S + 1) + i_idx, :]) * 0.25
                all_vertices[:, vid, :] = 0.7 * all_vertices[:, vid, :] + 0.3 * neighbors

    vertices_flat = all_vertices.reshape(-1, 3)

    tri_local = []
    for jj in range(S):
        for ii in range(S):
            v0 = jj * (S + 1) + ii;
            v1 = v0 + 1;
            v2 = (jj + 1) * (S + 1) + ii;
            v3 = v2 + 1
            tri_local.append([v0, v1, v2])
            tri_local.append([v1, v3, v2])
    tri_local = np.array(tri_local, dtype=np.int32)

    offsets = np.arange(n_quads, dtype=np.int32) * (S + 1) * (S + 1)
    all_tris = tri_local[None, :, :] + offsets[:, None, None]
    triangles_flat = all_tris.reshape(-1, 3)

    # 顶点合并
    if len(vertices_flat) < 2000000:
        precision = 1e-6
        quantized = np.round(vertices_flat / precision).astype(np.int64)
        keys = quantized[:, 0] * 1000003 + quantized[:, 1] * 999983 + quantized[:, 2]
        _, unique_idx, inverse_idx = np.unique(keys, return_index=True, return_inverse=True)
        vertices_flat = vertices_flat[unique_idx]
        triangles_flat = inverse_idx[triangles_flat]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_flat.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles_flat.astype(np.int32))
    mesh.compute_vertex_normals()

    z_vals = np.asarray(mesh.vertices)[:, 2]
    z_valid = z_vals[z_vals > 0]
    z_min = z_valid.min() if len(z_valid) > 0 else 0
    z_max = z_valid.max() if len(z_valid) > 0 else 1
    rng = max(z_max - z_min, 1e-3)
    t = np.clip((z_vals - z_min) / rng, 0, 1)
    colors = np.zeros((len(z_vals), 3), dtype=np.float64)
    colors[:, 0] = np.clip(t * 2.5 - 1.0, 0, 1)
    colors[:, 1] = np.clip(1.0 - np.abs(t - 0.5) * 3, 0, 1)
    colors[:, 2] = np.clip(1.0 - t * 2.5, 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


# ═══════════════════════════════════════════════
#  TSDF 体素融合器
# ═══════════════════════════════════════════════
#
#  核心思想：
#  - 维护一个 3D 体素网格，每个体素存储：
#    - TSDF 值（到最近表面的截断有符号距离）
#    - 权重（观测次数/置信度）
#    - 颜色（可选）
#  - 每来一帧深度图，就把它融合进体素网格
#  - 用 Marching Cubes 提取等值面（TSDF=0 的面）
#
#  相比单帧面片的优势：
#  - 多帧融合去噪（同一区域多次观测取平均）
#  - 即使相机不动，持续融合也能让表面越来越精细
#  - 如果相机移动（配合 ICP 配准），能建出完整 360° 模型

class TSDFVolume:
    """
    CPU 版 TSDF 体素融合

    体素网格覆盖一个固定的 3D 空间范围
    每帧深度图投影到体素空间，更新 TSDF 值
    """

    def __init__(self, vol_bounds, voxel_size=0.005, trunc_margin=0.015):
        """
        参数:
            vol_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] 单位 m
            voxel_size: 体素大小 (m)，越小越精细但越慢
            trunc_margin: TSDF 截断距离 (m)
        """
        self.vol_bounds = np.array(vol_bounds, dtype=np.float32)
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin

        # 计算体素网格尺寸
        self.vol_dim = np.ceil(
            (self.vol_bounds[:, 1] - self.vol_bounds[:, 0]) / voxel_size
        ).astype(np.int32)

        self.vol_origin = self.vol_bounds[:, 0].copy()

        print(f"  [TSDF] 体素网格: {self.vol_dim} = {np.prod(self.vol_dim):,} 体素")
        print(f"  [TSDF] 体素大小: {voxel_size * 1000:.1f}mm | 截断距离: {trunc_margin * 1000:.1f}mm")

        # TSDF 值和权重
        self.tsdf_vol = np.ones(self.vol_dim, dtype=np.float32)  # 初始化为 1（远离表面）
        self.weight_vol = np.zeros(self.vol_dim, dtype=np.float32)
        self.color_vol = np.zeros((*self.vol_dim, 3), dtype=np.float32)

        # 预计算所有体素中心的世界坐标
        self._precompute_voxel_coords()

        self.n_fused = 0

    def _precompute_voxel_coords(self):
        """预计算体素中心坐标（只算一次）"""
        xv = np.arange(self.vol_dim[0]) * self.voxel_size + self.vol_origin[0] + self.voxel_size * 0.5
        yv = np.arange(self.vol_dim[1]) * self.voxel_size + self.vol_origin[1] + self.voxel_size * 0.5
        zv = np.arange(self.vol_dim[2]) * self.voxel_size + self.vol_origin[2] + self.voxel_size * 0.5

        # (Nx, Ny, Nz, 3)
        self.vox_coords = np.stack(
            np.meshgrid(xv, yv, zv, indexing='ij'), axis=-1
        ).astype(np.float32)

    def integrate(self, depth_mm, vmask, fx, fy, cx, cy,
                  cam_pose=None, color_img=None):
        """
        融合一帧深度图

        参数:
            depth_mm: 深度图 (H, W)，单位 mm
            vmask:    有效掩码
            fx, fy, cx, cy: 内参
            cam_pose: 4x4 相机位姿矩阵（世界→相机）
                      None 表示相机在原点
            color_img: 可选的彩色图 (H, W, 3)，0~255
        """
        H, W = depth_mm.shape
        depth_m = depth_mm / 1000.0

        # 如果没有位姿，假设相机在原点
        if cam_pose is None:
            cam_pose = np.eye(4, dtype=np.float32)

        cam_pose_inv = np.linalg.inv(cam_pose)

        # ─── 将体素坐标变换到相机坐标系 ───
        # vox_coords: (Nx, Ny, Nz, 3)
        vox_flat = self.vox_coords.reshape(-1, 3)  # (N, 3)

        # 世界坐标 → 相机坐标
        R = cam_pose_inv[:3, :3]
        t = cam_pose_inv[:3, 3]
        cam_pts = vox_flat @ R.T + t  # (N, 3)

        # 相机坐标 → 像素坐标
        cam_z = cam_pts[:, 2]
        valid_z = cam_z > 0.01  # 在相机前方

        pix_x = np.zeros_like(cam_z)
        pix_y = np.zeros_like(cam_z)
        pix_x[valid_z] = cam_pts[valid_z, 0] * fx / cam_z[valid_z] + cx
        pix_y[valid_z] = -cam_pts[valid_z, 1] * fy / cam_z[valid_z] + cy

        # 四舍五入到像素坐标
        pix_xi = np.round(pix_x).astype(np.int32)
        pix_yi = np.round(pix_y).astype(np.int32)

        # 在图像范围内的掩码
        in_bounds = (valid_z &
                     (pix_xi >= 0) & (pix_xi < W) &
                     (pix_yi >= 0) & (pix_yi < H))

        # 查询对应深度值
        depth_vals = np.zeros_like(cam_z)
        depth_vals[in_bounds] = depth_m[pix_yi[in_bounds], pix_xi[in_bounds]]

        # 有效掩码查询
        vmask_vals = np.zeros_like(cam_z, dtype=bool)
        vmask_vals[in_bounds] = vmask[pix_yi[in_bounds], pix_xi[in_bounds]] > 0

        # ─── 计算 TSDF ───
        # SDF = 深度图上的深度 - 体素到相机的距离
        sdf = depth_vals - cam_z

        # 只更新：在图像内、有有效深度、SDF 在截断范围内的体素
        valid_update = in_bounds & vmask_vals & (depth_vals > 0) & (sdf > -self.trunc_margin)

        # 截断 SDF
        tsdf = np.clip(sdf / self.trunc_margin, -1.0, 1.0)

        # ─── 加权平均融合 ───
        # 展平索引
        flat_idx = np.arange(len(vox_flat))
        update_idx = flat_idx[valid_update]

        if len(update_idx) == 0:
            return

        # 转回 3D 索引
        idx_3d = np.unravel_index(update_idx, self.vol_dim)

        # 旧值
        old_tsdf = self.tsdf_vol[idx_3d]
        old_weight = self.weight_vol[idx_3d]

        # 新观测
        new_tsdf = tsdf[valid_update]
        new_weight = 1.0  # 每帧权重为 1

        # 加权平均更新
        total_weight = old_weight + new_weight
        self.tsdf_vol[idx_3d] = (old_tsdf * old_weight + new_tsdf * new_weight) / total_weight
        self.weight_vol[idx_3d] = np.minimum(total_weight, 50.0)  # 权重上限

        # 可选：融合颜色
        if color_img is not None:
            old_color = self.color_vol[idx_3d[0], idx_3d[1], idx_3d[2], :]
            new_color = color_img[pix_yi[valid_update], pix_xi[valid_update], :].astype(np.float32) / 255.0
            self.color_vol[idx_3d[0], idx_3d[1], idx_3d[2], :] = (
                    (old_color * old_weight[:, None] + new_color * new_weight) / total_weight[:, None]
            )

        self.n_fused += 1

    def extract_mesh(self, weight_thresh=2.0, apply_color=True):
        """
        用 Marching Cubes 提取等值面

        参数:
            weight_thresh: 最小权重阈值（低于此值的体素被忽略）
        """
        # 低权重区域设为正值（远离表面），避免产生噪声面片
        tsdf_filtered = self.tsdf_vol.copy()
        tsdf_filtered[self.weight_vol < weight_thresh] = 1.0

        # 检查是否有零交叉（表面存在）
        has_surface = np.any(tsdf_filtered < 0) and np.any(tsdf_filtered > 0)
        if not has_surface:
            return None

        try:
            from skimage.measure import marching_cubes

            verts, faces, normals, _ = marching_cubes(
                tsdf_filtered,
                level=0.0,
                spacing=[self.voxel_size] * 3
            )

            # 偏移到世界坐标
            verts += self.vol_origin

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

            # 着色
            z_vals = verts[:, 2]
            z_valid = z_vals[z_vals > 0]
            z_min = z_valid.min() if len(z_valid) > 0 else 0
            z_max = z_valid.max() if len(z_valid) > 0 else 1
            rng = max(z_max - z_min, 1e-3)
            t = np.clip((z_vals - z_min) / rng, 0, 1)
            colors = np.zeros((len(z_vals), 3), dtype=np.float64)
            colors[:, 0] = np.clip(t * 2.0, 0, 1)
            colors[:, 1] = np.clip(1.0 - np.abs(t - 0.4) * 3, 0.1, 0.8)
            colors[:, 2] = np.clip(1.0 - t * 1.5, 0, 1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            # 简单后处理
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()

            return mesh

        except ImportError:
            print("[Warning] skimage 未安装，无法使用 Marching Cubes")
            print("  pip install scikit-image")
            return None

    def reset(self):
        """清空融合结果"""
        self.tsdf_vol[:] = 1.0
        self.weight_vol[:] = 0.0
        self.color_vol[:] = 0.0
        self.n_fused = 0
        print("  [TSDF] 已重置")


# ═══════════════════════════════════════════════
#  简单 ICP 帧间配准
# ═══════════════════════════════════════════════

def frame_to_pointcloud(depth_mm, vmask, fx, fy, cx, cy, downsample=4):
    """深度图转点云（用于 ICP）"""
    if downsample > 1:
        depth_ds = depth_mm[::downsample, ::downsample]
        vmask_ds = vmask[::downsample, ::downsample]
    else:
        depth_ds = depth_mm
        vmask_ds = vmask

    H, W = depth_ds.shape
    z = depth_ds / 1000.0
    u = np.arange(W, dtype=np.float32) * downsample
    v = np.arange(H, dtype=np.float32) * downsample
    uu, vv = np.meshgrid(u, v)

    valid = vmask_ds > 0
    x = (uu[valid] - cx) * z[valid] / fx
    y = -(vv[valid] - cy) * z[valid] / fy
    pts = np.stack([x, y, z[valid]], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
    )
    return pcd


def align_icp(source_pcd, target_pcd, init_pose=None, max_dist=0.02):
    """
    用 ICP 对齐两帧点云
    返回 4x4 变换矩阵
    """
    if init_pose is None:
        init_pose = np.eye(4)

    if len(source_pcd.points) < 100 or len(target_pcd.points) < 100:
        return init_pose, False

    # Point-to-Plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=max_dist,
        init=init_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=30
        )
    )

    success = result.fitness > 0.3 and result.inlier_rmse < 0.01
    return result.transformation, success


# ═══════════════════════════════════════════════
#  主扫描器（双模式）
# ═══════════════════════════════════════════════

class Depth3DScanner:
    MIN_DEPTH = 0
    MAX_DEPTH = 3000
    MAX_HOLE_PX = 100
    GF_RADIUS = 4
    GF_EPS = 50.0

    # 柏林面片参数
    DOWNSAMPLE = 3
    SUBDIVISIONS = 2
    DEPTH_JUMP_MM = 50.0

    # TSDF 参数
    VOXEL_SIZE = 0.004  # 体素大小 (m)，4mm
    TRUNC_MARGIN = 0.012  # 截断距离 (m)，12mm
    VOL_BOUNDS = [  # 融合空间范围 (m)
        [-0.5, 0.5],  # X: 左右 1m
        [-0.4, 0.4],  # Y: 上下 0.8m
        [0.1, 1.5],  # Z: 深度 0.1~1.5m
    ]
    EXTRACT_EVERY = 5  # 每 N 帧提取一次 Mesh
    ICP_ENABLED = False  # 是否启用 ICP 配准（相机不动时关闭）

    @staticmethod
    def remove_flying_pixels(d, depth_thresh=100.0, erode_px=2):
        diff_x = np.abs(np.diff(d, axis=1, append=0))
        diff_y = np.abs(np.diff(d, axis=0, append=0))
        edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        edge_dilated = cv2.dilate(edge, kernel)
        d_out = d.copy()
        d_out[edge_dilated > 0] = 0.0
        return d_out

    def __init__(self):
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())
        self.width, self.height = 640, 480
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0
        self.first_frame = True
        self.ema_depth = None
        self.EMA_ALPHA = 0.5

    def init_device(self):
        if not self.sdk.initialize():
            print("[Error] SDK init failed");
            return False
        if not self.sdk.open_device():
            print("[Error] open_device failed");
            return False
        if not self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH):
            print("[Error] create depth stream failed");
            return False
        if not self.sdk.start_stream():
            print("[Error] start stream failed");
            return False
        return True

    def close(self):
        try:
            self.sdk.cleanup()
        except Exception:
            pass

    def process_depth(self, raw):
        d = raw.astype(np.float32)
        d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0
        valid = (d > 0).astype(np.uint8)
        valid_clean = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        d[valid_clean == 0] = 0.0
        d = self.remove_flying_pixels(d, depth_thresh=100.0, erode_px=2)
        d_filled = fill_small_holes(d, max_hole_px=self.MAX_HOLE_PX)
        valid_mask = (d > 0).astype(np.uint8)

        if self.ema_depth is None:
            self.ema_depth = d_filled.copy()
        else:
            both = (d_filled > 0) & (self.ema_depth > 0)
            self.ema_depth[both] = (self.EMA_ALPHA * d_filled[both]
                                    + (1 - self.EMA_ALPHA) * self.ema_depth[both])
            new_v = (d_filled > 0) & (self.ema_depth == 0)
            self.ema_depth[new_v] = d_filled[new_v]
            self.ema_depth[d_filled == 0] = 0.0

        d_ema = self.ema_depth.copy()
        d_for_gf = _fill_all_holes_for_filter(d_ema)
        guide = cv2.normalize(d_for_gf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_gf = guided_filter(guide, d_for_gf, r=self.GF_RADIUS, eps=self.GF_EPS)
        d_gf[valid_mask == 0] = 0.0
        vmask = ((d_gf > self.MIN_DEPTH) & (d_gf < self.MAX_DEPTH)).astype(np.uint8)
        return d_gf, vmask

    def run(self):
        if not self.init_device():
            return

        # ─── 初始化 TSDF ───
        tsdf = TSDFVolume(
            self.VOL_BOUNDS,
            voxel_size=self.VOXEL_SIZE,
            trunc_margin=self.TRUNC_MARGIN
        )

        # ─── Open3D 窗口 ───
        vis = o3d.visualization.Visualizer()
        vis.create_window("3D 扫描器 (柏林实时 + TSDF融合)", width=1280, height=800)

        # 实时预览用的 mesh
        preview_mesh = o3d.geometry.TriangleMesh()
        preview_mesh.vertices = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        preview_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 0, 0]], dtype=np.int32))
        vis.add_geometry(preview_mesh)

        ro = vis.get_render_option()
        ro.background_color = np.array([0.05, 0.05, 0.1])
        ro.show_coordinate_frame = True
        ro.mesh_show_back_face = True
        ro.light_on = True

        # ─── 模式 ───
        # "live"  = 柏林平滑实时预览（每帧更新，不融合）
        # "fuse"  = TSDF 融合模式（持续融合，定期提取 mesh）
        mode = "live"

        # ICP 相关
        cam_pose = np.eye(4, dtype=np.float64)
        prev_pcd = None

        print("=" * 60)
        print("  3D 扫描器 — 柏林平滑实时 + TSDF 多帧融合")
        print("=" * 60)
        print("  模式:")
        print("    [L] 实时预览（柏林平滑面片，默认）")
        print("    [F] 开始/停止 TSDF 融合")
        print("    [E] 提取融合后的 Mesh")
        print("    [R] 重置 TSDF")
        print("  通用:")
        print("    q     - 退出")
        print("    s     - 保存当前模型 (.ply)")
        print("    o     - 保存 .obj")
        print("    w     - 线框/实体")
        print("    d     - 切换下采样")
        print("    1-4   - 柏林细分次数")
        print("    i     - 切换 ICP 配准")
        print("    space - 暂停")
        print("=" * 60)

        frame_count = 0
        fps_timer = time.perf_counter()
        fps_display = 0.0
        paused = False
        last_mesh = None
        fuse_count = 0
        wireframe = False

        try:
            while True:
                if not paused:
                    res = self.sdk.capture_depth_frame()
                    if res is None:
                        if not vis.poll_events():
                            break
                        vis.update_renderer()
                        continue
                    raw, _ = res

                    t0 = time.perf_counter()
                    depth, vmask = self.process_depth(raw)

                    if mode == "live":
                        # ─── 柏林平滑实时预览 ───
                        new_mesh = depth_to_perlin_mesh(
                            depth, vmask,
                            self.fx, self.fy, self.cx, self.cy,
                            downsample=self.DOWNSAMPLE,
                            subdivisions=self.SUBDIVISIONS,
                            depth_jump_thresh=self.DEPTH_JUMP_MM
                        )
                        if new_mesh is not None:
                            preview_mesh.vertices = new_mesh.vertices
                            preview_mesh.triangles = new_mesh.triangles
                            preview_mesh.vertex_colors = new_mesh.vertex_colors
                            preview_mesh.vertex_normals = new_mesh.vertex_normals
                            vis.update_geometry(preview_mesh)
                            last_mesh = new_mesh

                    elif mode == "fuse":
                        # ─── TSDF 融合模式 ───

                        # 可选 ICP 配准
                        if self.ICP_ENABLED:
                            cur_pcd = frame_to_pointcloud(
                                depth, vmask,
                                self.fx, self.fy, self.cx, self.cy,
                                downsample=6
                            )
                            if prev_pcd is not None and len(cur_pcd.points) > 100:
                                transform, success = align_icp(cur_pcd, prev_pcd, max_dist=0.02)
                                if success:
                                    cam_pose = cam_pose @ np.linalg.inv(transform)
                            prev_pcd = cur_pcd

                        # 融合当前帧
                        tsdf.integrate(
                            depth, vmask,
                            self.fx, self.fy, self.cx, self.cy,
                            cam_pose=cam_pose if self.ICP_ENABLED else None
                        )
                        fuse_count += 1

                        # 定期提取 mesh
                        if fuse_count % self.EXTRACT_EVERY == 0:
                            fused_mesh = tsdf.extract_mesh(weight_thresh=2.0)
                            if fused_mesh is not None:
                                preview_mesh.vertices = fused_mesh.vertices
                                preview_mesh.triangles = fused_mesh.triangles
                                preview_mesh.vertex_colors = fused_mesh.vertex_colors
                                preview_mesh.vertex_normals = fused_mesh.vertex_normals
                                vis.update_geometry(preview_mesh)
                                last_mesh = fused_mesh

                    if self.first_frame:
                        vis.reset_view_point(True)
                        self.first_frame = False

                    # FPS
                    frame_count += 1
                    dt = time.perf_counter() - t0
                    now = time.perf_counter()
                    if now - fps_timer >= 2.0:
                        fps_display = frame_count / (now - fps_timer)
                        n_v = len(preview_mesh.vertices) if preview_mesh else 0
                        n_t = len(preview_mesh.triangles) if preview_mesh else 0
                        mode_str = "实时柏林" if mode == "live" else f"TSDF融合(帧{tsdf.n_fused})"
                        print(f"  [{mode_str}] FPS: {fps_display:.1f} | "
                              f"顶点: {n_v:,} | 面片: {n_t:,} | "
                              f"处理: {dt * 1000:.1f}ms")
                        frame_count = 0
                        fps_timer = now

                    # 深度预览
                    disp = np.clip(depth / self.MAX_DEPTH * 255, 0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET)
                    mode_text = "LIVE" if mode == "live" else f"FUSE ({tsdf.n_fused})"
                    cv2.putText(depth_color, f"{mode_text} FPS:{fps_display:.0f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow("Depth", depth_color)

                # Open3D 事件
                if not vis.poll_events():
                    break
                vis.update_renderer()

                # 按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"  [{'暂停' if paused else '继续'}]")
                elif key == ord('l'):
                    mode = "live"
                    print("  [模式] 实时柏林预览")
                elif key == ord('f'):
                    if mode == "fuse":
                        mode = "live"
                        print(f"  [模式] 停止融合 (共融合 {tsdf.n_fused} 帧)，切回实时预览")
                    else:
                        mode = "fuse"
                        fuse_count = 0
                        print("  [模式] 开始 TSDF 融合... (再按 F 停止)")
                elif key == ord('e'):
                    print("  [提取] 正在提取融合 Mesh...")
                    fused_mesh = tsdf.extract_mesh(weight_thresh=2.0)
                    if fused_mesh is not None:
                        preview_mesh.vertices = fused_mesh.vertices
                        preview_mesh.triangles = fused_mesh.triangles
                        preview_mesh.vertex_colors = fused_mesh.vertex_colors
                        preview_mesh.vertex_normals = fused_mesh.vertex_normals
                        vis.update_geometry(preview_mesh)
                        last_mesh = fused_mesh
                        n_v = len(fused_mesh.vertices)
                        n_t = len(fused_mesh.triangles)
                        print(f"  [提取] 完成: {n_v:,} 顶点, {n_t:,} 面片")
                    else:
                        print("  [提取] 无有效数据")
                elif key == ord('r'):
                    tsdf.reset()
                    cam_pose = np.eye(4, dtype=np.float64)
                    prev_pcd = None
                elif key == ord('s'):
                    if last_mesh:
                        fname = f"scan_{mode}_{int(time.time())}.ply"
                        o3d.io.write_triangle_mesh(fname, last_mesh)
                        print(f"  [保存] {fname}")
                elif key == ord('o'):
                    if last_mesh:
                        fname = f"scan_{mode}_{int(time.time())}.obj"
                        o3d.io.write_triangle_mesh(fname, last_mesh)
                        print(f"  [保存] {fname}")
                elif key == ord('d'):
                    cycle = {2: 3, 3: 4, 4: 6, 6: 2}
                    self.DOWNSAMPLE = cycle.get(self.DOWNSAMPLE, 3)
                    print(f"  [下采样] {self.DOWNSAMPLE}x")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    self.SUBDIVISIONS = key - ord('0')
                    print(f"  [柏林细分] {self.SUBDIVISIONS}")
                elif key == ord('w'):
                    wireframe = not wireframe
                    ro.mesh_show_wireframe = wireframe
                    print(f"  [显示] {'线框' if wireframe else '实体'}")
                elif key == ord('i'):
                    self.ICP_ENABLED = not self.ICP_ENABLED
                    if not self.ICP_ENABLED:
                        cam_pose = np.eye(4, dtype=np.float64)
                        prev_pcd = None
                    print(f"  [ICP] {'开启（相机可移动）' if self.ICP_ENABLED else '关闭（相机固定）'}")

        finally:
            self.close()
            vis.destroy_window()
            cv2.destroyAllWindows()
            print("已退出。")


if __name__ == "__main__":
    scanner = Depth3DScanner()
    scanner.run()