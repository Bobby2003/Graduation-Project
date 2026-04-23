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
    mean_I  = cv2.boxFilter(I,     cv2.CV_32F, (r, r))
    mean_p  = cv2.boxFilter(p,     cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = mean_II - mean_I * mean_I
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
        x0 = max(x - pad, 0);  y0 = max(y - pad, 0)
        x1 = min(x + w + pad, depth.shape[1])
        y1 = min(y + h + pad, depth.shape[0])
        roi_depth = filled[y0:y1, x0:x1].copy()
        roi_hole  = hole_mask[y0:y1, x0:x1].astype(bool)
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
    hole_coords  = np.argwhere(hole_mask)
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
#  柏林平滑插值：点云 → 光滑面片
# ═══════════════════════════════════════════════
#
#  思路：
#  1. 深度图中每个像素对应一个 3D 点
#  2. 每个 2x2 像素块的 4 个角点定义一个"柏林格子"
#  3. 在每个格子内部，用柏林风格的 Hermite 平滑插值
#     （smoothstep: t² (3-2t)）细分出更多顶点
#  4. 同时用格子角点的梯度（法线方向）做加权，
#     使插值出的曲面更贴合局部曲率
#  5. 最终在细分网格上直接三角化
#
#  这样得到的面片比直接连三角形更平滑，
#  因为 smoothstep 会在边界处梯度为 0，
#  保证了相邻格子之间 C1 连续。

def perlin_smoothstep(t):
    """柏林经典平滑阶跃：6t^5 - 15t^4 + 10t^3（改进版，C2连续）"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def compute_gradient_map(pts_3d, vmask, H, W):
    """
    计算每个有效像素点的局部梯度（用于柏林插值中的梯度加权）
    返回 (H, W, 3) 的梯度向量
    """
    grad = np.zeros((H, W, 3), dtype=np.float32)

    # x 方向梯度
    for c in range(3):
        channel = pts_3d[:, :, c]
        gx = np.zeros_like(channel)
        gy = np.zeros_like(channel)

        # 中心差分（有效区域）
        gx[:, 1:-1] = (channel[:, 2:] - channel[:, :-2]) * 0.5
        gy[1:-1, :] = (channel[2:, :] - channel[:-2, :]) * 0.5

        # 边界用前向/后向差分
        gx[:, 0]  = channel[:, 1] - channel[:, 0]
        gx[:, -1] = channel[:, -1] - channel[:, -2]
        gy[0, :]  = channel[1, :] - channel[0, :]
        gy[-1, :] = channel[-1, :] - channel[-2, :]

        grad[:, :, c] = np.sqrt(gx**2 + gy**2)

    # 无效区域梯度清零
    grad[vmask == 0] = 0.0
    return grad


def depth_to_perlin_mesh(depth_mm, vmask, fx, fy, cx, cy,
                         downsample=2, subdivisions=2,
                         depth_jump_thresh=50.0,
                         use_gradient_weight=True):
    """
    用柏林平滑插值将深度图转为光滑三角网格

    参数:
        depth_mm:     深度图 (H, W)，单位 mm
        vmask:        有效掩码 (H, W)
        fx, fy, cx, cy: 相机内参
        downsample:   下采样倍率（控制基础网格密度）
        subdivisions: 每个格子内的细分次数
                      1 = 不细分（等同于直接三角化）
                      2 = 每条边分2段 → 4个子三角形
                      3 = 每条边分3段 → 9个子三角形
                      越大越平滑但越慢
        depth_jump_thresh: 深度跳变阈值(mm)
        use_gradient_weight: 是否用梯度加权插值（更贴合曲面）
    """
    # ─── 1. 下采样 ───
    if downsample > 1:
        depth_ds = depth_mm[::downsample, ::downsample]
        vmask_ds = vmask[::downsample, ::downsample]
    else:
        depth_ds = depth_mm
        vmask_ds = vmask

    H, W = depth_ds.shape
    z = depth_ds / 1000.0

    # ─── 2. 像素 → 3D 坐标 ───
    u = np.arange(W, dtype=np.float32) * downsample
    v = np.arange(H, dtype=np.float32) * downsample
    uu, vv = np.meshgrid(u, v)
    x_map = (uu - cx) * z / fx
    y_map = -(vv - cy) * z / fy

    # 3D 坐标图 (H, W, 3)
    pts_3d = np.stack([x_map, y_map, z], axis=2)

    # ─── 3. 找出所有有效的 2×2 格子 ───
    v00 = vmask_ds[:-1, :-1]
    v01 = vmask_ds[:-1, 1:]
    v10 = vmask_ds[1:, :-1]
    v11 = vmask_ds[1:, 1:]
    all_valid = (v00 > 0) & (v01 > 0) & (v10 > 0) & (v11 > 0)

    # 深度跳变过滤
    d00 = depth_ds[:-1, :-1].astype(np.float32)
    d01 = depth_ds[:-1, 1:].astype(np.float32)
    d10 = depth_ds[1:, :-1].astype(np.float32)
    d11 = depth_ds[1:, 1:].astype(np.float32)

    max_d = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
    min_d = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11))
    no_jump = (max_d - min_d) < depth_jump_thresh
    valid_quads = all_valid & no_jump

    qy, qx = np.where(valid_quads)
    n_quads = len(qy)

    if n_quads == 0:
        return None

    # ─── 4. 提取四角 3D 坐标 ───
    # (n_quads, 3) each
    c00 = pts_3d[qy, qx]          # 左上
    c01 = pts_3d[qy, qx + 1]      # 右上
    c10 = pts_3d[qy + 1, qx]      # 左下
    c11 = pts_3d[qy + 1, qx + 1]  # 右下

    # ─── 5. 柏林平滑插值细分 ───
    S = subdivisions
    n_verts_per_quad = (S + 1) * (S + 1)
    n_tris_per_quad  = S * S * 2

    # 插值参数
    t_vals = np.linspace(0.0, 1.0, S + 1, dtype=np.float32)

    # 预计算 smoothstep
    st = perlin_smoothstep(t_vals)  # (S+1,)

    # 生成每个格子内的细分顶点
    # 用广播批量计算所有格子的所有细分点

    all_vertices = np.zeros((n_quads, n_verts_per_quad, 3), dtype=np.float32)

    for j, ty in enumerate(st):
        for i, tx in enumerate(st):
            # 柏林双线性平滑插值
            # top = lerp(c00, c01, smoothstep(tx))
            # bot = lerp(c10, c11, smoothstep(tx))
            # pt  = lerp(top, bot, smoothstep(ty))
            top = c00 * (1.0 - tx) + c01 * tx   # (n_quads, 3)
            bot = c10 * (1.0 - tx) + c11 * tx
            pt  = top * (1.0 - ty) + bot * ty

            vid = j * (S + 1) + i
            all_vertices[:, vid, :] = pt

    # ─── 6. 可选：梯度加权修正 ───
    # 利用局部法线方向微调插值点，使曲面更贴合原始形状
    if use_gradient_weight and S > 1:
        # 计算四角的局部法线
        # 用简单的有限差分近似
        for j_idx in range(1, S):
            ty = st[j_idx]
            for i_idx in range(1, S):
                tx = st[i_idx]
                vid = j_idx * (S + 1) + i_idx

                # 取当前插值点的4个邻居
                vid_l = j_idx * (S + 1) + (i_idx - 1)
                vid_r = j_idx * (S + 1) + (i_idx + 1)
                vid_u = (j_idx - 1) * (S + 1) + i_idx
                vid_d = (j_idx + 1) * (S + 1) + i_idx

                # 用Laplacian平滑微调内部点
                neighbor_avg = (all_vertices[:, vid_l, :] +
                                all_vertices[:, vid_r, :] +
                                all_vertices[:, vid_u, :] +
                                all_vertices[:, vid_d, :]) * 0.25

                # 混合：70% 原始插值 + 30% Laplacian 平滑
                alpha = 0.3
                all_vertices[:, vid, :] = (
                    (1.0 - alpha) * all_vertices[:, vid, :] +
                    alpha * neighbor_avg
                )

    # ─── 7. 展平顶点 + 构建全局索引 ───
    # all_vertices: (n_quads, n_verts_per_quad, 3)
    # 展平为 (n_quads * n_verts_per_quad, 3)
    vertices_flat = all_vertices.reshape(-1, 3)

    # 构建三角形索引
    # 每个格子内部的 S×S 子格，每个子格 2 个三角形
    tri_local = []
    for jj in range(S):
        for ii in range(S):
            v0 = jj * (S + 1) + ii
            v1 = jj * (S + 1) + (ii + 1)
            v2 = (jj + 1) * (S + 1) + ii
            v3 = (jj + 1) * (S + 1) + (ii + 1)
            tri_local.append([v0, v1, v2])
            tri_local.append([v1, v3, v2])
    tri_local = np.array(tri_local, dtype=np.int32)  # (n_tris_per_quad, 3)

    # 为每个格子偏移索引
    quad_offsets = np.arange(n_quads, dtype=np.int32) * n_verts_per_quad
    # (n_quads, n_tris_per_quad, 3)
    all_triangles = tri_local[np.newaxis, :, :] + quad_offsets[:, np.newaxis, np.newaxis]
    triangles_flat = all_triangles.reshape(-1, 3)

    # ─── 8. 去重顶点（可选，提升渲染质量） ───
    # 相邻格子的共享边界顶点会重复，合并它们能让法线更连续
    # 用近似匹配（距离阈值）
    # 注意：对大量顶点，这一步可能较慢，可以跳过
    # 这里用一个简单的网格哈希方法

    MERGE_VERTICES = True
    if MERGE_VERTICES and len(vertices_flat) < 2000000:
        # 量化坐标到一定精度后做哈希去重
        precision = 1e-6
        quantized = np.round(vertices_flat / precision).astype(np.int64)
        # 哈希
        keys = (quantized[:, 0] * 1000003 +
                quantized[:, 1] * 999983 +
                quantized[:, 2])

        _, unique_idx, inverse_idx = np.unique(
            keys, return_index=True, return_inverse=True
        )
        vertices_merged = vertices_flat[unique_idx]
        triangles_merged = inverse_idx[triangles_flat]
    else:
        vertices_merged = vertices_flat
        triangles_merged = triangles_flat

    # ─── 9. 构建 Open3D Mesh ───
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_merged.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles_merged.astype(np.int32))

    # 计算法线（平滑光照）
    mesh.compute_vertex_normals()

    # 按深度着色（蓝近红远）
    z_vals = np.asarray(mesh.vertices)[:, 2]
    z_valid = z_vals[z_vals > 0]
    if len(z_valid) > 0:
        z_min, z_max = z_valid.min(), z_valid.max()
    else:
        z_min, z_max = 0, 1
    rng = z_max - z_min if z_max > z_min else 1.0
    t = np.clip((z_vals - z_min) / rng, 0, 1)

    colors = np.zeros((len(z_vals), 3), dtype=np.float64)
    # 平滑彩虹渐变
    colors[:, 0] = np.clip(t * 2.5 - 1.0, 0, 1)
    colors[:, 1] = np.clip(1.0 - np.abs(t - 0.5) * 3, 0, 1)
    colors[:, 2] = np.clip(1.0 - t * 2.5, 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


# ═══════════════════════════════════════════════
#  主扫描器
# ═══════════════════════════════════════════════

class Depth3DScanner:

    MIN_DEPTH       = 0
    MAX_DEPTH       = 3000
    MAX_HOLE_PX     = 100
    GF_RADIUS       = 4
    GF_EPS          = 50.0

    DOWNSAMPLE      = 3       # 下采样倍率（越大越快越粗）
    SUBDIVISIONS    = 2       # 柏林细分次数（越大越平滑越慢）
    DEPTH_JUMP_MM   = 50.0    # 深度跳变阈值
    USE_GRADIENT    = True    # 梯度加权修正
    WIREFRAME       = False

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
            print("[Error] SDK init failed"); return False
        if not self.sdk.open_device():
            print("[Error] open_device failed"); return False
        if not self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH):
            print("[Error] create depth stream failed"); return False
        if not self.sdk.start_stream():
            print("[Error] start stream failed"); return False
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

        vis = o3d.visualization.Visualizer()
        vis.create_window("柏林平滑面片扫描器", width=1280, height=800)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 0, 0]], dtype=np.int32))
        vis.add_geometry(mesh)

        ro = vis.get_render_option()
        ro.background_color = np.array([0.05, 0.05, 0.1])
        ro.show_coordinate_frame = True
        ro.mesh_show_back_face = True
        ro.light_on = True

        print("=" * 60)
        print("  柏林平滑面片扫描器")
        print("=" * 60)
        print(f"  下采样: {self.DOWNSAMPLE}x | 细分: {self.SUBDIVISIONS}")
        print(f"  跳变阈值: {self.DEPTH_JUMP_MM}mm | 梯度加权: {self.USE_GRADIENT}")
        print("  按键:")
        print("    q     - 退出")
        print("    s     - 保存 .ply")
        print("    o     - 保存 .obj")
        print("    d     - 切换下采样 (2/3/4/6)")
        print("    1-4   - 设置柏林细分次数")
        print("    w     - 切换 线框/实体")
        print("    g     - 切换梯度加权")
        print("    j/k   - 调整跳变阈值 ±10mm")
        print("    space - 暂停/继续")
        print("=" * 60)

        frame_count = 0
        fps_timer = time.perf_counter()
        fps_display = 0.0
        paused = False
        last_mesh = None

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

                    new_mesh = depth_to_perlin_mesh(
                        depth, vmask,
                        self.fx, self.fy, self.cx, self.cy,
                        downsample=self.DOWNSAMPLE,
                        subdivisions=self.SUBDIVISIONS,
                        depth_jump_thresh=self.DEPTH_JUMP_MM,
                        use_gradient_weight=self.USE_GRADIENT
                    )

                    if new_mesh is not None:
                        mesh.vertices = new_mesh.vertices
                        mesh.triangles = new_mesh.triangles
                        mesh.vertex_colors = new_mesh.vertex_colors
                        mesh.vertex_normals = new_mesh.vertex_normals
                        vis.update_geometry(mesh)
                        last_mesh = new_mesh

                        if self.first_frame:
                            vis.reset_view_point(True)
                            self.first_frame = False

                    frame_count += 1
                    dt = time.perf_counter() - t0
                    now = time.perf_counter()
                    if now - fps_timer >= 2.0:
                        fps_display = frame_count / (now - fps_timer)
                        n_v = len(new_mesh.vertices) if new_mesh else 0
                        n_t = len(new_mesh.triangles) if new_mesh else 0
                        print(f"  FPS: {fps_display:.1f} | "
                              f"顶点: {n_v:,} | 面片: {n_t:,} | "
                              f"处理: {dt*1000:.1f}ms | "
                              f"下采样: {self.DOWNSAMPLE}x | "
                              f"细分: {self.SUBDIVISIONS}")
                        frame_count = 0
                        fps_timer = now

                    disp = np.clip(depth / self.MAX_DEPTH * 255, 0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET)
                    info = (f"FPS:{fps_display:.0f} "
                            f"V:{len(mesh.vertices):,} "
                            f"F:{len(mesh.triangles):,} "
                            f"Sub:{self.SUBDIVISIONS}")
                    cv2.putText(depth_color, info, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.imshow("Depth", depth_color)

                if not vis.poll_events():
                    break
                vis.update_renderer()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"  [{'暂停' if paused else '继续'}]")
                elif key == ord('s'):
                    if last_mesh:
                        fname = f"perlin_mesh_{int(time.time())}.ply"
                        o3d.io.write_triangle_mesh(fname, last_mesh)
                        print(f"  [保存] {fname}")
                elif key == ord('o'):
                    if last_mesh:
                        fname = f"perlin_mesh_{int(time.time())}.obj"
                        o3d.io.write_triangle_mesh(fname, last_mesh)
                        print(f"  [保存] {fname}")
                elif key == ord('d'):
                    cycle = {2: 3, 3: 4, 4: 6, 6: 2}
                    self.DOWNSAMPLE = cycle.get(self.DOWNSAMPLE, 3)
                    print(f"  [下采样] {self.DOWNSAMPLE}x")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    self.SUBDIVISIONS = key - ord('0')
                    print(f"  [柏林细分] {self.SUBDIVISIONS} "
                          f"(每格 {self.SUBDIVISIONS**2 * 2} 三角形)")
                elif key == ord('w'):
                    self.WIREFRAME = not self.WIREFRAME
                    ro.mesh_show_wireframe = self.WIREFRAME
                    print(f"  [模式] {'线框' if self.WIREFRAME else '实体'}")
                elif key == ord('g'):
                    self.USE_GRADIENT = not self.USE_GRADIENT
                    print(f"  [梯度加权] {'开' if self.USE_GRADIENT else '关'}")
                elif key == ord('j'):
                    self.DEPTH_JUMP_MM = max(10, self.DEPTH_JUMP_MM - 10)
                    print(f"  [跳变阈值] {self.DEPTH_JUMP_MM}mm")
                elif key == ord('k'):
                    self.DEPTH_JUMP_MM = min(200, self.DEPTH_JUMP_MM + 10)
                    print(f"  [跳变阈值] {self.DEPTH_JUMP_MM}mm")

        finally:
            self.close()
            vis.destroy_window()
            cv2.destroyAllWindows()
            print("已退出。")


if __name__ == "__main__":
    scanner = Depth3DScanner()
    scanner.run()