import time
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path


# ═══════════════════════════════════════════════
#  工具函数（滤波相关，保持不变）
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
    valid_mask = (d > 0).astype(np.uint8)
    _, nearest_idx = distance_transform_edt(
        invalid, return_distances=True, return_indices=True
    )
    result[invalid] = d[nearest_idx[0][invalid], nearest_idx[1][invalid]]
    return result


# ═══════════════════════════════════════════════
#  增量网格重建器
# ═══════════════════════════════════════════════

class IncrementalMeshReconstructor:
    """
    增量式网格重建器
    - 维护一个全局的顶点网格地图 (H, W, 3)
    - 每帧更新有效的深度像素
    - 使用历史信息填补空洞
    - 实现增量三角化
    """

    def __init__(self, width, height, fx, fy, cx, cy,
                 temporal_decay=0.8,  # 历史权重衰减
                 hole_fill_radius=5,  # 空洞填充半径
                 depth_consistency_thresh=30.0):  # 深度一致性阈值 (mm)

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # 全局网格状态
        self.global_vertex_map = np.zeros((height, width, 3), dtype=np.float32)
        self.global_depth_map = np.zeros((height, width), dtype=np.float32)
        self.validity_map = np.zeros((height, width), dtype=bool)  # 当前帧有效
        self.age_map = np.zeros((height, width), dtype=np.float32)  # 历史年龄

        self.temporal_decay = temporal_decay
        self.hole_fill_radius = hole_fill_radius
        self.depth_consistency_thresh = depth_consistency_thresh

        # 缓存坐标网格
        self.uv_grid = np.meshgrid(np.arange(width), np.arange(height))
        self.uv_coords = np.stack(self.uv_grid[::-1], axis=-1).astype(np.float32)  # (H, W, 2)

    def depth_to_vertex_map(self, depth_mm):
        """深度图 → 顶点图 (H, W, 3)，单位米"""
        H, W = depth_mm.shape
        z = depth_mm / 1000.0
        x = (self.uv_grid[0] - self.cx) * z / self.fx
        y = -(self.uv_grid[1] - self.cy) * z / self.fy
        vertices = np.stack([x, y, z], axis=-1)
        return vertices

    def update_global_map(self, new_depth_mm, new_vmask):
        """
        更新全局网格状态
        - 用新帧替换有效区域
        - 对无效区域保持历史数据，但降低权重
        """
        H, W = new_depth_mm.shape

        # 生成新顶点图
        new_vertex_map = self.depth_to_vertex_map(new_depth_mm)

        # 更新有效性
        current_valid = (new_depth_mm > 0) & (new_vmask > 0)

        # 衰减历史数据
        self.age_map *= self.temporal_decay
        self.age_map[current_valid] = 1.0  # 新数据年龄为1

        # 替换有效区域，保留无效区域的历史数据
        mask_new = current_valid
        self.global_depth_map[mask_new] = new_depth_mm[mask_new]
        self.global_vertex_map[mask_new] = new_vertex_map[mask_new]
        self.validity_map[mask_new] = True

        # 对无效区域，如果历史数据较新，可以考虑保留
        mask_invalid = ~current_valid
        # 这里可以加入更多启发式规则，比如根据年龄决定是否保留
        # 目前简单地保持历史数据

    def fill_holes_with_history(self):
        """
        用历史数据填充当前帧的空洞
        使用最近邻插值，但考虑年龄权重
        """
        H, W = self.height, self.width

        # 找出当前无效但历史上有效的像素
        current_invalid = ~self.validity_map
        has_history = (self.age_map > 0.1)  # 年龄阈值

        fillable = current_invalid & has_history

        if not fillable.any():
            return

        # 获取可填充区域的坐标
        fill_coords = np.argwhere(fillable)

        # 获取当前有效区域的坐标
        valid_coords = np.argwhere(self.validity_map)

        if len(valid_coords) == 0:
            return  # 没有有效点可参考

        # 用KD树找最近邻
        tree = cKDTree(valid_coords)
        distances, indices = tree.query(fill_coords)

        # 只填充距离较近的点（防止跨越大的空隙）
        nearby_mask = distances < self.hole_fill_radius

        if nearby_mask.any():
            nearby_fill_coords = fill_coords[nearby_mask]
        nearby_ref_indices = indices[nearby_mask]

        ref_coords = valid_coords[nearby_ref_indices]

        # 复制参考点的数据
        self.global_depth_map[nearby_fill_coords[:, 0], nearby_fill_coords[:, 1]] = \
            self.global_depth_map[ref_coords[:, 0], ref_coords[:, 1]]

        self.global_vertex_map[nearby_fill_coords[:, 0], nearby_fill_coords[:, 1]] = \
            self.global_vertex_map[ref_coords[:, 0], ref_coords[:, 1]]

        self.validity_map[nearby_fill_coords[:, 0], nearby_fill_coords[:, 1]] = True

    def build_mesh_from_current_state(self):
        """
        从当前全局状态构建网格
        使用深度图三角化，但基于当前有效性和填充后的数据
        """
        # 应用孔洞填充
        self.fill_holes_with_history()

        vertex_map = self.global_vertex_map
        depth_map = self.global_depth_map
        valid = self.validity_map

        H, W = vertex_map.shape[:2]

        # 生成顶点数组
        all_vertices = vertex_map.reshape(-1, 3)  # (H*W, 3)
        vertex_idx = np.arange(H * W).reshape(H, W)

        # 构建三角形（向量化）
        v00 = valid[:-1, :-1]
        v01 = valid[:-1, 1:]
        v10 = valid[1:, :-1]
        v11 = valid[1:, 1:]

        d00 = depth_map[:-1, :-1].astype(np.float32)
        d01 = depth_map[:-1, 1:].astype(np.float32)
        d10 = depth_map[1:, :-1].astype(np.float32)
        d11 = depth_map[1:, 1:].astype(np.float32)

        thresh = self.depth_consistency_thresh

        # 上三角 (00, 01, 10)
        ok_upper = (v00 & v01 & v10
                    & (np.abs(d00 - d01) < thresh)
                    & (np.abs(d00 - d10) < thresh)
                    & (np.abs(d01 - d10) < thresh))

        i00 = vertex_idx[:-1, :-1]
        i01 = vertex_idx[:-1, 1:]
        i10 = vertex_idx[1:, :-1]
        i11 = vertex_idx[1:, 1:]

        tri_upper = np.stack([i00[ok_upper], i01[ok_upper], i10[ok_upper]], axis=-1)

        # 下三角 (01, 11, 10)
        ok_lower = (v01 & v11 & v10
                    & (np.abs(d01 - d11) < thresh)
                    & (np.abs(d01 - d10) < thresh)
                    & (np.abs(d11 - d10) < thresh))

        tri_lower = np.stack([i01[ok_lower], i11[ok_lower], i10[ok_lower]], axis=-1)

        triangles = np.concatenate([tri_upper, tri_lower], axis=0) if (len(tri_upper) + len(
            tri_lower)) > 0 else np.zeros((0, 3), dtype=np.int32)

        # 创建网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

        # 计算法线
        mesh.compute_vertex_normals()

        # 简单的颜色（基于深度）
        z = vertex_map[:, :, 2].reshape(-1)
        max_z = z[z > 0].max() if (z > 0).any() else 1.0
        t = np.clip(z / max_z, 0, 1)
        colors = np.zeros((H * W, 3))
        colors[:, 0] = 1.0 - t  # R
        colors[:, 2] = t  # B
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # 移除未使用的顶点
        mesh.remove_unreferenced_vertices()

        return mesh

    def reset(self):
        """重置全局状态"""
        self.global_vertex_map[:] = 0
        self.global_depth_map[:] = 0
        self.validity_map[:] = False
        self.age_map[:] = 0


# ═══════════════════════════════════════════════
#  主类（支持增量重建）
# ═══════════════════════════════════════════════

class Depth3DScanner:

    @staticmethod
    def remove_flying_pixels(d, depth_thresh=100.0, erode_px=2):
        diff_x = np.abs(np.diff(d, axis=1, append=0))
        diff_y = np.abs(np.diff(d, axis=0, append=0))
        edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)
        edge_dilated = cv2.dilate(
            edge, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        )
        d_out = d.copy()
        d_out[edge_dilated > 0] = 0.0
        return d_out

    # ── 可调参数 ──────────────────────────────
    MIN_DEPTH = 0
    MAX_DEPTH = 3000
    MAX_HOLE_PX = 100
    GF_RADIUS = 4
    GF_EPS = 50.0
    POINT_SIZE = 2.0

    # ─────────────────────────────────────────

    def __init__(self):
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())
        self.width, self.height = 640, 480
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0

        xx = np.arange(self.width)
        yy = np.arange(self.height)
        self.u, self.v = np.meshgrid(xx, yy)
        self.first_frame = True
        self.ema_depth = None
        self.EMA_ALPHA = 0.5

        # ── 增量网格重建器 ──
        self.incremental_reconstructor = IncrementalMeshReconstructor(
            width=self.width, height=self.height,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

    # ── 硬件 ──────────────────────────────────
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

    # ── 单帧深度处理 ──────────────────────────
    def process(self, raw):
        d = raw.astype(np.float32)
        d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0

        valid = (d > 0).astype(np.uint8)
        valid_clean = cv2.morphologyEx(
            valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        d[valid_clean == 0] = 0.0
        d = self.remove_flying_pixels(d, depth_thresh=100.0, erode_px=2)
        d_filled = fill_small_holes(d, max_hole_px=self.MAX_HOLE_PX)
        valid_mask = (d > 0).astype(np.uint8)

        if self.ema_depth is None:
            self.ema_depth = d_filled.copy()
        else:
            both_valid = (d_filled > 0) & (self.ema_depth > 0)
            self.ema_depth[both_valid] = (
                    self.EMA_ALPHA * d_filled[both_valid]
                    + (1.0 - self.EMA_ALPHA) * self.ema_depth[both_valid]
            )
            new_valid = (d_filled > 0) & (self.ema_depth == 0)
            self.ema_depth[new_valid] = d_filled[new_valid]
            self.ema_depth[d_filled == 0] = 0.0

        d_ema = self.ema_depth.copy()
        d_for_gf = _fill_all_holes_for_filter(d_ema)
        guide = cv2.normalize(d_for_gf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_gf = guided_filter(guide, d_for_gf, r=self.GF_RADIUS, eps=self.GF_EPS)
        d_gf[valid_mask == 0] = 0.0
        vmask = ((d_gf > self.MIN_DEPTH) & (d_gf < self.MAX_DEPTH)).astype(np.uint8)

        return d_gf, vmask

    # ══════════════════════════════════════════
    #  ★ 暴露的接口
    # ══════════════════════════════════════════

    def capture_raw(self):
        res = self.sdk.capture_depth_frame()
        if res is None:
            return None
        return res

    def get_depth(self, raw):
        return self.process(raw)

    def get_pointcloud(self, depth_mm, vmask):
        sel = (depth_mm > 0) & (vmask > 0)
        if not sel.any():
            return None, None
        z = depth_mm[sel] / 1000.0
        u = self.u[sel]
        v = self.v[sel]
        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy
        pts = np.stack([x, y, z], axis=-1)
        t = np.clip(z / (self.MAX_DEPTH / 1000.0), 0.0, 1.0)
        col = np.zeros_like(pts)
        col[:, 0] = 1.0 - t
        col[:, 2] = t
        return pts, col

    def update_incremental_mesh(self, depth_mm, vmask):
        """
        更新增量网格状态
        这是核心方法：将新帧融合到全局网格中
        """
        self.incremental_reconstructor.update_global_map(depth_mm, vmask)

    def get_incremental_mesh(self):
        """
        获取当前完整的增量网格
        这个网格包含了历史累积的信息
        """
        return self.incremental_reconstructor.build_mesh_from_current_state()

    def reset_mesh(self):
        """重置增量网格状态"""
        self.incremental_reconstructor.reset()

    # ══════════════════════════════════════════
    #  主循环（增量重建模式）
    # ══════════════════════════════════════════

    def run(self):
        if not self.init_device():
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window("LazyMan 3D Scanner - Incremental Mesh", width=1280, height=800)

        # 只使用网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
        vis.add_geometry(mesh)

        ro = vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.08])
        ro.point_size = self.POINT_SIZE
        ro.show_coordinate_frame = True
        ro.mesh_show_back_face = True

        print("增量重建模式运行中 | 'q' 退出  's' 保存  'r' 重置")

        frame_count = 0
        try:
            while True:
                # ── 采集 ──
                res = self.capture_raw()
                if res is None:
                    continue
                raw, _ = res

                # ── 处理深度 ──
                t0 = time.perf_counter()
                depth, vmask = self.get_depth(raw)

                # ── 更新增量网格 ──
                self.update_incremental_mesh(depth, vmask)

                # ── 获取当前完整网格 ──
                current_mesh = self.get_incremental_mesh()

                # ── 更新可视化 ──
                mesh.vertices = current_mesh.vertices
                mesh.triangles = current_mesh.triangles
                mesh.vertex_normals = current_mesh.vertex_normals
                mesh.vertex_colors = current_mesh.vertex_colors

                dt = (time.perf_counter() - t0) * 1000
                frame_count += 1
                if frame_count % 30 == 0:
                    n_tri = len(mesh.triangles)
                    n_vert = len(mesh.vertices)
                    print(f"  [增量重建] {dt:.1f}ms | V:{n_vert} T:{n_tri}")

                vis.update_geometry(mesh)

                if self.first_frame:
                    vis.reset_view_point(True)
                    self.first_frame = False

                if not vis.poll_events():
                    break
                vis.update_renderer()

                # ── 深度预览 ──
                disp = np.clip(depth / self.MAX_DEPTH * 255, 0, 255).astype(np.uint8)
                cv2.imshow("Depth Preview",
                           cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("退出")
                    break
                elif key == ord('s'):
                    fname = f"incremental_mesh_{int(time.time())}.ply"
                    o3d.io.write_triangle_mesh(fname, mesh)
                    print(f"[已保存] {fname}")
                elif key == ord('r'):
                    self.reset_mesh()
                    print("[重置网格]")

        finally:
            self.close()
            vis.destroy_window()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    scanner = Depth3DScanner()
    scanner.run()