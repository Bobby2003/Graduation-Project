import time
import os
import cv2
import numpy as np
import open3d as o3d
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

# ─────────────────────────────────────────────
#  引导滤波
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  只填充小空洞（连通域面积 <= max_hole_px）
#  大空洞（物体边缘/反光区）保持为 0，不污染边缘
# ─────────────────────────────────────────────
def fill_small_holes(depth, max_hole_px=200):
    mask_invalid = (depth == 0).astype(np.uint8)
    if mask_invalid.sum() == 0:
        return depth

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_invalid, connectivity=8
    )

    filled = depth.copy()
    kernel = np.ones((3, 3), np.uint8)

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

        # ★ 直接在 float32 上做最近邻填充，完全不引入中间值
        roi_filled = _nearest_neighbor_fill(roi_depth, roi_hole)

        roi_depth[roi_hole] = roi_filled[roi_hole]
        filled[y0:y1, x0:x1] = roi_depth

    return filled

def _nearest_neighbor_fill(roi, hole_mask):
    """
    空洞像素用周围最近的有效像素值填充
    不做任何插值，不产生中间深度值
    """
    result = roi.copy()
    if not hole_mask.any():
        return result

    # 有效像素的坐标和值
    valid_mask = (roi > 0) & (~hole_mask)
    if not valid_mask.any():
        return result  # 周围全是空洞，放弃填充

    valid_coords = np.argwhere(valid_mask)   # shape (N, 2)
    hole_coords  = np.argwhere(hole_mask)    # shape (M, 2)

    # 用 KD 树找每个空洞像素最近的有效像素
    from scipy.spatial import cKDTree
    tree = cKDTree(valid_coords)
    _, idx = tree.query(hole_coords)

    nearest_vals = roi[valid_coords[idx, 0], valid_coords[idx, 1]]
    result[hole_coords[:, 0], hole_coords[:, 1]] = nearest_vals

    return result

def statistical_filter(d, ksize=15, n_sigma=1.5):
    mask = (d > 0).astype(np.float32)
    sum_d    = cv2.boxFilter(d,     -1, (ksize, ksize), normalize=False)
    sum_mask = cv2.boxFilter(mask,  -1, (ksize, ksize), normalize=False)
    mean_d   = np.where(sum_mask > 0, sum_d / (sum_mask + 1e-6), 0.0)
    sum_d2   = cv2.boxFilter(d * d, -1, (ksize, ksize), normalize=False)
    mean_d2  = np.where(sum_mask > 0, sum_d2 / (sum_mask + 1e-6), 0.0)
    std_d    = np.sqrt(np.maximum(mean_d2 - mean_d**2, 0.0))
    outlier  = (d > 0) & (np.abs(d - mean_d) > n_sigma * std_d + 1.0)
    d_out    = d.copy()
    d_out[outlier] = 0.0
    return d_out

def _fill_all_holes_for_filter(d):
    """
    把所有0值用最近邻填满，只用于引导滤波前的临时处理。
    目的：防止0值污染boxFilter均值，产生伪深度层。
    """
    if (d > 0).all():
        return d

    result = d.copy()
    invalid = result == 0

    # 用距离变换找最近有效像素
    valid_mask = (d > 0).astype(np.uint8)
    # 对每个无效像素，找最近有效像素坐标
    from scipy.ndimage import distance_transform_edt
    _, nearest_idx = distance_transform_edt(
        invalid, return_distances=True, return_indices=True
    )
    result[invalid] = d[nearest_idx[0][invalid], nearest_idx[1][invalid]]
    return result
# ─────────────────────────────────────────────
#  主类
# ─────────────────────────────────────────────
class Depth3DScanner:

    @staticmethod
    def remove_flying_pixels(d, depth_thresh=100.0, erode_px=2):
        """
        去除飞点：
        1. 计算相邻像素深度差，找出深度不连续的边缘
        2. 把边缘区域腐蚀掉 erode_px 个像素
        depth_thresh: 深度差超过多少mm算边缘（默认100mm）
        erode_px:     边缘向内腐蚀几个像素（默认2px）
        """
        # 计算上下左右相邻像素的深度差
        diff_x = np.abs(np.diff(d, axis=1, append=0))
        diff_y = np.abs(np.diff(d, axis=0, append=0))

        # 深度差超过阈值 = 边缘
        edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)

        # 把边缘区域膨胀一下，覆盖飞点范围
        edge_dilated = cv2.dilate(
            edge,
            np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        )

        # 边缘区域清零
        d_out = d.copy()
        d_out[edge_dilated > 0] = 0.0
        return d_out

    # ── 可调参数 ──────────────────────────────
    MIN_DEPTH    = 100    # mm  有效最近距离
    MAX_DEPTH    = 3000   # mm  有效最远距离
    MAX_HOLE_PX  = 100    # px  超过此面积的空洞不填（保护边缘）
    GF_RADIUS    = 4      # 引导滤波半径（小 = 保细节）
    GF_EPS       = 50.0   # 引导滤波正则化（小 = 保边缘）
    POINT_SIZE   = 2.0    # 点云点大小
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
        self.ema_depth = None  # EMA 深度缓存
        self.EMA_ALPHA = 0.5 # 越小越平滑，越大响应越快（0.3~0.5 之间调）

    # ── 硬件 ──────────────────────────────────
    def init_device(self):
        if not self.sdk.initialize():
            print("[Error] SDK init failed");  return False
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



    # ── 单帧深度处理（无时间滤波，彻底消除拖尾）──
    def process(self, raw):
        # 1. 转换 + 范围截断
        d = raw.astype(np.float32)
        d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0

        # 2. 去除孤立噪点
        valid = (d > 0).astype(np.uint8)
        valid_clean = cv2.morphologyEx(
            valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        d[valid_clean == 0] = 0.0

        # 2.5 去除飞点
        d = self.remove_flying_pixels(d, depth_thresh=100.0, erode_px=2)

        # 3. 只填小空洞
        d_filled = fill_small_holes(d, max_hole_px=self.MAX_HOLE_PX)

        # ★ 保存真实有效掩码（填洞之前的边界）
        valid_mask = (d > 0).astype(np.uint8)  # 注意：用 d 不用 d_filled

        # 4. EMA
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

        # 5. ★ 修复引导滤波：空洞区域用邻近值临时填满，滤完再抠掉
        d_for_gf = _fill_all_holes_for_filter(d_ema)  # 临时填满，0不参与均值
        guide = cv2.normalize(d_for_gf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_gf = guided_filter(guide, d_for_gf, r=self.GF_RADIUS, eps=self.GF_EPS)
        d_gf[valid_mask == 0] = 0.0  # 滤完再抠掉，边界干净

        # 6. 最终掩码
        vmask = ((d_gf > self.MIN_DEPTH) & (d_gf < self.MAX_DEPTH)).astype(np.uint8)

        return d_gf, vmask


    # ── 深度 → 点云 ───────────────────────────
    def to_pointcloud(self, depth_mm, mask):
        sel = (depth_mm > 0) & (mask > 0)
        if not sel.any():
            return None, None

        z   = depth_mm[sel] / 1000.0
        u   = self.u[sel]
        v   = self.v[sel]
        x   = (u - self.cx) * z / self.fx
        y   = -(v - self.cy) * z / self.fy
        pts = np.stack([x, y, z], axis=-1)

        # 颜色：近红 → 远蓝
        t   = np.clip(z / (self.MAX_DEPTH / 1000.0), 0.0, 1.0)
        col = np.zeros_like(pts)
        col[:, 0] = 1.0 - t   # R
        col[:, 2] = t          # B
        return pts, col

    # ── 主循环 ────────────────────────────────
    def run(self):
        if not self.init_device():
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window("LazyMan 3D Scanner", width=1280, height=800)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(pcd)

        ro = vis.get_render_option()
        ro.background_color      = np.array([0.08, 0.08, 0.08])
        ro.point_size            = self.POINT_SIZE
        ro.show_coordinate_frame = True

        print("运行中 | OpenCV 窗口聚焦后：'q' 退出   's' 保存点云")

        try:
            while True:
                # ── 采集 ──────────────────────
                res = self.sdk.capture_depth_frame()
                if res is None:
                    continue
                raw, _ = res

                # ── 处理 ──────────────────────
                depth, vmask = self.process(raw)

                # ── 点云 ──────────────────────
                pts, col = self.to_pointcloud(depth, vmask)
                if pts is None:
                    pts = np.zeros((1, 3))
                    col = np.zeros((1, 3))

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(col)
                vis.update_geometry(pcd)

                if self.first_frame and len(pts) > 1:
                    vis.reset_view_point(True)
                    self.first_frame = False

                # ── Open3D 渲染 ───────────────
                if not vis.poll_events():
                    break
                vis.update_renderer()

                # ── OpenCV 深度预览 ───────────
                disp = np.clip(
                    depth / self.MAX_DEPTH * 255, 0, 255
                ).astype(np.uint8)
                cv2.imshow("Depth Preview",
                           cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET))

                # ── 按键 ─────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("退出")
                    break
                elif key == ord('s'):
                    print("\n[触发保存] 正在计算保存路径...")
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    folder_path = os.path.join(script_dir, "pointcloud")
                    
                    # 检查数据并生成文件名
                    if len(pcd.points) < 5:
                        print("[错误] 点云为空或点数太少，取消保存！")
                        continue

                    fname = "pointcloud_{}.ply".format(int(time.time()))
                    save_path = os.path.join(folder_path, fname)
                    
                    # 执行保存
                    success = o3d.io.write_point_cloud(save_path, pcd)
                    
                    if success:
                        print(f" [成功] 点云已存入项目文件夹:")
                        print(f" 路径: {save_path}")
                    else:
                        print(f" [失败] 无法写入文件，请检查磁盘权限。")

        finally:
            self.close()
            vis.destroy_window()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = Depth3DScanner()
    scanner.run()
