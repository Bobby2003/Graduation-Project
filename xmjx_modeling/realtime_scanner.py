import cv2
import numpy as np
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path


# ─────────────────────────────
# Guided Filter
# ─────────────────────────────
def guided_filter(guide_u8, src, r=4, eps=50.0):
    I = guide_u8.astype(np.float32)
    p = src.astype(np.float32)

    mean_I  = cv2.boxFilter(I,     cv2.CV_32F, (r, r))
    mean_p  = cv2.boxFilter(p,     cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    return mean_a * I + mean_b


# ─────────────────────────────
# 局部最近邻填洞（核心恢复）
# ─────────────────────────────
def _nearest_neighbor_fill(roi, hole_mask):
    result = roi.copy()

    valid_mask = (roi > 0) & (~hole_mask)
    if not valid_mask.any():
        return result

    from scipy.spatial import cKDTree

    valid_coords = np.argwhere(valid_mask)
    hole_coords  = np.argwhere(hole_mask)

    tree = cKDTree(valid_coords)
    _, idx = tree.query(hole_coords)

    nearest_vals = roi[valid_coords[idx, 0], valid_coords[idx, 1]]
    result[hole_coords[:, 0], hole_coords[:, 1]] = nearest_vals

    return result


def fill_small_holes(depth, max_hole_px=100):
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

        hole_mask = (labels == label_id)

        x, y, w, h = (
            stats[label_id, cv2.CC_STAT_LEFT],
            stats[label_id, cv2.CC_STAT_TOP],
            stats[label_id, cv2.CC_STAT_WIDTH],
            stats[label_id, cv2.CC_STAT_HEIGHT],
        )

        pad = 4
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, depth.shape[1])
        y1 = min(y + h + pad, depth.shape[0])

        roi_depth = filled[y0:y1, x0:x1].copy()
        roi_hole  = hole_mask[y0:y1, x0:x1]

        roi_filled = _nearest_neighbor_fill(roi_depth, roi_hole)

        roi_depth[roi_hole] = roi_filled[roi_hole]
        filled[y0:y1, x0:x1] = roi_depth

    return filled


# ─────────────────────────────
# 引导滤波辅助填充（避免0污染）
# ─────────────────────────────
def fill_all_for_filter(d):
    if (d > 0).all():
        return d

    from scipy.ndimage import distance_transform_edt

    result = d.copy()
    invalid = result == 0

    _, idx = distance_transform_edt(
        invalid, return_distances=True, return_indices=True
    )

    result[invalid] = d[idx[0][invalid], idx[1][invalid]]
    return result


class DepthScanner:

    # ── 参数集中管理 ──
    MIN_DEPTH   = 100
    MAX_DEPTH   = 3000
    MAX_HOLE_PX = 100

    GF_RADIUS = 4
    GF_EPS    = 50.0

    EMA_ALPHA = 1.0

    def __init__(self):
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())

        self.width, self.height = 640, 480
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5

        xx = np.arange(self.width)
        yy = np.arange(self.height)
        self.u, self.v = np.meshgrid(xx, yy)

        self.ema_depth = None

    def init(self):
        return (self.sdk.initialize() and
                self.sdk.open_device() and
                self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH) and
                self.sdk.start_stream())

    def remove_flying_pixels(self, d, depth_thresh=100.0, erode_px=2):
        diff_x = np.abs(np.diff(d, axis=1, append=0))
        diff_y = np.abs(np.diff(d, axis=0, append=0))

        edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)

        edge = cv2.dilate(
            edge,
            np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        )

        d_out = d.copy()
        d_out[edge > 0] = 0.0
        return d_out

    def process(self, raw):
        d = raw.astype(np.float32)

        # 1. 深度裁剪
        d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0

        # 2. 去孤立点
        valid = (d > 0).astype(np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        d[valid == 0] = 0.0

        # 3. 飞点
        d = self.remove_flying_pixels(d)

        # 4. 小洞填充
        d_filled = fill_small_holes(d, self.MAX_HOLE_PX)

        # ★ 原始有效mask
        valid_mask = (d > 0).astype(np.uint8)

        # 5. EMA
        if self.ema_depth is None:
            self.ema_depth = d_filled.copy()
        else:
            both_valid = (d_filled > 0) & (self.ema_depth > 0)

            self.ema_depth[both_valid] = (
                self.EMA_ALPHA * d_filled[both_valid]
                + (1 - self.EMA_ALPHA) * self.ema_depth[both_valid]
            )

            new_valid = (d_filled > 0) & (self.ema_depth == 0)
            self.ema_depth[new_valid] = d_filled[new_valid]

            self.ema_depth[d_filled == 0] = 0.0

        d_ema = self.ema_depth.copy()

        # 6. 引导滤波
        d_temp = fill_all_for_filter(d_ema)
        guide = cv2.normalize(d_temp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        d_gf = guided_filter(guide, d_temp,
                             r=self.GF_RADIUS,
                             eps=self.GF_EPS)

        d_gf[valid_mask == 0] = 0.0

        # 7. 最终mask
        final_mask = (
            (d_gf > self.MIN_DEPTH) &
            (d_gf < self.MAX_DEPTH)
        ).astype(np.uint8)

        return d_gf, final_mask

    def to_pointcloud(self, depth, mask):
        sel = (depth > 0) & (mask > 0)
        if not sel.any():
            return None, None

        z = depth[sel] / 1000.0
        u = self.u[sel]
        v = self.v[sel]

        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy

        pts = np.stack([x, y, z], axis=-1)

        # 恢复深度颜色映射
        t = np.clip(z / (self.MAX_DEPTH / 1000.0), 0.0, 1.0)
        col = np.zeros_like(pts)
        col[:, 0] = 1.0 - t
        col[:, 2] = t

        return pts, col

    def get_pointcloud(self):
        res = self.sdk.capture_depth_frame()
        if res is None:
            return None

        raw, _ = res
        depth, mask = self.process(raw)
        return self.to_pointcloud(depth, mask)
    

    def get_rgb_and_depth(self):
        res = self.sdk.capture_depth_frame()
        if res is None:
            return None

        raw, _ = res
        
        # 1. 滤波处理后的深度图
        depth, mask = self.process(raw)
        
        # 2. 清理残影，有效 mask 外的深度置零
        depth_clean = depth.copy()
        depth_clean[mask == 0] = 0.0
        
        # 3. 生成基于深度的科学热力伪彩色图 
        # 将深度限制在可显视范围内，并翻转，使得：近处发红，远处发蓝
        t = np.clip(1.0 - depth_clean / self.MAX_DEPTH, 0.0, 1.0)
        color_8u = (t * 255).astype(np.uint8)
        
        # 应用 OpenCV 的 Jet 色表
        color_bgr = cv2.applyColorMap(color_8u, cv2.COLORMAP_JET)
        
        # 转换到 Open3D 所需的 RGB 格式
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        # 把没有扫描到/被过滤掉的黑洞区域设为黑色
        color_rgb[mask == 0] = [0, 0, 0]
        
        # 4. Open3D 期望的深度输入是毫米为单位的 float32 或者 uint16
        depth_f32 = depth_clean.astype(np.float32)

        return color_rgb, depth_f32

    def close(self):
        self.sdk.cleanup()