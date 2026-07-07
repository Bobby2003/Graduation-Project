"""
实时 SLAM: 深度相机 + 外置IMU，无 Open3D 版本

核心设计:
  1. 相机和 IMU 使用 time.monotonic() 软件时序同步
  2. IMU 环形缓冲 + 最近邻查询
  3. NumPy/SciPy 自写 point-to-point ICP
  4. 体素哈希全局地图，定容，不爆内存
  5. OpenCV 显示 RGB + 简单俯视地图
  6. 手写 PLY 保存点云
"""

import cv2
import numpy as np
import struct
import math
import threading
import collections
import time
import os
import sys

from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

try:
    from orbbec_sdk import OrbbecCameraSDK
except ImportError:
    print("找不到 orbbec_sdk.py")
    sys.exit(1)

import serial
import serial.tools.list_ports

# ═══════════════════════════════════════════════════════════
#  深度处理
# ═══════════════════════════════════════════════════════════

def guided_filter(guide_u8, src, r=4, eps=50.0):
    I = guide_u8.astype(np.float32)
    p = src.astype(np.float32)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))

    a = (mean_Ip - mean_I * mean_p) / (mean_II - mean_I * mean_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    return mean_a * I + mean_b

def fill_small_holes(depth, max_hole_px=100):
    mask_inv = (depth == 0).astype(np.uint8)

    if mask_inv.sum() == 0:
        return depth

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inv, 8)
    filled = depth.copy()

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > max_hole_px:
            continue

        hole = labels == i
        x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]

        p = 4
        y0, y1 = max(y - p, 0), min(y + h + p, depth.shape[0])
        x0, x1 = max(x - p, 0), min(x + w + p, depth.shape[1])

        roi = filled[y0:y1, x0:x1].copy()
        rh = hole[y0:y1, x0:x1]
        rv = (roi > 0) & (~rh)

        if rv.any():
            vc = np.argwhere(rv)
            hc = np.argwhere(rh)

            _, idx = cKDTree(vc).query(hc)
            roi[hc[:, 0], hc[:, 1]] = roi[vc[idx, 0], vc[idx, 1]]

            filled[y0:y1, x0:x1] = roi

    return filled

def process_depth(raw, ema_depth, min_d=100, max_d=3000):
    d = raw.astype(np.float32)

    d[(d < min_d) | (d > max_d)] = 0.0

    v = (d > 0).astype(np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    d[v == 0] = 0.0

    dx = np.abs(np.diff(d, axis=1, append=0))
    dy = np.abs(np.diff(d, axis=0, append=0))

    edge = ((dx > 100) | (dy > 100)).astype(np.uint8)
    edge = cv2.dilate(edge, np.ones((5, 5), np.uint8))
    d[edge > 0] = 0.0

    d = fill_small_holes(d, 100)

    if ema_depth is None:
        ema_depth = d.copy()
    else:
        both = (d > 0) & (ema_depth > 0)
        ema_depth[both] = 0.5 * d[both] + 0.5 * ema_depth[both]
        ema_depth[(d > 0) & (ema_depth == 0)] = d[(d > 0) & (ema_depth == 0)]
        ema_depth[d == 0] = 0.0

    de = ema_depth.copy()
    inv = de == 0

    if inv.any() and np.any(~inv):
        _, ni = distance_transform_edt(inv, return_distances=True, return_indices=True)
        de[inv] = ema_depth[ni[0][inv], ni[1][inv]]

    guide = cv2.normalize(de, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dg = guided_filter(guide, de)

    vmask = ((dg > min_d) & (dg < max_d) & (d > 0)).astype(np.uint8)
    dg[vmask == 0] = 0.0

    return dg, vmask, ema_depth

# ═══════════════════════════════════════════════════════════
#  无 Open3D 点云工具
# ═══════════════════════════════════════════════════════════

def voxel_downsample_np(points, colors=None, voxel_size=0.02):
    """
    纯 numpy 体素降采样。
    每个体素取均值点。
    """
    if points is None or len(points) == 0:
        if colors is None:
            return np.empty((0, 3), np.float32), None
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

    points = np.asarray(points, dtype=np.float32)

    voxel = np.floor(points / voxel_size).astype(np.int32)
    _, inverse = np.unique(voxel, axis=0, return_inverse=True)

    n = int(inverse.max()) + 1

    pts_ds = np.zeros((n, 3), dtype=np.float32)
    cnt = np.bincount(inverse).astype(np.float32)

    np.add.at(pts_ds, inverse, points)
    pts_ds /= cnt[:, None]

    if colors is None:
        return pts_ds, None

    colors = np.asarray(colors, dtype=np.float32)
    col_ds = np.zeros((n, 3), dtype=np.float32)
    np.add.at(col_ds, inverse, colors)
    col_ds /= cnt[:, None]

    return pts_ds, col_ds

def transform_points(pts, T):
    """
    点云刚体变换。
    pts: Nx3
    T: 4x4
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t

def best_fit_transform(A, B):
    """
    求 A -> B 的最优刚体变换。
    A, B: Nx3
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t

def icp_point_to_point(
        source_pts,
        target_pts,
        init_T=None,
        max_iterations=25,
        tolerance=1e-4,
        max_correspondence_distance=0.15,
        sample_size=3500):
    """
    纯 NumPy/SciPy ICP。
    source_pts 配准到 target_pts。

    返回:
      T_total, fitness, rmse
    """
    if source_pts is None or target_pts is None:
        return np.eye(4), 0.0, 1e9

    if len(source_pts) < 50 or len(target_pts) < 50:
        return np.eye(4), 0.0, 1e9

    src = np.asarray(source_pts, dtype=np.float64)
    tgt = np.asarray(target_pts, dtype=np.float64)

    if sample_size is not None and len(src) > sample_size:
        idx = np.random.choice(len(src), sample_size, replace=False)
        src = src[idx]

    if sample_size is not None and len(tgt) > sample_size * 2:
        idx = np.random.choice(len(tgt), sample_size * 2, replace=False)
        tgt = tgt[idx]

    T_total = np.eye(4, dtype=np.float64) if init_T is None else init_T.astype(np.float64).copy()

    src_trans = transform_points(src, T_total)

    tree = cKDTree(tgt)

    prev_rmse = None
    fitness = 0.0
    rmse = 1e9

    for _ in range(max_iterations):
        dists, indices = tree.query(src_trans, k=1)

        mask = dists < max_correspondence_distance

        if mask.sum() < 30:
            break

        src_corr = src_trans[mask]
        tgt_corr = tgt[indices[mask]]

        T_delta, _, _ = best_fit_transform(src_corr, tgt_corr)

        src_trans = transform_points(src_trans, T_delta)
        T_total = T_delta @ T_total

        rmse = float(np.sqrt(np.mean(dists[mask] ** 2)))
        fitness = float(mask.sum() / max(1, len(src_trans)))

        if prev_rmse is not None and abs(prev_rmse - rmse) < tolerance:
            break

        prev_rmse = rmse

    return T_total, fitness, rmse

def save_ply(path, points, colors=None):
    """
    保存 ASCII PLY。
    colors 可以是 [0,1] float 或 uint8。
    """
    points = np.asarray(points)

    if len(points) == 0:
        return False

    has_color = colors is not None and len(colors) == len(points)

    if has_color:
        colors = np.asarray(colors)

        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if has_color:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

    return True

def statistical_outlier_remove(points, colors=None, nb_neighbors=20, std_ratio=1.5):
    """
    简化版统计离群点去除。
    """
    points = np.asarray(points)

    if len(points) < nb_neighbors + 1:
        return points, colors

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)

    mean_d = np.mean(dists[:, 1:], axis=1)

    mu = np.mean(mean_d)
    sigma = np.std(mean_d)

    threshold = mu + std_ratio * sigma

    mask = mean_d < threshold

    if colors is None:
        return points[mask], None

    return points[mask], colors[mask]

def render_topdown(points, colors=None, width=800, height=800, scale=120.0):
    """
    用 OpenCV 渲染简单俯视地图。
    使用 X-Z 平面投影。
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if points is None or len(points) == 0:
        cv2.putText(canvas, "Map empty", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return canvas

    pts = np.asarray(points)

    # 点太多时抽样显示，避免 OpenCV 太卡
    max_show = 150000
    if len(pts) > max_show:
        idx = np.random.choice(len(pts), max_show, replace=False)
        pts = pts[idx]
        if colors is not None and len(colors) == len(points):
            colors_show = colors[idx]
        else:
            colors_show = None
    else:
        colors_show = colors

    x = pts[:, 0]
    z = pts[:, 2]

    u = (x * scale + width / 2).astype(np.int32)
    v = (z * scale + height / 2).astype(np.int32)

    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    u = u[valid]
    v = v[valid]

    if colors_show is not None and len(colors_show) == len(pts):
        c = np.clip(colors_show[valid] * 255, 0, 255).astype(np.uint8)
        c = c[:, ::-1]  # RGB -> BGR
        canvas[v, u] = c
    else:
        canvas[v, u] = (255, 255, 255)

    # 坐标轴
    cv2.line(canvas, (width // 2, 0), (width // 2, height), (60, 60, 60), 1)
    cv2.line(canvas, (0, height // 2), (width, height // 2), (60, 60, 60), 1)

    cv2.circle(canvas, (width // 2, height // 2), 5, (0, 0, 255), -1)

    cv2.putText(canvas, "Top View X-Z", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return canvas

# ═══════════════════════════════════════════════════════════
#  IMU 模块
# ═══════════════════════════════════════════════════════════

HEADER_1, HEADER_2 = 0x7E, 0x23
FUNC_EULER, FUNC_QUAT, FUNC_RAW = 0x26, 0x16, 0x04
ACCEL_SCALE = 16.0 / 32767.0

def calc_checksum(data):
    return sum(data) & 0xFF

class FrameParser:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data):
        self.buf.extend(data)
        frames = []

        while True:
            idx = -1

            for i in range(len(self.buf) - 1):
                if self.buf[i] == HEADER_1 and self.buf[i + 1] == HEADER_2:
                    idx = i
                    break

            if idx == -1:
                self.buf.clear()
                break

            if idx > 0:
                del self.buf[:idx]

            if len(self.buf) < 3:
                break

            fl = self.buf[2]

            if len(self.buf) < fl:
                break

            frame = bytes(self.buf[:fl])
            del self.buf[:fl]

            if calc_checksum(frame[:-1]) != frame[-1]:
                continue

            frames.append((frame[3], frame[4:-1]))

        return frames

def quat_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

def find_ch340_port():
    for p in serial.tools.list_ports.comports():
        desc = p.description.upper()
        if any(k in desc for k in ("CH340", "CH341", "USB-SERIAL")):
            return p.device
    return None

IMUSample = collections.namedtuple("IMUSample", ["t", "R", "accel"])

class IMUReader:
    """
    带时间戳环形缓冲的 IMU 读取器。
    """

    RING_SIZE = 500

    def __init__(self):
        self._lock = threading.Lock()
        self._ring = collections.deque(maxlen=self.RING_SIZE)

        self._R = np.eye(3, dtype=np.float64)
        self._running = False
        self._ready = False

        self._vel = np.zeros(3, dtype=np.float64)
        self._pos = np.zeros(3, dtype=np.float64)

        self._accel_buf = []
        self._accel_hp = np.zeros(3, dtype=np.float64)
        self._accel_prev = np.zeros(3, dtype=np.float64)

        self._last_t = None

    @property
    def is_ready(self):
        return self._ready

    def start(self, port=None, baudrate=115200):
        port = port or find_ch340_port()

        if port is None:
            print("未找到 CH340，IMU 禁用")
            return False

        self._running = True

        th = threading.Thread(
            target=self._loop,
            args=(port, baudrate),
            daemon=True
        )
        th.start()

        print(f"IMU -> {port}")
        return True

    def stop(self):
        self._running = False

    def interpolate_at(self, t_query):
        """
        最近邻查询 IMU 姿态。
        """
        with self._lock:
            if not self._ring:
                return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

            best = min(self._ring, key=lambda s: abs(s.t - t_query))

            return best.R.copy(), self._pos.copy()

    def get_delta_transform(self, t0, t1):
        """
        获取 t0 -> t1 的 IMU 增量变换。
        """
        R0, p0 = self.interpolate_at(t0)
        R1, p1 = self.interpolate_at(t1)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R1 @ R0.T
        T[:3, 3] = p1 - p0

        return T

    def correct_position(self, correction):
        with self._lock:
            self._pos += correction
            self._vel *= 0.8

    def reset_position(self):
        with self._lock:
            self._pos[:] = 0.0
            self._vel[:] = 0.0

    def _update_position(self, accel_raw, R, dt):
        aw = R @ accel_raw
        al = aw - np.array([0.0, 0.0, 9.81], dtype=np.float64)

        self._accel_hp = 0.95 * (self._accel_hp + al - self._accel_prev)
        self._accel_prev = al.copy()

        self._accel_buf.append(np.linalg.norm(self._accel_hp))

        if len(self._accel_buf) > 20:
            self._accel_buf.pop(0)

        # 静止检测
        if len(self._accel_buf) == 20 and np.std(self._accel_buf) < 0.15:
            self._vel[:] = 0.0
        else:
            self._vel += self._accel_hp * dt
            self._pos += self._vel * dt

    def _loop(self, port, baudrate):
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            parser = FrameParser()

            while self._running:
                raw = ser.read(256)

                if not raw:
                    continue

                now = time.monotonic()

                for func, payload in parser.feed(raw):
                    with self._lock:
                        dt = (now - self._last_t) if self._last_t else 0.02
                        dt = min(max(dt, 0.001), 0.1)
                        self._last_t = now

                        accel = None

                        if func == FUNC_QUAT and len(payload) >= 16:
                            w, x, y, z = struct.unpack_from("<ffff", payload, 0)
                            self._R = quat_to_matrix(w, x, y, z)
                            self._ready = True

                        elif func == FUNC_EULER and len(payload) >= 12:
                            roll, pitch, yaw = struct.unpack_from("<fff", payload, 0)

                            cr, sr = math.cos(roll), math.sin(roll)
                            cp, sp = math.cos(pitch), math.sin(pitch)
                            cy, sy = math.cos(yaw), math.sin(yaw)

                            self._R = np.array([
                                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                                [-sp, cp * sr, cp * cr],
                            ], dtype=np.float64)

                            self._ready = True

                        elif func == FUNC_RAW and len(payload) >= 6:
                            ax = struct.unpack_from("<h", payload, 0)[0] * ACCEL_SCALE * 9.81
                            ay = struct.unpack_from("<h", payload, 2)[0] * ACCEL_SCALE * 9.81
                            az = struct.unpack_from("<h", payload, 4)[0] * ACCEL_SCALE * 9.81

                            accel = np.array([ax, ay, az], dtype=np.float64)

                            self._update_position(accel, self._R, dt)

                        self._ring.append(
                            IMUSample(
                                t=now,
                                R=self._R.copy(),
                                accel=accel if accel is not None else np.zeros(3)
                            )
                        )

        except Exception as e:
            print(f"IMU 异常: {e}")

# ═══════════════════════════════════════════════════════════
#  时序同步
# ═══════════════════════════════════════════════════════════

class TimeSynchronizer:
    """
    纯软件时序对齐:
      相机帧到达时打 monotonic 时间戳
      IMU 样本也用 monotonic 时间戳
      相机时间 + offset 后查询 IMU 环形缓冲
    """

    def __init__(self, initial_offset_ms=0.0):
        self._offset = initial_offset_ms / 1000.0
        self._history = []

    @property
    def offset_ms(self):
        return self._offset * 1000.0

    def cam_to_imu_time(self, t_cam):
        return t_cam + self._offset

    def update_offset(self, R_icp_delta, R_imu_delta, step=0.0002):
        R_diff = R_icp_delta @ R_imu_delta.T

        cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(cos_a)

        self._history.append(angle)

        if len(self._history) > 100:
            self._history.pop(0)

        if len(self._history) >= 20:
            recent = np.mean(self._history[-20:])
            older = np.mean(self._history[:10]) if len(self._history) >= 30 else recent

            if recent > older * 1.2 and recent > 0.02:
                self._offset += step
            elif recent < older * 0.8:
                self._offset -= step * 0.5

# ═══════════════════════════════════════════════════════════
#  体素哈希全局地图
# ═══════════════════════════════════════════════════════════

class VoxelHashMap:
    """
    每个体素存一个代表点。
    新点进入空体素 -> 插入。
    新点进入已有体素 -> 加权修正。
    """

    def __init__(self, voxel_size=0.008, max_voxels=800_000):
        self.vs = voxel_size
        self.inv_vs = 1.0 / voxel_size

        self.max_n = max_voxels

        self.pos = np.empty((max_voxels, 3), dtype=np.float32)
        self.col = np.empty((max_voxels, 3), dtype=np.float32)
        self.cnt = np.zeros(max_voxels, dtype=np.int32)

        self.n = 0
        self._map = {}

    def integrate(self, pts, colors):
        if pts is None or len(pts) == 0:
            return 0, 0

        pts = np.asarray(pts, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)

        vk = np.floor(pts * self.inv_vs).astype(np.int32)

        added = 0
        updated = 0

        for i in range(len(pts)):
            key = (int(vk[i, 0]), int(vk[i, 1]), int(vk[i, 2]))

            if key in self._map:
                idx = self._map[key]

                c = self.cnt[idx]
                w_old = min(c, 10)
                total = w_old + 1.0

                self.pos[idx] = (self.pos[idx] * w_old + pts[i]) / total
                self.col[idx] = (self.col[idx] * w_old + colors[i]) / total

                self.cnt[idx] = min(c + 1, 20)

                updated += 1

            else:
                if self.n >= self.max_n:
                    continue

                self._map[key] = self.n

                self.pos[self.n] = pts[i]
                self.col[self.n] = colors[i]
                self.cnt[self.n] = 1

                self.n += 1
                added += 1

        return added, updated

    def get_points_colors(self):
        if self.n == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32)
            )

        return self.pos[:self.n].copy(), np.clip(self.col[:self.n].copy(), 0, 1)

# ═══════════════════════════════════════════════════════════
#  主 SLAM 系统
# ═══════════════════════════════════════════════════════════

class RealtimeSLAM:

    MIN_DEPTH = 100
    MAX_DEPTH = 3000

    def __init__(self):
        print("=" * 60)
        print("  SLAM + IMU + NumPy ICP + 无 Open3D")
        print("=" * 60)

        # 深度相机 SDK 路径
        sdk_path = r"C:\毕设\奥比中光Win64-Release\奥比中光Win64-Release\sdk\libs"

        if os.path.exists(sdk_path) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(sdk_path)

        self.sdk = OrbbecCameraSDK(sdk_path)

        if not self.sdk.initialize() or not self.sdk.open_device(0):
            raise RuntimeError("深度相机初始化失败")

        self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH)
        self.sdk.start_stream()

        # RGB 摄像头
        self.cam = cv2.VideoCapture(0)

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cam.isOpened():
            print("警告: RGB 摄像头打开失败，后续可能没有彩色图")

        # 相机内参
        self.W, self.H = 640, 480
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5

        xx, yy = np.meshgrid(np.arange(self.W), np.arange(self.H))

        self.u_grid = xx.astype(np.float32)
        self.v_grid = yy.astype(np.float32)

        # IMU + 同步器
        self.imu = IMUReader()
        self.imu_ok = self.imu.start()
        self.sync = TimeSynchronizer(initial_offset_ms=0.0)

        # 全局地图
        self.gmap = VoxelHashMap(voxel_size=0.008, max_voxels=800_000)

        # ICP 参数
        self.icp_voxel = 0.02
        self.max_corr = 0.15

        # 状态
        self.T_global = np.eye(4, dtype=np.float64)
        self.prev_icp_pts = None
        self.prev_time = None
        self.prev_imu_R = None

        self.ema_depth = None

        self.frame_count = 0
        self.icp_ok_count = 0

        # 显示窗口名
        self.rgb_window_name = "SLAM View"
        self.map_window_name = "SLAM Map TopView"

    def _grab(self):
        depth_res = self.sdk.capture_depth_frame(timeout=500)

        ret, color = self.cam.read()
        t_cam = time.monotonic()

        if not depth_res:
            return None

        raw, info = depth_res

        raw = np.fliplr(raw).copy()

        if not ret or color is None:
            color = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        else:
            color = cv2.resize(color, (self.W, self.H))
            color = np.fliplr(color).copy()

        return raw, color, t_cam

    def _to_points(self, depth_mm, vmask, color_bgr):
        sel = (depth_mm > 0) & (vmask > 0)

        if sel.sum() < 500:
            return None, None, None

        z = depth_mm[sel] / 1000.0
        x = (self.u_grid[sel] - self.cx) * z / self.fx
        y = -(self.v_grid[sel] - self.cy) * z / self.fy

        pts = np.stack([x, y, z], axis=-1).astype(np.float32)

        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        col = rgb[sel].astype(np.float32) / 255.0

        # ICP 降采样点云
        pts_ds, _ = voxel_downsample_np(pts, None, voxel_size=self.icp_voxel)

        return pts, col, pts_ds

    def _transform_pts(self, pts, T):
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)

        return (pts @ R.T) + t

    def _update_vis(self):
        pts, cols = self.gmap.get_points_colors()

        img = render_topdown(
            pts,
            cols,
            width=800,
            height=800,
            scale=120.0
        )

        cv2.imshow(self.map_window_name, img)

    def _show_info(self, color, t_frame_ms, fitness, rmse=0.0):
        info_lines = [
            f"Map: {self.gmap.n / 1000:.0f}K pts | Frame: {self.frame_count}",
            f"ICP fit: {fitness:.2f} | RMSE: {rmse:.3f} | {t_frame_ms:.0f}ms",
            f"Sync offset: {self.sync.offset_ms:.1f}ms",
        ]

        if self.imu_ok and self.imu.is_ready:
            _, pos = self.imu.interpolate_at(time.monotonic())
            info_lines.append(f"IMU pos: {pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}")
        else:
            info_lines.append("IMU: N/A")

        for i, line in enumerate(info_lines):
            cv2.putText(
                color,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1
            )

        cv2.imshow(self.rgb_window_name, color)

    def _save_snapshot(self):
        out_dir = os.path.join(os.getcwd(), "outdata")
        os.makedirs(out_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"slam_{ts}.ply")

        pts, cols = self.gmap.get_points_colors()

        if len(pts) > 0:
            save_ply(path, pts, cols)
            print(f"快照已保存 -> {path} ({len(pts)} pts)")
        else:
            print("地图为空，跳过保存")

    def _shutdown(self):
        print("\n正在关闭...")

        try:
            self.imu.stop()
        except Exception:
            pass

        try:
            self.sdk.cleanup()
        except Exception:
            pass

        try:
            self.cam.release()
        except Exception:
            pass

        pts, cols = self.gmap.get_points_colors()

        if len(pts) > 0:
            print("正在进行离群点过滤...")

            pts, cols = statistical_outlier_remove(
                pts,
                cols,
                nb_neighbors=20,
                std_ratio=1.5
            )

            out_dir = os.path.join(os.getcwd(), "outdata")
            os.makedirs(out_dir, exist_ok=True)

            path = os.path.join(out_dir, "slam_final.ply")
            save_ply(path, pts, cols)

            print(f"最终点云 -> {path} ({len(pts)} pts)")

        cv2.destroyAllWindows()

        print("\n统计:")
        print(f"  总帧数: {self.frame_count}")
        print(f"  ICP 成功: {self.icp_ok_count}")
        print(f"  地图体素: {self.gmap.n}")
        print(f"  时间偏移: {self.sync.offset_ms:.2f} ms")

    def run(self):
        print("按 Q/ESC 退出，S 保存，R 重置 IMU 位移")
        print(f"IMU: {'OK' if self.imu_ok else 'N/A'}")

        try:
            while True:
                t_loop_start = time.monotonic()

                grab = self._grab()

                if grab is None:
                    continue

                raw, color, t_cam = grab

                depth_mm, vmask, self.ema_depth = process_depth(
                    raw,
                    self.ema_depth,
                    self.MIN_DEPTH,
                    self.MAX_DEPTH
                )

                pts_local, col_local, pts_icp = self._to_points(
                    depth_mm,
                    vmask,
                    color
                )

                if pts_local is None or pts_icp is None or len(pts_icp) < 100:
                    continue

                self.frame_count += 1

                fitness = 0.0
                rmse = 0.0

                # ─── 第一帧 ───
                if self.prev_icp_pts is None:
                    self.gmap.integrate(pts_local, col_local)

                    self.prev_icp_pts = pts_icp
                    self.prev_time = t_cam

                    if self.imu_ok and self.imu.is_ready:
                        self.prev_imu_R, _ = self.imu.interpolate_at(
                            self.sync.cam_to_imu_time(t_cam)
                        )

                    self._update_vis()

                    t_frame_ms = (time.monotonic() - t_loop_start) * 1000.0
                    self._show_info(color, t_frame_ms, 1.0, 0.0)

                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break
                    elif key == ord("s"):
                        self._save_snapshot()

                    continue

                # ─── IMU 提供 ICP 初值 ───
                T_init = np.eye(4, dtype=np.float64)
                R_imu_delta = np.eye(3, dtype=np.float64)

                if self.imu_ok and self.imu.is_ready and self.prev_time is not None:
                    t0_imu = self.sync.cam_to_imu_time(self.prev_time)
                    t1_imu = self.sync.cam_to_imu_time(t_cam)

                    T_init = self.imu.get_delta_transform(t0_imu, t1_imu)
                    R_imu_delta = T_init[:3, :3].copy()

                # ─── 纯 NumPy/SciPy ICP ───
                try:
                    T_icp, fitness, rmse = icp_point_to_point(
                        source_pts=pts_icp,
                        target_pts=self.prev_icp_pts,
                        init_T=T_init,
                        max_iterations=25,
                        tolerance=1e-4,
                        max_correspondence_distance=self.max_corr,
                        sample_size=3500
                    )
                except Exception as e:
                    print("ICP 异常:", e)

                    T_icp = np.eye(4, dtype=np.float64)
                    fitness = 0.0
                    rmse = 1e9

                # ─── 配准成功 ───
                if fitness > 0.25:
                    if self.imu_ok and self.imu.is_ready:
                        self.sync.update_offset(T_icp[:3, :3], R_imu_delta)

                        drift = T_icp[:3, 3] - T_init[:3, 3]
                        self.imu.correct_position(drift * 0.3)

                    self.T_global = self.T_global @ T_icp

                    pts_global = self._transform_pts(pts_local, self.T_global)

                    added, updated = self.gmap.integrate(pts_global, col_local)

                    self.icp_ok_count += 1

                    self.prev_icp_pts = pts_icp

                else:
                    # ICP 失败时只更新参考帧，避免一直卡死在旧帧
                    self.prev_icp_pts = pts_icp

                self.prev_time = t_cam

                if self.imu_ok and self.imu.is_ready:
                    self.prev_imu_R, _ = self.imu.interpolate_at(
                        self.sync.cam_to_imu_time(t_cam)
                    )

                # ─── 地图窗口，每 3 帧更新一次 ───
                if self.frame_count % 3 == 0:
                    self._update_vis()

                # ─── RGB 信息窗口 ───
                t_frame_ms = (time.monotonic() - t_loop_start) * 1000.0
                self._show_info(color, t_frame_ms, fitness, rmse)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                elif key == ord("s"):
                    self._save_snapshot()

                elif key == ord("r"):
                    if self.imu_ok:
                        self.imu.reset_position()
                        print("IMU 位移已重置")

        except KeyboardInterrupt:
            pass

        finally:
            self._shutdown()

# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    slam = RealtimeSLAM()
    slam.run()