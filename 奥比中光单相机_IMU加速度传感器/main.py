"""
实时 SLAM + 9轴IMU融合 + 完整深度处理流水线
IMU 提供旋转+位移初值，ICP 修正漂移
"""

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import serial.tools.list_ports
import struct
import math
import threading
import os
import sys
import time

from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

try:
    from orbbec_sdk import OrbbecCameraSDK
except ImportError:
    print("❌ 找不到 orbbec_sdk.py")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
#  深度处理函数
# ═══════════════════════════════════════════════════════════

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


def fill_small_holes(depth, max_hole_px=200):
    mask_invalid = (depth == 0).astype(np.uint8)
    if mask_invalid.sum() == 0:
        return depth
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_invalid, connectivity=8)
    filled = depth.copy()
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] > max_hole_px:
            continue
        hole_mask = (labels == label_id).astype(np.uint8)
        x, y, w, h = (stats[label_id, cv2.CC_STAT_LEFT],
                      stats[label_id, cv2.CC_STAT_TOP],
                      stats[label_id, cv2.CC_STAT_WIDTH],
                      stats[label_id, cv2.CC_STAT_HEIGHT])
        pad = 4
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1 = min(x + w + pad, depth.shape[1])
        y1 = min(y + h + pad, depth.shape[0])
        roi_depth = filled[y0:y1, x0:x1].copy()
        roi_hole  = hole_mask[y0:y1, x0:x1].astype(bool)
        roi_filled = _nearest_neighbor_fill(roi_depth, roi_hole)
        roi_depth[roi_hole] = roi_filled[roi_hole]
        filled[y0:y1, x0:x1] = roi_depth
    return filled


def _fill_all_holes_for_filter(d):
    if (d > 0).all():
        return d
    result = d.copy()
    invalid = result == 0
    _, nearest_idx = distance_transform_edt(invalid, return_distances=True, return_indices=True)
    result[invalid] = d[nearest_idx[0][invalid], nearest_idx[1][invalid]]
    return result


def remove_flying_pixels(d, depth_thresh=100.0, erode_px=2):
    diff_x = np.abs(np.diff(d, axis=1, append=0))
    diff_y = np.abs(np.diff(d, axis=0, append=0))
    edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)
    edge_dilated = cv2.dilate(
        edge, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8))
    d_out = d.copy()
    d_out[edge_dilated > 0] = 0.0
    return d_out


# ═══════════════════════════════════════════════════════════
#  IMU 模块（增强版：含位移估计）
# ═══════════════════════════════════════════════════════════

HEADER_1, HEADER_2 = 0x7E, 0x23
FUNC_EULER, FUNC_QUAT, FUNC_RAW = 0x26, 0x16, 0x04
ACCEL_SCALE = 16.0 / 32767.0


def calc_checksum(data: bytes) -> int:
    return sum(data) & 0xFF


class FrameParser:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)
        frames = []
        while True:
            idx = next((i for i in range(len(self.buf) - 1)
                        if self.buf[i] == HEADER_1 and self.buf[i+1] == HEADER_2), -1)
            if idx == -1:
                self.buf.clear(); break
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < 3:
                break
            frame_len = self.buf[2]
            if len(self.buf) < frame_len:
                break
            frame = bytes(self.buf[:frame_len])
            del self.buf[:frame_len]
            if calc_checksum(frame[:-1]) != frame[-1]:
                continue
            frames.append((frame[3], frame[4:-1]))
        return frames


def quat_to_rotation_matrix(w, x, y, z) -> np.ndarray:
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def find_ch340_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.upper() for k in ('CH340', 'CH341', 'USB-SERIAL')):
            return p.device
    return None


class IMUReader:
    def __init__(self, port=None, baudrate=115200):
        self.port, self.baudrate = port or find_ch340_port(), baudrate
        self._lock = threading.Lock()
        self._R = np.eye(3)
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._ready = False
        self._running, self._thread = False, None
        self._accel_buf = []
        self._last_t = None

    def start(self):
        if self.port is None:
            print("⚠️  未找到 CH340，IMU 辅助已禁用")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"✅ IMU 线程已启动 → {self.port}")
        return True

    def stop(self):
        self._running = False

    @property
    def rotation_matrix(self):
        with self._lock:
            return self._R.copy()

    @property
    def position(self):
        with self._lock:
            return self._pos.copy()

    @property
    def is_ready(self):
        return self._ready

    def correct_position(self, correction: np.ndarray):
        """ICP 成功后校正 IMU 位移漂移"""
        with self._lock:
            self._pos += correction
            self._vel *= 0.8

    def _update_position(self, accel_raw: np.ndarray, dt: float):
        """改进版位移积分"""
        accel_world = self._R @ accel_raw
        accel_linear = accel_world - np.array([0.0, 0.0, 9.81])

        # 高通滤波
        alpha = 0.95
        if not hasattr(self, '_accel_hp'):
            self._accel_hp = np.zeros(3)
            self._accel_prev = accel_linear.copy()

        self._accel_hp = alpha * (self._accel_hp + accel_linear - self._accel_prev)
        self._accel_prev = accel_linear.copy()

        # 静止检测
        self._accel_buf.append(np.linalg.norm(self._accel_hp))
        if len(self._accel_buf) > 20:
            self._accel_buf.pop(0)

        is_static = (len(self._accel_buf) == 20 and np.std(self._accel_buf) < 0.15)

        if is_static:
            self._vel[:] = 0.0
            self._pos *= 0.95
        else:
            self._vel += self._accel_hp * dt
            self._pos += self._vel * dt

    def _loop(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, bytesize=8, stopbits=1, parity='N', timeout=1)
            parser = FrameParser()
            while self._running:
                raw = ser.read(256)
                if not raw:
                    continue

                now = time.time()

                for func, payload in parser.feed(raw):
                    with self._lock:
                        dt = (now - self._last_t) if self._last_t else 0.02
                        dt = min(dt, 0.1)
                        self._last_t = now

                        if func == FUNC_QUAT and len(payload) >= 16:
                            w, x, y, z = struct.unpack_from('<ffff', payload, 0)
                            self._R = quat_to_rotation_matrix(w, x, y, z)
                            self._ready = True
                        elif func == FUNC_EULER and len(payload) >= 12:
                            roll, pitch, yaw = struct.unpack_from('<fff', payload, 0)
                            cr, sr = math.cos(roll), math.sin(roll)
                            cp, sp = math.cos(pitch), math.sin(pitch)
                            cy, sy = math.cos(yaw), math.sin(yaw)
                            self._R = np.array([
                                [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                [  -sp,          cp*sr,          cp*cr],
                            ], dtype=np.float64)
                            self._ready = True
                        elif func == FUNC_RAW and len(payload) >= 6:
                            ax = struct.unpack_from('<h', payload, 0)[0] * ACCEL_SCALE * 9.81
                            ay = struct.unpack_from('<h', payload, 2)[0] * ACCEL_SCALE * 9.81
                            az = struct.unpack_from('<h', payload, 4)[0] * ACCEL_SCALE * 9.81
                            self._update_position(np.array([ax, ay, az]), dt)
        except Exception as e:
            print(f"⚠️  IMU 线程异常: {e}")


# ═══════════════════════════════════════════════════════════
#  SLAM 主系统
# ═══════════════════════════════════════════════════════════

class RealTimeSLAMGPU:

    MIN_DEPTH   = 100
    MAX_DEPTH   = 3000
    MAX_HOLE_PX = 100
    GF_RADIUS   = 4
    GF_EPS      = 50.0
    EMA_ALPHA   = 0.5

    def __init__(self):
        print("=" * 60)

        # GPU
        if o3c.cuda.is_available():
            self.device = o3c.Device("CUDA:0")
            print("✅ 成功接管 NVIDIA GPU")
        else:
            self.device = o3c.Device("CPU:0")
            print("⚠️  未检测到 CUDA，使用 CPU")
        print("=" * 60)

        # 深度相机
        self.sdk_path = r"C:\毕设\奥比中光Win64-Release\奥比中光Win64-Release\sdk\libs"
        if os.path.exists(self.sdk_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(self.sdk_path)
        self.sdk = OrbbecCameraSDK(self.sdk_path)
        if not self.sdk.initialize() or not self.sdk.open_device(0):
            raise RuntimeError("❌ 深度相机初始化失败！")
        self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH)
        self.sdk.start_stream()

        # RGB 相机
        self.color_cam = cv2.VideoCapture(0)
        self.color_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.color_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 相机内参
        self.width, self.height = 640, 480
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5
        xx = np.arange(self.width)
        yy = np.arange(self.height)
        self.u_grid, self.v_grid = np.meshgrid(xx, yy)

        # ICP 参数
        self.icp_voxel_size    = 0.02
        self.global_voxel_size = 0.01
        self.max_corr_dist     = 0.15

        # 状态
        self.global_pcd_legacy     = o3d.geometry.PointCloud()
        self.global_transformation = o3c.Tensor(np.eye(4), dtype=o3c.float64, device=self.device)
        self.prev_icp_pcd_t        = None
        self.prev_imu_R            = None
        self.prev_imu_pos          = None
        self.ema_depth             = None
        self.success_count         = 0
        self.is_first_frame        = True

        # IMU
        self.imu = IMUReader()
        self.imu_enabled = self.imu.start()

        # 可视化
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Real-time SLAM + IMU", width=1024, height=768)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.15])
        opt.point_size = 2.5
    def _process_depth(self, raw: np.ndarray):
        d = raw.astype(np.float32)
        d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0

        valid = (d > 0).astype(np.uint8)
        valid_clean = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        d[valid_clean == 0] = 0.0
        d = remove_flying_pixels(d, depth_thresh=100.0, erode_px=2)
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
            self.ema_depth[(d_filled > 0) & (self.ema_depth == 0)] = d_filled[(d_filled > 0) & (self.ema_depth == 0)]
            self.ema_depth[d_filled == 0] = 0.0

        d_ema = self.ema_depth.copy()
        d_for_gf = _fill_all_holes_for_filter(d_ema)
        guide = cv2.normalize(d_for_gf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_gf = guided_filter(guide, d_for_gf, r=self.GF_RADIUS, eps=self.GF_EPS)
        d_gf[valid_mask == 0] = 0.0

        vmask = ((d_gf > self.MIN_DEPTH) & (d_gf < self.MAX_DEPTH)).astype(np.uint8)
        return d_gf, vmask

    def _depth_to_pcd_tensor(self, depth_mm: np.ndarray, vmask: np.ndarray, color_img: np.ndarray):
        sel = (depth_mm > 0) & (vmask > 0)
        if not sel.any():
            return None

        z = depth_mm[sel] / 1000.0
        x = (self.u_grid[sel] - self.cx) * z / self.fx
        y = -(self.v_grid[sel] - self.cy) * z / self.fy
        x = x

        pts = np.stack([x, y, z], axis=-1).astype(np.float32)
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        col = color_rgb[sel].astype(np.float32) / 255.0

        pcd_t = o3d.t.geometry.PointCloud()
        pcd_t.point.positions = o3c.Tensor(pts, dtype=o3c.float32, device=self.device)
        pcd_t.point.colors    = o3c.Tensor(col, dtype=o3c.float32, device=self.device)
        return pcd_t

    def grab_and_process(self):
        depth_res = self.sdk.capture_depth_frame(timeout=1000)
        ret, color_img = self.color_cam.read()
        if not depth_res or not ret:
            return None, None, None

        raw, _ = depth_res
        raw = np.fliplr(raw).copy()  # 加 .copy()
        color_img = np.fliplr(color_img).copy()  # 加 .copy()
        depth_mm, vmask = self._process_depth(raw)

        pcd_dense_t = self._depth_to_pcd_tensor(depth_mm, vmask, color_img)
        if pcd_dense_t is None or pcd_dense_t.point.positions.shape[0] < 500:
            return None, None, None

        pcd_icp = pcd_dense_t.voxel_down_sample(self.icp_voxel_size)
        pcd_icp.estimate_normals(max_nn=30, radius=self.icp_voxel_size * 2)

        return pcd_dense_t, pcd_icp, color_img

    def get_imu_init_transform(self) -> np.ndarray:
        """IMU 提供旋转+位移初值"""
        if not self.imu_enabled or not self.imu.is_ready:
            return np.eye(4)

        if self.prev_imu_R is None or self.prev_imu_pos is None:
            return np.eye(4)

        # 旋转增量
        delta_R = self.imu.rotation_matrix @ self.prev_imu_R.T

        # 位移增量
        delta_pos = self.imu.position - self.prev_imu_pos

        T = np.eye(4)
        T[:3, :3] = delta_R
        T[:3, 3] = delta_pos
        return T

    def run(self):
        print("👉 请拿起相机移动。按 Q 或 ESC 退出。")
        try:
            while True:
                pcd_dense_t, pcd_icp_t, color_preview = self.grab_and_process()
                if pcd_dense_t is None:
                    continue

                # IMU 信息显示
                if self.imu_enabled and self.imu.is_ready:
                    R = self.imu.rotation_matrix
                    pos = self.imu.position
                    roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                    pitch = math.degrees(math.asin(-R[2, 0]))
                    yaw   = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                    cv2.putText(color_preview,
                                f"IMU R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(color_preview,
                                f"Pos X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f}m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(color_preview, "IMU: N/A",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("GPU Tracking View", color_preview)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break# 第一帧
                if self.is_first_frame:
                    self.global_pcd_legacy += pcd_dense_t.to_legacy()
                    self.vis.add_geometry(self.global_pcd_legacy)
                    vc = self.vis.get_view_control()
                    vc.set_up([0, -1, 0]); vc.set_front([0, 0, -1])
                    vc.set_lookat([0, 0, 1]); vc.set_zoom(0.8)
                    self.prev_icp_pcd_t = pcd_icp_t.clone()
                    if self.imu_enabled and self.imu.is_ready:
                        self.prev_imu_R = self.imu.rotation_matrix
                        self.prev_imu_pos = self.imu.position
                    self.is_first_frame = False
                    continue

                # ICP 配准
                T_init = self.get_imu_init_transform()
                icp_result = o3d.t.pipelines.registration.icp(
                    source=pcd_icp_t,
                    target=self.prev_icp_pcd_t,
                    max_correspondence_distance=self.max_corr_dist,
                    init_source_to_target=o3c.Tensor(T_init, dtype=o3c.float64, device=self.device),
                    estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )

                # 配准成功
                if icp_result.fitness > 0.30:
                    T_icp = icp_result.transformation.cpu().numpy()

                    # 用 ICP 结果校正 IMU 位移漂移
                    if self.imu_enabled and self.imu.is_ready:
                        imu_translation = T_init[:3, 3]
                        icp_translation = T_icp[:3, 3]
                        correction = icp_translation - imu_translation
                        self.imu.correct_position(correction * 0.3)  # 部分修正，避免过度

                    self.global_transformation = self.global_transformation.matmul(
                        icp_result.transformation)

                    pcd_dense_t.transform(self.global_transformation)
                    self.global_pcd_legacy += pcd_dense_t.to_legacy()
                    self.success_count += 1

                    if self.success_count % 4 == 0:
                        new_pcd = self.global_pcd_legacy.voxel_down_sample(self.global_voxel_size)
                        self.vis.remove_geometry(self.global_pcd_legacy, reset_bounding_box=False)
                        self.global_pcd_legacy = new_pcd
                        self.vis.add_geometry(self.global_pcd_legacy, reset_bounding_box=False)

                    self.vis.poll_events()
                    self.vis.update_renderer()
                    self.prev_icp_pcd_t = pcd_icp_t.clone()

                if self.imu_enabled and self.imu.is_ready:
                    self.prev_imu_R = self.imu.rotation_matrix
                    self.prev_imu_pos = self.imu.position

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        print("\n🛑 停止采集，正在保存点云...")
        self.imu.stop()
        self.sdk.cleanup()
        self.color_cam.release()
        cv2.destroyAllWindows()
        self.vis.destroy_window()

        if len(self.global_pcd_legacy.points) > 0:
            print("🧽 执行最终降噪...")
            self.global_pcd_legacy, _ = self.global_pcd_legacy.remove_statistical_outlier(
                nb_neighbors=30, std_ratio=1.5)

            output_dir = os.path.join(os.getcwd(), "outdata")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "realtime_slam_result.ply")
            o3d.io.write_point_cloud(output_path, self.global_pcd_legacy)
            print(f"🎉 已保存至: {output_path}")


if __name__ == "__main__":
    slam = RealTimeSLAMGPU()
    slam.run()