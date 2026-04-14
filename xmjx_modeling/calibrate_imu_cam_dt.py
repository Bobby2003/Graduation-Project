import os
import time
import struct
import threading
from collections import deque
from bisect import bisect_left

from datetime import datetime
import shutil

import numpy as np
import serial
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from realtime_scanner import DepthScanner

class IMUCameraDTCalibratorDual:
    def __init__(self):
        self.scanner = DepthScanner()
        self.input_color_is_bgr = False
        self.depth_scale = None
        self.depth_trunc = 5.0

        # 与 rot / TSDF 统一
        width, height = 640, 480
        fx, fy = 525.0, 525.0
        cx, cy = 319.5, 239.5
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # IMU
        self.serial_port = "COM7"
        self.baud_rate = 115200
        self.ser = None
        self.is_imu_running = False
        self.imu_thread = None
        self.lock = threading.Lock()

        # 双映射零点
        self.initial_inv_new = None
        self.initial_inv_legacy = None

        # IMU缓存（同一时间戳，两套旋转）
        self.imu_buf_maxlen = 50000
        self.imu_ts_buf = deque(maxlen=self.imu_buf_maxlen)
        self.imu_R_buf_new = deque(maxlen=self.imu_buf_maxlen)
        self.imu_R_buf_legacy = deque(maxlen=self.imu_buf_maxlen)

        # 相机样本参数
        self.min_info_trace = 3e5
        self.min_rot_deg = 1.0
        self.max_rot_deg = 12.0
        self.max_cam_pairs = 10000

        # 视觉里程计缓存
        self.prev_rgbd = None
        self.prev_cam_ts = None
        self.total_frames = 0
        self.good_odom = 0

        # 相机增量样本: (t0, t1, dR_cam, info_trace)
        self.cam_samples = []

        # reject统计
        self.rej_info = 0
        self.rej_cam_rot = 0

        # 路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.npy_dir = os.path.join(self.script_dir, "npy")
        os.makedirs(self.npy_dir, exist_ok=True)

        self.r_ci_path = os.path.join(self.npy_dir, "imu_to_cam_rot.npy")
        self.save_dt_path = os.path.join(self.npy_dir, "imu_to_cam_dt.npy")
        self.save_flag_path = os.path.join(self.npy_dir, "imu_euler_legacy_flag.npy")
        self.save_raw_path = os.path.join(self.npy_dir, "imu_cam_dt_scan_dual.npz")
        self.flag_path = os.path.join(self.npy_dir, "imu_euler_legacy_flag.npy")
        
        self.use_legacy_euler_mapping = False
        
        self.R_ci = np.eye(3, dtype=np.float64)

        self.init_rot_path = os.path.join(self.npy_dir, "imu_to_cam_rot_init.npy")
        self.load_imu_cam_extrinsic_for_dt()

        self.load_imu_legacy_flag()

        self.save_rot_path = os.path.join(self.npy_dir, "imu_to_cam_rot.npy")

        
        

    # ---------- 工具 ----------
    def load_imu_legacy_flag(self):
        if os.path.exists(self.flag_path):
            try:
                v = int(np.load(self.flag_path))
                self.use_legacy_euler_mapping = (v == 1)
                print(f"✅ 已加载欧拉映射标志: legacy={self.use_legacy_euler_mapping}")
            except Exception as e:
                print(f"⚠️ legacy flag 加载失败: {e}，默认 legacy=False")
                self.use_legacy_euler_mapping = False
        else:
            print("⚠️ 未找到 imu_euler_legacy_flag.npy，默认 legacy=False")
            self.use_legacy_euler_mapping = False

    def load_imu_cam_extrinsic_for_dt(self):
    #dt 求解时优先使用粗对齐外参 imu_to_cam_rot_init.npy，
    #避免 dt <-> rot 相互耦合导致第一次迭代跑偏。
        if os.path.exists(self.init_rot_path):
            try:
                R_ci = np.load(self.init_rot_path)
                R_ci = self.normalize_rotation(R_ci)
                if self.is_valid_rotation(R_ci):
                    self.R_ci = R_ci
                    print(f"✅ dt求解使用粗对齐外参: {self.init_rot_path}")
                    print(self.R_ci)
                    return
                else:
                    print("⚠️ imu_to_cam_rot_init.npy 无效，回退到 imu_to_cam_rot.npy")
            except Exception as e:
                print(f"⚠️ imu_to_cam_rot_init.npy 加载失败: {e}，回退到 imu_to_cam_rot.npy")

        # 回退到正式 rot
        if os.path.exists(self.save_rot_path):
            try:
                R_ci = np.load(self.save_rot_path)
                R_ci = self.normalize_rotation(R_ci)
                if self.is_valid_rotation(R_ci):
                    self.R_ci = R_ci
                    print(f"✅ dt求解回退使用正式外参: {self.save_rot_path}")
                    print(self.R_ci)
                else:
                    print("⚠️ imu_to_cam_rot.npy 无效，使用单位阵")
            except Exception as e:
                print(f"⚠️ imu_to_cam_rot.npy 加载失败: {e}，使用单位阵")
        else:
            print("⚠️ 未找到 imu_to_cam_rot_init.npy / imu_to_cam_rot.npy，使用单位阵")

    def normalize_rotation(self, Rm):
        U, _, Vt = np.linalg.svd(Rm)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    def is_valid_rotation(self, Rm):
        if Rm.shape != (3, 3):
            return False
        should_be_I = Rm.T @ Rm
        det = np.linalg.det(Rm)
        return np.allclose(should_be_I, np.eye(3), atol=1e-3) and np.isclose(det, 1.0, atol=1e-3)

    def rot_deg(self, Rm):
        rv = R.from_matrix(self.normalize_rotation(Rm)).as_rotvec()
        return float(np.degrees(np.linalg.norm(rv)))

    def load_imu_cam_extrinsic(self):
        if os.path.exists(self.r_ci_path):
            try:
                rci = np.load(self.r_ci_path)
                rci = self.normalize_rotation(rci)
                if self.is_valid_rotation(rci):
                    self.R_ci = rci
                    print(f"✅ 已加载 R_ci: {self.r_ci_path}")
                else:
                    print("⚠️ R_ci 无效，使用单位阵")
            except Exception as e:
                print(f"⚠️ 加载 R_ci 失败: {e}，使用单位阵")
        else:
            print("⚠️ 未找到 imu_to_cam_rot.npy，使用单位阵")

    # ---------- 图像预处理 ----------
    def preprocess_frame(self, color_np, depth_np):
        if color_np is None or depth_np is None:
            return None

        if self.input_color_is_bgr and color_np.ndim == 3 and color_np.shape[2] == 3:
            color_np = color_np[:, :, ::-1].copy()

        color_np = np.ascontiguousarray(color_np)
        depth_np = np.ascontiguousarray(depth_np)

        if self.depth_scale is None:
            valid = depth_np[depth_np > 0]
            vmax = float(np.percentile(valid, 95)) if valid.size > 0 else 0.0
            self.depth_scale = 1000.0 if vmax > 20 else 1.0
            print(f"[DEBUG] auto depth_scale={self.depth_scale}, p95={vmax:.3f}")

        if depth_np.dtype not in (np.uint16, np.float32, np.float64):
            depth_np = depth_np.astype(np.float32)

        if np.issubdtype(depth_np.dtype, np.floating):
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        h, w = depth_np.shape[:2]
        if (w != self.intrinsic.width) or (h != self.intrinsic.height):
            fx, fy = self.intrinsic.get_focal_length()
            cx, cy = self.intrinsic.get_principal_point()
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            print(f"[INFO] intrinsic resized to {w}x{h}")

        valid_ratio = float(np.count_nonzero(depth_np)) / depth_np.size
        if valid_ratio < 0.01:
            return None

        color_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )
        return rgbd

    # ---------- IMU ----------
    def read_exact(self, ser, n):
        data = b""
        while len(data) < n and self.is_imu_running:
            chunk = ser.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data if len(data) == n else None

    def start_imu(self):
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
        self.is_imu_running = True
        self.imu_thread = threading.Thread(target=self._imu_worker, daemon=True)
        self.imu_thread.start()
        print("✅ IMU线程启动")

    def stop_imu(self):
        self.is_imu_running = False
        if self.imu_thread and self.imu_thread.is_alive():
            self.imu_thread.join(timeout=0.5)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def euler_to_R(self, r_rad, p_rad, y_rad, legacy=False):
        if legacy:
            Rm = R.from_euler("xyz", [-p_rad, y_rad, -r_rad], degrees=False).as_matrix()
        else:
            Rm = R.from_euler("xyz", [r_rad, p_rad, y_rad], degrees=False).as_matrix()
        return self.normalize_rotation(Rm)

    def _imu_worker(self):
        while self.is_imu_running:
            try:
                h1 = self.read_exact(self.ser, 1)
                if h1 != b"\x7e":
                    continue
                h2 = self.read_exact(self.ser, 1)
                if h2 != b"\x23":
                    continue

                lb = self.read_exact(self.ser, 1)
                if lb is None:
                    continue
                length = lb[0]
                if length < 3:
                    continue

                payload = self.read_exact(self.ser, length - 3)
                if payload is None or len(payload) < 13:
                    continue
                if payload[0] != 0x26:
                    continue

                data = payload[1:13]
                r_rad = struct.unpack("<f", data[0:4])[0]
                p_rad = struct.unpack("<f", data[4:8])[0]
                y_rad = struct.unpack("<f", data[8:12])[0]

                R_new = self.euler_to_R(r_rad, p_rad, y_rad, legacy=False)
                R_legacy = self.euler_to_R(r_rad, p_rad, y_rad, legacy=True)

                if (not self.is_valid_rotation(R_new)) or (not self.is_valid_rotation(R_legacy)):
                    continue

                ts = time.time()

                with self.lock:
                    if self.initial_inv_new is None:
                        self.initial_inv_new = np.linalg.inv(R_new)
                        print("🎯 IMU零点(new)建立")
                    if self.initial_inv_legacy is None:
                        self.initial_inv_legacy = np.linalg.inv(R_legacy)
                        print("🎯 IMU零点(legacy)建立")

                    rel_new = self.normalize_rotation(self.initial_inv_new @ R_new)
                    rel_legacy = self.normalize_rotation(self.initial_inv_legacy @ R_legacy)

                    self.imu_ts_buf.append(ts)
                    self.imu_R_buf_new.append(rel_new)
                    self.imu_R_buf_legacy.append(rel_legacy)

            except Exception:
                time.sleep(0.002)

    # ---------- 旋转插值 ----------
    def interpolate_rotation(self, R0, R1, alpha):
        dR = self.normalize_rotation(R0.T @ R1)
        rv = R.from_matrix(dR).as_rotvec()
        return self.normalize_rotation(R0 @ R.from_rotvec(alpha * rv).as_matrix())

    def get_imu_rotation_at(self, t_query, mode="new"):
        with self.lock:
            ts_list = list(self.imu_ts_buf)
            if mode == "new":
                R_list = list(self.imu_R_buf_new)
            else:
                R_list = list(self.imu_R_buf_legacy)

        n = len(ts_list)
        if n < 2:
            return None
        if t_query < ts_list[0] or t_query > ts_list[-1]:
            return None

        idx = bisect_left(ts_list, t_query)
        if idx == 0:
            return R_list[0]
        if idx >= n:
            return R_list[-1]

        t0, t1 = ts_list[idx - 1], ts_list[idx]
        R0, R1 = R_list[idx - 1], R_list[idx]
        if t1 <= t0:
            return R0

        alpha = float(np.clip((t_query - t0) / (t1 - t0), 0.0, 1.0))
        return self.interpolate_rotation(R0, R1, alpha)

    # ---------- dt评估 ----------
    def evaluate_dt(self, dt, mode="new"):
        residuals = []
        used = 0

        for (t0, t1, dR_cam, info_trace) in self.cam_samples:
            cam_deg = self.rot_deg(dR_cam)
            if cam_deg < 1.0:
                continue

            R_imu_0 = self.get_imu_rotation_at(t0 + dt, mode=mode)
            R_imu_1 = self.get_imu_rotation_at(t1 + dt, mode=mode)
            if R_imu_0 is None or R_imu_1 is None:
                continue

            dR_imu = self.normalize_rotation(R_imu_0.T @ R_imu_1)
            dR_cam_pred = self.normalize_rotation(self.R_ci @ dR_imu @ self.R_ci.T)

            dR_err = self.normalize_rotation(dR_cam_pred.T @ dR_cam)
            err_deg = self.rot_deg(dR_err)

            residuals.append(err_deg)
            used += 1

        if used < 20:
            return np.inf, np.inf, used

        residuals = np.array(residuals, dtype=np.float64)
        med = float(np.median(residuals))
        p95 = float(np.percentile(residuals, 95))
        return med, p95, used

    def solve_best_dt_for_mode(self, mode="new"):
        coarse_dts = np.arange(-0.30, 0.301, 0.01)
        fine_span = 0.06
        fine_step = 0.002
        coarse_stats = []
        for dt in coarse_dts:
            med, p95, used = self.evaluate_dt(dt, mode=mode)
            coarse_stats.append((dt, med, p95, used))
        coarse_stats = np.array(coarse_stats, dtype=np.float64)

        best_idx = int(np.argmin(coarse_stats[:, 1]))
        best_coarse_dt = float(coarse_stats[best_idx, 0])

        fine_min = best_coarse_dt - 0.05
        fine_max = best_coarse_dt + 0.05
        fine_dts = np.arange(best_coarse_dt - fine_span, best_coarse_dt + fine_span + 1e-9, fine_step)
        fine_stats = []
        for dt in fine_dts:
            med, p95, used = self.evaluate_dt(dt, mode=mode)
            fine_stats.append((dt, med, p95, used))
        fine_stats = np.array(fine_stats, dtype=np.float64)

        best_idx_f = int(np.argmin(fine_stats[:, 1]))
        best_dt = float(fine_stats[best_idx_f, 0])
        best_med = float(fine_stats[best_idx_f, 1])
        best_p95 = float(fine_stats[best_idx_f, 2])
        best_used = int(fine_stats[best_idx_f, 3])

        return {
            "mode": mode,
            "best_dt": best_dt,
            "best_med": best_med,
            "best_p95": best_p95,
            "best_used": best_used,
            "coarse_stats": coarse_stats,
            "fine_stats": fine_stats,
            "fine_edge": (best_idx_f == 0 or best_idx_f == len(fine_stats)-1),
        }

    # ---------- 主流程 ----------
    def run(self):
        if not self.scanner.init():
            print("❌ 相机初始化失败")
            return

        try:
            self.start_imu()
        except Exception as e:
            print(f"❌ IMU 打开失败: {e}")
            self.scanner.close()
            return

        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = 0.10

        print("\n===== 开始采集 dt 标定样本（双映射）=====")
        print("动作：连续多轴旋转 60~120 秒（yaw/pitch/roll）")
        print("按 Ctrl+C 结束\n")

        last_print = time.time()

        try:
            while True:
                data = self.scanner.get_rgb_and_depth()
                if data is None:
                    continue

                if len(self.imu_ts_buf) < 50:
                    continue

                color_np, depth_np = data
                rgbd = self.preprocess_frame(color_np, depth_np)
                if rgbd is None:
                    continue

                cam_ts = time.time()
                self.total_frames += 1

                if self.prev_rgbd is None:
                    self.prev_rgbd = rgbd
                    self.prev_cam_ts = cam_ts
                    continue

                success, delta, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    self.prev_rgbd,
                    rgbd,
                    self.intrinsic,
                    np.eye(4, dtype=np.float64),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option
                )

                if success:
                    self.good_odom += 1
                    dR_cam = self.normalize_rotation(delta[:3, :3])
                    od_deg = self.rot_deg(dR_cam)
                    info_trace = float(np.trace(info)) if info is not None else 0.0

                    ok = True
                    if info_trace < self.min_info_trace:
                        self.rej_info += 1
                        ok = False
                    if not (self.min_rot_deg <= od_deg <= self.max_rot_deg):
                        self.rej_cam_rot += 1
                        ok = False

                    if ok and len(self.cam_samples) < self.max_cam_pairs:
                        self.cam_samples.append((self.prev_cam_ts, cam_ts, dR_cam, info_trace))

                self.prev_rgbd = rgbd
                self.prev_cam_ts = cam_ts

                now = time.time()
                if now - last_print > 1.0:
                    print(f"[STAT] frames={self.total_frames}, good_odom={self.good_odom}, "
                          f"pairs={len(self.cam_samples)}, rej_info={self.rej_info}, rej_cam_rot={self.rej_cam_rot}, "
                          f"imu_buf={len(self.imu_ts_buf)}")
                    last_print = now

        except KeyboardInterrupt:
            print("\n🛑 用户停止采集，开始求解 dt ...")

        finally:
            self.stop_imu()
            self.scanner.close()

        if len(self.cam_samples) < 40:
            print(f"❌ 样本不足: {len(self.cam_samples)}（建议>=200）")
            return

        print("\n===== 求解 new 映射 =====")
        ret_new = self.solve_best_dt_for_mode("new")
        print(f"[new] dt={ret_new['best_dt']*1000:.1f} ms, med={ret_new['best_med']:.2f}°, "
              f"p95={ret_new['best_p95']:.2f}°, used={ret_new['best_used']}")

        print("\n===== 求解 legacy 映射 =====")
        ret_legacy = self.solve_best_dt_for_mode("legacy")
        print(f"[legacy] dt={ret_legacy['best_dt']*1000:.1f} ms, med={ret_legacy['best_med']:.2f}°, "
              f"p95={ret_legacy['best_p95']:.2f}°, used={ret_legacy['best_used']}")

        # 自动选更优
        pick_legacy = (ret_legacy["best_med"] < ret_new["best_med"])
        best = ret_legacy if pick_legacy else ret_new

        print("\n========== 最终选择 ==========")
        print(f"mode = {'legacy(True)' if pick_legacy else 'new(False)'}")
        print(f"best dt = {best['best_dt']*1000:.1f} ms")
        print(f"median = {best['best_med']:.2f} deg")
        print(f"p95    = {best['best_p95']:.2f} deg")
        print(f"used   = {best['best_used']}")
        if best["fine_edge"]:
            print("⚠️ 最优落在细扫边界，建议扩大范围重扫。")
        print("=============================")

        # 保存
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) dt 历史版 + latest
        dt_hist_path = os.path.join(self.npy_dir, f"imu_to_cam_dt_{ts}.npy")
        np.save(dt_hist_path, np.array(best["best_dt"], dtype=np.float64))
        shutil.copyfile(dt_hist_path, self.save_dt_path)

        print(f"✅ dt 历史版: {dt_hist_path}")
        print(f"✅ dt 最新版: {self.save_dt_path}")

        # 2) flag 历史版 + latest
        picked_legacy = 1 if (best["mode"] == "legacy") else 0
        flag_hist_path = os.path.join(self.npy_dir, f"imu_euler_legacy_flag_{ts}.npy")
        np.save(flag_hist_path, np.array(picked_legacy, dtype=np.int32))
        shutil.copyfile(flag_hist_path, self.save_flag_path)

        print(f"✅ flag 历史版: {flag_hist_path}")
        print(f"✅ flag 最新版: {self.save_flag_path}")

        # 3) raw 历史版 + latest
        raw_hist_path = os.path.join(self.npy_dir, f"imu_cam_dt_scan_dual_{ts}.npz")
        np.savez(
            raw_hist_path,
            best_dt=np.array(best["best_dt"], dtype=np.float64),
            best_med=np.array(best["best_med"], dtype=np.float64),
            best_p95=np.array(best["best_p95"], dtype=np.float64),
            best_used=np.array(best["best_used"], dtype=np.int32),
        )
        shutil.copyfile(raw_hist_path, self.save_raw_path)

        print(f"✅ raw 历史版: {raw_hist_path}")
        print(f"✅ raw 最新版: {self.save_raw_path}")


if __name__ == "__main__":
    app = IMUCameraDTCalibratorDual()
    app.run()