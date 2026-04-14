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

class IMUCameraRotCalibrator:
    def __init__(self):
        # ====== 相机 ======
        self.scanner = DepthScanner()
        self.input_color_is_bgr = False
        self.depth_scale = None
        self.depth_trunc = 5.0

        self.last_imu_R = None
        self.max_imu_jump_deg = 20.0

        # !!! 与 dt / TSDF 统一（建议三脚本一致）
        width, height = 640, 480
        fx, fy = 525.0, 525.0
        cx, cy = 319.5, 239.5
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # ====== IMU 串口 ======
        self.serial_port = "COM7"
        self.baud_rate = 115200
        self.ser = None
        self.is_imu_running = False
        self.imu_thread = None
        self.lock = threading.Lock()

        self.initial_imu_inv = None

        # 是否使用旧欧拉映射（务必和 dt / TSDF 一致）
        self.use_legacy_euler_mapping = False

        # ====== IMU缓存（时间戳+旋转）======
        self.imu_buf_maxlen = 50000
        self.imu_ts_buf = deque(maxlen=self.imu_buf_maxlen)  # float ts
        self.imu_R_buf = deque(maxlen=self.imu_buf_maxlen)   # (3,3)

        # ====== 采样阈值（可调） ======
        self.min_info_trace = 3e5
        self.min_rot_deg = 0.8
        self.max_rot_deg = 12.0
        self.max_pairs = 8000

        # ====== 数据缓存 ======
        self.prev_rgbd = None
        self.prev_cam_ts = None

        self.cam_dRs = []
        self.imu_dRs = []
        self.weights = []

        self.total_frames = 0
        self.good_odom = 0
        self.rej_info = 0
        self.rej_cam_rot = 0
        self.rej_imu_unavailable = 0
        self.rej_imu_rot = 0

        # ====== 输出路径 ======
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.npy_dir = os.path.join(self.script_dir, "npy")
        os.makedirs(self.npy_dir, exist_ok=True)

        self.save_rot_path = os.path.join(self.npy_dir, "imu_to_cam_rot.npy")
        self.save_raw_path = os.path.join(self.npy_dir, "imu_cam_calib_pairs.npz")
        self.flag_path = os.path.join(self.npy_dir, "imu_euler_legacy_flag.npy")

        # dt 输入路径（来自 calibrate_imu_cam_dt.py）
        self.dt_path = os.path.join(self.npy_dir, "imu_to_cam_dt.npy")

        # ===== 临时强制 dt=0，用于粗对齐 rot =====
        self.imu_delay = 0.0
        print("⚠️ 当前为粗对齐模式：强制使用 dt = 0 ms")

        self.load_imu_legacy_flag()

    # ----------------------
    # 基础工具
    # ----------------------
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

    def load_imu_cam_dt(self):
        if os.path.exists(self.dt_path):
            try:
                dt = float(np.load(self.dt_path))
                self.imu_delay = dt
                print(f"✅ 已加载 IMU->CAM 时间偏移 dt: {dt * 1000:.1f} ms")
            except Exception as e:
                print(f"⚠️ dt 加载失败: {e}，使用 dt=0")
                self.imu_delay = 0.0
        else:
            print("⚠️ 未找到 imu_to_cam_dt.npy，使用 dt=0")

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

    # ----------------------
    # 图像预处理
    # ----------------------
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
            print(f"[DEBUG] depth p95={vmax:.3f}, auto depth_scale={self.depth_scale}")

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

    # ----------------------
    # IMU 串口线程
    # ----------------------
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
        print("✅ IMU 线程启动")

    def stop_imu(self):
        self.is_imu_running = False
        if self.imu_thread is not None and self.imu_thread.is_alive():
            self.imu_thread.join(timeout=0.5)
        if self.ser and self.ser.is_open:
            self.ser.close()

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

                if self.use_legacy_euler_mapping:
                    cur_R = R.from_euler("xyz", [-p_rad, y_rad, -r_rad], degrees=False).as_matrix()
                else:
                    cur_R = R.from_euler("xyz", [r_rad, p_rad, y_rad], degrees=False).as_matrix()

                cur_R = self.normalize_rotation(cur_R)
                if not self.is_valid_rotation(cur_R):
                    continue

                ts = time.time()

                with self.lock:
                    if self.initial_imu_inv is None:
                        self.initial_imu_inv = np.linalg.inv(cur_R)
                        print("🎯 IMU 零点完成")

                    rel_R = self.normalize_rotation(self.initial_imu_inv @ cur_R)
                    self.imu_ts_buf.append(ts)
                    self.imu_R_buf.append(rel_R)

            except Exception:
                time.sleep(0.002)

    # ----------------------
    # IMU 时间插值
    # ----------------------
    def interpolate_rotation(self, R0, R1, alpha):
        dR = self.normalize_rotation(R0.T @ R1)
        rv = R.from_matrix(dR).as_rotvec()
        return self.normalize_rotation(R0 @ R.from_rotvec(alpha * rv).as_matrix())

    def get_imu_rotation_at(self, t_query):
        with self.lock:
            ts_list = list(self.imu_ts_buf)
            R_list = list(self.imu_R_buf)

        n = len(ts_list)
        if n < 2:
            return None

        # 查询点在缓存范围外，不外推
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

    # ----------------------
    # 外参求解
    # ----------------------
    def solve_rotation_extrinsic(self, cam_dRs, imu_dRs, weights=None):
        """
        解 R_ci，使 dR_cam ≈ R_ci * dR_imu * R_ci^T
        """
        a_list, b_list, w_list = [], [], []

        for i, (Rc, Ri) in enumerate(zip(cam_dRs, imu_dRs)):
            wc = R.from_matrix(Rc).as_rotvec()
            wi = R.from_matrix(Ri).as_rotvec()
            ac, ai = np.linalg.norm(wc), np.linalg.norm(wi)
            if ac < np.deg2rad(0.8) or ai < np.deg2rad(0.8):
                continue

            b_list.append(wc / ac)  # cam axis
            a_list.append(wi / ai)  # imu axis
            w_list.append(1.0 if weights is None else float(weights[i]))

        if len(a_list) < 20:
            raise RuntimeError(f"有效样本不足: {len(a_list)}")

        A = np.asarray(a_list)  # Nx3
        B = np.asarray(b_list)  # Nx3
        W = np.asarray(w_list).reshape(-1, 1)

        H = (W * B).T @ A
        U, _, Vt = np.linalg.svd(H)
        R_ci = U @ Vt
        if np.linalg.det(R_ci) < 0:
            U[:, -1] *= -1
            R_ci = U @ Vt

        return self.normalize_rotation(R_ci)

    def residual_deg(self, R_ci, Rc, Ri):
        pred = self.normalize_rotation(R_ci @ Ri @ R_ci.T)
        d = self.normalize_rotation(pred.T @ Rc)
        return self.rot_deg(d)

    def robust_solve(self, cam_dRs, imu_dRs, weights):
        # 第一次拟合
        R_ci_1 = self.solve_rotation_extrinsic(cam_dRs, imu_dRs, weights)
        res = np.array([self.residual_deg(R_ci_1, Rc, Ri) for Rc, Ri in zip(cam_dRs, imu_dRs)])

        # 去掉极端离群，再拟合一次
        inlier = res < 15.0
        if np.sum(inlier) >= 20:
            cam2 = [cam_dRs[i] for i in range(len(cam_dRs)) if inlier[i]]
            imu2 = [imu_dRs[i] for i in range(len(imu_dRs)) if inlier[i]]
            w2 = [weights[i] for i in range(len(weights)) if inlier[i]]

            R_ci_2 = self.solve_rotation_extrinsic(cam2, imu2, w2)
            res2 = np.array([self.residual_deg(R_ci_2, Rc, Ri) for Rc, Ri in zip(cam2, imu2)])
            return R_ci_2, res2, int(np.sum(inlier)), len(cam_dRs)
        else:
            return R_ci_1, res, len(cam_dRs), len(cam_dRs)

    # ----------------------
    # 运行
    # ----------------------
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

        print("\n===== 开始采集外参样本（dt 对齐版）=====")
        print(f"当前 dt = {self.imu_delay * 1000:.1f} ms")
        print("动作要求：慢速多轴旋转(yaw/pitch/roll)，少平移，建议 60~120 秒")
        print("按 Ctrl+C 结束并自动求解\n")

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

                # 尽量靠近取帧时间
                cam_ts = time.time()
                self.total_frames += 1

                if self.prev_rgbd is None:
                    self.prev_rgbd = rgbd
                    self.prev_cam_ts = cam_ts
                    continue

                # 视觉增量
                success, delta, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    self.prev_rgbd,
                    rgbd,
                    self.intrinsic,
                    np.eye(4, dtype=np.float64),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option
                )

                if success and info is not None:
                    self.good_odom += 1

                    dR_cam = self.normalize_rotation(delta[:3, :3])
                    od_deg = self.rot_deg(dR_cam)
                    trans = float(np.linalg.norm(delta[:3, 3]))
                    info_trace = float(np.trace(info))

                    ok = True

                    # 可疑帧：平移偏大但旋转很小，通常不是纯旋转样本
                    if trans > 0.01 and od_deg < 2.0:
                        self.rej_cam_rot += 1
                        ok = False

                    if info_trace < self.min_info_trace:
                        self.rej_info += 1
                        ok = False

                    if not (self.min_rot_deg <= od_deg <= self.max_rot_deg):
                        self.rej_cam_rot += 1
                        ok = False

                    # IMU 按时间对齐取姿态：t0+dt, t1+dt
                    if ok:
                        t0 = self.prev_cam_ts + self.imu_delay
                        t1 = cam_ts + self.imu_delay
                        R0 = self.get_imu_rotation_at(t0)
                        R1 = self.get_imu_rotation_at(t1)

                        if R0 is None or R1 is None:
                            self.rej_imu_unavailable += 1
                            ok = False
                        else:
                            dR_imu = self.normalize_rotation(R0.T @ R1)
                            imu_deg = self.rot_deg(dR_imu)

                            if not (self.min_rot_deg <= imu_deg <= self.max_rot_deg):
                                self.rej_imu_rot += 1
                                ok = False

                            if self.last_imu_R is not None:
                                jump = self.rot_deg(self.normalize_rotation(self.last_imu_R.T @ R1))
                                if jump > self.max_imu_jump_deg:
                                    self.rej_imu_rot += 1
                                    ok = False

                            self.last_imu_R = R1.copy()

                    if ok and len(self.cam_dRs) < self.max_pairs:
                        self.cam_dRs.append(dR_cam)
                        self.imu_dRs.append(dR_imu)
                        weight = info_trace * min(od_deg, 5.0)
                        self.weights.append(weight)

                # 更新上一帧
                self.prev_rgbd = rgbd
                self.prev_cam_ts = cam_ts

                now = time.time()
                if now - last_print > 1.0:
                    print(
                        f"[STAT] total={self.total_frames}, good_odom={self.good_odom}, pairs={len(self.cam_dRs)}, "
                        f"rej_info={self.rej_info}, rej_cam_rot={self.rej_cam_rot}, "
                        f"rej_imu_na={self.rej_imu_unavailable}, rej_imu_rot={self.rej_imu_rot}, "
                        f"imu_buf={len(self.imu_ts_buf)}"
                    )
                    last_print = now

        except KeyboardInterrupt:
            print("\n🛑 用户停止采集，开始求解...")

        finally:
            self.stop_imu()
            self.scanner.close()

        # 保存原始样本
        if len(self.cam_dRs) > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_hist_path = os.path.join(self.npy_dir, f"imu_cam_calib_pairs_{ts}.npz")

            np.savez(
                raw_hist_path,
                cam_dRs=np.asarray(self.cam_dRs),
                imu_dRs=np.asarray(self.imu_dRs),
                weights=np.asarray(self.weights, dtype=np.float64),
                imu_delay=np.array(self.imu_delay, dtype=np.float64),
                use_legacy_euler_mapping=np.array(int(self.use_legacy_euler_mapping), dtype=np.int32),
            )

            shutil.copyfile(raw_hist_path, self.save_raw_path)

            print(f"✅ raw 历史版: {raw_hist_path}")
            print(f"✅ raw 最新版: {self.save_raw_path}")

        # 求解
        if len(self.cam_dRs) < 20:
            print(f"❌ 样本不足，无法求解。当前 pairs={len(self.cam_dRs)}")
            return

        try:
            R_ci, residuals, nin, ntotal = self.robust_solve(self.cam_dRs, self.imu_dRs, self.weights)
            med = float(np.median(residuals))
            p95 = float(np.percentile(residuals, 95))

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            rot_hist_path = os.path.join(self.npy_dir, f"imu_to_cam_rot_{ts}.npy")
            rot_init_path = os.path.join(self.npy_dir, "imu_to_cam_rot_init.npy")

            np.save(rot_hist_path, R_ci)
            shutil.copyfile(rot_hist_path, self.save_rot_path)
            np.save(rot_init_path, R_ci)

            print("\n========== 标定结果 ==========")
            print(f"总样本: {ntotal}, 内点: {nin}")
            print(f"median residual: {med:.2f} deg")
            print(f"p95 residual   : {p95:.2f} deg")
            print(f" rot 历史版: {rot_hist_path}")
            print(f" rot 最新版: {self.save_rot_path}")
            print(f" rot 初始粗对齐版: {rot_init_path}")
            print("R_ci (IMU->CAM) =")
            print(R_ci)
            print("==============================")

        except Exception as e:
            print(f"❌ 求解失败: {e}")

if __name__ == "__main__":
    app = IMUCameraRotCalibrator()
    app.run()