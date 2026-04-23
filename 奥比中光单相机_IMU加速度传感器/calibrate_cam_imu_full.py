import cv2
import numpy as np
import serial
import serial.tools.list_ports
import struct
import math
import time
import threading
import json
import os

# ============================================================
# 参数区
# ============================================================

# 你的 6x6 如果是“内角点”，填 (6, 6)
# 如果是“方格数”，应该填 (5, 5)
CHESSBOARD_SIZE = (5, 5)
SQUARE_SIZE = 0.025  # 米

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 先用近似内参；如果有真实标定值请替换
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5

CAMERA_MATRIX = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

BAUDRATE = 115200
IMU_OUTPUT_HZ = 100
# 姿态来源开关：建议只保留一种，避免四元数和欧拉角同时进入 att_samples
USE_QUAT_ATTITUDE = True
USE_EULER_ATTITUDE = False

# 前几秒静止用于估计 gyro bias
STATIC_BIAS_SECONDS = 3.0

# 相机采样间隔，避免重复姿态太多
CAM_POSE_MIN_DT = 0.04

# 阶段1：旋转外参 + 时间偏移
ROT_MIN_CAMERA_POSES = 80
ROT_PAIR_STEP = 5
ROT_MIN_ANGLE_DEG = 3.0
ROT_MAX_ANGLE_DEG = 80.0
OFFSET_MIN_MS = -300
OFFSET_MAX_MS = 300
OFFSET_STEP_MS = 5
ROT_ACCEPT_ERR_DEG = 5.0
ROT_ACCEPT_MIN_PAIRS = 20

# 阶段2：杆臂/平移外参
LEVER_MIN_SYNC_FRAMES = 80
LEVER_ACCEPT_RESIDUAL = 1.8   # m/s^2
LEVER_MAX_NORM = 0.30         # 最大允许 30cm
GRAVITY_MIN = 7.0
GRAVITY_MAX = 12.5

# 平滑窗口
SMOOTH_WIN_CAM = 5
SMOOTH_WIN_IMU = 5

OUT_DIR = "calib_out"
OUT_JSON = "cam_imu_full_calib.json"

# ============================================================
# 串口协议
# 根据你给的协议文件
# ============================================================

HEADER_1, HEADER_2 = 0x7E, 0x23
FUNC_RAW = 0x04
FUNC_QUAT = 0x16
FUNC_EULER = 0x26

G = 9.80665
ACCEL_SCALE = 16.0 / 32767.0 * G
GYRO_SCALE = 2000.0 / 32767.0 * math.pi / 180.0
MAG_SCALE = 800.0 / 32767.0

def calc_checksum(data):
    return sum(data) & 0xFF

def build_set_rate_packet(hz=100):
    hz = int(max(10, min(100, hz)))
    pkt = bytearray([0x7E, 0x23, 0x07, 0x60, hz, 0x5F])
    pkt.append(calc_checksum(pkt))
    return bytes(pkt)

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

            func = frame[3]
            payload = frame[4:-1]
            frames.append((func, payload))

        return frames

def quat_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

def euler_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)

def parse_raw_payload(payload):
    if len(payload) < 18:
        return None

    ax, ay, az, gx, gy, gz, mx, my, mz = struct.unpack_from("<hhhhhhhhh", payload, 0)

    accel = np.array([ax, ay, az], dtype=np.float64) * ACCEL_SCALE
    gyro = np.array([gx, gy, gz], dtype=np.float64) * GYRO_SCALE
    mag = np.array([mx, my, mz], dtype=np.float64) * MAG_SCALE
    return accel, gyro, mag

def find_ch340_port():
    for p in serial.tools.list_ports.comports():
        desc = p.description.upper()
        if any(k in desc for k in ("CH340", "CH341", "USB-SERIAL", "SERIAL")):
            return p.device
    return None

class IMURecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.ser = None

        self.att_samples = []   # (t, R)
        self.raw_samples = []   # dict: {t, accel, gyro, mag}
        self.gyro_bias = np.zeros(3, dtype=np.float64)

        self.start_time = None

    def start(self, port=None, baudrate=115200):
        port = port or find_ch340_port()
        if port is None:
            print("未找到 IMU 串口（CH340/USB-SERIAL）")
            return False

        print(f"IMU 串口: {port}")
        self.running = True
        self.start_time = time.monotonic()

        th = threading.Thread(target=self._loop, args=(port, baudrate), daemon=True)
        th.start()
        return True

    def stop(self):
        self.running = False
        try:
            if self.ser is not None:
                self.ser.close()
        except:
            pass

    def _loop(self, port, baudrate):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(0.2)
            self.ser.write(build_set_rate_packet(IMU_OUTPUT_HZ))
            parser = FrameParser()

            while self.running:
                raw = self.ser.read(256)
                if not raw:
                    continue

                now = time.monotonic()

                for func, payload in parser.feed(raw):
                    if func == FUNC_RAW:
                        parsed = parse_raw_payload(payload)
                        if parsed is not None:
                            accel, gyro, mag = parsed
                            with self.lock:
                                self.raw_samples.append({
                                    "t": now,
                                    "accel": accel.copy(),
                                    "gyro": gyro.copy(),
                                    "mag": mag.copy()
                                })
                                if len(self.raw_samples) > 50000:
                                    self.raw_samples = self.raw_samples[-25000:]

                    elif USE_QUAT_ATTITUDE and func == FUNC_QUAT and len(payload) >= 16:
                        w, x, y, z = struct.unpack_from("<ffff", payload, 0)
                        R = quat_to_matrix(w, x, y, z)
                        with self.lock:
                            self.att_samples.append((now, R.copy()))
                            if len(self.att_samples) > 50000:
                                self.att_samples = self.att_samples[-25000:]

                    elif USE_EULER_ATTITUDE and func == FUNC_EULER and len(payload) >= 12:
                        roll, pitch, yaw = struct.unpack_from("<fff", payload, 0)
                        R = euler_to_matrix(roll, pitch, yaw)
                        with self.lock:
                            self.att_samples.append((now, R.copy()))
                            if len(self.att_samples) > 50000:
                                self.att_samples = self.att_samples[-25000:]

        except Exception as e:
            print("IMU 线程异常:", e)

    def estimate_gyro_bias(self, duration=3.0):
        with self.lock:
            raws = list(self.raw_samples)

        if len(raws) < 10:
            return False

        t0 = raws[0]["t"]
        vals = [s["gyro"] for s in raws if s["t"] - t0 <= duration]

        if len(vals) < 10:
            return False

        self.gyro_bias = np.mean(np.asarray(vals, dtype=np.float64), axis=0)
        return True

    def get_att_samples(self):
        with self.lock:
            return list(self.att_samples)

    def get_raw_samples(self):
        with self.lock:
            out = []
            for s in self.raw_samples:
                out.append({
                    "t": s["t"],
                    "accel": s["accel"].copy(),
                    "gyro": s["gyro"].copy(),
                    "mag": s["mag"].copy(),
                })
            return out

# ============================================================
# 数学工具
# ============================================================

def skew(v):
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ], dtype=np.float64)

def rot_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return math.acos(c)

def rot_to_vec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)

def moving_average(data, window=5):
    data = np.asarray(data, dtype=np.float64)
    if len(data) == 0:
        return data.copy()
    if window <= 1:
        return data.copy()

    k = np.ones(window, dtype=np.float64) / float(window)
    out = np.zeros_like(data)

    for d in range(data.shape[1]):
        out[:, d] = np.convolve(data[:, d], k, mode="same")

    # 边缘补偿
    half = window // 2
    for i in range(min(half, len(data))):
        out[i] = np.mean(data[:i + half + 1], axis=0)
        out[-1 - i] = np.mean(data[max(0, len(data) - 1 - i - half):], axis=0)

    return out

def derivative(values, times):
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    out = np.zeros_like(values)
    n = len(values)

    if n < 3:
        return out

    for i in range(1, n - 1):
        dt = times[i + 1] - times[i - 1]
        if dt <= 1e-8:
            out[i] = out[i - 1]
        else:
            out[i] = (values[i + 1] - values[i - 1]) / dt

    out[0] = out[1]
    out[-1] = out[-2]
    return out

def interp_vec3(times, values, t_query):
    if len(times) == 0:
        return None

    if t_query < times[0] or t_query > times[-1]:
        return None

    idx = np.searchsorted(times, t_query)
    if idx == 0:
        return values[0].copy()
    if idx >= len(times):
        return values[-1].copy()

    t0 = times[idx - 1]
    t1 = times[idx]
    v0 = values[idx - 1]
    v1 = values[idx]

    if abs(t1 - t0) < 1e-9:
        return v0.copy()

    s = (t_query - t0) / (t1 - t0)
    return (1.0 - s) * v0 + s * v1

def nearest_rotation(att_samples, t_query):
    if len(att_samples) == 0:
        return None

    times = np.array([s[0] for s in att_samples], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - t_query)))
    return att_samples[idx][1]

def solve_rotation_kabsch(cam_vecs, imu_vecs):
    A = np.asarray(cam_vecs, dtype=np.float64)
    B = np.asarray(imu_vecs, dtype=np.float64)

    if len(A) < 5:
        return None

    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    return R

# ============================================================
# 相机棋盘格
# ============================================================

def create_chessboard_object_points(chessboard_size, square_size):
    cols, rows = chessboard_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, 0] = grid[:, 0] * square_size
    objp[:, 1] = grid[:, 1] * square_size
    objp[:, 2] = 0.0
    return objp

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=0.08):
    axis = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
    ])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    o = tuple(imgpts[0])
    x = tuple(imgpts[1])
    y = tuple(imgpts[2])
    z = tuple(imgpts[3])

    cv2.line(img, o, x, (0, 0, 255), 3)
    cv2.line(img, o, y, (0, 255, 0), 3)
    cv2.line(img, o, z, (255, 0, 0), 3)

def camera_pose_from_pnp(R_cb, t_cb):
    t_cb = np.asarray(t_cb, dtype=np.float64).reshape(3, 1)
    R_wc = R_cb.T
    p_wc = -R_cb.T @ t_cb
    return R_wc, p_wc.reshape(3)

# ============================================================
# 阶段1：旋转外参 + 时间偏移
# ============================================================

def build_camera_relative_rotations(camera_poses, pair_step=5):
    rels = []
    min_angle = math.radians(ROT_MIN_ANGLE_DEG)
    max_angle = math.radians(ROT_MAX_ANGLE_DEG)

    n = len(camera_poses)
    for i in range(0, n - pair_step):
        j = i + pair_step

        R_cb_i = camera_poses[i]["R_cb"]
        R_cb_j = camera_poses[j]["R_cb"]

        R_bc_i = R_cb_i.T
        R_bc_j = R_cb_j.T

        R_cam_delta = R_bc_i.T @ R_bc_j
        a = rot_angle(R_cam_delta)

        if a < min_angle or a > max_angle:
            continue

        rels.append((camera_poses[i]["t"], camera_poses[j]["t"], R_cam_delta))

    return rels

def evaluate_rotation_error(R_ci, cam_rel_list, imu_rel_list):
    errs = []

    for R_cam, R_imu in zip(cam_rel_list, imu_rel_list):
        R_pred = R_ci @ R_imu @ R_ci.T
        R_err = R_cam @ R_pred.T
        errs.append(rot_angle(R_err))

    if len(errs) == 0:
        return 1e9

    return float(np.mean(errs))

def calibrate_time_and_rotation(camera_poses, att_samples):
    cam_rels = build_camera_relative_rotations(camera_poses, ROT_PAIR_STEP)

    if len(cam_rels) < 10:
        return None

    best = None

    for offset_ms in range(OFFSET_MIN_MS, OFFSET_MAX_MS + 1, OFFSET_STEP_MS):
        offset = offset_ms / 1000.0

        for imu_mode in [0, 1]:
            cam_rel_list = []
            imu_rel_list = []
            cam_vecs = []
            imu_vecs = []

            for t_i, t_j, R_cam_delta in cam_rels:
                R_imu_i = nearest_rotation(att_samples, t_i + offset)
                R_imu_j = nearest_rotation(att_samples, t_j + offset)
                if R_imu_i is None or R_imu_j is None:
                    continue

                if imu_mode == 0:
                    R_imu_delta = R_imu_i.T @ R_imu_j
                else:
                    R_imu_delta = R_imu_j @ R_imu_i.T

                a_cam = rot_angle(R_cam_delta)
                a_imu = rot_angle(R_imu_delta)

                if abs(a_cam - a_imu) > math.radians(20):
                    continue

                cam_rel_list.append(R_cam_delta)
                imu_rel_list.append(R_imu_delta)
                cam_vecs.append(rot_to_vec(R_cam_delta))
                imu_vecs.append(rot_to_vec(R_imu_delta))

            if len(cam_rel_list) < 10:
                continue

            R_ci = solve_rotation_kabsch(cam_vecs, imu_vecs)
            if R_ci is None:
                continue

            mean_err = evaluate_rotation_error(R_ci, cam_rel_list, imu_rel_list)

            item = {
                "offset_ms": offset_ms,
                "imu_mode": imu_mode,
                "R_ci": R_ci,
                "mean_error_deg": math.degrees(mean_err),
                "num_pairs": len(cam_rel_list)
            }

            if best is None or item["mean_error_deg"] < best["mean_error_deg"]:
                best = item

    return best

# ============================================================
# 阶段2：杆臂 t_ci
# ============================================================

def build_lever_dataset(camera_poses, raw_samples, gyro_bias, offset_s):
    if len(camera_poses) < 20 or len(raw_samples) < 20:
        return None

    cam_times = np.array([p["t"] for p in camera_poses], dtype=np.float64)
    R_wc_list = np.array([p["R_wc"] for p in camera_poses], dtype=np.float64)
    p_wc_list = np.array([p["p_wc"] for p in camera_poses], dtype=np.float64)

    imu_times = np.array([s["t"] for s in raw_samples], dtype=np.float64)
    imu_acc = np.array([s["accel"] for s in raw_samples], dtype=np.float64)
    imu_gyro = np.array([s["gyro"] - gyro_bias for s in raw_samples], dtype=np.float64)

    sync_times = []
    sync_R_wc = []
    sync_p_wc = []
    sync_acc = []
    sync_gyro = []

    for t_cam, R_wc, p_wc in zip(cam_times, R_wc_list, p_wc_list):
        tq = t_cam + offset_s
        acc = interp_vec3(imu_times, imu_acc, tq)
        gyro = interp_vec3(imu_times, imu_gyro, tq)

        if acc is None or gyro is None:
            continue

        sync_times.append(t_cam)
        sync_R_wc.append(R_wc)
        sync_p_wc.append(p_wc)
        sync_acc.append(acc)
        sync_gyro.append(gyro)

    if len(sync_times) < LEVER_MIN_SYNC_FRAMES:
        return None

    sync_times = np.asarray(sync_times, dtype=np.float64)
    sync_R_wc = np.asarray(sync_R_wc, dtype=np.float64)
    sync_p_wc = np.asarray(sync_p_wc, dtype=np.float64)
    sync_acc = np.asarray(sync_acc, dtype=np.float64)
    sync_gyro = np.asarray(sync_gyro, dtype=np.float64)

    # 平滑后做差分
    p_s = moving_average(sync_p_wc, SMOOTH_WIN_CAM)
    v = derivative(p_s, sync_times)
    v_s = moving_average(v, SMOOTH_WIN_CAM)
    a_cam_w = derivative(v_s, sync_times)
    a_cam_w = moving_average(a_cam_w, SMOOTH_WIN_CAM)

    gyro_s = moving_average(sync_gyro, SMOOTH_WIN_IMU)
    alpha = derivative(gyro_s, sync_times)
    alpha = moving_average(alpha, SMOOTH_WIN_IMU)

    trim = max(SMOOTH_WIN_CAM, SMOOTH_WIN_IMU) + 2
    if len(sync_times) <= 2 * trim + 5:
        return None

    keep = slice(trim, len(sync_times) - trim)

    return {
        "times": sync_times[keep],
        "R_wc": sync_R_wc[keep],
        "p_wc": sync_p_wc[keep],
        "a_cam_w": a_cam_w[keep],
        "acc_i": sync_acc[keep],
        "gyro_i": gyro_s[keep],
        "alpha_i": alpha[keep],
    }

def solve_lever_arm_with_bias(dataset, R_ci):
    R_wc_list = dataset["R_wc"]
    a_cam_w_list = dataset["a_cam_w"]
    acc_i_list = dataset["acc_i"]
    gyro_i_list = dataset["gyro_i"]
    alpha_i_list = dataset["alpha_i"]

    A_list = []
    b_list = []

    for R_wc, a_cam_w, acc_i, gyro_i, alpha_i in zip(
        R_wc_list, a_cam_w_list, acc_i_list, gyro_i_list, alpha_i_list
    ):
        omega_c = R_ci @ gyro_i
        alpha_c = R_ci @ alpha_i
        acc_i_c = R_ci @ acc_i

        M = skew(alpha_c) + skew(omega_c) @ skew(omega_c)

        # M r + R_wc.T g + c = R_wc.T a_cam_w - acc_i_c
        A_k = np.hstack([
            M,
            R_wc.T,
            np.eye(3)
        ])
        b_k = (R_wc.T @ a_cam_w - acc_i_c).reshape(3, 1)

        A_list.append(A_k)
        b_list.append(b_k)

    A = np.vstack(A_list)
    b = np.vstack(b_list)

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    r_ic_c = x[0:3].reshape(3)   # IMU -> Camera, expressed in camera frame
    g_w = x[3:6].reshape(3)
    c_bias = x[6:9].reshape(3)   # 吸收加速度常值误差项

    errs = []
    for R_wc, a_cam_w, acc_i, gyro_i, alpha_i in zip(
        R_wc_list, a_cam_w_list, acc_i_list, gyro_i_list, alpha_i_list
    ):
        omega_c = R_ci @ gyro_i
        alpha_c = R_ci @ alpha_i
        acc_i_c = R_ci @ acc_i

        M = skew(alpha_c) + skew(omega_c) @ skew(omega_c)
        lhs = M @ r_ic_c + R_wc.T @ g_w + c_bias
        rhs = R_wc.T @ a_cam_w - acc_i_c
        errs.append(np.linalg.norm(lhs - rhs))

    residual_mean = float(np.mean(errs))
    return r_ic_c, g_w, c_bias, residual_mean, int(rank)

def observability_progress(dataset, R_ci):
    if dataset is None:
        return 0.0, {}

    n = len(dataset["times"])
    if n < 20:
        return 0.0, {}

    gyro_c = np.asarray([R_ci @ g for g in dataset["gyro_i"]], dtype=np.float64)
    alpha_c = np.asarray([R_ci @ a for a in dataset["alpha_i"]], dtype=np.float64)

    gyro_p95 = np.percentile(np.abs(gyro_c), 95, axis=0)
    alpha_p95 = np.percentile(np.abs(alpha_c), 95, axis=0)

    frames_score = min(1.0, n / 180.0)
    gyro_score = np.mean(np.minimum(1.0, gyro_p95 / np.array([0.8, 0.8, 0.8])))
    alpha_score = np.mean(np.minimum(1.0, alpha_p95 / np.array([2.5, 2.5, 2.5])))

    A_list = []
    for R_wc, gyro_i, alpha_i in zip(dataset["R_wc"], dataset["gyro_i"], dataset["alpha_i"]):
        omega_c = R_ci @ gyro_i
        alpha_c = R_ci @ alpha_i
        M = skew(alpha_c) + skew(omega_c) @ skew(omega_c)
        A_k = np.hstack([M, R_wc.T, np.eye(3)])
        A_list.append(A_k)

    A = np.vstack(A_list)
    col_norm = np.linalg.norm(A, axis=0) + 1e-12
    A_n = A / col_norm[None, :]
    rank = np.linalg.matrix_rank(A_n)
    sv = np.linalg.svd(A_n / math.sqrt(max(1, len(A_n))), compute_uv=False)
    smin = float(np.min(sv)) if len(sv) > 0 else 0.0

    rank_score = rank / 9.0
    smin_score = min(1.0, smin / 0.12)

    total = (
        0.25 * frames_score +
        0.20 * gyro_score +
        0.20 * alpha_score +
        0.20 * rank_score +
        0.15 * smin_score
    )
    total = float(np.clip(total, 0.0, 1.0))

    info = {
        "frames": n,
        "gyro_p95": gyro_p95,
        "alpha_p95": alpha_p95,
        "rank": rank,
        "smin": smin,
        "frames_score": frames_score,
        "gyro_score": float(gyro_score),
        "alpha_score": float(alpha_score),
        "rank_score": float(rank_score),
        "smin_score": float(smin_score),
    }
    return total, info

# ============================================================
# 绘图
# ============================================================

def draw_progress_bar(img, x, y, w, h, ratio, color, text):
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 2)
    fill_w = int(w * ratio)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_text(img, txt, pos, color=(255, 255, 255), scale=0.55, thick=2):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def terminal_bar(pct, width=40):
    n = int(width * pct / 100.0)
    return "[" + "#" * n + "-" * (width - n) + "]"

def opencv_vec_to_yup(v):
    return np.array([v[0], -v[1], v[2]], dtype=np.float64)

# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 78)
    print(" Camera-IMU 全自动标定：time offset + R_ci + t_ci")
    print("=" * 78)
    print("使用说明：")
    print("1) 启动后前 3 秒保持设备静止，用于估计 gyro bias")
    print("2) 然后保持棋盘格固定不动，手持“相机+IMU”整体运动")
    print("3) 需要做 yaw / pitch / roll / 多轴组合旋转，并带加速减速")
    print("4) 棋盘格尽量始终保持在画面内")
    print("5) 程序会显示进度条；只有激励充分后才自动输出最终结果")
    print("=" * 78)

    obj_points = create_chessboard_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)

    imu = IMURecorder()
    if not imu.start(baudrate=BAUDRATE):
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("相机打开失败")
        imu.stop()
        return

    camera_poses = []
    last_pose_time = 0.0

    rot_done = False
    final_done = False
    gyro_bias_done = False

    rot_result = None
    lever_result = None

    last_term_print = 0.0
    last_rot_try = 0.0
    last_progress_try = 0.0
    last_rot_try_pose_count = 0
    ROT_TRY_INTERVAL = 8.0
    ROT_TRY_NEW_POSES = 40

    win_name = "Camera-IMU Full Calibration"

    try:
        while True:
            ret, frame = cap.read()
            t_cam = time.monotonic()

            if not ret or frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ----------------------------
            # 前3秒估计 gyro bias
            # ----------------------------
            elapsed = t_cam - imu.start_time if imu.start_time is not None else 0.0
            if (not gyro_bias_done) and elapsed >= STATIC_BIAS_SECONDS:
                gyro_bias_done = imu.estimate_gyro_bias(STATIC_BIAS_SECONDS)
                if gyro_bias_done:
                    print(f"gyro_bias = {imu.gyro_bias}")
                    print("已完成静止零偏估计，开始正式采集。")
                else:
                    print("gyro bias 估计失败，继续尝试...")

            # ----------------------------
            # 棋盘格检测
            # ----------------------------
            found, corners = cv2.findChessboardCorners(
                gray,
                CHESSBOARD_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            pose_ok = False
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

                cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners_refined, found)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners_refined,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if ok:
                    pose_ok = True
                    R_cb, _ = cv2.Rodrigues(rvec)
                    R_wc, p_wc = camera_pose_from_pnp(R_cb, tvec)

                    draw_axes(display, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, axis_len=0.08)

                    if gyro_bias_done and (t_cam - last_pose_time) > CAM_POSE_MIN_DT:
                        camera_poses.append({
                            "t": t_cam,
                            "R_cb": R_cb.copy(),
                            "t_cb": tvec.reshape(3).copy(),
                            "R_wc": R_wc.copy(),
                            "p_wc": p_wc.copy()
                        })
                        if len(camera_poses) > 10000:
                            camera_poses = camera_poses[-5000:]
                        last_pose_time = t_cam

            att_samples = imu.get_att_samples()
            raw_samples = imu.get_raw_samples()

            # ----------------------------
            # 阶段1：旋转外参与时间偏移
            # ----------------------------
            stage1_progress = 0.0

            if not rot_done:
                pose_score = min(1.0, len(camera_poses) / 120.0)

                if len(raw_samples) > 20:
                    raw_gyro = np.array([s["gyro"] - imu.gyro_bias for s in raw_samples[-1500:]], dtype=np.float64)
                    if len(raw_gyro) > 5:
                        gyro_p95 = np.percentile(np.abs(raw_gyro), 95, axis=0)
                        axis_score = np.mean(np.minimum(1.0, gyro_p95 / np.array([0.8, 0.8, 0.8])))
                    else:
                        axis_score = 0.0
                else:
                    axis_score = 0.0

                att_score = min(1.0, len(att_samples) / 800.0)
                stage1_progress = float(np.clip(0.45 * pose_score + 0.30 * axis_score + 0.25 * att_score, 0.0, 1.0))

                if (
                        gyro_bias_done and
                        len(camera_poses) >= ROT_MIN_CAMERA_POSES and
                        len(att_samples) >= 500 and
                        (t_cam - last_rot_try) > ROT_TRY_INTERVAL and
                        (len(camera_poses) - last_rot_try_pose_count) >= ROT_TRY_NEW_POSES
                ):
                    last_rot_try = t_cam
                    last_rot_try_pose_count = len(camera_poses)

                    print("\n尝试求解阶段1：time offset + R_ci ...")
                    best = calibrate_time_and_rotation(camera_poses, att_samples)

                    if best is not None:
                        print(f"阶段1候选: offset={best['offset_ms']} ms, pairs={best['num_pairs']}, err={best['mean_error_deg']:.3f} deg")
                        if best["mean_error_deg"] <= ROT_ACCEPT_ERR_DEG and best["num_pairs"] >= ROT_ACCEPT_MIN_PAIRS:
                            rot_done = True
                            rot_result = best
                            print("阶段1完成：已得到 time offset 和 R_ci")
                        else:
                            print("阶段1尚未通过，继续收集更多旋转激励。")

            # ----------------------------
            # 阶段2：lever arm / t_ci
            # ----------------------------
            stage2_progress = 0.0
            obs_info = {}
            dataset = None

            if rot_done and not final_done and (t_cam - last_progress_try) > 0.8:
                last_progress_try = t_cam

                dataset = build_lever_dataset(
                    camera_poses,
                    raw_samples,
                    imu.gyro_bias,
                    rot_result["offset_ms"] / 1000.0
                )

                if dataset is not None:
                    stage2_progress, obs_info = observability_progress(dataset, rot_result["R_ci"])

                    if stage2_progress >= 0.999:
                        print("\n激励与可观测性已满足，开始求最终外参 t_ci ...")
                        r_ic_c, g_w, c_bias, residual, rank = solve_lever_arm_with_bias(dataset, rot_result["R_ci"])

                        lever_norm = np.linalg.norm(r_ic_c)
                        g_norm = np.linalg.norm(g_w)

                        sane = (
                            rank >= 9 and
                            residual <= LEVER_ACCEPT_RESIDUAL and
                            lever_norm <= LEVER_MAX_NORM and
                            GRAVITY_MIN <= g_norm <= GRAVITY_MAX
                        )

                        print(f"阶段2候选: residual={residual:.4f}, |r|={lever_norm:.4f} m, |g|={g_norm:.4f}, rank={rank}")

                        if sane:
                            final_done = True

                            # r_ic_c: IMU -> Camera in OpenCV camera frame
                            # t_ci_cv: IMU原点在Camera坐标系下的位置 = Camera -> IMU = -r_ic_c
                            t_ci_cv = -r_ic_c
                            t_ci_yup = opencv_vec_to_yup(t_ci_cv)

                            lever_result = {
                                "r_ic_c": r_ic_c,
                                "t_ci_cv": t_ci_cv,
                                "t_ci_yup": t_ci_yup,
                                "g_w": g_w,
                                "c_bias": c_bias,
                                "residual": residual,
                                "rank": rank,
                            }

                            os.makedirs(OUT_DIR, exist_ok=True)
                            out_path = os.path.join(OUT_DIR, OUT_JSON)

                            result_json = {
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "chessboard_size": list(CHESSBOARD_SIZE),
                                "square_size_m": SQUARE_SIZE,
                                "camera_matrix": CAMERA_MATRIX.tolist(),
                                "dist_coeffs": DIST_COEFFS.reshape(-1).tolist(),
                                "gyro_bias_radps": imu.gyro_bias.tolist(),
                                "time_offset_ms": rot_result["offset_ms"],
                                "imu_mode": rot_result["imu_mode"],
                                "rotation_error_deg": rot_result["mean_error_deg"],
                                "rotation_pairs": rot_result["num_pairs"],
                                "R_ci_imu_to_opencv_camera": rot_result["R_ci"].tolist(),
                                "r_ic_c_imu_to_camera_origin_in_cv_m": r_ic_c.tolist(),
                                "t_ci_cv_camera_to_imu_in_cv_m": t_ci_cv.tolist(),
                                "t_ci_yup_camera_to_imu_in_yup_m": t_ci_yup.tolist(),
                                "gravity_world_est_mps2": g_w.tolist(),
                                "constant_bias_term_camera_mps2": c_bias.tolist(),
                                "lever_residual_mps2": residual,
                                "lever_rank": rank,
                                "synced_frame_count": int(len(dataset["times"])),
                            }

                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(result_json, f, indent=4, ensure_ascii=False)

                            print("\n" + "=" * 78)
                            print("最终标定成功")
                            print("=" * 78)
                            print(f"time_offset_ms = {rot_result['offset_ms']}")
                            print(f"rotation_error_deg = {rot_result['mean_error_deg']:.4f}")
                            print(f"lever_residual = {residual:.4f} m/s^2")
                            print()

                            print("R_ci (IMU -> OpenCV Camera):")
                            print(rot_result["R_ci"])
                            print()

                            print("OpenCV 相机坐标系下：")
                            print("self.R_ci_cv = np.array([")
                            for row in rot_result["R_ci"]:
                                print(f"    [{row[0]:+.9f}, {row[1]:+.9f}, {row[2]:+.9f}],")
                            print("], dtype=np.float64)")
                            print()

                            print(f"self.t_ci_cv = np.array([{t_ci_cv[0]:+.6f}, {t_ci_cv[1]:+.6f}, {t_ci_cv[2]:+.6f}], dtype=np.float64)")
                            print()

                            print("如果你的 SLAM 相机坐标系是 X右 Y上 Z前，则建议：")
                            print(f"self.t_ci = np.array([{t_ci_yup[0]:+.6f}, {t_ci_yup[1]:+.6f}, {t_ci_yup[2]:+.6f}], dtype=np.float64)")
                            print()
                            print(f"结果已保存到: {out_path}")
                            print("=" * 78 + "\n")
                        else:
                            print("虽然进度接近完成，但当前残差/秩/重力范围不理想，继续采集更多多轴激励。")

            # ----------------------------
            # 总进度
            # ----------------------------
            if not rot_done:
                total_progress = 0.5 * stage1_progress
            else:
                total_progress = 0.5 + 0.5 * stage2_progress

            total_progress = float(np.clip(total_progress, 0.0, 1.0))

            # ----------------------------
            # 界面显示
            # ----------------------------
            status_color = (0, 255, 0) if pose_ok else (0, 0, 255)
            draw_text(display, f"Chessboard: {'FOUND' if pose_ok else 'NOT FOUND'}", (10, 25), status_color, 0.65, 2)
            draw_text(display, f"Camera poses: {len(camera_poses)}", (10, 50), (255, 255, 0), 0.6, 2)
            draw_text(display, f"IMU raw: {len(raw_samples)}", (10, 75), (255, 255, 0), 0.6, 2)
            draw_text(display, f"IMU att: {len(att_samples)}", (10, 100), (255, 255, 0), 0.6, 2)

            if not gyro_bias_done:
                remain = max(0.0, STATIC_BIAS_SECONDS - elapsed)
                draw_text(display, f"KEEP STILL for gyro bias... {remain:.1f}s", (10, 130), (0, 200, 255), 0.65, 2)
            else:
                draw_text(display, f"gyro_bias done", (10, 130), (0, 255, 0), 0.65, 2)

            draw_progress_bar(
                display, 10, 155, 420, 22,
                total_progress, (0, 200, 255),
                f"TOTAL PROGRESS: {total_progress * 100:.1f}%"
            )

            draw_progress_bar(
                display, 10, 195, 420, 18,
                stage1_progress if not rot_done else 1.0, (0, 255, 0),
                f"Stage1 Rotation+Offset: {(stage1_progress if not rot_done else 1.0) * 100:.1f}%"
            )

            draw_progress_bar(
                display, 10, 230, 420, 18,
                stage2_progress if rot_done else 0.0, (255, 180, 0),
                f"Stage2 Translation Lever-arm: {(stage2_progress if rot_done else 0.0) * 100:.1f}%"
            )

            if rot_done:
                draw_text(display, f"offset = {rot_result['offset_ms']} ms", (10, 265), (0, 255, 0), 0.58, 2)
                draw_text(display, f"rot err = {rot_result['mean_error_deg']:.3f} deg", (10, 290), (0, 255, 0), 0.58, 2)

            if len(obs_info) > 0:
                gyro_p95 = obs_info["gyro_p95"]
                alpha_p95 = obs_info["alpha_p95"]
                draw_text(display, f"lever frames = {obs_info['frames']}", (10, 320), (255, 255, 255), 0.54, 2)
                draw_text(display, f"rank = {obs_info['rank']}/9, smin = {obs_info['smin']:.4f}", (10, 345), (255, 255, 255), 0.54, 2)
                draw_text(display, f"gyro p95 = [{gyro_p95[0]:.2f}, {gyro_p95[1]:.2f}, {gyro_p95[2]:.2f}] rad/s", (10, 370), (255, 255, 255), 0.52, 2)
                draw_text(display, f"alpha p95 = [{alpha_p95[0]:.2f}, {alpha_p95[1]:.2f}, {alpha_p95[2]:.2f}] rad/s^2", (10, 395), (255, 255, 255), 0.52, 2)

            if final_done and lever_result is not None:
                draw_text(display, "CALIBRATION SUCCESS", (10, 430), (0, 255, 0), 0.8, 2)
                t_ci_cv = lever_result["t_ci_cv"]
                draw_text(display, f"t_ci_cv = [{t_ci_cv[0]:+.3f}, {t_ci_cv[1]:+.3f}, {t_ci_cv[2]:+.3f}] m", (10, 460), (0, 255, 0), 0.56, 2)

            cv2.imshow(win_name, display)

            # 终端打印进度
            if (t_cam - last_term_print) > 1.5:
                last_term_print = t_cam
                pct = total_progress * 100.0
                print(f"\r{terminal_bar(pct)} {pct:5.1f}%  poses={len(camera_poses)} raw={len(raw_samples)} att={len(att_samples)}", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        print()
        cap.release()
        imu.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()