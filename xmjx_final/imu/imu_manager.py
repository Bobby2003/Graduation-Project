import os
import threading
import numpy as np
from bisect import bisect_left
from scipy.spatial.transform import Rotation as R

from common.types import IMUSample

class IMUManager:
    """
    只接受“已经对齐到 device timestamp”的 IMU 数据。
    tracking 查询时，也只用 device timestamp。

    设计原则：
    1. 不在这里偷偷用 host time 冒充 device time
    2. 如果时间基准没统一，宁可不用 IMU
    """

    def __init__(
        self,
        npy_dir="npy",
        imu_cam_rot_filename="imu_to_cam_rot.npy",
        imu_cam_dt_filename="imu_to_cam_dt.npy",
        imu_legacy_flag_filename="imu_euler_legacy_flag.npy",
    ):
        self._lock = threading.Lock()

        self.timestamps = []   # device timestamp
        self.rotations = []    # 相对姿态 R_imu0_to_imu_t, 3x3

        self.R_ci = np.eye(3, dtype=np.float64)   # IMU -> CAM
        self.imu_delay = 0.0                      # t_imu + dt ≈ t_cam
        self.use_legacy_euler_mapping = False

        self.npy_dir = npy_dir
        self.imu_cam_rot_path = os.path.join(npy_dir, imu_cam_rot_filename)
        self.imu_cam_dt_path = os.path.join(npy_dir, imu_cam_dt_filename)
        self.imu_legacy_flag_path = os.path.join(npy_dir, imu_legacy_flag_filename)

        self.load_imu_legacy_flag()
        self.load_imu_cam_extrinsic()
        self.load_imu_cam_dt()

    # ---------- rotation utils ----------
    @staticmethod
    def normalize_rotation(Rm):
        U, _, Vt = np.linalg.svd(Rm)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    @staticmethod
    def is_valid_rotation(Rm):
        if Rm.shape != (3, 3):
            return False
        should_be_I = Rm.T @ Rm
        det = np.linalg.det(Rm)
        return np.allclose(should_be_I, np.eye(3), atol=1e-3) and np.isclose(det, 1.0, atol=1e-3)

    @staticmethod
    def rot_deg(Rm):
        rv = R.from_matrix(IMUManager.normalize_rotation(Rm)).as_rotvec()
        return float(np.degrees(np.linalg.norm(rv)))

    @staticmethod
    def interpolate_rotation(R0, R1, alpha):
        dR = IMUManager.normalize_rotation(R0.T @ R1)
        rv = R.from_matrix(dR).as_rotvec()
        return IMUManager.normalize_rotation(R0 @ R.from_rotvec(alpha * rv).as_matrix())

    # ---------- calibration ----------
    def load_imu_cam_extrinsic(self):
        if os.path.exists(self.imu_cam_rot_path):
            try:
                R_ci = np.load(self.imu_cam_rot_path)
                R_ci = self.normalize_rotation(R_ci)
                if self.is_valid_rotation(R_ci):
                    self.R_ci = R_ci
                    print(f"✅ 已加载 IMU->CAM 外参: {self.imu_cam_rot_path}")
                    print(self.R_ci)
                else:
                    print("⚠️ imu_to_cam_rot.npy 无效，使用单位阵")
            except Exception as e:
                print(f"⚠️ 外参加载失败: {e}，使用单位阵")
        else:
            print("⚠️ 未找到 imu_to_cam_rot.npy，使用单位阵")

    def load_imu_cam_dt(self):
        if os.path.exists(self.imu_cam_dt_path):
            try:
                dt = float(np.load(self.imu_cam_dt_path))
                self.imu_delay = dt
                print(f"✅ 已加载 IMU->CAM 时间偏移 dt: {dt * 1000:.1f} ms")
            except Exception as e:
                print(f"⚠️ dt 加载失败: {e}，使用 0")
                self.imu_delay = 0.0
        else:
            print("⚠️ 未找到 imu_to_cam_dt.npy，默认 dt=0")

    def load_imu_legacy_flag(self):
        if os.path.exists(self.imu_legacy_flag_path):
            try:
                v = int(np.load(self.imu_legacy_flag_path))
                self.use_legacy_euler_mapping = (v == 1)
                print(f"✅ 已加载欧拉映射标志: legacy={self.use_legacy_euler_mapping}")
            except Exception as e:
                print(f"⚠️ legacy flag 加载失败: {e}，默认 legacy=False")
                self.use_legacy_euler_mapping = False
        else:
            print("⚠️ 未找到 imu_euler_legacy_flag.npy，默认 legacy=False")
            self.use_legacy_euler_mapping = False

    # ---------- data ingestion ----------
    def push_rotation(self, timestamp: float, R_imu: np.ndarray):
        """
        推入一个“已对齐到 device timestamp”的 IMU姿态样本。
        R_imu 应为某个统一参考系下的相对旋转。
        """
        R_imu = self.normalize_rotation(R_imu)
        if not self.is_valid_rotation(R_imu):
            return

        with self._lock:
            if self.timestamps and timestamp < self.timestamps[-1]:
                # 简单保护：要求时间非降序
                return
            self.timestamps.append(float(timestamp))
            self.rotations.append(R_imu)

    def push_sample(self, sample: IMUSample):
        """
        当前只缓存 accel/gyro 的外壳接口。
        如果你未来做预积分，可以在这里扩展。
        """
        # 目前 tracker 用的是 rotation 查询，不直接用 accel/gyro
        pass

    # ---------- query ----------
    def get_rotation_at(self, cam_device_timestamp: float):
        """
        查询与相机帧时间对齐的 IMU 姿态。
        使用: t_imu + dt ≈ t_cam
        因此查询 imu 时间为: t_cam - dt
        """
        t_query = float(cam_device_timestamp) - float(self.imu_delay)

        with self._lock:
            ts_list = self.timestamps
            R_list = self.rotations

            n = len(ts_list)
            if n < 2:
                return None

            if t_query < ts_list[0] or t_query > ts_list[-1]:
                return None

            idx = bisect_left(ts_list, t_query)
            if idx == 0:
                return R_list[0].copy()
            if idx >= n:
                return R_list[-1].copy()

            t0, t1 = ts_list[idx - 1], ts_list[idx]
            R0, R1 = R_list[idx - 1], R_list[idx]

        if t1 <= t0:
            return R0.copy()

        alpha = float(np.clip((t_query - t0) / (t1 - t0), 0.0, 1.0))
        return self.interpolate_rotation(R0, R1, alpha)

    def get_delta_rotation(self, t0_cam: float, t1_cam: float):
        """
        返回 IMU 在 [t0_cam, t1_cam] 对应的增量旋转（先查姿态，再做相对旋转）
        """
        R0 = self.get_rotation_at(t0_cam)
        R1 = self.get_rotation_at(t1_cam)

        if R0 is None or R1 is None:
            return None

        dR_imu = self.normalize_rotation(R0.T @ R1)
        if not self.is_valid_rotation(dR_imu):
            return None
        return dR_imu

    def imu_delta_to_cam_delta(self, dR_imu):
        """
        IMU 坐标系增量 -> 相机坐标系增量
        """
        return self.normalize_rotation(self.R_ci @ dR_imu @ self.R_ci.T)

    def get_measurements_near(self, timestamp, window=0.01):
        """
        为兼容旧接口保留；当前仅返回附近姿态索引信息，不返回原始 accel/gyro。
        """
        with self._lock:
            result = []
            for ts, Rm in zip(self.timestamps, self.rotations):
                if abs(ts - timestamp) <= window:
                    result.append((ts, Rm))
            return result

    def size(self):
        with self._lock:
            return len(self.timestamps)