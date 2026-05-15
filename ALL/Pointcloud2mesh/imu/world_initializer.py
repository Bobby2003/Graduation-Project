from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WorldInitResult:
    success: bool
    T_world_to_camera: np.ndarray
    T_camera_to_world: np.ndarray
    T_world_to_imu: Optional[np.ndarray]
    imu_quaternion_wxyz: Optional[np.ndarray]
    reason: str


def rotation_to_euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    """
    Return roll/pitch/yaw degrees for R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    R = _normalize_rotation(R)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def pose_debug_dict(T: np.ndarray) -> dict:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = _normalize_rotation(T[:3, :3])
    return {
        "translation": T[:3, 3].copy(),
        "euler_roll_pitch_yaw_deg": rotation_to_euler_zyx_deg(R),
        "x_axis_world": R[:, 0].copy(),
        "y_axis_world": R[:, 1].copy(),
        "z_axis_world": R[:, 2].copy(),
    }


def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def quaternion_wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm([w, x, y, z])
    if n <= 1e-12 or not np.isfinite(n):
        raise ValueError("Invalid quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(R: np.ndarray, t=None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _normalize_rotation(R)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


class IMUWorldInitializer:
    """
    One-shot world-frame initializer.

    The fixed rotation R_cam_imu maps IMU coordinates into camera coordinates.
    Translation is intentionally ignored because the IMU is glued to the camera
    and the requirement is only to align axes.
    """

    def __init__(
        self,
        provider=None,
        enabled: bool = False,
        required: bool = False,
        R_cam_imu=None,
        R_reconstruction_imu_world=None,
        quaternion_convention: str = "imu_to_world",
        initial_alignment_mode: Optional[str] = None,
        sample_timeout_sec: float = 0.5,
        zero_initial_camera_rotation: bool = True,
        logger=None,
    ):
        self.provider = provider
        self.enabled = bool(enabled)
        self.required = bool(required)
        self.R_cam_imu = _normalize_rotation(
            np.eye(3, dtype=np.float64) if R_cam_imu is None else np.asarray(R_cam_imu, dtype=np.float64)
        )
        self.R_reconstruction_imu_world = _normalize_rotation(
            np.eye(3, dtype=np.float64)
            if R_reconstruction_imu_world is None
            else np.asarray(R_reconstruction_imu_world, dtype=np.float64)
        )
        self.quaternion_convention = str(quaternion_convention).lower().strip()
        self.initial_alignment_mode = self._resolve_initial_alignment_mode(
            initial_alignment_mode,
            zero_initial_camera_rotation,
        )
        self.sample_timeout_sec = float(sample_timeout_sec)
        self.zero_initial_camera_rotation = bool(zero_initial_camera_rotation)
        self.logger = logger

        self.last_result: Optional[WorldInitResult] = None
        self.last_debug_info: Optional[dict] = None
        self._initialized = False

    @staticmethod
    def _fmt_vec(v) -> str:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"

    @staticmethod
    def _fmt_mat(R) -> str:
        return np.asarray(R, dtype=np.float64).round(6).tolist()

    @staticmethod
    def _resolve_initial_alignment_mode(mode, zero_initial_camera_rotation: bool) -> str:
        if mode is None or str(mode).strip() == "":
            return (
                "align_initial_camera_to_forward"
                if zero_initial_camera_rotation
                else "imu_absolute_world"
            )

        resolved = str(mode).lower().strip()
        valid_modes = {
            "imu_absolute_world",
            "align_initial_camera_to_forward",
            "disabled",
        }
        if resolved not in valid_modes:
            raise ValueError(
                "initial_alignment_mode must be one of "
                "'imu_absolute_world', 'align_initial_camera_to_forward', or 'disabled'"
            )
        return resolved

    def _log_rotation_debug(self, label: str, R: np.ndarray):
        if self.logger is None:
            return
        self.logger.pose(
            f"{label}: rpy_deg={self._fmt_vec(rotation_to_euler_zyx_deg(R))}",
            force=True,
        )

    def _log_warning(self, msg: str):
        if self.logger is not None and hasattr(self.logger, "warning"):
            self.logger.warning(msg, force=True)

    def _log_pose_debug(self, label: str, T: np.ndarray):
        if self.logger is None:
            return
        info = pose_debug_dict(T)
        self.logger.pose(
            f"{label}: "
            f"t={self._fmt_vec(info['translation'])} "
            f"rpy_deg={self._fmt_vec(info['euler_roll_pitch_yaw_deg'])} "
            f"x={self._fmt_vec(info['x_axis_world'])} "
            f"y={self._fmt_vec(info['y_axis_world'])} "
            f"z={self._fmt_vec(info['z_axis_world'])}",
            force=True,
        )

    def _set_fallback_result(self, fallback: WorldInitResult):
        self.last_result = fallback
        self.last_debug_info = {
            "quaternion_convention": self.quaternion_convention,
            "initial_alignment_mode": self.initial_alignment_mode,
            "imu_quaternion_wxyz": None,
            "R_cam_imu": self.R_cam_imu.copy(),
            "R_reconstruction_imu_world": self.R_reconstruction_imu_world.copy(),
            "camera_pose": pose_debug_dict(fallback.T_camera_to_world),
            "imu_pose": None,
            "world_to_camera": pose_debug_dict(fallback.T_world_to_camera),
            "reason": fallback.reason,
        }
        self._initialized = True

        if self.logger is not None:
            self.logger.pose(
                "IMU world init failed/fallback: "
                f"reason={fallback.reason} "
                f"enabled={self.enabled} "
                f"required={self.required} "
                f"convention={self.quaternion_convention}",
                force=True,
            )
        self._log_pose_debug("Fallback camera pose camera_to_world", fallback.T_camera_to_world)
        self._log_pose_debug("Fallback TSDF extrinsic world_to_camera", fallback.T_world_to_camera)
        return fallback

    def close(self):
        if self.provider is not None and hasattr(self.provider, "close"):
            self.provider.close()

    def initialize_world_to_camera(self, frame_timestamp=None) -> WorldInitResult:
        if self._initialized and self.last_result is not None:
            return self.last_result

        fallback = WorldInitResult(
            success=False,
            T_world_to_camera=np.eye(4, dtype=np.float64),
            T_camera_to_world=np.eye(4, dtype=np.float64),
            T_world_to_imu=None,
            imu_quaternion_wxyz=None,
            reason="imu_world_init_disabled",
        )

        if not self.enabled:
            return self._set_fallback_result(fallback)

        if self.provider is None:
            fallback.reason = "imu_provider_missing"
            self._set_fallback_result(fallback)
            if self.required:
                raise RuntimeError(fallback.reason)
            return fallback

        sample = self.provider.read_quaternion_sample(timeout_sec=self.sample_timeout_sec)
        if sample is None:
            fallback.reason = "imu_quaternion_timeout"
            self._set_fallback_result(fallback)
            if self.required:
                raise RuntimeError(fallback.reason)
            return fallback

        R_from_quat = quaternion_wxyz_to_rotation(sample.quaternion_wxyz)

        if self.quaternion_convention == "imu_to_world":
            R_world_imu = R_from_quat
        elif self.quaternion_convention == "world_to_imu":
            R_world_imu = R_from_quat.T
        else:
            raise ValueError(
                "quaternion_convention must be 'imu_to_world' or 'world_to_imu'"
            )

        R_imu_cam = self.R_cam_imu.T
        R_world_cam = _normalize_rotation(R_world_imu @ R_imu_cam)

        mode = self.initial_alignment_mode
        if mode == "imu_absolute_world":
            R_recon_world = self.R_reconstruction_imu_world
        elif mode == "align_initial_camera_to_forward":
            R_target_cam = np.eye(3, dtype=np.float64)
            R_recon_world = _normalize_rotation(R_target_cam @ R_world_cam.T)
        elif mode == "disabled":
            R_recon_world = np.eye(3, dtype=np.float64)
        else:
            raise ValueError(
                "initial_alignment_mode must be one of "
                "'imu_absolute_world', 'align_initial_camera_to_forward', or 'disabled'"
            )

        R_reconstruction_camera = _normalize_rotation(R_recon_world @ R_world_cam)
        R_reconstruction_imu = _normalize_rotation(
            R_recon_world @ R_world_imu
        )

        T_camera_to_world = make_transform(R_reconstruction_camera)
        T_world_to_camera = np.linalg.inv(T_camera_to_world)
        T_world_to_imu = make_transform(R_reconstruction_imu)

        result = WorldInitResult(
            success=True,
            T_world_to_camera=T_world_to_camera,
            T_camera_to_world=T_camera_to_world,
            T_world_to_imu=T_world_to_imu,
            imu_quaternion_wxyz=sample.quaternion_wxyz.copy(),
            reason="ok",
        )

        self.last_result = result
        self.last_debug_info = {
            "quaternion_convention": self.quaternion_convention,
            "initial_alignment_mode": mode,
            "imu_quaternion_wxyz": sample.quaternion_wxyz.copy(),
            "R_cam_imu": self.R_cam_imu.copy(),
            "R_world_imu": R_world_imu.copy(),
            "R_world_cam": R_world_cam.copy(),
            "R_recon_world": R_recon_world.copy(),
            "R_reconstruction_imu_world": R_recon_world.copy(),
            "zero_initial_camera_rotation": self.zero_initial_camera_rotation,
            "camera_pose": pose_debug_dict(T_camera_to_world),
            "imu_pose": pose_debug_dict(T_world_to_imu),
            "world_to_camera": pose_debug_dict(T_world_to_camera),
        }
        self._initialized = True

        if self.logger is not None:
            self.logger.pose(
                "IMU world init: "
                f"raw_imu_quaternion_wxyz={self._fmt_vec(sample.quaternion_wxyz)} "
                f"convention={self.quaternion_convention} "
                f"initial_alignment_mode={mode} "
                f"zero_initial_camera_rotation={self.zero_initial_camera_rotation}",
                force=True,
            )
            self.logger.pose(
                "IMU world init matrices: "
                f"R_cam_imu={self._fmt_mat(self.R_cam_imu)}",
                force=True,
            )
        self._log_rotation_debug("R_world_imu", R_world_imu)
        self._log_rotation_debug("R_world_cam", R_world_cam)
        self._log_rotation_debug("R_recon_world/correction", R_recon_world)
        self._log_pose_debug("IMU pose imu_to_world", T_world_to_imu)
        self._log_pose_debug("Camera pose camera_to_world", T_camera_to_world)
        self._log_pose_debug("TSDF extrinsic world_to_camera", T_world_to_camera)

        camera_rpy = rotation_to_euler_zyx_deg(T_camera_to_world[:3, :3])
        if mode == "align_initial_camera_to_forward" and np.max(np.abs(camera_rpy)) > 1.0:
            self._log_warning(
                "IMU initial alignment expected camera_to_world rpy close to "
                f"[0, 0, 0], got {self._fmt_vec(camera_rpy)}"
            )

        return result

    def make_imu_pose_from_camera_pose(self, T_camera_to_world: np.ndarray) -> np.ndarray:
        T_camera_to_imu = make_transform(self.R_cam_imu)
        return np.asarray(T_camera_to_world, dtype=np.float64).reshape(4, 4) @ T_camera_to_imu

    def get_last_init_info(self) -> Optional[dict]:
        if self.last_result is None:
            return None
        return {
            "success": self.last_result.success,
            "reason": self.last_result.reason,
            "imu_quaternion_wxyz": (
                None
                if self.last_result.imu_quaternion_wxyz is None
                else self.last_result.imu_quaternion_wxyz.copy()
            ),
            "T_world_to_camera": self.last_result.T_world_to_camera.copy(),
            "T_camera_to_world": self.last_result.T_camera_to_world.copy(),
            "T_world_to_imu": (
                None
                if self.last_result.T_world_to_imu is None
                else self.last_result.T_world_to_imu.copy()
            ),
            "debug": self.last_debug_info,
        }
