import numpy as np
import open3d as o3d

from common.types import RGBDFrame, TrackingResult
from input.rgbd_preprocessor import RGBDPreprocessor

class Tracker:
    """
    基于 RGBD Odometry 的 tracking 前端。
    时间统一使用 frame.device_timestamp。
    """

    def __init__(
        self,
        imu_manager=None,
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        input_color_is_bgr=False,
        depth_scale=None,
        depth_trunc=2.0,
        min_valid_pixels=1000,

        max_rotation_deg_per_frame=12.0,
        rotation_dominant_angle_deg=0.25,

        max_translation_when_rotating=0.008,
        min_info_trace=5e5,
        min_translation_per_frame=0.0008,
        max_translation_per_frame=0.03,
        trans_smooth_alpha=0.3,

        min_imu_rot_deg_per_frame=0.30,

        imu_as_vo_init_only=True,
        allow_imu_fallback_when_vo_fails=True,

        debug_print_odom=True,
    ):
        self.imu_manager = imu_manager
        self.min_valid_pixels = min_valid_pixels

        self.preprocessor = RGBDPreprocessor(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            input_color_is_bgr=input_color_is_bgr,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )

        self.option = o3d.pipelines.odometry.OdometryOption()
        self.option.depth_diff_max = 0.07

        self.max_rotation_deg_per_frame = max_rotation_deg_per_frame
        self.rotation_dominant_angle_deg = rotation_dominant_angle_deg

        self.max_translation_when_rotating = max_translation_when_rotating
        self.min_info_trace = min_info_trace
        self.min_translation_per_frame = min_translation_per_frame
        self.max_translation_per_frame = max_translation_per_frame
        self.trans_smooth_alpha = trans_smooth_alpha

        self.min_imu_rot_deg_per_frame = min_imu_rot_deg_per_frame

        self.imu_as_vo_init_only = imu_as_vo_init_only
        self.allow_imu_fallback_when_vo_fails = allow_imu_fallback_when_vo_fails

        self.debug_print_odom = debug_print_odom

        self.prev_rgbd = None
        self.prev_frame = None
        self.prev_cam_ts = None

        # 保存 world -> current_camera 的外参，直接兼容你原型的 TSDF integrate 用法
        self.T_c_w = np.eye(4, dtype=np.float64)

        self.last_trans = np.zeros(3, dtype=np.float64)

    # ---------- utils ----------
    @staticmethod
    def normalize_rotation(Rm):
        U, _, Vt = np.linalg.svd(Rm)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    @staticmethod
    def rot_deg(Rm):
        trace = np.trace(Rm)
        trace = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))

    def reset(self):
        self.prev_rgbd = None
        self.prev_frame = None
        self.prev_cam_ts = None
        self.T_c_w = np.eye(4, dtype=np.float64)
        self.last_trans = np.zeros(3, dtype=np.float64)

    def get_intrinsic(self):
        return self.preprocessor.get_intrinsic()

    def _is_frame_trackable(self, frame: RGBDFrame) -> bool:
        return frame.has_valid_depth and frame.valid_pixel_count >= self.min_valid_pixels

    def _make_lost_result(self, frame: RGBDFrame, reason: str, extras=None):
        ext = {} if extras is None else dict(extras)
        ext["reason"] = reason
        return TrackingResult(
            frame_id=frame.frame_id,
            timestamp=frame.device_timestamp,
            success=False,
            T_wc=self.T_c_w.copy(),
            score=0.0,
            mode="lost",
            extras=ext,
        )

    def _compute_imu_delta(self, t0, t1):
        if self.imu_manager is None:
            return np.eye(3, dtype=np.float64), 0.0, False

        dR_imu = self.imu_manager.get_delta_rotation(t0, t1)
        if dR_imu is None:
            return np.eye(3, dtype=np.float64), 0.0, False

        dR_imu_cam = self.imu_manager.imu_delta_to_cam_delta(dR_imu)
        imu_rot_deg = self.imu_manager.rot_deg(dR_imu_cam)

        ok = (
            self.min_imu_rot_deg_per_frame <= imu_rot_deg <= self.max_rotation_deg_per_frame
        )
        return dR_imu_cam, imu_rot_deg, ok

    def track(self, frame: RGBDFrame) -> TrackingResult:
        frame.validate()

        if not self._is_frame_trackable(frame):
            return self._make_lost_result(
                frame,
                "not_enough_valid_pixels",
                {"valid_pixel_count": frame.valid_pixel_count},
            )

        current_rgbd, rgbd_info = self.preprocessor.preprocess(frame.color, frame.depth)
        if current_rgbd is None:
            return self._make_lost_result(frame, "rgbd_preprocess_failed", rgbd_info)

        cam_ts = float(frame.device_timestamp)

        # ---------- 第一帧 ----------
        if self.prev_rgbd is None:
            self.prev_rgbd = current_rgbd
            self.prev_frame = frame
            self.prev_cam_ts = cam_ts
            self.T_c_w = np.eye(4, dtype=np.float64)

            return TrackingResult(
                frame_id=frame.frame_id,
                timestamp=frame.device_timestamp,
                success=True,
                T_wc=self.T_c_w.copy(),
                score=1.0,
                mode="init",
                extras={
                    "valid_pixel_count": frame.valid_pixel_count,
                    "rgbd_info": rgbd_info,
                },
            )

        # ---------- IMU delta 仅用于 VO 初值 / 失败兜底 ----------
        dR_imu_cam, imu_rot_deg, imu_delta_available = self._compute_imu_delta(
            self.prev_cam_ts, cam_ts
        )

        init_delta = np.eye(4, dtype=np.float64)
        if self.imu_as_vo_init_only and imu_delta_available:
            init_delta[:3, :3] = dR_imu_cam

        success, delta_refined, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            self.prev_rgbd,
            current_rgbd,
            self.preprocessor.get_intrinsic(),
            init_delta,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.option
        )

        bad_frame = False
        fused_delta = np.eye(4, dtype=np.float64)
        fused_R = np.eye(3, dtype=np.float64)

        info_trace = float(np.trace(info)) if (success and info is not None) else 0.0
        vo_rot_deg = 0.0
        trans_refined = np.zeros(3, dtype=np.float64)
        accepted_t = False
        odom_t_norm = 0.0

        # ---------- 旋转策略 ----------
        # VO 好 -> 用 VO
        # VO 差/失败 -> 小角度 IMU 兜底
        if success and info is not None:
            dR_vo = self.normalize_rotation(delta_refined[:3, :3])
            vo_rot_deg = self.rot_deg(dR_vo)

            vo_rot_ok = (vo_rot_deg <= self.max_rotation_deg_per_frame)
            vo_info_ok = (info_trace >= self.min_info_trace)

            if vo_rot_ok and vo_info_ok:
                fused_R = dR_vo
            else:
                if (
                    self.allow_imu_fallback_when_vo_fails
                    and imu_delta_available
                    and imu_rot_deg <= 5.0
                ):
                    fused_R = dR_imu_cam
                else:
                    bad_frame = True
        else:
            if (
                self.allow_imu_fallback_when_vo_fails
                and imu_delta_available
                and imu_rot_deg <= 5.0
            ):
                fused_R = dR_imu_cam
            else:
                bad_frame = True

        if bad_frame:
            self.prev_rgbd = current_rgbd
            self.prev_frame = frame
            self.prev_cam_ts = cam_ts

            return self._make_lost_result(
                frame,
                "odometry_failed_or_gated",
                {
                    "imu_rot_deg": imu_rot_deg,
                    "imu_delta_available": imu_delta_available,
                    "info_trace": info_trace,
                    "vo_success": success,
                },
            )

        fused_delta[:3, :3] = fused_R

        # ---------- 平移策略 ----------
        # 平移只信 VO，不信 IMU
        if success and info is not None:
            raw_t = delta_refined[:3, 3].copy()
            odom_t_norm = float(np.linalg.norm(raw_t))

            # 异常大平移，整帧拒绝
            if odom_t_norm > 0.05:
                self.prev_rgbd = current_rgbd
                self.prev_frame = frame
                self.prev_cam_ts = cam_ts

                return self._make_lost_result(
                    frame,
                    "translation_too_large",
                    {
                        "odom_t_norm": odom_t_norm,
                        "imu_rot_deg": imu_rot_deg,
                        "info_trace": info_trace,
                    },
                )

            if (
                info_trace >= self.min_info_trace
                and self.min_translation_per_frame <= odom_t_norm <= self.max_translation_per_frame
            ):
                rot_used_for_trans_gate = self.rot_deg(fused_R)

                if rot_used_for_trans_gate >= self.rotation_dominant_angle_deg:
                    accepted_t = (odom_t_norm <= self.max_translation_when_rotating)
                else:
                    accepted_t = True

                if accepted_t:
                    trans_refined = (
                        self.trans_smooth_alpha * raw_t
                        + (1.0 - self.trans_smooth_alpha) * self.last_trans
                    )
                    self.last_trans = trans_refined.copy()
                else:
                    self.last_trans[:] = 0.0
            else:
                self.last_trans[:] = 0.0
        else:
            self.last_trans[:] = 0.0

        fused_delta[:3, 3] = trans_refined

        if self.debug_print_odom:
            if success and info is not None:
                print(
                    f"[ODOM] frame={frame.frame_id}, "
                    f"vo_rot={vo_rot_deg:.2f}deg, "
                    f"imu_rot={imu_rot_deg:.2f}deg, "
                    f"info={info_trace:.1f}, "
                    f"t_raw={odom_t_norm:.4f}m, "
                    f"accepted_t={accepted_t}, "
                    f"t_use={np.linalg.norm(trans_refined):.4f}m, "
                    f"imu_delta_ok={imu_delta_available}"
                )
            else:
                print(
                    f"[ODOM] frame={frame.frame_id}, "
                    f"failed, imu_rot={imu_rot_deg:.2f}deg, "
                    f"imu_delta_ok={imu_delta_available}"
                )

        # ---------- 位姿链更新 ----------
        # world -> current_camera
        self.T_c_w = fused_delta @ self.T_c_w

        self.prev_rgbd = current_rgbd
        self.prev_frame = frame
        self.prev_cam_ts = cam_ts

        score = 0.0
        if info_trace > 0:
            score = min(1.0, info_trace / (self.min_info_trace * 2.0))

        return TrackingResult(
            frame_id=frame.frame_id,
            timestamp=frame.device_timestamp,
            success=True,
            T_wc=self.T_c_w.copy(),
            score=score,
            mode="tracking",
            extras={
                "valid_pixel_count": frame.valid_pixel_count,
                "rgbd_info": rgbd_info,
                "imu_rot_deg": imu_rot_deg,
                "imu_delta_available": imu_delta_available,
                "vo_rot_deg": vo_rot_deg,
                "info_trace": info_trace,
                "translation_norm": float(np.linalg.norm(trans_refined)),
            },
        )