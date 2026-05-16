import time
import numpy as np
import open3d as o3d

from ..common.types import RGBDFrame, TrackingResult
from ..input.rgbd_preprocessor import RGBDPreprocessor

class Tracker:
    """
    基于 RGBD Odometry 的 tracking 前端。
    时间统一使用 frame.device_timestamp。
    """

    def __init__(
        self,
        imu_manager=None,
        width=320,
        height=240,
        fx=262.5,
        fy=262.5,
        cx=159.75,
        cy=119.75,
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

        imu_world_initializer=None,
        debug_print_odom=False,

        # profiling
        enable_profile=True,
        profile_print_interval=30,
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
            convert_rgb_to_intensity=True,
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
        self.enable_profile = enable_profile
        self.profile_print_interval = max(1, int(profile_print_interval))

        self.prev_rgbd = None
        self.prev_pcd = None
        self.prev_frame = None
        self.prev_cam_ts = None

        # 保存 world -> current_camera 的外参，兼容 TSDF integrate 用法
        self.T_c_w = np.eye(4, dtype=np.float64)
        self.imu_world_initializer = imu_world_initializer

        self.last_trans = np.zeros(3, dtype=np.float64)
        self.icp_max_correspondence_distance = 0.07
        self.icp_max_iteration = 20
        self.icp_min_fitness = 0.08
        self.icp_max_rmse = 0.06
        self.icp_voxel_size = 0.025
        
        # print(
        #     f"[Tracker.__init__] width={width}, height={height}, "
        #     f"fx={fx}, fy={fy}, cx={cx}, cy={cy}, "
        #     f"depth_trunc={depth_trunc}, min_valid_pixels={min_valid_pixels}, "
        #     f"enable_profile={self.enable_profile}, "
        #     f"profile_print_interval={self.profile_print_interval}"
        # )


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
        self.prev_pcd = None
        self.prev_frame = None
        self.prev_cam_ts = None
        self.T_c_w = np.eye(4, dtype=np.float64)
        self.last_trans = np.zeros(3, dtype=np.float64)

    def reset_lost_counters_for_resume(self) -> None:
        """Symmetry with GpuICPTracker; cpu_rgbd path has no lost_count streak."""

    def get_intrinsic(self):
        return self.preprocessor.get_intrinsic()

    def _initial_world_to_camera(self, frame: RGBDFrame):
        if self.imu_world_initializer is None:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.pose(
                    f"tracker initial pose: frame={frame.frame_id} "
                    "IMU initializer missing; using identity world_to_camera",
                    frame_id=frame.frame_id,
                    force=True,
                )
            return np.eye(4, dtype=np.float64), {
                "imu_world_init_success": False,
                "imu_world_init_reason": "initializer_missing",
            }

        result = self.imu_world_initializer.initialize_world_to_camera(
            frame_timestamp=frame.device_timestamp,
        )
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.pose(
                f"tracker initial pose: frame={frame.frame_id} "
                f"imu_success={result.success} "
                f"reason={result.reason} "
                f"T_wc_r0={result.T_world_to_camera[0].tolist()} "
                f"T_wc_r1={result.T_world_to_camera[1].tolist()} "
                f"T_wc_r2={result.T_world_to_camera[2].tolist()}",
                frame_id=frame.frame_id,
                force=True,
            )
        return result.T_world_to_camera.copy(), {
            "imu_world_init_success": result.success,
            "imu_world_init_reason": result.reason,
            "imu_quaternion_wxyz": (
                None
                if result.imu_quaternion_wxyz is None
                else result.imu_quaternion_wxyz.copy()
            ),
        }

    def _is_frame_trackable(self, frame: RGBDFrame) -> bool:
        return frame.has_valid_depth and frame.valid_pixel_count >= self.min_valid_pixels

    def _make_lost_result(self, frame: RGBDFrame, reason: str, extras=None):
        ext = {} if extras is None else dict(extras)
        ext["reason"] = reason
        ext["tracker_backend"] = "cpu_rgbd"
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

    def _need_print(self, frame_id: int) -> bool:
        return frame_id % self.profile_print_interval == 0

    def _make_depth_pcd(self, rgbd, frame_id=None):
        """
        CPU fallback tracking uses depth ICP instead of legacy RGBD odometry.
        PCAC currently provides pseudo-color depth visualization, not a real RGB
        texture stream, so color-based odometry is not reliable here.
        """
        if rgbd is None:
            return None, {"reason": "rgbd_is_none"}

        depth = np.asarray(rgbd.depth)
        if depth.size == 0:
            return None, {"reason": "depth_empty"}

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(self.preprocessor.depth_trunc))
        valid_count = int(np.count_nonzero(valid))

        if valid_count < self.min_valid_pixels:
            return None, {
                "reason": "too_few_valid_depth_for_cpu_icp",
                "valid_count": valid_count,
                "min_valid_pixels": self.min_valid_pixels,
                "depth_min": float(depth[valid].min()) if valid_count > 0 else None,
                "depth_max": float(depth[valid].max()) if valid_count > 0 else None,
            }

        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth))
        try:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_img,
                self.preprocessor.get_intrinsic(),
                depth_scale=1.0,
                depth_trunc=float(self.preprocessor.depth_trunc),
                stride=2,
            )
        except Exception as e:
            return None, {"reason": "create_depth_pcd_failed", "error": repr(e)}

        if pcd is None or len(pcd.points) == 0:
            return None, {
                "reason": "depth_pcd_empty",
                "valid_count": valid_count,
            }

        try:
            pcd = pcd.voxel_down_sample(self.icp_voxel_size)
        except Exception:
            pass

        if pcd is None or len(pcd.points) == 0:
            return None, {
                "reason": "depth_pcd_empty_after_downsample",
                "valid_count": valid_count,
            }

        try:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=20)
            )
        except Exception:
            pass

        return pcd, {
            "valid_count": valid_count,
            "pcd_points": len(pcd.points),
            "icp_voxel_size": self.icp_voxel_size,
            "frame_id": frame_id,
        }

    def _run_depth_icp(self, source_pcd, target_pcd):
        if source_pcd is None or target_pcd is None:
            return None, "missing_pcd"
        if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
            return None, "empty_pcd"

        used_point_to_plane = False
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        try:
            if target_pcd.has_normals():
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
                used_point_to_plane = True
        except Exception:
            pass

        try:
            return (
                o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    self.icp_max_correspondence_distance,
                    np.eye(4, dtype=np.float64),
                    estimation,
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.icp_max_iteration
                    ),
                ),
                None,
            )
        except Exception as e:
            if used_point_to_plane:
                try:
                    return (
                        o3d.pipelines.registration.registration_icp(
                            source_pcd,
                            target_pcd,
                            self.icp_max_correspondence_distance,
                            np.eye(4, dtype=np.float64),
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                max_iteration=self.icp_max_iteration
                            ),
                        ),
                        f"point_to_plane_failed:{repr(e)}",
                    )
                except Exception as e2:
                    return None, f"point_to_point_failed:{repr(e2)}"
            return None, repr(e)

    def _maybe_print_profile(
        self,
        frame_id,
        t_validate_ms,
        t_pre_ms,
        t_imu_ms,
        t_odom_ms,
        t_post_ms,
        t_total_ms,
    ):
        if not self.enable_profile:
            return
        if not self._need_print(frame_id):
            return

        print(
            f"[TRACK_PROFILE] frame={frame_id} "
            f"validate={t_validate_ms:.2f}ms "
            f"pre={t_pre_ms:.2f}ms "
            f"imu={t_imu_ms:.2f}ms "
            f"odom={t_odom_ms:.2f}ms "
            f"post={t_post_ms:.2f}ms "
            f"total={t_total_ms:.2f}ms"
        )

    def _maybe_print_rgbd_shape(self, frame_id, current_rgbd):
        if current_rgbd is None:
            return
        if not self._need_print(frame_id):
            return

        print(
            "tracker rgbd:",
            np.asarray(current_rgbd.color).shape,
            np.asarray(current_rgbd.depth).shape,
        )

    def _maybe_print_odom_summary(
        self,
        frame_id,
        success,
        vo_rot_deg,
        imu_rot_deg,
        info_trace,
        odom_t_norm,
        accepted_t,
        trans_refined,
        imu_delta_available,
    ):
        if not self.debug_print_odom:
            return
        if not self._need_print(frame_id):
            return

        if success:
            print(
                f"[ODOM] frame={frame_id}, "
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
                f"[ODOM] frame={frame_id}, "
                f"failed, imu_rot={imu_rot_deg:.2f}deg, "
                f"imu_delta_ok={imu_delta_available}"
            )

    def track(self, frame: RGBDFrame) -> TrackingResult:
        t0 = time.perf_counter()

        # ---------- validate ----------
        frame.validate()
        t1 = time.perf_counter()

        if not self._is_frame_trackable(frame):
            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (t_end - t0) * 1000.0,
            )
            return self._make_lost_result(
                frame,
                "not_enough_valid_pixels",
                {"valid_pixel_count": frame.valid_pixel_count},
            )

        # ---------- preprocess ----------
        current_rgbd, rgbd_info = self.preprocessor.preprocess(frame.color, frame.depth)
        t2 = time.perf_counter()

        self._maybe_print_rgbd_shape(frame.frame_id, current_rgbd)

        if current_rgbd is None:
            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                0.0,
                0.0,
                0.0,
                (t_end - t0) * 1000.0,
            )
            return self._make_lost_result(frame, "rgbd_preprocess_failed", rgbd_info)

        current_pcd, pcd_info = self._make_depth_pcd(current_rgbd, frame_id=frame.frame_id)
        if current_pcd is None:
            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                0.0,
                0.0,
                0.0,
                (t_end - t0) * 1000.0,
            )
            return self._make_lost_result(
                frame,
                "depth_pcd_failed",
                {
                    "valid_pixel_count": frame.valid_pixel_count,
                    "rgbd_info": rgbd_info,
                    "pcd_info": pcd_info,
                },
            )

        cam_ts = float(frame.device_timestamp)

        # ---------- 第一帧 ----------
        if self.prev_rgbd is None:
            self.T_c_w, imu_init_info = self._initial_world_to_camera(frame)
            if (
                not imu_init_info.get("imu_world_init_success", False)
                and getattr(self.imu_world_initializer, "required", False)
            ):
                return self._make_lost_result(
                    frame,
                    "imu_world_init_failed",
                    {
                        "valid_pixel_count": frame.valid_pixel_count,
                        "rgbd_info": rgbd_info,
                        **imu_init_info,
                    },
                )

            self.prev_rgbd = current_rgbd
            self.prev_pcd = current_pcd
            self.prev_frame = frame
            self.prev_cam_ts = cam_ts

            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                0.0,
                0.0,
                0.0,
                (t_end - t0) * 1000.0,
            )

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
                    "pcd_info": pcd_info,
                    "tracker_backend": "cpu_rgbd",
                    **imu_init_info,
                },
            )

        # ---------- IMU delta ----------
        dR_imu_cam, imu_rot_deg, imu_delta_available = self._compute_imu_delta(
            self.prev_cam_ts, cam_ts
        )
        t3 = time.perf_counter()

        # ---------- depth ICP odometry ----------
        init_delta = np.eye(4, dtype=np.float64)
        if self.imu_as_vo_init_only and imu_delta_available:
            init_delta[:3, :3] = dR_imu_cam

        odom_error = None
        result, icp_error = self._run_depth_icp(current_pcd, self.prev_pcd)
        if result is None:
            success = False
            delta_refined = np.eye(4, dtype=np.float64)
            info = None
            odom_error = icp_error
            icp_fitness = 0.0
            icp_rmse = 0.0
        else:
            icp_fitness = float(result.fitness)
            icp_rmse = float(result.inlier_rmse)
            success = (
                icp_fitness >= self.icp_min_fitness
                and icp_rmse <= self.icp_max_rmse
            )
            # Open3D ICP returns current_camera -> previous_camera.
            # Mapper needs world -> current_camera, so use the inverse delta.
            try:
                delta_refined = np.linalg.inv(np.asarray(result.transformation, dtype=np.float64))
            except Exception as e:
                success = False
                delta_refined = np.eye(4, dtype=np.float64)
                odom_error = f"icp_transform_inverse_failed:{repr(e)}"
            if success:
                info = np.eye(6, dtype=np.float64) * (self.min_info_trace / 6.0)
            else:
                info = np.eye(6, dtype=np.float64) * max(icp_fitness, 0.0)
        t4 = time.perf_counter()

        # ---------- post / gating ----------
        bad_frame = False
        fused_delta = np.eye(4, dtype=np.float64)
        fused_R = np.eye(3, dtype=np.float64)

        info_trace = float(np.trace(info)) if (success and info is not None) else 0.0
        vo_rot_deg = 0.0
        trans_refined = np.zeros(3, dtype=np.float64)
        accepted_t = False
        odom_t_norm = 0.0

        # 旋转策略：
        # 1. VO 好：优先用 VO 旋转
        # 2. VO 差/失败：允许用小角度 IMU 兜底
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
            t5 = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (t3 - t2) * 1000.0,
                (t4 - t3) * 1000.0,
                (t5 - t4) * 1000.0,
                (t5 - t0) * 1000.0,
            )

            self._maybe_print_odom_summary(
                frame_id=frame.frame_id,
                success=False,
                vo_rot_deg=vo_rot_deg,
                imu_rot_deg=imu_rot_deg,
                info_trace=info_trace,
                odom_t_norm=odom_t_norm,
                accepted_t=accepted_t,
                trans_refined=trans_refined,
                imu_delta_available=imu_delta_available,
            )

            return self._make_lost_result(
                frame,
                "odometry_failed_or_gated",
                {
                    "imu_rot_deg": imu_rot_deg,
                    "imu_delta_available": imu_delta_available,
                    "info_trace": info_trace,
                    "vo_success": success,
                    "odom_error": odom_error,
                    "icp_fitness": icp_fitness,
                    "icp_rmse": icp_rmse,
                    "pcd_info": pcd_info,
                },
            )

        fused_delta[:3, :3] = fused_R

        # 平移策略：
        # 1. 平移只信 VO
        # 2. IMU 不提供平移
        if success and info is not None:
            raw_t = delta_refined[:3, 3].copy()
            odom_t_norm = float(np.linalg.norm(raw_t))

            # 明显异常的大平移，直接拒绝
            if odom_t_norm > 0.05:
                self.last_trans[:] = 0.0

                t5 = time.perf_counter()
                self._maybe_print_profile(
                    frame.frame_id,
                    (t1 - t0) * 1000.0,
                    (t2 - t1) * 1000.0,
                    (t3 - t2) * 1000.0,
                    (t4 - t3) * 1000.0,
                    (t5 - t4) * 1000.0,
                    (t5 - t0) * 1000.0,
                )

                self._maybe_print_odom_summary(
                    frame_id=frame.frame_id,
                    success=False,
                    vo_rot_deg=vo_rot_deg,
                    imu_rot_deg=imu_rot_deg,
                    info_trace=info_trace,
                    odom_t_norm=odom_t_norm,
                    accepted_t=False,
                    trans_refined=np.zeros(3, dtype=np.float64),
                    imu_delta_available=imu_delta_available,
                )

                return self._make_lost_result(
                    frame,
                    "translation_too_large",
                    {
                        "odom_t_norm": odom_t_norm,
                        "imu_rot_deg": imu_rot_deg,
                        "info_trace": info_trace,
                        "icp_fitness": icp_fitness,
                        "icp_rmse": icp_rmse,
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

        self._maybe_print_odom_summary(
            frame_id=frame.frame_id,
            success=True,
            vo_rot_deg=vo_rot_deg,
            imu_rot_deg=imu_rot_deg,
            info_trace=info_trace,
            odom_t_norm=odom_t_norm,
            accepted_t=accepted_t,
            trans_refined=trans_refined,
            imu_delta_available=imu_delta_available,
        )

        # ---------- 位姿链更新 ----------
        # world -> current_camera
        self.T_c_w = fused_delta @ self.T_c_w

        self.prev_rgbd = current_rgbd
        self.prev_pcd = current_pcd
        self.prev_frame = frame
        self.prev_cam_ts = cam_ts

        score = 0.0
        if info_trace > 0:
            score = min(1.0, info_trace / (self.min_info_trace * 2.0))

        t5 = time.perf_counter()
        self._maybe_print_profile(
            frame.frame_id,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            (t4 - t3) * 1000.0,
            (t5 - t4) * 1000.0,
            (t5 - t0) * 1000.0,
        )

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
                "pcd_info": pcd_info,
                "imu_rot_deg": imu_rot_deg,
                "imu_delta_available": imu_delta_available,
                "vo_rot_deg": vo_rot_deg,
                "info_trace": info_trace,
                "translation_norm": float(np.linalg.norm(trans_refined)),
                "odom_error": odom_error,
                "icp_fitness": icp_fitness,
                "icp_rmse": icp_rmse,
                "tracker_backend": "cpu_rgbd",
            },
        )