# tracking/gpu_icp_tracker.py

import time
import math
import numpy as np
import open3d as o3d

try:
    import cv2
except Exception:
    cv2 = None

from ..common.types import RGBDFrame, TrackingResult

class GpuICPTracker:
    """
    CUDA Tensor ICP tracker.

    - 独立于旧 tracker.py
    - 不使用 IMU
    - 不使用 Open3D legacy odometry
    - Open3D 内部计算尽量全走 CUDA tensor API
    - 输出 TrackingResult，保持后续 TrackingWorker / Mapper 不需要大改
    """

    def __init__(
        self,

        width=320,
        height=240,
        fx=262.5,
        fy=262.5,
        cx=159.75,
        cy=119.75,

        input_color_is_bgr=False,
        depth_scale=None,
        depth_trunc=3.0,
        min_valid_pixels=500,

        max_rotation_deg_per_frame=20.0,
        rotation_dominant_angle_deg=0.25,
        max_translation_when_rotating=0.015,
        min_translation_per_frame=0.0008,
        max_translation_per_frame=0.05,
        trans_smooth_alpha=0.3,

        # GPU / ICP 参数
        device="CUDA:0",
        tracking_voxel_size=0.015,
        tracking_pcd_stride=2,
        icp_max_correspondence_distance=0.05,
        icp_max_iteration=20,
        normal_radius=0.04,
        normal_max_nn=30,

        # ICP 质量门限
        min_icp_fitness=0.15,
        max_icp_rmse=0.04,

        # 是否允许 point-to-plane 失败后回退 point-to-point
        allow_point_to_point_fallback=True,

        # local map / frame-to-model tracking
        local_map=None,
        use_map_tracking=True,
        map_tracking_min_points=2000,
        map_tracking_max_corr=0.10,

        map_min_icp_fitness=0.22,
        map_max_icp_rmse=0.045,
        map_max_delta_trans=0.06,
        map_max_delta_rot_deg=10.0,
        map_icp_cooldown_after_fail=10,
        relocalize_try_interval=5,
        lost_print_interval=30,
        relocalize_use_full_map_fallback=True,
        relocalize_full_map_max_corr=0.30,
        relocalize_full_map_min_fitness=0.08,
        relocalize_full_map_max_rmse=0.09,
        imu_world_initializer=None,

        logger=None,

    ):
        self.width = int(width)
        self.height = int(height)

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        self.input_color_is_bgr = bool(input_color_is_bgr)
        self.depth_scale = depth_scale
        self.depth_trunc = float(depth_trunc)
        self.min_valid_pixels = int(min_valid_pixels)

        self.max_rotation_deg_per_frame = float(max_rotation_deg_per_frame)
        self.rotation_dominant_angle_deg = float(rotation_dominant_angle_deg)
        self.max_translation_when_rotating = float(max_translation_when_rotating)
        self.min_translation_per_frame = float(min_translation_per_frame)
        self.max_translation_per_frame = float(max_translation_per_frame)
        self.trans_smooth_alpha = float(trans_smooth_alpha)

        self.logger = logger

        self.device = o3d.core.Device(device)

        self.tracking_voxel_size = float(tracking_voxel_size)
        self.tracking_pcd_stride = max(1, int(tracking_pcd_stride))
        self.icp_max_correspondence_distance = float(icp_max_correspondence_distance)
        self.icp_max_iteration = int(icp_max_iteration)
        self.normal_radius = float(normal_radius)
        self.normal_max_nn = int(normal_max_nn)

        self.min_icp_fitness = float(min_icp_fitness)
        self.max_icp_rmse = float(max_icp_rmse)

        self.allow_point_to_point_fallback = bool(allow_point_to_point_fallback)

        self.local_map = local_map
        self.use_map_tracking = bool(use_map_tracking)
        self.map_tracking_min_points = int(map_tracking_min_points)
        self.map_tracking_max_corr = float(map_tracking_max_corr)

        # frame-to-model 比 frame-to-frame 更危险：
        # 一旦错配，会直接把当前帧吸到错误地图位置。
        # 所以这里单独用更严格的接受门限。
        self.map_min_icp_fitness = float(map_min_icp_fitness)
        self.map_max_icp_rmse = float(map_max_icp_rmse)

        # frame-to-model 单次相对上一次位姿的最大跳变。
        # 超过就认为可能错配，不允许 integrate。
        self.map_max_delta_trans = float(map_max_delta_trans)
        self.map_max_delta_rot_deg = float(map_max_delta_rot_deg)

        self.intrinsic_t = o3d.core.Tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=o3d.core.Dtype.Float32,
            device=self.device,
        )

        self.identity_extrinsic_t = o3d.core.Tensor.eye(
            4,
            dtype=o3d.core.Dtype.Float32,
            device=self.device,
        )

        self.prev_pcd = None
        self.prev_frame = None
        self.prev_cam_ts = None

        self.T_c_w = np.eye(4, dtype=np.float64)
        self.last_trans = np.zeros(3, dtype=np.float64)
        self.last_delta_icp = np.eye(4, dtype=np.float64)

        # tracking lost / relocalization state
        self.is_lost = False
        self.lost_count = 0
        self.last_lost_reason = None

        self.map_icp_cooldown_frames = 0
        self.map_icp_cooldown_after_fail = max(0, int(map_icp_cooldown_after_fail))

        self.relocalize_try_interval = max(1, int(relocalize_try_interval))
        self.lost_print_interval = max(1, int(lost_print_interval))
        self.relocalize_use_full_map_fallback = bool(relocalize_use_full_map_fallback)
        self.relocalize_full_map_max_corr = float(relocalize_full_map_max_corr)
        self.relocalize_full_map_min_fitness = float(relocalize_full_map_min_fitness)
        self.relocalize_full_map_max_rmse = float(relocalize_full_map_max_rmse)
        self.imu_world_initializer = imu_world_initializer

        if self.logger is not None:
            self.logger.status(
                f"initialized: "
                f"device={self.device}, "
                f"size={self.width}x{self.height}, "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
                f"depth_scale={self.depth_scale}, "
                f"depth_trunc={self.depth_trunc}, "
                f"voxel={self.tracking_voxel_size}, "
                f"max_corr={self.icp_max_correspondence_distance}, "
                f"iter={self.icp_max_iteration}, "
                f"normal_radius={self.normal_radius}, "
                f"normal_max_nn={self.normal_max_nn}, "
                f"fallback_p2p={self.allow_point_to_point_fallback}, "
                f"use_map_tracking={self.use_map_tracking}, "
                f"map_min_points={self.map_tracking_min_points}, "
                f"map_max_corr={self.map_tracking_max_corr}",
                force=True,
            )

    # ============================================================
    # 基础工具
    # ============================================================

    def reset(self):
        self.prev_pcd = None
        self.prev_frame = None
        self.prev_cam_ts = None
        self.T_c_w = np.eye(4, dtype=np.float64)
        self.last_trans[:] = 0.0
        self.last_delta_icp = np.eye(4, dtype=np.float64)

        self.is_lost = False
        self.lost_count = 0
        self.last_lost_reason = None
        self.map_icp_cooldown_frames = 0

    def reset_lost_counters_for_resume(self) -> None:
        """
        Called when external logic resumes tracking after a hard pause.
        Clears lost streak without discarding prev_pcd / pose so relocalization can proceed.
        """
        self.lost_count = 0
        self.is_lost = False
        self.last_lost_reason = None
        self.map_icp_cooldown_frames = 0

    def get_intrinsic(self):
        """
        兼容旧接口。
        某些外部模块可能会调用 tracker.get_intrinsic()。
        这里尽量返回 legacy PinholeCameraIntrinsic。
        如果你的环境 legacy 可用，这个没问题。
        """
        return o3d.camera.PinholeCameraIntrinsic(
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )

    def _initial_world_to_camera(self, frame: RGBDFrame):
        if self.imu_world_initializer is None:
            if self.logger is not None:
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
        if self.logger is not None:
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
        v = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(v)))

    def _maybe_print_profile(
        self,
        frame_id,
        t_validate_ms,
        t_pre_ms,
        t_pcd_ms,
        t_icp_ms,
        t_post_ms,
        t_total_ms,
        tracking_mode="unknown",
    ):
        if self.logger is None:
            return

        self.logger.profile(
            f"frame={frame_id} "
            f"mode={tracking_mode} "
            f"validate={t_validate_ms:.2f}ms "
            f"pre={t_pre_ms:.2f}ms "
            f"pcd={t_pcd_ms:.2f}ms "
            f"icp={t_icp_ms:.2f}ms "
            f"post={t_post_ms:.2f}ms "
            f"total={t_total_ms:.2f}ms",
            frame_id=frame_id,
        )

    def _make_lost_result(self, frame: RGBDFrame, reason: str, extras=None):
        ext = {} if extras is None else dict(extras)
        ext["reason"] = reason
        ext["tracker_backend"] = "gpu_icp"

        return TrackingResult(
            frame_id=frame.frame_id,
            timestamp=frame.device_timestamp,
            success=False,
            T_wc=self.T_c_w.copy(),
            score=0.0,
            mode="lost",
            extras=ext,
        )
    
    def _enter_lost(self, frame: RGBDFrame, reason: str):
        """
        进入 lost 状态。这里只负责状态和日志，不负责返回 TrackingResult。
        """
        was_lost = self.is_lost

        self.is_lost = True
        self.lost_count += 1
        self.last_lost_reason = reason

        if (not was_lost) or (self.lost_count % self.lost_print_interval == 0):
            if self.logger is not None:
                self.logger.tracking_state(
                    f"tracking lost: frame={frame.frame_id} "
                    f"lost_count={self.lost_count} "
                    f"reason={reason}. "
                    f"Mapping paused. Move camera slowly back to a previously mapped area.",
                    frame_id=frame.frame_id,
                    force=not was_lost,
                )

    def _leave_lost(self, frame: RGBDFrame, tracking_mode: str, fitness: float, rmse: float):
        """
        从 lost 状态恢复。
        """
        if self.is_lost and self.logger is not None:
            self.logger.relocalization(
                f"relocalized: frame={frame.frame_id} "
                f"mode={tracking_mode} "
                f"fitness={fitness:.3f} "
                f"rmse={rmse:.5f}. "
                f"Mapping resumed.",
                frame_id=frame.frame_id,
                force=True,
            )

        self.is_lost = False
        self.lost_count = 0
        self.last_lost_reason = None

    def _log_relocalize_attempt(self, frame: RGBDFrame, status: str, target_points=0, reason=None):
        if self.logger is None:
            return
        self.logger.relocalization(
            f"attempt: frame={frame.frame_id} "
            f"lost_count={self.lost_count} "
            f"status={status} "
            f"target_points={target_points} "
            f"reason={reason}",
            frame_id=frame.frame_id,
            force=status in ("try", "success"),
        )

    def _is_frame_trackable(self, frame: RGBDFrame) -> bool:
        return frame.has_valid_depth and frame.valid_pixel_count >= self.min_valid_pixels

    # ============================================================
    # depth 预处理
    # ============================================================

    def _auto_depth_scale(self, depth_np: np.ndarray) -> float:
        """
        根据 depth dtype / 数值范围自动估计 depth_scale。

        约定：
        - uint16 通常是毫米，scale=1000
        - float32 / float64 如果最大值很大，也可能是毫米
        - float32 / float64 如果数值在几米范围内，scale=1
        """
        if self.depth_scale is not None:
            return float(self.depth_scale)

        if depth_np.dtype == np.uint16:
            return 1000.0

        finite = depth_np[np.isfinite(depth_np)]
        finite = finite[finite > 0]

        if finite.size == 0:
            return 1.0

        p95 = float(np.percentile(finite, 95))

        # 经验判断：float 深度如果 95 分位数大于 20，大概率是毫米
        if p95 > 20.0:
            return 1000.0

        return 1.0

    def _resize_depth_if_needed(self, depth_np: np.ndarray):
        h, w = depth_np.shape[:2]

        if w == self.width and h == self.height:
            return np.ascontiguousarray(depth_np), {
                "resized": False,
                "source_shape": (h, w),
                "target_shape": (self.height, self.width),
            }

        if cv2 is None:
            # 常见 640x480 -> 320x240 的整数降采样 fallback
            if w % self.width == 0 and h % self.height == 0:
                sx = w // self.width
                sy = h // self.height
                depth_small = depth_np[::sy, ::sx]
                return np.ascontiguousarray(depth_small), {
                    "resized": True,
                    "resize_backend": "numpy_stride",
                    "source_shape": (h, w),
                    "target_shape": (self.height, self.width),
                }

            raise RuntimeError(
                "cv2 is not available, and input size cannot be integer-stride resized. "
                f"input={(h, w)}, target={(self.height, self.width)}"
            )

        # 深度图建议最近邻，避免插值制造虚假深度
        depth_small = cv2.resize(
            depth_np,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )

        return np.ascontiguousarray(depth_small), {
            "resized": True,
            "resize_backend": "cv2_nearest",
            "source_shape": (h, w),
            "target_shape": (self.height, self.width),
        }

    def _preprocess_depth_to_tensor_image(self, depth_np: np.ndarray):
        """
        输入 CPU numpy depth。
        输出 CUDA o3d.t.geometry.Image。
        """
        if depth_np is None:
            return None, {"reason": "depth_is_none"}

        if not isinstance(depth_np, np.ndarray):
            return None, {
                "reason": "depth_not_numpy",
                "depth_type": str(type(depth_np)),
            }

        if depth_np.ndim != 2:
            return None, {
                "reason": "depth_not_2d",
                "shape": tuple(depth_np.shape),
            }

        depth_np = np.ascontiguousarray(depth_np)

        depth_scale = self._auto_depth_scale(depth_np)

        if np.issubdtype(depth_np.dtype, np.floating):
            depth_np = np.nan_to_num(
                depth_np,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

        elif depth_np.dtype != np.uint16:
            depth_np = depth_np.astype(np.float32)

        depth_np, resize_info = self._resize_depth_if_needed(depth_np)

        valid_count = int(np.count_nonzero(depth_np))
        valid_ratio = float(valid_count) / float(depth_np.size)

        if valid_count < self.min_valid_pixels:
            return None, {
                "reason": "too_few_valid_depth_after_resize",
                "valid_count": valid_count,
                "valid_ratio": valid_ratio,
                "depth_scale": depth_scale,
                **resize_info,
            }

        if depth_np.dtype == np.uint16:
            depth_dtype = o3d.core.Dtype.UInt16
        else:
            depth_np = depth_np.astype(np.float32, copy=False)
            depth_dtype = o3d.core.Dtype.Float32

        depth_tensor = o3d.core.Tensor(
            depth_np,
            dtype=depth_dtype,
            device=self.device,
        )

        depth_img = o3d.t.geometry.Image(depth_tensor)

        info = {
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
            "depth_scale": depth_scale,
            "depth_trunc": self.depth_trunc,
            "depth_dtype": str(depth_np.dtype),
            "device": str(self.device),
            **resize_info,
        }

        return depth_img, info

    # ============================================================
    # CUDA 点云构建与 ICP
    # ============================================================

    def _estimate_normals_safe(self, pcd):
        """
        不同 Open3D 构建下 estimate_normals 的 Python 参数绑定可能略有差异。
        做两种调用兼容。
        """
        try:
            pcd.estimate_normals(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
            return True, None
        except TypeError:
            try:
                pcd.estimate_normals(
                    self.normal_max_nn,
                    self.normal_radius,
                )
                return True, None
            except Exception as e:
                return False, repr(e)
        except Exception as e:
            return False, repr(e)
    
    def _pcd_num_points(self, pcd):
        if pcd is None:
            return 0

        try:
            return int(pcd.point["positions"].shape[0])
        except Exception:
            pass

        try:
            return int(pcd.point.positions.shape[0])
        except Exception:
            pass

        return 0

    def _make_pcd_from_depth_image(self, depth_img, depth_scale):
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_img,
            self.intrinsic_t,
            self.identity_extrinsic_t,
            depth_scale=float(depth_scale),
            depth_max=float(self.depth_trunc),
            stride=self.tracking_pcd_stride,
            with_normals=False,
        )

        try:
            n0 = int(pcd.point["positions"].shape[0])
        except Exception:
            n0 = 0

        if n0 <= 0:
            return None, {
                "reason": "empty_pointcloud_before_downsample",
                "num_points_raw": n0,
            }

        if self.tracking_voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.tracking_voxel_size)

        try:
            n1 = int(pcd.point["positions"].shape[0])
        except Exception:
            n1 = 0

        if n1 <= 0:
            return None, {
                "reason": "empty_pointcloud_after_downsample",
                "num_points_raw": n0,
                "num_points_down": n1,
            }

        ok_normals, normal_error = self._estimate_normals_safe(pcd)

        info = {
            "num_points_raw": n0,
            "num_points_down": n1,
            "normals_ok": ok_normals,
        }

        if normal_error is not None:
            info["normal_error"] = normal_error

        return pcd, info

    def _make_init_tensor(self):
        """
        使用上一帧的 ICP delta 作为当前帧 ICP 初值。
        这样快速移动时，比每次 identity 初始化更稳。
        """
        init_np = self.last_delta_icp.astype(np.float32, copy=True)

        return o3d.core.Tensor(
            init_np,
            dtype=o3d.core.Dtype.Float32,
            device=self.device,
        )

    def _run_point_to_plane_icp(self, source_pcd, target_pcd, init=None, max_corr=None):
        if init is None:
            init = self._make_init_tensor()

        if max_corr is None:
            max_corr = self.icp_max_correspondence_distance

        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=self.icp_max_iteration,
        )

        return o3d.t.pipelines.registration.icp(
            source_pcd,
            target_pcd,
            float(max_corr),
            init,
            estimation,
            criteria,
        )

    def _run_point_to_point_icp(self, source_pcd, target_pcd, init=None, max_corr=None):
        if init is None:
            init = self._make_init_tensor()

        if max_corr is None:
            max_corr = self.icp_max_correspondence_distance

        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=self.icp_max_iteration,
        )

        return o3d.t.pipelines.registration.icp(
            source_pcd,
            target_pcd,
            float(max_corr),
            init,
            estimation,
            criteria,
        )

    def _run_icp_with_fallback(self, source_pcd, target_pcd, init=None, max_corr=None):
        """
        source = prev_pcd
        target = current_pcd

        返回：
            result, estimation_name, error_info
        """
        try:
            result = self._run_point_to_plane_icp(
                source_pcd,
                target_pcd,
                init=init,
                max_corr=max_corr,
            )
            return result, "point_to_plane", None

        except Exception as e1:
            p2l_error = repr(e1)

            if not self.allow_point_to_point_fallback:
                return None, "point_to_plane_failed", {
                    "point_to_plane_error": p2l_error,
                }

            try:
                result = self._run_point_to_point_icp(
                    source_pcd,
                    target_pcd,
                    init=init,
                    max_corr=max_corr,
                )
                return result, "point_to_point_fallback", {
                    "point_to_plane_error": p2l_error,
                }

            except Exception as e2:
                return None, "icp_failed", {
                    "point_to_plane_error": p2l_error,
                    "point_to_point_error": repr(e2),
                }
    
    def _prepare_local_map_target(self):
        """
        准备 frame-to-model ICP 的 target。

        当前约定：
        - self.T_c_w: world -> camera
        - local_map: world 坐标点云
        - frame-to-model ICP:
            source = current_pcd, camera 坐标
            target = local_map target, world 坐标
            init = T_w_c, camera -> world
        """
        if self.map_icp_cooldown_frames > 0:
            self.map_icp_cooldown_frames -= 1
            return None, None, 0, f"map_icp_cooldown:{self.map_icp_cooldown_frames}"

        if not self.use_map_tracking:
            return None, None, 0, "map_tracking_disabled"

        if self.local_map is None:
            return None, None, 0, "local_map_is_none"

        try:
            if not self.local_map.has_enough_points():
                return None, None, 0, "local_map_not_enough_points"
        except Exception as e:
            return None, None, 0, f"local_map_has_enough_points_failed:{repr(e)}"

        try:
            T_w_c_init = np.linalg.inv(self.T_c_w)
            camera_pos_world = T_w_c_init[:3, 3].copy()

            target_pcd = self.local_map.get_tracking_target(camera_pos_world)
            if target_pcd is None or target_pcd.is_empty():
                return None, None, 0, "local_map_target_empty"

            try:
                target_pcd = target_pcd.to(self.device)
            except Exception:
                pass

            target_points = self._pcd_num_points(target_pcd)
            if target_points < self.map_tracking_min_points:
                return None, None, target_points, "local_map_target_too_few_points"

            # point-to-plane ICP 需要 target normals
            ok_normals, normal_error = self._estimate_normals_safe(target_pcd)
            if not ok_normals:
                # 不直接失败，因为后面还有 point-to-point fallback
                if self.logger is not None:
                    self.logger.warning(
                        f"local_map target normal failed: {normal_error}"
                    )

            init = o3d.core.Tensor(
                T_w_c_init.astype(np.float32),
                dtype=o3d.core.Dtype.Float32,
                device=self.device,
            )

            return target_pcd, init, target_points, "ok"

        except Exception as e:
            return None, None, 0, f"local_map_prepare_failed:{repr(e)}"

    def _prepare_full_map_relocalize_target(self):
        if not self.relocalize_use_full_map_fallback:
            return None, None, 0, "full_map_relocalize_disabled"

        if self.local_map is None:
            return None, None, 0, "local_map_is_none"

        try:
            target_pcd = self.local_map.get_tracking_target(camera_pos_world=None)
            if target_pcd is None or target_pcd.is_empty():
                return None, None, 0, "full_map_target_empty"

            try:
                target_pcd = target_pcd.to(self.device)
            except Exception:
                pass

            target_points = self._pcd_num_points(target_pcd)
            if target_points < self.map_tracking_min_points:
                return None, None, target_points, "full_map_target_too_few_points"

            ok_normals, normal_error = self._estimate_normals_safe(target_pcd)
            if not ok_normals and self.logger is not None:
                self.logger.warning(
                    f"full map target normal failed: {normal_error}"
                )

            T_w_c_init = np.linalg.inv(self.T_c_w)
            init = o3d.core.Tensor(
                T_w_c_init.astype(np.float32),
                dtype=o3d.core.Dtype.Float32,
                device=self.device,
            )

            return target_pcd, init, target_points, "ok"
        except Exception as e:
            return None, None, 0, f"full_map_prepare_failed:{repr(e)}"

    # ============================================================
    # 主 tracking
    # ============================================================

    def track(self, frame: RGBDFrame) -> TrackingResult:
        t0 = time.perf_counter()

        # ---------------- validate ----------------
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

            self._enter_lost(frame, "not_enough_valid_pixels")

            return self._make_lost_result(
                frame,
                "not_enough_valid_pixels",
                {
                    "valid_pixel_count": frame.valid_pixel_count,
                    "min_valid_pixels": self.min_valid_pixels,
                },
            )

        # ---------------- preprocess depth -> CUDA Image ----------------
        depth_img, depth_info = self._preprocess_depth_to_tensor_image(frame.depth)
        t2 = time.perf_counter()

        if depth_img is None:
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
            self._enter_lost(frame, "depth_preprocess_failed")

            return self._make_lost_result(
                frame,
                "depth_preprocess_failed",
                depth_info,
            )

        # ---------------- CUDA Image -> CUDA PointCloud ----------------
        current_pcd, pcd_info = self._make_pcd_from_depth_image(
            depth_img,
            depth_scale=depth_info["depth_scale"],
        )
        t3 = time.perf_counter()

        if current_pcd is None:
            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (t3 - t2) * 1000.0,
                0.0,
                0.0,
                (t_end - t0) * 1000.0,
            )

            self._enter_lost(frame, "pointcloud_failed")

            return self._make_lost_result(
                frame,
                "pointcloud_failed",
                {
                    "depth_info": depth_info,
                    "pcd_info": pcd_info,
                },
            )

        cam_ts = float(frame.device_timestamp)

        # ---------------- first frame ----------------
        if self.prev_pcd is None:
            self.T_c_w, imu_init_info = self._initial_world_to_camera(frame)
            if (
                not imu_init_info.get("imu_world_init_success", False)
                and getattr(self.imu_world_initializer, "required", False)
            ):
                t_end = time.perf_counter()
                self._maybe_print_profile(
                    frame.frame_id,
                    (t1 - t0) * 1000.0,
                    (t2 - t1) * 1000.0,
                    (t3 - t2) * 1000.0,
                    0.0,
                    0.0,
                    (t_end - t0) * 1000.0,
                    tracking_mode="imu_init_failed",
                )
                self._enter_lost(
                    frame,
                    imu_init_info.get("imu_world_init_reason", "imu_world_init_failed"),
                )
                return self._make_lost_result(
                    frame,
                    "imu_world_init_failed",
                    {
                        "valid_pixel_count": frame.valid_pixel_count,
                        "depth_info": depth_info,
                        "pcd_info": pcd_info,
                        "tracker_backend": "gpu_icp",
                        **imu_init_info,
                    },
                )

            self.prev_pcd = current_pcd
            self.prev_frame = frame
            self.prev_cam_ts = cam_ts
            self.last_trans[:] = 0.0
            self.last_delta_icp = np.eye(4, dtype=np.float64)

            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (t3 - t2) * 1000.0,
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
                    "depth_info": depth_info,
                    "pcd_info": pcd_info,
                    "tracker_backend": "gpu_icp",
                    **imu_init_info,
                },
            )

        # ---------------- ICP ----------------
        tracking_mode = "frame_to_frame"
        use_local_map = False
        local_map_points = 0
        local_map_status = "not_used"
        map_icp_rejected_reason = None

        result = None
        icp_estimation = None
        icp_error_info = None

        # lost 状态下按间隔尝试 relocalize，避免每帧刷屏和误吸附。
        allow_map_try = True
        if self.is_lost:
            allow_map_try = (self.lost_count % self.relocalize_try_interval == 0)

        if allow_map_try:
            map_target_pcd, map_init, local_map_points, local_map_status = (
                self._prepare_local_map_target()
            )
            if self.is_lost:
                self._log_relocalize_attempt(
                    frame,
                    "try",
                    target_points=local_map_points,
                    reason=local_map_status,
                )
        else:
            map_target_pcd, map_init = None, None
            local_map_status = f"lost_relocalize_wait:{self.lost_count}"

        # 1) 先尝试 frame-to-model
        if map_target_pcd is not None and map_init is not None:
            map_result, map_estimation, map_error_info = self._run_icp_with_fallback(
                current_pcd,
                map_target_pcd,
                init=map_init,
                max_corr=self.map_tracking_max_corr,
            )

            if map_result is not None:
                map_fitness = float(map_result.fitness)
                map_rmse = float(map_result.inlier_rmse)

                if (
                    map_fitness >= self.map_min_icp_fitness
                    and map_rmse <= self.map_max_icp_rmse
                ):
                    result = map_result
                    icp_estimation = map_estimation
                    icp_error_info = map_error_info
                    tracking_mode = "frame_to_model"
                    use_local_map = True
                else:
                    map_icp_rejected_reason = (
                        f"map_icp_bad_quality:"
                        f"fitness={map_fitness:.4f},rmse={map_rmse:.5f},"
                        f"need_fitness>={self.map_min_icp_fitness:.3f},"
                        f"need_rmse<={self.map_max_icp_rmse:.5f}"
                    )

                    # 基本可视为 0 correspondence / 极差对齐，进入 cooldown
                    if map_fitness <= 1e-6:
                        self.map_icp_cooldown_frames = self.map_icp_cooldown_after_fail
            else:
                map_icp_rejected_reason = "map_icp_failed"
                self.map_icp_cooldown_frames = self.map_icp_cooldown_after_fail

        # 1.5) lost 后局部地图对不上时，尝试整张 local map 做宽松重定位。
        if result is None and self.is_lost and allow_map_try:
            full_target, full_init, full_points, full_status = (
                self._prepare_full_map_relocalize_target()
            )
            self._log_relocalize_attempt(
                frame,
                "try_full_map",
                target_points=full_points,
                reason=full_status,
            )

            if full_target is not None and full_init is not None:
                full_result, full_estimation, full_error_info = self._run_icp_with_fallback(
                    current_pcd,
                    full_target,
                    init=full_init,
                    max_corr=self.relocalize_full_map_max_corr,
                )

                if full_result is not None:
                    full_fitness = float(full_result.fitness)
                    full_rmse = float(full_result.inlier_rmse)
                    if (
                        full_fitness >= self.relocalize_full_map_min_fitness
                        and full_rmse <= self.relocalize_full_map_max_rmse
                    ):
                        result = full_result
                        icp_estimation = f"full_map_{full_estimation}"
                        icp_error_info = full_error_info
                        tracking_mode = "full_map_relocalize"
                        use_local_map = True
                        local_map_points = full_points
                        local_map_status = full_status
                        map_icp_rejected_reason = None
                        self.map_icp_cooldown_frames = 0
                    else:
                        map_icp_rejected_reason = (
                            f"full_map_bad_quality:"
                            f"fitness={full_fitness:.4f},rmse={full_rmse:.5f},"
                            f"need_fitness>={self.relocalize_full_map_min_fitness:.3f},"
                            f"need_rmse<={self.relocalize_full_map_max_rmse:.5f}"
                        )
                else:
                    map_icp_rejected_reason = "full_map_icp_failed"

        # 2) 还没 lost 时，允许 fallback 到 frame-to-frame
        if result is None and not self.is_lost:
            result, icp_estimation, icp_error_info = self._run_icp_with_fallback(
                self.prev_pcd,
                current_pcd,
            )
            tracking_mode = "frame_to_frame"
            use_local_map = False

            if icp_error_info is None:
                icp_error_info = {}

            if map_icp_rejected_reason is not None:
                icp_error_info["map_icp_rejected_reason"] = map_icp_rejected_reason
                icp_error_info["local_map_status"] = local_map_status
                icp_error_info["local_map_points"] = local_map_points

        # 3) 已经处于 lost 状态时，不允许 frame-to-frame 伪恢复
        if result is None and self.is_lost:
            t4 = time.perf_counter()

            self.last_trans[:] = 0.0
            self.last_delta_icp = np.eye(4, dtype=np.float64)

            self._enter_lost(
                frame,
                map_icp_rejected_reason or local_map_status or "relocalization_failed",
            )

            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (t3 - t2) * 1000.0,
                (t4 - t3) * 1000.0,
                0.0,
                (t_end - t0) * 1000.0,
                tracking_mode="lost",
            )

            return self._make_lost_result(
                frame,
                "tracking_lost_relocalizing",
                {
                    "depth_info": depth_info,
                    "pcd_info": pcd_info,
                    "tracking_mode": "lost",
                    "use_local_map": False,
                    "local_map_points": local_map_points,
                    "local_map_status": local_map_status,
                    "map_icp_rejected_reason": map_icp_rejected_reason,
                    "lost_count": self.lost_count,
                },
            )

        t4 = time.perf_counter()

        if result is None:
            # 跟踪失败时不要更新 prev_pcd。
            # 否则会把“未知位姿的当前帧”变成新的参考帧，
            # 后续即使 frame-to-frame 成功，也会漏掉丢失期间的真实运动。
            # self.prev_pcd = current_pcd
            # self.prev_frame = frame
            # self.prev_cam_ts = cam_ts
            self.last_trans[:] = 0.0
            self.last_delta_icp = np.eye(4, dtype=np.float64)

            self._enter_lost(frame,"gpu_icp_failed")

            t_end = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (t3 - t2) * 1000.0,
                (t4 - t3) * 1000.0,
                0.0,
                (t_end - t0) * 1000.0,
                tracking_mode=tracking_mode,
            )

            return self._make_lost_result(
                frame,
                "gpu_icp_failed",
                {
                    "depth_info": depth_info,
                    "pcd_info": pcd_info,
                    "icp_estimation": icp_estimation,
                    "icp_error_info": icp_error_info,
                    "tracking_mode": tracking_mode,
                    "use_local_map": use_local_map,
                    "local_map_points": local_map_points,
                    "local_map_status": local_map_status,
                    "lost_count": self.lost_count,
                    "map_icp_rejected_reason": map_icp_rejected_reason,
                },
            )

        # 注意：Open3D 0.19 测试中 transformation 返回 CPU:0 Float64 Tensor
        icp_transform = result.transformation.cpu().numpy().astype(np.float64)

        old_T_c_w = self.T_c_w.copy()

        if use_local_map:
            # frame-to-model:
            # icp_transform 是 camera -> world，即 T_w_c_new
            # 但本工程传给 mapper 的 TrackingResult.T_wc 实际作为 TSDF extrinsic 使用，
            # 也就是 world -> camera。
            T_w_c_new = icp_transform
            T_c_w_new = np.linalg.inv(T_w_c_new)

            # 用相对位姿变化做 gating / 日志统计
            delta_refined = T_c_w_new @ np.linalg.inv(old_T_c_w)
        else:
            # frame-to-frame:
            # 保持原来的语义，prev -> current 的 delta
            delta_refined = icp_transform

        dR = self.normalize_rotation(delta_refined[:3, :3])
        vo_rot_deg = self.rot_deg(dR)

        raw_t = delta_refined[:3, 3].copy()
        raw_t_norm = float(np.linalg.norm(raw_t))

        fitness = float(result.fitness)
        inlier_rmse = float(result.inlier_rmse)

        # ---------------- gating ----------------
        bad_frame = False
        reason = None

        if vo_rot_deg > self.max_rotation_deg_per_frame:
            bad_frame = True
            reason = "rotation_too_large"

        if raw_t_norm > self.max_translation_per_frame:
            bad_frame = True
            reason = "translation_too_large"

        if fitness < self.min_icp_fitness:
            bad_frame = True
            reason = "icp_fitness_too_low"

        if inlier_rmse > self.max_icp_rmse:
            bad_frame = True
            reason = "icp_rmse_too_high"

        if use_local_map:
            if raw_t_norm > self.map_max_delta_trans:
                bad_frame = True
                reason = "map_delta_translation_too_large"

            if vo_rot_deg > self.map_max_delta_rot_deg:
                bad_frame = True
                reason = "map_delta_rotation_too_large"

            if fitness < self.map_min_icp_fitness:
                bad_frame = True
                reason = "map_icp_fitness_too_low"

            if inlier_rmse > self.map_max_icp_rmse:
                bad_frame = True
                reason = "map_icp_rmse_too_high"

        if bad_frame:
            # 被 gating 判定为坏帧时，不更新 prev_pcd。
            # 系统保留最后一个可信参考帧，
            # 用户把摄像头移回已建区域时，有机会重新对上。
            # self.prev_pcd = current_pcd
            # self.prev_frame = frame
            # self.prev_cam_ts = cam_ts
            self.last_trans[:] = 0.0
            self.last_delta_icp = np.eye(4, dtype=np.float64)
            self._enter_lost(frame,reason or "gpu_icp_rejected")

            t5 = time.perf_counter()
            self._maybe_print_profile(
                frame.frame_id,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (t3 - t2) * 1000.0,
                (t4 - t3) * 1000.0,
                (t5 - t4) * 1000.0,
                (t5 - t0) * 1000.0,
                tracking_mode=tracking_mode,
            )

            if self.logger is not None:
                self.logger.debug(
                    f"reject: frame={frame.frame_id} "
                    f"reason={reason} "
                    f"rot={vo_rot_deg:.3f}deg "
                    f"t_raw={raw_t_norm:.5f}m "
                    f"fitness={fitness:.3f} "
                    f"rmse={inlier_rmse:.5f} "
                    f"est={icp_estimation}",
                    frame_id=frame.frame_id,
                )

            return self._make_lost_result(
                frame,
                reason or "gpu_icp_rejected",
                {
                    "valid_pixel_count": frame.valid_pixel_count,
                    "depth_info": depth_info,
                    "pcd_info": pcd_info,
                    "vo_rot_deg": vo_rot_deg,
                    "translation_norm_raw": raw_t_norm,
                    "fitness": fitness,
                    "inlier_rmse": inlier_rmse,
                    "icp_estimation": icp_estimation,
                    "icp_error_info": icp_error_info,
                    "tracking_mode": tracking_mode,
                    "use_local_map": use_local_map,
                    "local_map_points": local_map_points,
                    "local_map_status": local_map_status,
                    "lost_count": self.lost_count,
                    "map_icp_rejected_reason": map_icp_rejected_reason,
                },
            )

        if self.is_lost:
            self._log_relocalize_attempt(
                frame,
                "success",
                target_points=local_map_points,
                reason=tracking_mode,
            )

        # ---------------- translation smoothing / pose update ----------------
        trans_refined = raw_t.copy()
        accepted_t = False

        if use_local_map:
            # frame-to-model 已经是 current frame 对齐 local map 得到的绝对位姿。
            # 这里不要再因为旋转而清零平移，否则又会把真实抬高/移动吃掉。
            accepted_t = True
            trans_refined = raw_t.copy()

            # 直接使用 frame-to-model 的绝对位姿结果
            self.T_c_w = T_c_w_new.copy()

            # last_delta_icp 仍然存相对 delta，用于日志/后续 fallback 初值
            self.last_delta_icp = delta_refined.copy()
            self.last_delta_icp[:3, :3] = self.normalize_rotation(
                self.last_delta_icp[:3, :3]
            )

            self.last_trans = trans_refined.copy()

        else:
            # 原来的 frame-to-frame 平移平滑逻辑
            if self.min_translation_per_frame <= raw_t_norm <= self.max_translation_per_frame:
                if vo_rot_deg >= self.rotation_dominant_angle_deg:
                    accepted_t = raw_t_norm <= self.max_translation_when_rotating
                else:
                    accepted_t = True

                if accepted_t:
                    trans_refined = (
                        self.trans_smooth_alpha * raw_t
                        + (1.0 - self.trans_smooth_alpha) * self.last_trans
                    )
                    self.last_trans = trans_refined.copy()
                else:
                    trans_refined[:] = 0.0
                    self.last_trans[:] = 0.0
                    self.last_delta_icp = np.eye(4, dtype=np.float64)
            else:
                trans_refined[:] = 0.0
                self.last_trans[:] = 0.0
                self.last_delta_icp = np.eye(4, dtype=np.float64)

            fused_delta = np.eye(4, dtype=np.float64)
            fused_delta[:3, :3] = dR
            fused_delta[:3, 3] = trans_refined

            self.T_c_w = fused_delta @ self.T_c_w
            self.last_delta_icp = delta_refined.copy()
            self.last_delta_icp[:3, :3] = self.normalize_rotation(
                self.last_delta_icp[:3, :3]
            )
        self._leave_lost(
            frame,
            tracking_mode=tracking_mode,
            fitness=fitness,
            rmse=inlier_rmse,
        )
        self.prev_pcd = current_pcd
        self.prev_frame = frame
        self.prev_cam_ts = cam_ts

        t5 = time.perf_counter()
        self._maybe_print_profile(
            frame.frame_id,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            (t4 - t3) * 1000.0,
            (t5 - t4) * 1000.0,
            (t5 - t0) * 1000.0,
            tracking_mode=tracking_mode,
        )

        if self.logger is not None:
            num_raw = pcd_info.get("num_points_raw", -1)
            num_down = pcd_info.get("num_points_down", -1)

            self.logger.debug(
                f"frame={frame.frame_id} "
                f"points={num_raw}->{num_down} "
                f"rot={vo_rot_deg:.3f}deg "
                f"t_raw={raw_t_norm:.5f}m "
                f"t_use={np.linalg.norm(trans_refined):.5f}m "
                f"accepted_t={accepted_t} "
                f"fitness={fitness:.3f} "
                f"rmse={inlier_rmse:.5f} "
                f"est={icp_estimation} "
                f"mode={tracking_mode} "
                f"map_pts={local_map_points} "
                f"map_status={local_map_status}",
                frame_id=frame.frame_id,
            )

        return TrackingResult(
            frame_id=frame.frame_id,
            timestamp=frame.device_timestamp,
            success=True,
            T_wc=self.T_c_w.copy(),
            score=fitness,
            mode="tracking",
            extras={
                "valid_pixel_count": frame.valid_pixel_count,
                "depth_info": depth_info,
                "pcd_info": pcd_info,
                "vo_rot_deg": vo_rot_deg,
                "tracking_mode": tracking_mode,
                "use_local_map": use_local_map,
                "local_map_points": local_map_points,
                "local_map_status": local_map_status,

                # 兼容旧 mapper / 日志里可能读取 info_trace 的逻辑。
                # 注意：这里不是真正 information matrix trace。
                "info_trace": fitness * 1e6,

                "translation_norm": float(np.linalg.norm(trans_refined)),
                "translation_norm_raw": raw_t_norm,
                "fitness": fitness,
                "inlier_rmse": inlier_rmse,
                "icp_estimation": icp_estimation,
                "icp_error_info": icp_error_info,
                "map_icp_rejected_reason": map_icp_rejected_reason,
                "is_lost": self.is_lost,
                "lost_count": self.lost_count,
                "tracker_backend": "gpu_icp",
            },
        )