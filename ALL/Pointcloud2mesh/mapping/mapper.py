import os
import shutil
import time
import threading
import copy
from datetime import datetime

import numpy as np
import open3d as o3d

from ..common.types import MapPacket, MapSnapshot
from ..input.rgbd_preprocessor import RGBDPreprocessor

class Mapper:
    """
    TSDF 建图模块。
    使用 TrackingResult 中的 T_wc
    直接作为 Open3D integrate 的 extrinsic。
    """

    def __init__(
        self,
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,

        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        input_color_is_bgr=False,
        depth_scale=None,
        depth_trunc=2.0,

        integrate_only_when_motion=True,
        min_integrate_rot_deg=0.30,
        max_integrate_rot_deg=8.0,
        min_integrate_trans_m=0.0008,

        mesh_update_interval=10,
        mesh_min_vertices_to_show=10,

        model_dir="model",

        mapping_lost_print_interval=30,
        max_mapping_pose_jump_trans=0.08,
        max_mapping_pose_jump_rot_deg=12.0,
        local_map_frame_pcd_voxel_size=0.03,
        local_map_frame_pcd_stride=4,
        apply_output_axis_transform=False,
        R_output_from_reconstruction=None,
        auto_align_output_yaw=False,
        auto_align_min_vertices=500,
        auto_align_min_angle_deg=2.0,
        output_translate_to_positive=False,

        local_map=None,
        logger=None,
    ):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=color_type,
        )

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

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.local_map = local_map
        self._lock = threading.RLock()

        self.integrate_only_when_motion = integrate_only_when_motion
        self.min_integrate_rot_deg = min_integrate_rot_deg
        self.max_integrate_rot_deg = max_integrate_rot_deg
        self.min_integrate_trans_m = min_integrate_trans_m

        self.mesh_update_interval = mesh_update_interval
        self.mesh_min_vertices_to_show = mesh_min_vertices_to_show

        self.local_map_frame_pcd_voxel_size = float(local_map_frame_pcd_voxel_size)
        self.local_map_frame_pcd_stride = max(1, int(local_map_frame_pcd_stride))
        self.apply_output_axis_transform = bool(apply_output_axis_transform)
        self.R_output_from_reconstruction = self.normalize_rotation(
            np.eye(3, dtype=np.float64)
            if R_output_from_reconstruction is None
            else np.asarray(R_output_from_reconstruction, dtype=np.float64)
        )
        self.auto_align_output_yaw = bool(auto_align_output_yaw)
        self.auto_align_min_vertices = max(3, int(auto_align_min_vertices))
        self.auto_align_min_angle_deg = float(auto_align_min_angle_deg)
        self.output_translate_to_positive = bool(output_translate_to_positive)

        self.total_packets = 0
        self.integrated_frames = 0
        self.last_mesh = None
        self.last_mesh_vertex_count = 0

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = logger
        self.last_profile = {}

        self.mapping_paused_by_tracking_lost = False
        self.mapping_lost_print_interval = max(1, int(mapping_lost_print_interval))

        # Mapper 自己的位姿连续性保护。
        # 即使 tracker 返回 success=True，只要相对上一次融合位姿跳得太大，
        # 也拒绝 integrate，避免把错误位姿写进 TSDF。
        self.last_integrated_T_wc = None

        # 这里的 T_wc 实际是 world -> camera。
        # 下面这两个阈值是“相邻两次 integrate”允许的最大跳变。
        self.max_mapping_pose_jump_trans = float(max_mapping_pose_jump_trans)
        self.max_mapping_pose_jump_rot_deg = float(max_mapping_pose_jump_rot_deg)

        self.mapping_pose_reject_count = 0

    def _auto_align_yaw(self, mesh):
        if not self.auto_align_output_yaw or len(mesh.vertices) < self.auto_align_min_vertices:
            return mesh

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        xz = vertices[:, [0, 2]]
        xz = xz - np.mean(xz, axis=0, keepdims=True)

        cov = xz.T @ xz / max(1, xz.shape[0] - 1)
        vals, vecs = np.linalg.eigh(cov)
        if not np.all(np.isfinite(vals)) or vals[-1] <= 1e-12:
            return mesh

        principal = vecs[:, int(np.argmax(vals))]
        angle = float(np.arctan2(principal[1], principal[0]))

        # PCA direction is signless; choose the nearest equivalent angle.
        while angle > np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi

        angle_deg = abs(float(np.degrees(angle)))
        if angle_deg < self.auto_align_min_angle_deg:
            return mesh

        c = float(np.cos(-angle))
        s = float(np.sin(-angle))
        R_y = np.asarray(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_y
        center = mesh.get_center()
        mesh.translate(-center)
        mesh.transform(T)
        mesh.translate(center)

        if self.logger is not None:
            self.logger.pose(
                f"output yaw auto-aligned: angle_deg={np.degrees(-angle):.3f}",
                force=True,
            )

        return mesh

    def _apply_output_axis_transform(self, mesh):
        out = copy.deepcopy(mesh)

        if self.apply_output_axis_transform:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = self.R_output_from_reconstruction
            out.transform(T)

        out = self._auto_align_yaw(out)

        if self.output_translate_to_positive and len(out.vertices) > 0:
            min_bound = out.get_min_bound()
            out.translate(-min_bound)

        return out

    def extract_output_mesh(self):
        with self._lock:
            mesh = self.volume.extract_triangle_mesh()
            if len(mesh.vertices) > 0:
                mesh = self._apply_output_axis_transform(mesh)
                mesh.compute_vertex_normals()
            return mesh

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

    def _print_profile(
        self,
        frame_id,
        preprocess_ms,
        gate_ms,
        integrate_ms,
        extract_mesh_ms,
        normal_ms,
        total_ms,
        do_integrate,
        mesh_updated,
        mesh_vertices,
        reason=None,
    ):
        if self.logger is None:
            return

        reason_str = "" if reason is None else f" reason={reason}"

        self.logger.profile(
            f"frame={frame_id} "
            f"preprocess={preprocess_ms:.2f}ms "
            f"gate={gate_ms:.2f}ms "
            f"integrate={integrate_ms:.2f}ms "
            f"extract_mesh={extract_mesh_ms:.2f}ms "
            f"normal={normal_ms:.2f}ms "
            f"total={total_ms:.2f}ms "
            f"do_integrate={do_integrate} "
            f"mesh_updated={mesh_updated} "
            f"mesh_vertices={mesh_vertices} "
            f"integrated_frames={self.integrated_frames} "
            f"total_packets={self.total_packets}"
            f"{reason_str}",
            frame_id=frame_id,
            force=mesh_updated,
        )

    def _guess_depth_scale_for_local_map(self, depth_np):
        """
        local_map 点云生成用的 depth_scale。

        注意：
        - 如果 config 里显式传了 depth_scale，就用 config 的。
        - 如果没有，就根据 depth dtype 和数值范围粗略判断。
        """
        if self.depth_scale is not None:
            return float(self.depth_scale)

        if depth_np is None:
            return 1000.0

        if np.issubdtype(depth_np.dtype, np.integer):
            return 1000.0

        # float depth:
        # 如果 p95 很大，通常说明是毫米。
        # 如果 p95 小于几十，通常说明已经是米。
        valid = depth_np[np.isfinite(depth_np)]
        valid = valid[valid > 0]
        if valid.size == 0:
            return 1000.0

        p95 = float(np.percentile(valid, 95))
        if p95 > 20.0:
            return 1000.0

        return 1.0

    def _create_frame_pcd_for_local_map(self, frame):
        """
        根据当前原始 depth 生成一个 camera 坐标系下的 tensor 点云。
        注意：这里只做轻量点云，用于 local map tracking，不是最终 mesh。
        """
        depth_np = frame.depth
        if depth_np is None:
            return None

        depth = depth_np
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        depth_img = o3d.t.geometry.Image(depth)

        intrinsic = o3d.core.Tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=o3d.core.Dtype.Float64,
        )

        depth_scale = self._guess_depth_scale_for_local_map(depth_np)

        try:
            pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
                depth_img,
                intrinsic,
                o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64),
                depth_scale=float(depth_scale),
                depth_max=float(self.depth_trunc),
                stride=self.local_map_frame_pcd_stride,
                with_normals=False,
            )
        except TypeError:
            # 兼容部分 Open3D 版本参数名差异
            pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
                depth_img,
                intrinsic,
                o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64),
                depth_scale=float(depth_scale),
                depth_max=float(self.depth_trunc),
                stride=self.local_map_frame_pcd_stride,
            )

        if pcd is None or pcd.is_empty():
            return None

        try:
            pcd = pcd.voxel_down_sample(self.local_map_frame_pcd_voxel_size)
        except Exception:
            pass

        return pcd

    def _should_integrate(self, packet: MapPacket) -> bool:
        tracking = packet.tracking

        if not tracking.success:
            return False

        # Bootstrap must run until the first mesh is actually extracted.
        # Otherwise a stationary start can integrate a few frames, stop on the
        # motion gate, and never reach mesh_update_interval.
        if self.last_mesh is None:
            return True

        if not self.integrate_only_when_motion:
            return True

        extras = tracking.extras if tracking.extras is not None else {}

        rot_used = float(extras.get("vo_rot_deg", 0.0))
        trans_used = float(extras.get("translation_norm", 0.0))

        moving = (
            (trans_used >= self.min_integrate_trans_m)
            or (rot_used >= self.min_integrate_rot_deg)
        )
        rot_too_fast = (rot_used > self.max_integrate_rot_deg)

        return moving and (not rot_too_fast)
    
    def _check_mapping_pose_jump(self, T_wc, frame_id=None):
        """
        检查当前位姿相对上一次真正 integrate 的位姿是否跳变过大。

        注意：
        本工程里的 T_wc 实际作为 Open3D TSDF extrinsic 使用，
        即 world -> camera。
        """
        if self.last_integrated_T_wc is None:
            return True, {
                "reason": "first_integrated_pose",
                "trans": 0.0,
                "rot_deg": 0.0,
            }

        try:
            T_new = np.asarray(T_wc, dtype=np.float64)
            T_old = np.asarray(self.last_integrated_T_wc, dtype=np.float64)

            # old -> new 的相对变化
            delta = T_new @ np.linalg.inv(T_old)

            R = self.normalize_rotation(delta[:3, :3])
            rot_deg = self.rot_deg(R)
            trans = float(np.linalg.norm(delta[:3, 3]))

            ok = (
                trans <= self.max_mapping_pose_jump_trans
                and rot_deg <= self.max_mapping_pose_jump_rot_deg
            )

            info = {
                "reason": "ok" if ok else "mapping_pose_jump",
                "trans": trans,
                "rot_deg": rot_deg,
                "max_trans": self.max_mapping_pose_jump_trans,
                "max_rot_deg": self.max_mapping_pose_jump_rot_deg,
            }

            if not ok:
                self.mapping_pose_reject_count += 1
                if self.logger is not None:
                    self.logger.warning(
                        f"pose jump reject: frame={frame_id} "
                        f"trans={trans:.4f}m>{self.max_mapping_pose_jump_trans:.4f}m "
                        f"rot={rot_deg:.2f}deg>{self.max_mapping_pose_jump_rot_deg:.2f}deg "
                        f"reject_count={self.mapping_pose_reject_count}. "
                        f"Skip TSDF/local_map integration to avoid crossed model.",
                        frame_id=frame_id,
                    )

            return ok, info

        except Exception as e:
            if self.logger is not None:
                self.logger.warning(
                    f"pose jump check failed: frame={frame_id} error={repr(e)}",
                    frame_id=frame_id,
                )
            return False, {
                "reason": "pose_jump_check_failed",
                "error": repr(e),
            }

    def update(self, packet: MapPacket) -> MapSnapshot:
        with self._lock:
            return self._update_impl(packet)

    def _update_impl(self, packet: MapPacket) -> MapSnapshot:
        t0 = time.perf_counter()

        self.total_packets += 1

        frame = packet.frame
        tracking = packet.tracking

        if not tracking.success:
            reason = "tracking_lost"
            if tracking.extras is not None:
                reason = tracking.extras.get("reason", reason)

            lost_count = 0
            tracking_mode = "unknown"
            local_map_status = None
            if tracking.extras is not None:
                lost_count = int(tracking.extras.get("lost_count", 0))
                tracking_mode = tracking.extras.get("tracking_mode", "unknown")
                local_map_status = tracking.extras.get("local_map_status", None)

            # 第一次暂停时打印，之后每隔一段打印
            if (
                not self.mapping_paused_by_tracking_lost
                or frame.frame_id % self.mapping_lost_print_interval == 0
            ):
                if self.logger is not None:
                    self.logger.mapping_state(
                        f"mapping paused: frame={frame.frame_id} "
                        f"reason={reason} "
                        f"tracking_mode={tracking_mode} "
                        f"lost_count={lost_count} "
                        f"local_map_status={local_map_status}. "
                        f"Model integration is paused. Move camera back to mapped area.",
                        frame_id=frame.frame_id,
                        force=not self.mapping_paused_by_tracking_lost,
                    )

            self.mapping_paused_by_tracking_lost = True

            t_end = time.perf_counter()
            total_ms = (t_end - t0) * 1000.0

            self.last_profile = {
                "preprocess_ms": 0.0,
                "gate_ms": 0.0,
                "integrate_ms": 0.0,
                "extract_mesh_ms": 0.0,
                "normal_ms": 0.0,
                "total_ms": total_ms,
                "do_integrate": False,
                "mesh_updated": False,
                "mesh_vertices": self.last_mesh_vertex_count,
                "reason": reason,
                "tracking_lost": True,
            }

            return MapSnapshot(
                frame_id=frame.frame_id,
                timestamp=frame.device_timestamp,
                map_data={
                    "integrated": False,
                    "reason": reason,
                    "tracking_lost": True,
                    "message": "Tracking lost. Move camera slowly back to a previously mapped area.",
                    "integrated_frames": self.integrated_frames,
                    "profile": self.last_profile,
                },
                extras={
                    "mesh": self.last_mesh,
                    "profile": self.last_profile,
                    "delta_pcd_world": None,
                },
            )
        
        if self.mapping_paused_by_tracking_lost:
            if self.logger is not None:
                self.logger.mapping_state(
                    f"mapping resumed: frame={frame.frame_id}. "
                    f"Tracking recovered. Model integration resumed.",
                    frame_id=frame.frame_id,
                    force=True,
                )
            self.mapping_paused_by_tracking_lost = False

        # ------------------------------------------------------------
        # 1. RGBD preprocess
        # ------------------------------------------------------------
        t_pre0 = time.perf_counter()
        current_rgbd, rgbd_info = self.preprocessor.preprocess(frame.color, frame.depth)
        t_pre1 = time.perf_counter()

        preprocess_ms = (t_pre1 - t_pre0) * 1000.0

        if current_rgbd is None:
            t_end = time.perf_counter()
            total_ms = (t_end - t0) * 1000.0

            self.last_profile = {
                "preprocess_ms": preprocess_ms,
                "gate_ms": 0.0,
                "integrate_ms": 0.0,
                "extract_mesh_ms": 0.0,
                "normal_ms": 0.0,
                "total_ms": total_ms,
                "do_integrate": False,
                "mesh_updated": False,
                "mesh_vertices": self.last_mesh_vertex_count,
                "reason": "rgbd_preprocess_failed",
            }

            self._print_profile(
                frame_id=frame.frame_id,
                preprocess_ms=preprocess_ms,
                gate_ms=0.0,
                integrate_ms=0.0,
                extract_mesh_ms=0.0,
                normal_ms=0.0,
                total_ms=total_ms,
                do_integrate=False,
                mesh_updated=False,
                mesh_vertices=self.last_mesh_vertex_count,
                reason="rgbd_preprocess_failed",
            )

            return MapSnapshot(
                frame_id=frame.frame_id,
                timestamp=frame.device_timestamp,
                map_data={
                    "integrated": False,
                    "reason": "rgbd_preprocess_failed",
                    "rgbd_info": rgbd_info,
                    "integrated_frames": self.integrated_frames,
                    "profile": self.last_profile,
                },
            )

        # ------------------------------------------------------------
        # 2. integrate gate
        # ------------------------------------------------------------
        t_gate0 = time.perf_counter()
        do_integrate = self._should_integrate(packet)
        t_gate1 = time.perf_counter()

        gate_ms = (t_gate1 - t_gate0) * 1000.0

        # ------------------------------------------------------------
        # 3. TSDF integrate + local map update
        # ------------------------------------------------------------
        integrate_ms = 0.0
        delta_pcd_world = None
        pose_gate_info = None

        if do_integrate:
            # 注意：
            # 这里沿用你原来的语义。
            # tracking.T_wc 实际作为 Open3D TSDF extrinsic 使用，
            # 即 world -> camera。
            extrinsic = tracking.T_wc.copy()

            pose_ok, pose_gate_info = self._check_mapping_pose_jump(
                extrinsic,
                frame_id=frame.frame_id,
            )

            if not pose_ok:
                # 关键保护：
                # tracker 虽然 success=True，但 mapper 判断位姿相对上一次融合跳变过大。
                # 这一帧不写 TSDF，也不写 local_map。
                do_integrate = False
                integrate_ms = 0.0
            else:
                t_int0 = time.perf_counter()

                self.volume.integrate(
                    current_rgbd,
                    self.preprocessor.get_intrinsic(),
                    extrinsic,
                )

                # local_map 需要 camera -> world
                # 因此这里必须取逆。
                if self.local_map is not None:
                    try:
                        pcd_cam = self._create_frame_pcd_for_local_map(frame)
                        if pcd_cam is not None and not pcd_cam.is_empty():
                            T_w_c = np.linalg.inv(extrinsic)
                            delta_pcd_world = self.local_map.integrate_frame_pcd(
                                pcd_cam,
                                T_w_c,
                            )
                    except Exception as e:
                        if self.logger is not None:
                            self.logger.warning(
                                f"local_map update failed: {repr(e)}",
                                frame_id=frame.frame_id,
                            )

                t_int1 = time.perf_counter()
                integrate_ms = (t_int1 - t_int0) * 1000.0

                self.integrated_frames += 1
                self.last_integrated_T_wc = extrinsic.copy()

        # ------------------------------------------------------------
        # 4. mesh extraction
        # ------------------------------------------------------------
        extract_mesh_ms = 0.0
        normal_ms = 0.0
        mesh_updated = False
        extracted_vertices = 0

        should_extract_mesh = (
            do_integrate
            and self.integrated_frames > 0
            and (
                self.last_mesh is None
                or self.integrated_frames % self.mesh_update_interval == 0
            )
        )

        if should_extract_mesh:
            mesh_updated = True

            t_mesh0 = time.perf_counter()
            new_mesh = self.volume.extract_triangle_mesh()
            t_mesh1 = time.perf_counter()

            extract_mesh_ms = (t_mesh1 - t_mesh0) * 1000.0

            vnum = len(new_mesh.vertices)
            extracted_vertices = vnum

            if vnum > self.mesh_min_vertices_to_show:
                t_norm0 = time.perf_counter()
                new_mesh = self._apply_output_axis_transform(new_mesh)
                new_mesh.compute_vertex_normals()
                t_norm1 = time.perf_counter()

                normal_ms = (t_norm1 - t_norm0) * 1000.0

                self.last_mesh = new_mesh
                self.last_mesh_vertex_count = vnum

        # ------------------------------------------------------------
        # 5. total
        # ------------------------------------------------------------
        t_end = time.perf_counter()
        total_ms = (t_end - t0) * 1000.0

        self.last_profile = {
            "preprocess_ms": preprocess_ms,
            "gate_ms": gate_ms,
            "integrate_ms": integrate_ms,
            "extract_mesh_ms": extract_mesh_ms,
            "normal_ms": normal_ms,
            "total_ms": total_ms,
            "do_integrate": do_integrate,
            "mesh_updated": mesh_updated,
            "mesh_vertices": self.last_mesh_vertex_count,
            "extracted_vertices": extracted_vertices,
            "integrated_frames": self.integrated_frames,
            "total_packets": self.total_packets,
            "pose_gate_info": pose_gate_info,
        }

        self._print_profile(
            frame_id=frame.frame_id,
            preprocess_ms=preprocess_ms,
            gate_ms=gate_ms,
            integrate_ms=integrate_ms,
            extract_mesh_ms=extract_mesh_ms,
            normal_ms=normal_ms,
            total_ms=total_ms,
            do_integrate=do_integrate,
            mesh_updated=mesh_updated,
            mesh_vertices=self.last_mesh_vertex_count,
        )

        map_data = {
            "integrated": do_integrate,
            "integrated_frames": self.integrated_frames,
            "total_packets": self.total_packets,
            "last_frame_id": frame.frame_id,
            "mesh_vertex_count": self.last_mesh_vertex_count,
            "mesh_updated": mesh_updated,
            "pose_gate_info": pose_gate_info,
            "profile": self.last_profile,
        }

        return MapSnapshot(
            frame_id=frame.frame_id,
            timestamp=frame.device_timestamp,
            map_data=map_data,
            extras={
                "mesh": self.last_mesh,
                "rgbd_info": rgbd_info,
                "profile": self.last_profile,
                "delta_pcd_world": delta_pcd_world,
            },
        )

    def get_latest_mesh(self):
        with self._lock:
            return self.last_mesh

    def save_final_mesh(self, prefix="imu_fusion_model_final"):
        with self._lock:
            final_mesh = self.volume.extract_triangle_mesh()
            if len(final_mesh.vertices) == 0:
                if self.logger is not None:
                    self.logger.save("没有有效模型数据可保存")
                return False, None, None

            final_mesh = self._apply_output_axis_transform(final_mesh)
            final_mesh.compute_vertex_normals()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            hist_path = os.path.join(self.model_dir, f"{prefix}_{ts}.obj")
            latest_path = os.path.join(self.model_dir, f"{prefix}.obj")

            ok_hist = o3d.io.write_triangle_mesh(
                hist_path,
                final_mesh,
                write_vertex_normals=True,
            )

            if ok_hist:
                shutil.copyfile(hist_path, latest_path)
                if self.logger is not None:
                    self.logger.save(f"模型历史版: {hist_path}")
                    self.logger.save(f"模型最新版: {latest_path}")
                return True, hist_path, latest_path

            if self.logger is not None:
                self.logger.save("历史版模型保存失败")
            return False, None, None