import open3d as o3d
import numpy as np
import os
import time
import serial
import struct
import threading
from collections import deque
from bisect import bisect_left
from scipy.spatial.transform import Rotation as R

from datetime import datetime
import shutil

from realtime_scanner import DepthScanner

class TSDFSensorFusion:
    def __init__(self):
        self.scanner = DepthScanner()

        # 1) TSDF
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # 2) 相机内参
        width, height = 640, 480
        fx, fy = 525.0, 525.0
        cx, cy = 319.5, 239.5
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # 3) RGBD
        self.input_color_is_bgr = False
        self.depth_scale = None
        self.depth_trunc = 2.0

        # 4) IMU
        self.use_imu = False
        self.serial_port = "COM7"
        self.baud_rate = 115200
        self.ser = None
        self.imu_thread = None
        self.is_imu_running = False
        self.lock = threading.Lock()

        self.initial_rotation_inv = None
        self.use_legacy_euler_mapping = False

        # IMU缓存
        self.imu_buf_maxlen = 50000
        self.imu_ts_buf = deque(maxlen=self.imu_buf_maxlen)
        self.imu_R_buf = deque(maxlen=self.imu_buf_maxlen)

        # 路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # model: 重建输出
        self.model_dir = os.path.join(self.script_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # npy: 标定参数
        self.npy_dir = os.path.join(self.script_dir, "npy")
        os.makedirs(self.npy_dir, exist_ok=True)

        # 标定文件
        self.imu_cam_rot_path = os.path.join(self.npy_dir, "imu_to_cam_rot.npy")
        self.imu_cam_dt_path = os.path.join(self.npy_dir, "imu_to_cam_dt.npy")
        self.imu_legacy_flag_path = os.path.join(self.npy_dir, "imu_euler_legacy_flag.npy")

        self.R_ci = np.eye(3, dtype=np.float64)   # IMU -> CAM
        self.imu_delay = 0.0                      # t_imu + dt 对齐 t_cam

        # 5) 融合状态
        self.prev_rgbd = None
        self.prev_cam_ts = None

        # 保存 world -> current_camera 的外参，直接用于 TSDF integrate
        self.T_c_w = np.eye(4, dtype=np.float64)

        # 过滤参数
        self.max_rotation_deg_per_frame = 12.0
        self.rotation_dominant_angle_deg = 0.25

        self.max_translation_when_rotating = 0.008
        self.min_info_trace = 5e5
        self.min_translation_per_frame = 0.0008
        self.max_translation_per_frame = 0.03
        self.trans_smooth_alpha = 0.3
        self.last_trans = np.zeros(3, dtype=np.float64)

        # IMU去抖（可选）
        self.min_imu_rot_deg_per_frame = 0.30
        self.imu_rot_smooth_beta = 0.20
        self.imu_rot_smooth_beta_fast = 0.90
        self.imu_smooth_fast_deg = 0.60
        self.filtered_imu_rot = np.eye(3, dtype=np.float64)
        self.enable_imu_lowpass = False

        # 积分策略
        self.integrate_only_when_motion = True
        self.min_integrate_rot_deg = 0.30
        self.max_integrate_rot_deg = 8.0
        self.min_integrate_trans_m = 0.0008

        # IMU使用策略：只给VO初值 / VO失败时兜底，不做后融合修正
        self.imu_as_vo_init_only = True
        self.allow_imu_fallback_when_vo_fails = True

        # 可视化
        self.mesh_update_interval = 10
        self.mesh_min_vertices_to_show = 10

        # 调试
        self.debug_print_odom = True

        # 统计
        self.total_frames = 0
        self.integrated_frames = 0
        self.skipped_invalid_depth = 0

        self.load_imu_legacy_flag()
        self.load_imu_cam_extrinsic()
        self.load_imu_cam_dt()

    # ---------- 工具 ----------
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
                print(f"⚠️ dt加载失败: {e}，使用0")
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

    def imu_delta_to_cam_delta(self, dR_imu):
        return self.normalize_rotation(self.R_ci @ dR_imu @ self.R_ci.T)

    def read_exact(self, ser, n):
        data = b""
        while len(data) < n and self.is_imu_running:
            chunk = ser.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data if len(data) == n else None

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
            print(f"[DEBUG] depth dtype={depth_np.dtype}, p95={vmax:.3f}, depth_scale={self.depth_scale}")

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
            self.skipped_invalid_depth += 1
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

    # ---------- IMU线程 ----------
    def start_imu_thread(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            self.is_imu_running = True
            self.imu_thread = threading.Thread(target=self._imu_worker, daemon=True)
            self.imu_thread.start()
            print("✅ IMU后台线程启动")
            return True
        except Exception as e:
            print(f"⚠️ IMU打开失败，降级纯视觉。错误: {e}")
            self.use_imu = False
            return False

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

                data_bytes = payload[1:13]
                r_rad = struct.unpack("<f", data_bytes[0:4])[0]
                p_rad = struct.unpack("<f", data_bytes[4:8])[0]
                y_rad = struct.unpack("<f", data_bytes[8:12])[0]

                if self.use_legacy_euler_mapping:
                    current_R = R.from_euler("xyz", [-p_rad, y_rad, -r_rad], degrees=False).as_matrix()
                else:
                    current_R = R.from_euler("xyz", [r_rad, p_rad, y_rad], degrees=False).as_matrix()

                current_R = self.normalize_rotation(current_R)
                if not self.is_valid_rotation(current_R):
                    continue

                ts = time.time()

                with self.lock:
                    if self.initial_rotation_inv is None:
                        self.initial_rotation_inv = np.linalg.inv(current_R)
                        print("🎯 IMU零点标定完成")

                    relative_R = self.normalize_rotation(self.initial_rotation_inv @ current_R)
                    self.imu_ts_buf.append(ts)
                    self.imu_R_buf.append(relative_R)

            except Exception:
                time.sleep(0.002)

    # ---------- IMU插值 ----------
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

    # ---------- 主循环 ----------
    def run(self):
        if not self.scanner.init():
            print("❌ 相机初始化失败")
            return

        self.initial_rotation_inv = None
        self.prev_rgbd = None
        self.prev_cam_ts = None
        self.T_c_w = np.eye(4, dtype=np.float64)
        self.filtered_imu_rot = np.eye(3, dtype=np.float64)
        self.last_trans = np.zeros(3, dtype=np.float64)

        if self.use_imu:
            self.start_imu_thread()

        vis = o3d.visualization.Visualizer()
        vis.create_window("TSDF Fusion (Stable Mode)", width=1280, height=720)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2])
        opt.mesh_show_back_face = True

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(axes)

        self.mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(self.mesh)

        print("🚀 启动")
        print(f"模式: {'IMU辅助视觉' if self.use_imu else '纯视觉'}")
        print(f"当前 dt = {self.imu_delay * 1000:.1f} ms")
        print("策略: IMU仅用于VO初值/失败兜底，不做后融合修正")
        print("位姿链: 使用 world->camera，delta 左乘累计，integrate 直接用 extrinsic=T_c_w")

        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = 0.07

        initialized_view = False
        frame_idx = 0

        try:
            while True:
                result = self.scanner.get_rgb_and_depth()
                if result is None:
                    continue

                self.total_frames += 1
                color_np, depth_np = result
                current_rgbd = self.preprocess_frame(color_np, depth_np)
                if current_rgbd is None:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
                    continue

                cam_ts = time.time()
                do_integrate = False

                # 可选：当前姿态低通，仅做缓存，不直接主导
                current_imu_rot = None
                if self.use_imu:
                    q_ts = cam_ts + self.imu_delay
                    Rq = self.get_imu_rotation_at(q_ts)
                    if Rq is not None and self.is_valid_rotation(Rq):
                        current_imu_rot = self.normalize_rotation(Rq)

                        if self.enable_imu_lowpass:
                            dR_f = self.normalize_rotation(self.filtered_imu_rot.T @ current_imu_rot)
                            rv_f = R.from_matrix(dR_f).as_rotvec()
                            raw_deg_f = np.degrees(np.linalg.norm(rv_f))
                            beta = (
                                self.imu_rot_smooth_beta_fast
                                if raw_deg_f >= self.imu_smooth_fast_deg
                                else self.imu_rot_smooth_beta
                            )
                            rv_f = beta * rv_f
                            self.filtered_imu_rot = self.normalize_rotation(
                                self.filtered_imu_rot @ R.from_rotvec(rv_f).as_matrix()
                            )
                            current_imu_rot = self.filtered_imu_rot.copy()

                # ---------------- 第一帧 ----------------
                if self.prev_rgbd is None:
                    self.prev_rgbd = current_rgbd
                    self.prev_cam_ts = cam_ts
                    self.T_c_w = np.eye(4, dtype=np.float64)
                    do_integrate = True
                    print("✅ 第一帧基准建立")

                else:
                    # ---------------- IMU增量：仅用于VO初值/兜底 ----------------
                    dR_imu_cam = np.eye(3, dtype=np.float64)
                    imu_rot_deg = 0.0
                    imu_delta_available = False

                    if self.use_imu and self.prev_cam_ts is not None:
                        R0 = self.get_imu_rotation_at(self.prev_cam_ts + self.imu_delay)
                        R1 = self.get_imu_rotation_at(cam_ts + self.imu_delay)

                        if R0 is not None and R1 is not None:
                            dR_imu = self.normalize_rotation(R0.T @ R1)

                            if self.is_valid_rotation(dR_imu):
                                dR_imu_cam = self.imu_delta_to_cam_delta(dR_imu)
                                imu_rot_deg = self.rot_deg(dR_imu_cam)

                                if self.min_imu_rot_deg_per_frame <= imu_rot_deg <= self.max_rotation_deg_per_frame:
                                    imu_delta_available = True

                    # ---------------- VO 初值 ----------------
                    init_delta = np.eye(4, dtype=np.float64)
                    if self.use_imu and self.imu_as_vo_init_only and imu_delta_available:
                        init_delta[:3, :3] = dR_imu_cam

                    success, delta_refined, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                        self.prev_rgbd,
                        current_rgbd,
                        self.intrinsic,
                        init_delta,
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                        option
                    )

                    bad_frame = False
                    fused_delta = np.eye(4, dtype=np.float64)
                    fused_R = np.eye(3, dtype=np.float64)

                    info_trace = float(np.trace(info)) if (success and info is not None) else 0.0
                    vo_rot_deg = 0.0
                    trans_refined = np.zeros(3, dtype=np.float64)
                    accepted_t = False
                    odom_t_norm = 0.0

                    # ---------------- 旋转策略 ----------------
                    # VO好 -> 用VO
                    # VO差/失败 -> 小角度IMU兜底
                    # 不做 VO+IMU 后融合修正
                    if success and info is not None:
                        dR_vo = self.normalize_rotation(delta_refined[:3, :3])
                        vo_rot_deg = self.rot_deg(dR_vo)

                        vo_rot_ok = (vo_rot_deg <= self.max_rotation_deg_per_frame)
                        vo_info_ok = (info_trace >= self.min_info_trace)

                        if vo_rot_ok and vo_info_ok:
                            fused_R = dR_vo
                        else:
                            if (
                                self.use_imu
                                and self.allow_imu_fallback_when_vo_fails
                                and imu_delta_available
                                and imu_rot_deg <= 5.0
                            ):
                                fused_R = dR_imu_cam
                            else:
                                bad_frame = True
                    else:
                        if (
                            self.use_imu
                            and self.allow_imu_fallback_when_vo_fails
                            and imu_delta_available
                            and imu_rot_deg <= 5.0
                        ):
                            fused_R = dR_imu_cam
                        else:
                            bad_frame = True

                    if bad_frame:
                        self.prev_rgbd = current_rgbd
                        self.prev_cam_ts = cam_ts
                        if not vis.poll_events():
                            break
                        vis.update_renderer()
                        continue

                    fused_delta[:3, :3] = fused_R

                    # ---------------- 平移策略 ----------------
                    # 平移只信 VO，不信 IMU
                    if success and info is not None:
                        raw_t = delta_refined[:3, 3].copy()
                        odom_t_norm = float(np.linalg.norm(raw_t))

                        # 异常大平移，整帧丢弃
                        if odom_t_norm > 0.05:
                            self.prev_rgbd = current_rgbd
                            self.prev_cam_ts = cam_ts
                            if not vis.poll_events():
                                break
                            vis.update_renderer()
                            continue

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
                                f"[ODOM] vo_rot={vo_rot_deg:.2f}deg, "
                                f"imu_rot={imu_rot_deg:.2f}deg, "
                                f"info={info_trace:.1f}, "
                                f"t_raw={odom_t_norm:.4f}m, "
                                f"accepted_t={accepted_t}, "
                                f"t_use={np.linalg.norm(trans_refined):.4f}m, "
                                f"imu_delta_ok={imu_delta_available}"
                            )
                        else:
                            print(
                                f"[ODOM] failed, imu_rot={imu_rot_deg:.2f}deg, "
                                f"imu_delta_ok={imu_delta_available}"
                            )

                    # ---------------- 关键修正：位姿链更新 ----------------
                    # 保存 world -> current_camera
                    # 新增量左乘
                    self.T_c_w = fused_delta @ self.T_c_w

                    self.prev_rgbd = current_rgbd
                    self.prev_cam_ts = cam_ts

                    # ---------------- 是否积分 ----------------
                    if self.integrate_only_when_motion:
                        trans_used = float(np.linalg.norm(trans_refined))
                        rot_used = self.rot_deg(fused_R)
                        moving = (
                            (trans_used >= self.min_integrate_trans_m)
                            or (rot_used >= self.min_integrate_rot_deg)
                        )
                        rot_too_fast = (rot_used > self.max_integrate_rot_deg)
                        do_integrate = moving and (not rot_too_fast)
                    else:
                        do_integrate = True

                # ---------------- TSDF积分 ----------------
                if do_integrate:
                    extrinsic = self.T_c_w.copy()
                    self.volume.integrate(current_rgbd, self.intrinsic, extrinsic)
                    self.integrated_frames += 1

                frame_idx += 1

                if frame_idx % self.mesh_update_interval == 0:
                    new_mesh = self.volume.extract_triangle_mesh()
                    vnum = len(new_mesh.vertices)
                    print(
                        f"[STAT] total={self.total_frames}, integrated={self.integrated_frames}, "
                        f"mesh_vertices={vnum}, imu_buf={len(self.imu_ts_buf)}"
                    )

                    if vnum > self.mesh_min_vertices_to_show:
                        new_mesh.compute_vertex_normals()
                        self.mesh.vertices = new_mesh.vertices
                        self.mesh.triangles = new_mesh.triangles
                        self.mesh.vertex_colors = new_mesh.vertex_colors
                        self.mesh.vertex_normals = new_mesh.vertex_normals
                        vis.update_geometry(self.mesh)

                        if not initialized_view:
                            vis.reset_view_point(True)
                            initialized_view = True
                            print("🎯 已自动对焦模型")

                if not vis.poll_events():
                    break
                vis.update_renderer()

        finally:
            self.is_imu_running = False
            if self.imu_thread is not None and self.imu_thread.is_alive():
                self.imu_thread.join(timeout=0.5)
            if self.ser and self.ser.is_open:
                self.ser.close()

            self.scanner.close()
            vis.destroy_window()

        print("\n💾 保存最终模型...")
        final_mesh = self.volume.extract_triangle_mesh()
        if len(final_mesh.vertices) > 0:
            final_mesh.compute_vertex_normals()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            hist_path = os.path.join(self.model_dir, f"imu_fusion_model_final_{ts}.obj")
            latest_path = os.path.join(self.model_dir, "imu_fusion_model_final.obj")

            ok_hist = o3d.io.write_triangle_mesh(hist_path, final_mesh, write_vertex_normals=True)

            if ok_hist:
                shutil.copyfile(hist_path, latest_path)
                print(f"✅ 模型历史版: {hist_path}")
                print(f"✅ 模型最新版: {latest_path}")
            else:
                print("⚠️ 历史版模型保存失败")
        else:
            print("⚠️ 没有有效模型数据可保存")

if __name__ == "__main__":
    app = TSDFSensorFusion()
    app.run()