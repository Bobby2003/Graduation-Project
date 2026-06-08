import time
import threading
from pathlib import Path
from typing import Optional, Any

import numpy as np

from .config import PipelineConfig
from .common.shared_state import SharedState
from .common.queue_utils import LatestFrameSlot, BoundedDropQueue
from .common.types import RGBDFrame

from .input.input_adapter import InputAdapter

from .tracking.tracker import Tracker
from .tracking.gpu_icp_tracker import GpuICPTracker
from .tracking.tracking_worker import TrackingWorker

from .mapping.mapper import Mapper
from .mapping.mapping_worker import MappingWorker
from .mapping.local_map import LocalMap

from .rendering.renderer import Renderer
from .rendering.render_worker import RenderWorker

from .utils.log_redirect import LogRedirectManager
from .utils.module_logger import ModuleLogger
from .imu import IMUWorldInitializer, SerialIMUProvider

def resolve_output_dir(path_str: str, base_dir: Path) -> str:
    """
    将配置中的目录统一解析为绝对路径：
    - 如果本身是绝对路径，直接使用
    - 如果是相对路径，则相对于 base_dir
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def build_tracker(cfg: PipelineConfig, local_map=None, logger=None, app_logger=None, imu_world_initializer=None):
    cam = cfg.tracking_camera
    tcfg = cfg.tracking
    icfg = cfg.input
    lcfg = cfg.logging

    backend = str(tcfg.tracking_backend).lower().strip()

    if backend == "gpu_icp":
        return GpuICPTracker(
            width=cam.width,
            height=cam.height,
            fx=cam.fx,
            fy=cam.fy,
            cx=cam.cx,
            cy=cam.cy,

            input_color_is_bgr=icfg.input_color_is_bgr,
            depth_scale=icfg.depth_scale,
            depth_trunc=icfg.depth_trunc,
            min_valid_pixels=tcfg.min_valid_pixels,

            max_rotation_deg_per_frame=tcfg.max_rotation_deg_per_frame,
            rotation_dominant_angle_deg=tcfg.rotation_dominant_angle_deg,
            max_translation_when_rotating=tcfg.max_translation_when_rotating,
            min_translation_per_frame=tcfg.min_translation_per_frame,
            max_translation_per_frame=tcfg.max_translation_per_frame,
            trans_smooth_alpha=tcfg.trans_smooth_alpha,

            device=tcfg.gpu_device,
            tracking_voxel_size=tcfg.gpu_tracking_voxel_size,
            tracking_pcd_stride=tcfg.gpu_tracking_pcd_stride,
            icp_max_correspondence_distance=tcfg.gpu_icp_max_correspondence_distance,
            icp_max_iteration=tcfg.gpu_icp_max_iteration,
            normal_radius=tcfg.gpu_normal_radius,
            normal_max_nn=tcfg.gpu_normal_max_nn,
            min_icp_fitness=tcfg.gpu_min_icp_fitness,
            max_icp_rmse=tcfg.gpu_max_icp_rmse,
            allow_point_to_point_fallback=tcfg.gpu_allow_point_to_point_fallback,

            local_map=local_map,
            use_map_tracking=tcfg.use_map_tracking,
            map_tracking_min_points=tcfg.map_tracking_min_points,
            map_tracking_max_corr=tcfg.map_tracking_max_correspondence_distance,

            map_min_icp_fitness=tcfg.map_min_icp_fitness,
            map_max_icp_rmse=tcfg.map_max_icp_rmse,
            map_max_delta_trans=tcfg.map_max_delta_trans,
            map_max_delta_rot_deg=tcfg.map_max_delta_rot_deg,
            map_icp_cooldown_after_fail=tcfg.map_icp_cooldown_after_fail,
            relocalize_try_interval=tcfg.relocalize_try_interval,
            lost_print_interval=tcfg.lost_print_interval,
            relocalize_use_full_map_fallback=tcfg.relocalize_use_full_map_fallback,
            relocalize_full_map_max_corr=tcfg.relocalize_full_map_max_correspondence_distance,
            relocalize_full_map_min_fitness=tcfg.relocalize_full_map_min_fitness,
            relocalize_full_map_max_rmse=tcfg.relocalize_full_map_max_rmse,

            imu_world_initializer=imu_world_initializer,
            logger=logger,
        )

    elif backend == "cpu_rgbd":
        return Tracker(
            imu_manager=None,

            width=cam.width,
            height=cam.height,
            fx=cam.fx,
            fy=cam.fy,
            cx=cam.cx,
            cy=cam.cy,

            input_color_is_bgr=icfg.input_color_is_bgr,
            depth_scale=icfg.depth_scale,
            depth_trunc=icfg.depth_trunc,
            min_valid_pixels=tcfg.min_valid_pixels,

            max_rotation_deg_per_frame=tcfg.max_rotation_deg_per_frame,
            rotation_dominant_angle_deg=tcfg.rotation_dominant_angle_deg,

            max_translation_when_rotating=tcfg.max_translation_when_rotating,
            min_info_trace=tcfg.min_info_trace,
            min_translation_per_frame=tcfg.min_translation_per_frame,
            max_translation_per_frame=tcfg.max_translation_per_frame,
            trans_smooth_alpha=tcfg.trans_smooth_alpha,

            imu_world_initializer=imu_world_initializer,
            debug_print_odom=lcfg.debug.enabled,
            enable_profile=lcfg.profile.enabled,
            profile_print_interval=lcfg.profile.interval,
        )

    else:
        raise ValueError(
            f"Unknown tracking backend: {tcfg.tracking_backend}. "
            "Expected 'gpu_icp' or 'cpu_rgbd'."
        )

def build_mapper(cfg: PipelineConfig, local_map=None, logger=None):
    cam = cfg.mapping_camera
    mcfg = cfg.mapping
    icfg = cfg.input

    return Mapper(
        voxel_length=mcfg.voxel_length,
        sdf_trunc=mcfg.sdf_trunc,

        width=cam.width,
        height=cam.height,
        fx=cam.fx,
        fy=cam.fy,
        cx=cam.cx,
        cy=cam.cy,

        input_color_is_bgr=icfg.input_color_is_bgr,
        depth_scale=icfg.depth_scale,
        depth_trunc=icfg.depth_trunc,

        integrate_only_when_motion=mcfg.integrate_only_when_motion,
        min_integrate_rot_deg=mcfg.min_integrate_rot_deg,
        max_integrate_rot_deg=mcfg.max_integrate_rot_deg,
        min_integrate_trans_m=mcfg.min_integrate_trans_m,

        mesh_update_interval=mcfg.mesh_update_interval,
        mesh_min_vertices_to_show=mcfg.mesh_min_vertices_to_show,

        model_dir=mcfg.model_dir,

        mapping_lost_print_interval=mcfg.mapping_lost_print_interval,
        max_mapping_pose_jump_trans=mcfg.max_mapping_pose_jump_trans,
        max_mapping_pose_jump_rot_deg=mcfg.max_mapping_pose_jump_rot_deg,
        local_map_frame_pcd_voxel_size=mcfg.local_map_frame_pcd_voxel_size,
        local_map_frame_pcd_stride=mcfg.local_map_frame_pcd_stride,
        apply_output_axis_transform=mcfg.apply_output_axis_transform,
        R_output_from_reconstruction=mcfg.R_output_from_reconstruction,
        auto_align_output_yaw=mcfg.auto_align_output_yaw,
        auto_align_min_vertices=mcfg.auto_align_min_vertices,
        auto_align_min_angle_deg=mcfg.auto_align_min_angle_deg,
        output_translate_to_positive=mcfg.output_translate_to_positive,

        local_map=local_map,
        logger=logger,
    )

def build_renderer(cfg: PipelineConfig, logger=None):
    rcfg = cfg.render
    lcfg = cfg.logging

    return Renderer(
        window_name=rcfg.window_name,
        width=rcfg.width,
        height=rcfg.height,
        enable_log=lcfg.status.enabled,
        logger=logger,
    )

def build_imu_world_initializer(cfg: PipelineConfig, logger=None):
    icfg = cfg.imu
    if not icfg.enable_world_init:
        return IMUWorldInitializer(enabled=False, logger=logger)

    provider = SerialIMUProvider(
        port=icfg.serial_port,
        baudrate=icfg.serial_baudrate,
        timeout=icfg.serial_timeout_sec,
        read_window_sec=icfg.init_timeout_sec,
        logger=logger,
    )

    return IMUWorldInitializer(
        provider=provider,
        enabled=True,
        required=icfg.required,
        R_cam_imu=icfg.R_cam_imu,
        R_reconstruction_imu_world=icfg.R_reconstruction_imu_world,
        quaternion_convention=icfg.quaternion_convention,
        initial_alignment_mode=getattr(icfg, "initial_alignment_mode", None),
        sample_timeout_sec=icfg.init_timeout_sec,
        zero_initial_camera_rotation=icfg.zero_initial_camera_rotation,
        logger=logger,
    )

class RealtimeMappingPipeline:
    """
    实时 RGB-D / 深度点云建模 Pipeline API。

    支持三种运行方式：

    1. run_forever()
       等价于原 main_pipeline.py 的行为。

    2. start() + run_once()
       适合需要 Open3D 窗口的场景，由主线程驱动。

    3. start_background()
       适合其他脚本后台调用，要求 cfg.render.enabled=False。

    4. start_external() + submit_rgbd()
       适合外部脚本主动喂帧，不直接使用硬件 scanner。
    """

    def __init__(
        self,
        cfg: Optional[PipelineConfig] = None,
        scanner: Optional[Any] = None,
        base_dir: Optional[Path] = None,
        owns_scanner: bool = True,
    ):
        self.cfg = cfg if cfg is not None else PipelineConfig()

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        self.base_dir = Path(base_dir)

        self.external_scanner = scanner
        self.owns_scanner = owns_scanner

        self.scanner = None
        self.input_adapter = None

        self.shared_state = None
        self.latest_frame_slot = None
        self.mapping_queue = None
        self.stop_event = threading.Event()

        self.local_map = None

        self.imu_world_initializer = None
        self.tracker = None
        self.mapper = None
        self.renderer = None

        self.tracking_worker = None
        self.mapping_worker = None
        self.render_worker = None

        self.log_manager = LogRedirectManager()
        self.log_path = None

        self.app_logger = ModuleLogger("PIPELINE", self.cfg.logging)
        self.input_logger = ModuleLogger("INPUT", self.cfg.logging)
        self.tracker_logger = ModuleLogger("TRACKER", self.cfg.logging)
        self.mapper_logger = ModuleLogger("MAPPER", self.cfg.logging)
        self.local_map_logger = ModuleLogger("LOCAL_MAP", self.cfg.logging)
        self.render_logger = ModuleLogger("RENDER", self.cfg.logging)

        self._setup_done = False
        self._running = False
        self._input_mode = "scanner"

        self._background_thread = None
        self._background_thread_started = False

        self._frame_counter_external = 0

        self._prev_ts = None
        self._ts_debug_count = 0

        self._last_error = None

        # Center / external recovery gate (tracking & mapping integrate)
        self._center_recovery_gate_lock = threading.Lock()
        self._center_tracking_enabled = True
        self._center_mapping_enabled = True
        self._tracking_result_observer = None

    # ============================================================
    # 内部构建
    # ============================================================

    def _log_config(self):
        cfg = self.cfg
        self.app_logger.status(f"save_log={cfg.logging.save_log}", force=True)
        self.app_logger.status(f"log_dir={cfg.logging.log_dir}", force=True)
        self.app_logger.status(f"model_dir={cfg.mapping.model_dir}", force=True)
        self.app_logger.status(f"input_camera={cfg.input_camera}", force=True)
        self.app_logger.status(f"tracking_camera={cfg.tracking_camera}", force=True)
        self.app_logger.status(f"mapping_camera={cfg.mapping_camera}", force=True)
        self.app_logger.status(f"tracking_backend={cfg.tracking.tracking_backend}", force=True)
        self.app_logger.status(
            f"imu_world_init_enabled={cfg.imu.enable_world_init}",
            force=True,
        )
        self.app_logger.status(f"render_enabled={cfg.render.enabled}", force=True)

        if self.log_path is not None:
            self.app_logger.status(f"log_file={self.log_path}", force=True)

    def _create_default_scanner(self):
        """
        延迟导入 PCAC.UnifiedDepthScanner，避免没有 Orbbec SDK 时，
        外部输入模式仅 import pipeline_api 就失败。
        """
        from PCAC.UnifiedDepthScanner import UnifiedDepthScanner

        icam = self.cfg.input_camera

        return UnifiedDepthScanner(
            width=icam.width,
            height=icam.height,
            fx=icam.fx,
            fy=icam.fy,
            cx=icam.cx,
            cy=icam.cy,
        )
    
    def _create_runtime_objects(self, recreate_renderer: bool = True):
        """
        创建一整套“本次建模会话”的运行时对象。

        这些对象都属于当前进程内的一次建模状态：
        - SharedState
        - LatestFrameSlot
        - MappingQueue
        - stop_event
        - LocalMap
        - Tracker
        - Mapper / TSDF volume
        - Renderer / RenderWorker
        - TrackingWorker
        - MappingWorker

        reset_reconstruction() 会调用这个方法重新创建它们，
        从而达到“不退出进程、不删磁盘文件、清空当前内存模型并重新建模”的效果。
        """
        cfg = self.cfg

        # 新一轮建模使用全新的 stop_event。
        # 旧 stop_event 已经被 set，用于通知旧 worker 退出。
        self.stop_event = threading.Event()

        # 线程间共享状态：清空旧 tracking/map snapshot
        self.shared_state = SharedState()

        # 清空输入帧槽和 mapping 队列
        self.latest_frame_slot = LatestFrameSlot()
        self.mapping_queue = BoundedDropQueue(maxsize=cfg.mapping.queue_size)

        # 新 local map
        self.local_map = LocalMap(
            voxel_size=cfg.tracking.local_map_voxel_size,
            max_points=cfg.tracking.local_map_max_points,
            local_radius=cfg.tracking.local_map_radius,
            min_points_for_tracking=cfg.tracking.local_map_min_points_for_tracking,
            logger=self.local_map_logger,
        )

        self.imu_world_initializer = build_imu_world_initializer(
            cfg,
            logger=self.tracker_logger,
        )

        # 新 tracker：清空累计位姿、上一帧、lost 状态等
        self.tracker = build_tracker(
            cfg,
            local_map=self.local_map,
            logger=self.tracker_logger,
            app_logger=self.app_logger,
            imu_world_initializer=self.imu_world_initializer,
        )

        # 新 mapper：清空 TSDF volume、last_mesh、integrated_frames 等
        self.mapper = build_mapper(
            cfg,
            local_map=self.local_map,
            logger=self.mapper_logger,
        )

        # Renderer 是否重建：
        # - setup() 初次创建时需要建
        # - reset_reconstruction() 时建议先 close 旧 renderer，再重建，避免旧 mesh 残留
        self.render_worker = None

        if cfg.render.enabled:
            if recreate_renderer or self.renderer is None:
                self.renderer = build_renderer(cfg, logger=self.render_logger)

            self.render_worker = RenderWorker(
                shared_state=self.shared_state,
                renderer=self.renderer,
                stop_event=self.stop_event,
                render_fps=cfg.render.fps,
                enable_status_log=cfg.logging.status.enabled,
                status_log_interval=cfg.logging.status.interval,
                logger=self.render_logger,
            )
        else:
            self.renderer = None
            self.render_worker = None

        # 新 tracking worker
        self.tracking_worker = TrackingWorker(
            latest_frame_slot=self.latest_frame_slot,
            tracker=self.tracker,
            shared_state=self.shared_state,
            mapping_queue=self.mapping_queue,
            stop_event=self.stop_event,
            sleep_ms=cfg.tracking.sleep_ms,
            enable_log=cfg.logging.status.enabled,
            mapping_stride=cfg.tracking.output_mapping_stride,
            logger=self.tracker_logger,
            tracking_enabled_fn=self.get_center_tracking_enabled,
            tracking_result_observer=self._invoke_tracking_observer,
        )

        # 新 mapping worker
        self.mapping_worker = MappingWorker(
            mapping_queue=self.mapping_queue,
            mapper=self.mapper,
            shared_state=self.shared_state,
            stop_event=self.stop_event,
            enable_log=cfg.logging.status.enabled,
            logger=self.mapper_logger,
            mapping_enabled_fn=self.get_center_mapping_enabled,
        )

    def setup(self, input_mode: str = "scanner"):
        """
        初始化资源，但不启动 worker。

        input_mode:
        - "scanner": 从 scanner / UnifiedDepthScanner 取帧
        - "external": 外部通过 submit_frame / submit_rgbd 喂帧
        """
        if self._setup_done:
            return True

        input_mode = str(input_mode).lower().strip()
        if input_mode not in ("scanner", "external"):
            raise ValueError("input_mode must be 'scanner' or 'external'")

        self._input_mode = input_mode

        cfg = self.cfg

        # 修正输出目录
        cfg.mapping.model_dir = resolve_output_dir(cfg.mapping.model_dir, self.base_dir)
        cfg.logging.log_dir = resolve_output_dir(cfg.logging.log_dir, self.base_dir)

        # 日志重定向
        self.log_path = self.log_manager.start(
            log_dir=cfg.logging.log_dir,
            log_prefix=cfg.logging.log_prefix,
            enabled=cfg.logging.save_log,
        )

        self._log_config()

        # scanner 模式才初始化相机
        if input_mode == "scanner":
            if self.external_scanner is not None:
                self.scanner = self.external_scanner
            else:
                self.scanner = self._create_default_scanner()

            if not self.scanner.init():
                self.app_logger.warning("scanner init failed", force=True)
                try:
                    self.scanner.close()
                except Exception:
                    pass
                self.scanner = None
                self._cleanup_log_if_needed()
                return False

            self.input_adapter = InputAdapter(
                scanner=self.scanner,
                timestamp_unit=cfg.input.device_timestamp_unit,
                validate=cfg.input.validate_input_frame,
                print_timestamp_debug_once=cfg.input.print_timestamp_debug_once,
                logger=self.input_logger,
            )

        else:
            # external 模式不创建 scanner / input_adapter
            self.scanner = None
            self.input_adapter = None

        # 创建本次建模会话的运行时对象：
        # SharedState / Queue / LocalMap / Tracker / Mapper / Renderer / Workers
        self._create_runtime_objects(recreate_renderer=True)

        self._setup_done = True
        return True
    
    def _cleanup_log_if_needed(self):
        if self.cfg.logging.save_log:
            if self.log_path is not None:
                self.app_logger.status(f"Log file: {self.log_path}", force=True)
            self.log_manager.stop()

    # ============================================================
    # 生命周期
    # ============================================================

    def start(self, input_mode: str = "scanner") -> bool:
        """
        启动 tracking/mapping worker。

        注意：
        - 这个方法不会阻塞。
        - 如果启用了 render，需要外部主线程循环调用 run_once()。
        - 如果需要后台自动采集，使用 start_background()。
        """
        if self._running:
            return True

        ok = self.setup(input_mode=input_mode)
        if not ok:
            return False

        self.tracking_worker.start()
        self.mapping_worker.start()

        self._running = True

        if self.render_worker is not None:
            self.app_logger.status(
                "Render is enabled: call run_once() on MAIN thread",
                force=True,
            )

        self.app_logger.status("started", force=True)
        return True

    def start_background(self) -> bool:
        """
        后台启动完整采集循环。

        要求：
        - cfg.render.enabled 必须为 False
        因为 Open3D 可视化一般要求在主线程驱动。
        """
        if self.cfg.render.enabled:
            raise RuntimeError(
                "start_background() requires cfg.render.enabled=False. "
                "If render is enabled, use start() + run_once() in main thread."
            )

        if not self.start(input_mode="scanner"):
            return False

        if self._background_thread_started:
            return True

        self._background_thread = threading.Thread(
            target=self._background_loop,
            name="PipelineInputLoop",
            daemon=True,
        )
        self._background_thread.start()
        self._background_thread_started = True

        return True

    def start_external(self) -> bool:
        """
        外部喂帧模式。

        此模式不初始化 UnifiedDepthScanner。
        外部脚本需要调用 submit_frame() 或 submit_rgbd()。
        """
        return self.start(input_mode="external")

    def _background_loop(self):
        try:
            while self.is_running():
                ok = self.run_once()
                if not ok:
                    break
        except Exception as e:
            self._last_error = e
            self.app_logger.warning(f"background loop exception: {repr(e)}", force=True)
            self.stop_event.set()

    def run_once(self) -> bool:
        """
        执行一次主循环。

        scanner 模式下：
        - 刷新渲染
        - 读取一帧
        - 投递给 tracking

        external 模式下：
        - 只刷新渲染，不主动读取相机

        返回：
        - True: 继续
        - False: 应停止
        """
        if not self._running:
            return False

        if self.stop_event.is_set():
            return False

        try:
            # Open3D 渲染由主线程驱动
            if self.render_worker is not None:
                ok = self.render_worker.run_once()
                if not ok:
                    self.stop_event.set()
                    return False

            if self._input_mode == "external":
                time.sleep(0.001)
                return True

            frame = self.input_adapter.get_frame()
            if frame is None:
                time.sleep(0.001)
                return True

            self._debug_timestamp(frame)
            self.latest_frame_slot.put(frame)

            return True

        except Exception as e:
            self._last_error = e
            self.app_logger.warning(f"run_once exception: {repr(e)}", force=True)
            self.stop_event.set()
            return False

    def run_forever(self, save_mesh: bool = True, final_mesh_prefix: str = "fusion_model_final"):
        """
        等价于旧 main_pipeline.py 行为。
        """
        ok = self.start(input_mode="scanner")
        if not ok:
            return None

        self.app_logger.status(
            "Pipeline started. Press Ctrl+C to stop.",
            force=True,
        )

        try:
            while self.is_running():
                if not self.run_once():
                    break
        except KeyboardInterrupt:
            self.app_logger.status("Stopping pipeline by KeyboardInterrupt...", force=True)
        finally:
            return self.stop(save_mesh=save_mesh, final_mesh_prefix=final_mesh_prefix)

    def stop(
        self,
        save_mesh: bool = True,
        final_mesh_prefix: str = "fusion_model_final",
        join_timeout: float = 1.0,
    ) -> dict:
        """
        停止 pipeline 并释放资源。
        """
        result = {
            "stopped": False,
            "mesh_saved": False,
            "mesh_history_path": None,
            "mesh_latest_path": None,
            "tracking_stats": None,
            "mapping_stats": None,
            "timestamp_info": None,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "last_error": repr(self._last_error) if self._last_error is not None else None,
        }

        if not self._setup_done:
            return result

        self.stop_event.set()

        if self.tracking_worker is not None:
            self.tracking_worker.join(timeout=join_timeout)

        if self.mapping_worker is not None:
            self.mapping_worker.join(timeout=join_timeout)

        if self._background_thread is not None and self._background_thread.is_alive():
            self._background_thread.join(timeout=join_timeout)

        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as e:
                self.app_logger.warning(f"renderer close failed: {repr(e)}", force=True)

        if self.scanner is not None and self.owns_scanner:
            try:
                self.scanner.close()
            except Exception as e:
                self.app_logger.warning(f"scanner close failed: {repr(e)}", force=True)

        if self.imu_world_initializer is not None:
            try:
                self.imu_world_initializer.close()
            except Exception as e:
                self.app_logger.warning(f"imu close failed: {repr(e)}", force=True)

        if self.input_adapter is not None:
            try:
                result["timestamp_info"] = self.input_adapter.debug_timestamp_info()
                self.app_logger.status(
                    f"Input timestamp info: {result['timestamp_info']}",
                    force=True,
                )
            except Exception:
                pass

        if self.tracking_worker is not None:
            try:
                result["tracking_stats"] = self.tracking_worker.get_stats()
                self.app_logger.status(
                    f"Tracking stats: {result['tracking_stats']}",
                    force=True,
                )
            except Exception:
                pass

        if self.mapping_worker is not None:
            try:
                result["mapping_stats"] = self.mapping_worker.get_stats()
                self.app_logger.status(
                    f"Mapping stats: {result['mapping_stats']}",
                    force=True,
                )
            except Exception:
                pass

        if save_mesh and self.mapper is not None:
            try:
                ok, hist_path, latest_path = self.mapper.save_final_mesh(
                    prefix=final_mesh_prefix
                )
                result["mesh_saved"] = bool(ok)
                result["mesh_history_path"] = hist_path
                result["mesh_latest_path"] = latest_path
            except Exception as e:
                self._last_error = e
                result["last_error"] = repr(e)
                self.app_logger.warning(f"save mesh failed: {repr(e)}", force=True)

        self._running = False
        self._setup_done = False
        self._background_thread_started = False

        result["stopped"] = True
        self.app_logger.status("stopped", force=True)

        if self.cfg.logging.save_log:
            if self.log_path is not None:
                self.app_logger.status(f"Log file: {self.log_path}", force=True)
            self.log_manager.stop()

        return result

    def close(self):
        return self.stop(save_mesh=False)

    # ============================================================
    # 外部输入
    # ============================================================

    def submit_frame(self, frame: RGBDFrame):
        """
        外部直接提交 RGBDFrame。
        """
        if not self._running:
            raise RuntimeError("Pipeline is not running. Call start_external() first.")

        if frame is None:
            return

        if self.cfg.input.validate_input_frame:
            frame.validate()

        self.latest_frame_slot.put(frame)

    def submit_rgbd(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        extras: Optional[dict] = None,
    ):
        """
        外部提交 color/depth/mask。

        注意：
        - depth 单位应与 cfg.input.depth_scale 匹配。
        - 如果 depth 是毫米，cfg.input.depth_scale 可为 None，由预处理器自动判断。
        """
        if frame_id is None:
            self._frame_counter_external += 1
            frame_id = self._frame_counter_external

        host_ts = time.perf_counter()

        if timestamp is None:
            timestamp = host_ts

        color = np.asarray(color)
        depth = np.asarray(depth)

        if mask is None:
            mask = (depth > 0).astype(np.uint8)
        else:
            mask = np.asarray(mask)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8, copy=False)

        if color.dtype != np.uint8:
            color = color.astype(np.uint8, copy=False)

        frame = RGBDFrame(
            frame_id=int(frame_id),
            device_timestamp=float(timestamp),
            host_timestamp=float(host_ts),
            color=color,
            depth=depth,
            mask=mask,
            extras=extras or {
                "timestamp_source": "external",
            },
        )

        self.submit_frame(frame)

    # ============================================================
    # 查询接口
    # ============================================================

    def is_running(self) -> bool:
        return bool(self._running) and (not self.stop_event.is_set())

    def get_latest_tracking(self):
        if self.shared_state is None:
            return None
        return self.shared_state.get_latest_tracking()

    def get_latest_map(self):
        if self.shared_state is None:
            return None
        return self.shared_state.get_latest_map()

    def get_latest_mesh(self):
        """
        返回最新 mesh 引用。

        注意：
        - 这是内部对象引用，外部只建议读，不建议修改。
        """
        if self.mapper is None:
            return None
        return self.mapper.get_latest_mesh()

    def extract_output_mesh(self):
        """
        从 TSDF 抽取最终 mesh，并应用输出坐标修正。
        """
        if self.mapper is None:
            return None
        return self.mapper.extract_output_mesh()

    def get_current_extrinsic_world_to_camera(self):
        """
        返回当前 Open3D TSDF integrate 使用的 extrinsic。

        注意：
        当前工程中 TrackingResult.T_wc 字段名虽然叫 T_wc，
        但实际被 Mapper 作为 world -> camera extrinsic 使用。
        """
        tracking = self.get_latest_tracking()
        if tracking is None:
            return None
        return tracking.T_wc.copy()

    def get_current_camera_pose_world(self):
        """
        返回 camera -> world 位姿，即 world_to_camera 的逆。
        """
        T_c_w = self.get_current_extrinsic_world_to_camera()
        if T_c_w is None:
            return None
        return np.linalg.inv(T_c_w)

    def get_current_imu_pose_world(self):
        """
        返回 IMU -> world 位姿。

        该接口不实时读取 IMU；IMU 不参与后续 tracking。
        它使用当前 camera -> world 位姿和固定 R_cam_imu 推导 IMU 坐标系。
        """
        T_w_c = self.get_current_camera_pose_world()
        if T_w_c is None or self.imu_world_initializer is None:
            return None
        return self.imu_world_initializer.make_imu_pose_from_camera_pose(T_w_c)

    def get_imu_world_init_info(self):
        if self.imu_world_initializer is None:
            return None
        return self.imu_world_initializer.get_last_init_info()

    def get_current_coordinate_frames(self):
        """
        返回当前相机坐标系和 IMU 坐标系在 world 下的 4x4 位姿。
        """
        return {
            "camera_to_world": self.get_current_camera_pose_world(),
            "imu_to_world": self.get_current_imu_pose_world(),
            "imu_world_init": self.get_imu_world_init_info(),
        }

    def get_tracking_stats(self) -> dict:
        if self.tracking_worker is None:
            return {}
        return self.tracking_worker.get_stats()

    def get_mapping_stats(self) -> dict:
        if self.mapping_worker is None:
            return {}
        return self.mapping_worker.get_stats()

    def get_last_error(self):
        return self._last_error

    def set_tracking_result_observer(self, cb):
        """Optional callback(tracking: TrackingResult) from tracking thread after each track."""
        self._tracking_result_observer = cb

    def _invoke_tracking_observer(self, tracking):
        obs = self._tracking_result_observer
        if obs is None:
            return
        try:
            obs(tracking)
        except Exception:
            pass

    def set_center_recovery_gate(self, tracking_enabled: bool, mapping_enabled: bool):
        with self._center_recovery_gate_lock:
            self._center_tracking_enabled = bool(tracking_enabled)
            self._center_mapping_enabled = bool(mapping_enabled)

    def get_center_tracking_enabled(self) -> bool:
        with self._center_recovery_gate_lock:
            return bool(self._center_tracking_enabled)

    def get_center_mapping_enabled(self) -> bool:
        with self._center_recovery_gate_lock:
            return bool(self._center_mapping_enabled)

    def get_status(self) -> dict:
        tracking = self.get_latest_tracking()
        map_snapshot = self.get_latest_map()

        tracking_info = None
        if tracking is not None:
            tracking_info = {
                "frame_id": tracking.frame_id,
                "timestamp": tracking.timestamp,
                "success": tracking.success,
                "mode": tracking.mode,
                "score": tracking.score,
                "extras": tracking.extras,
            }

        map_info = None
        if map_snapshot is not None:
            map_info = {
                "frame_id": map_snapshot.frame_id,
                "timestamp": map_snapshot.timestamp,
                "map_data": map_snapshot.map_data,
            }

        return {
            "running": self.is_running(),
            "input_mode": self._input_mode,
            "render_enabled": self.cfg.render.enabled,
            "center_tracking_enabled": self.get_center_tracking_enabled(),
            "center_mapping_enabled": self.get_center_mapping_enabled(),
            "tracking": tracking_info,
            "mapping": map_info,
            "tracking_stats": self.get_tracking_stats(),
            "mapping_stats": self.get_mapping_stats(),
            "mapping_queue_size": self.mapping_queue.qsize() if self.mapping_queue is not None else None,
            "coordinate_frames": self.get_current_coordinate_frames(),
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "last_error": repr(self._last_error) if self._last_error is not None else None,
        }

    # ============================================================
    # 保存 / 重置辅助
    # ============================================================

    def save_mesh(self, prefix: str = "fusion_model_final"):
        """
        保存当前 TSDF 提取出的最终 mesh。

        建议：
        - 最稳妥是在 stop 后保存。
        - 如果运行中保存，建议给 Mapper 增加锁，见后文补丁。
        """
        if self.mapper is None:
            return False, None, None
        return self.mapper.save_final_mesh(prefix=prefix)
    
    def reset_reconstruction(
        self,
        reset_scanner_temporal_state: bool = True,
        restart_workers: bool = True,
        join_timeout: float = 1.0,
        rebuild_renderer: bool = True,
    ) -> dict:
        """
        清空当前进程内的建模数据，并从空白模型重新开始建模。

        重要：
        - 不删除磁盘上的模型文件
        - 不关闭整个进程
        - 不停止日志文件
        - scanner 模式下默认不重新打开相机
        - 会清空当前内存中的 TSDF / mesh / local_map / tracker pose / 队列 / shared_state

        适用场景：
        - 外部脚本调用 pipeline 后，希望丢弃当前扫描结果，重新开始扫描
        - GUI 点击“重新建模”
        - Web API 调用“reset reconstruction”

        参数：
        - reset_scanner_temporal_state:
            如果 scanner 有 reset_temporal_state()，是否重置它。
            对 UnifiedDepthScanner 来说，会清空 EMA 深度缓存。

        - restart_workers:
            如果 reset 前 pipeline 正在运行，reset 后是否自动重启 tracking/mapping worker。

        - join_timeout:
            等待旧 worker 退出的超时时间。

        - rebuild_renderer:
            是否重建 Renderer。
            True 更稳，可以避免旧 mesh 残留在 Open3D 窗口里。

        返回：
        dict，包含 reset 是否成功、reset 前后的运行状态等。
        """
        result = {
            "success": False,
            "was_running": self._running,
            "was_background": self._background_thread_started,
            "input_mode": self._input_mode,
            "workers_restarted": False,
            "background_restarted": False,
            "scanner_temporal_reset": False,
            "renderer_rebuilt": False,
            "last_error": None,
        }

        if not self._setup_done:
            result["last_error"] = "Pipeline is not setup yet."
            self.app_logger.warning(
                "reset_reconstruction ignored: pipeline is not setup",
                force=True,
            )
            return result

        self.app_logger.status("Reset reconstruction requested...", force=True)

        was_running = bool(self._running)
        was_background = bool(self._background_thread_started)
        input_mode = self._input_mode

        # ------------------------------------------------------------
        # 1) 停止旧 worker / 后台输入循环
        # ------------------------------------------------------------
        try:
            # 让旧 tracking/mapping/background loop 退出
            if self.stop_event is not None:
                self.stop_event.set()

            current_thread = threading.current_thread()

            if self.tracking_worker is not None and self.tracking_worker.is_alive():
                if current_thread is not self.tracking_worker:
                    self.tracking_worker.join(timeout=join_timeout)

            if self.mapping_worker is not None and self.mapping_worker.is_alive():
                if current_thread is not self.mapping_worker:
                    self.mapping_worker.join(timeout=join_timeout)

            if self._background_thread is not None and self._background_thread.is_alive():
                if current_thread is not self._background_thread:
                    self._background_thread.join(timeout=join_timeout)

        except Exception as e:
            self._last_error = e
            result["last_error"] = repr(e)
            self.app_logger.warning(
                f"failed while stopping old workers during reset: {repr(e)}",
                force=True,
            )
            return result

        # 标记当前旧运行态结束
        self._running = False
        self._background_thread_started = False

        # ------------------------------------------------------------
        # 2) 可选：重置 scanner 内部时序状态
        # ------------------------------------------------------------
        if (
            reset_scanner_temporal_state
            and self.scanner is not None
            and hasattr(self.scanner, "reset_temporal_state")
        ):
            try:
                self.scanner.reset_temporal_state()
                result["scanner_temporal_reset"] = True
                self.app_logger.status("scanner temporal state reset", force=True)
            except Exception as e:
                # scanner reset 失败不一定要终止整个 reconstruction reset
                self._last_error = e
                result["last_error"] = repr(e)
                self.app_logger.warning(f"scanner reset_temporal_state failed: {repr(e)}", force=True)

        # ------------------------------------------------------------
        # 3) 重建 / 清理 renderer，避免旧 mesh 残留
        # ------------------------------------------------------------
        if rebuild_renderer and self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as e:
                self.app_logger.warning(f"renderer close during reset failed: {repr(e)}", force=True)

            self.renderer = None
            self.render_worker = None
            result["renderer_rebuilt"] = True

        if self.imu_world_initializer is not None:
            try:
                self.imu_world_initializer.close()
            except Exception as e:
                self.app_logger.warning(f"imu close during reset failed: {repr(e)}", force=True)

        # ------------------------------------------------------------
        # 4) 丢弃旧建模会话对象引用
        # ------------------------------------------------------------
        self.shared_state = None
        self.latest_frame_slot = None
        self.mapping_queue = None

        self.local_map = None

        self.imu_world_initializer = None
        self.tracker = None
        self.mapper = None

        self.tracking_worker = None
        self.mapping_worker = None
        self.render_worker = None

        self._background_thread = None

        # ------------------------------------------------------------
        # 5) 创建一套新的建模会话对象
        # ------------------------------------------------------------
        try:
            self._create_runtime_objects(recreate_renderer=rebuild_renderer)
        except Exception as e:
            self._last_error = e
            result["last_error"] = repr(e)
            self.app_logger.warning(
                f"failed to recreate runtime objects: {repr(e)}",
                force=True,
            )
            return result

        # ------------------------------------------------------------
        # 6) 重置调试计数 / external frame counter
        # ------------------------------------------------------------
        self._prev_ts = None
        self._ts_debug_count = 0
        self._frame_counter_external = 0
        self._last_error = None

        # 保持原来的输入模式
        self._input_mode = input_mode

        # ------------------------------------------------------------
        # 7) 按需重启 worker
        # ------------------------------------------------------------
        if restart_workers and was_running:
            try:
                self.tracking_worker.start()
                self.mapping_worker.start()

                self._running = True
                result["workers_restarted"] = True

                # 如果 reset 前是后台模式，则恢复后台采集循环
                # 注意：render enabled 时本来就不允许 start_background()
                if was_background and input_mode == "scanner" and (not self.cfg.render.enabled):
                    self._background_thread = threading.Thread(
                        target=self._background_loop,
                        name="PipelineInputLoop",
                        daemon=True,
                    )
                    self._background_thread.start()
                    self._background_thread_started = True
                    result["background_restarted"] = True
                else:
                    self._background_thread = None
                    self._background_thread_started = False

            except Exception as e:
                self._last_error = e
                result["last_error"] = repr(e)
                self.app_logger.warning(f"failed to restart workers after reset: {repr(e)}", force=True)
                return result
        else:
            self._running = False
            self._background_thread = None
            self._background_thread_started = False

        result["success"] = True
        result["running"] = self.is_running()
        result["mapping_stats"] = self.get_mapping_stats()
        result["tracking_stats"] = self.get_tracking_stats()

        # Resume integrate/track paths after a full session rebuild unless callers opted out.
        self.set_center_recovery_gate(True, True)

        self.app_logger.status("Reconstruction reset complete.", force=True)
        return result

    def reset_reconstruction_and_save(
        self,
        prefix: str = "before_reset",
        reset_scanner_temporal_state: bool = True,
        restart_workers: bool = True,
        join_timeout: float = 1.0,
        rebuild_renderer: bool = True,
    ) -> dict:
        """
        先保存当前模型，再清空当前进程内建模数据并重新开始。

        注意：
        - 如果当前模型为空，保存可能失败，但仍会继续 reset。
        """
        save_ok, hist_path, latest_path = self.save_mesh(prefix=prefix)

        reset_result = self.reset_reconstruction(
            reset_scanner_temporal_state=reset_scanner_temporal_state,
            restart_workers=restart_workers,
            join_timeout=join_timeout,
            rebuild_renderer=rebuild_renderer,
        )

        return {
            "save": {
                "success": bool(save_ok),
                "history_path": hist_path,
                "latest_path": latest_path,
            },
            "reset": reset_result,
        }

    def clear_local_map(self):
        if self.local_map is not None:
            self.local_map.clear()

    def reset_tracker(self):
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            self.tracker.reset()

    def _debug_timestamp(self, frame: RGBDFrame):
        if self._prev_ts is not None and self._ts_debug_count < 20:
            dt = frame.device_timestamp - self._prev_ts
            self.app_logger.debug(
                f"timestamp: frame={frame.frame_id}, dt={dt:.6f}s",
                frame_id=frame.frame_id,
                force=True,
            )
            self._ts_debug_count += 1

        self._prev_ts = frame.device_timestamp