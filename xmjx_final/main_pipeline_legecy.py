import time
import threading
from pathlib import Path

from config import PipelineConfig
from common.shared_state import SharedState
from common.queue_utils import LatestFrameSlot, BoundedDropQueue

from input.input_adapter import InputAdapter

from tracking.tracker import Tracker
from tracking.gpu_icp_tracker import GpuICPTracker
from tracking.tracking_worker import TrackingWorker

from mapping.mapper import Mapper
from mapping.mapping_worker import MappingWorker
from mapping.local_map import LocalMap

from rendering.renderer import Renderer
from rendering.render_worker import RenderWorker

from utils.log_redirect import LogRedirectManager
from utils.module_logger import ModuleLogger

from UnifiedDepthScanner import UnifiedDepthScanner

def resolve_output_dir(path_str: str, base_dir: Path) -> str:
    """
    将配置中的目录统一解析为绝对路径：
    - 如果本身是绝对路径，直接使用
    - 如果是相对路径，则相对于当前 main_pipeline.py 所在目录
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def build_tracker(cfg: PipelineConfig, local_map=None, logger=None, app_logger=None):
    cam = cfg.tracking_camera
    tcfg = cfg.tracking
    icfg = cfg.input
    lcfg = cfg.logging

    backend = str(tcfg.tracking_backend).lower().strip()

    if app_logger is not None:
        app_logger.status(f"tracking_backend={backend}", force=True)
    else:
        print(f"[Main] tracking_backend={backend}")

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

            # 如果你已经在 config.py / gpu_icp_tracker.py 中加了 gpu_tracking_pcd_stride，
            # 这里会正常传入。
            # 如果你的 GpuICPTracker 还没加 tracking_pcd_stride 参数，
            # 需要先把 gpu_icp_tracker.py 补上对应参数。
            tracking_pcd_stride=getattr(tcfg, "gpu_tracking_pcd_stride", 2),

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

            logger=logger,
        )

    elif backend == "cpu_rgbd":
        # 兼容旧 CPU tracker。
        # 旧 Tracker 可能还在使用 debug_print_odom / enable_profile / profile_print_interval。
        # 这里从统一 LoggingConfig 映射过去，避免 config.py 再保留重复开关。
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

        local_map=local_map,
        logger=logger,
    )

def build_renderer(cfg: PipelineConfig):
    rcfg = cfg.render
    lcfg = cfg.logging

    return Renderer(
        window_name=rcfg.window_name,
        width=rcfg.width,
        height=rcfg.height,

        # Renderer 旧接口如果只有 enable_log，就用统一 status 开关映射过去。
        enable_log=lcfg.status.enabled,
    )

def main():
    cfg = PipelineConfig()
    base_dir = Path(__file__).resolve().parent

    # 统一修正输出目录，避免相对路径落到“执行目录”
    cfg.mapping.model_dir = resolve_output_dir(cfg.mapping.model_dir, base_dir)
    cfg.logging.log_dir = resolve_output_dir(cfg.logging.log_dir, base_dir)

    # ------------------------------------------------------------
    # 日志重定向
    # ------------------------------------------------------------
    # LogRedirectManager 只负责 stdout/stderr 文件双写。
    # ModuleLogger 负责日志分类开关。
    log_manager = LogRedirectManager()
    log_path = log_manager.start(
        log_dir=cfg.logging.log_dir,
        log_prefix=cfg.logging.log_prefix,
        enabled=cfg.logging.save_log,
    )

    # ------------------------------------------------------------
    # 模块 logger
    # ------------------------------------------------------------
    app_logger = ModuleLogger("APP", cfg.logging)
    input_logger = ModuleLogger("INPUT", cfg.logging)
    tracker_logger = ModuleLogger("TRACKER", cfg.logging)
    mapper_logger = ModuleLogger("MAPPER", cfg.logging)
    local_map_logger = ModuleLogger("LOCAL_MAP", cfg.logging)
    render_logger = ModuleLogger("RENDER", cfg.logging)

    app_logger.status(f"save_log={cfg.logging.save_log}", force=True)
    app_logger.status(f"log_dir={cfg.logging.log_dir}", force=True)
    app_logger.status(f"model_dir={cfg.mapping.model_dir}", force=True)
    app_logger.status(f"input_camera={cfg.input_camera}", force=True)
    app_logger.status(f"tracking_camera={cfg.tracking_camera}", force=True)
    app_logger.status(f"mapping_camera={cfg.mapping_camera}", force=True)
    app_logger.status(f"tracking_backend={cfg.tracking.tracking_backend}", force=True)

    if log_path is not None:
        app_logger.status(f"log_file={log_path}", force=True)

    icam = cfg.input_camera

    scanner = None
    input_adapter = None
    tracking_worker = None
    mapping_worker = None
    render_worker = None
    mapper = None

    try:
        scanner = UnifiedDepthScanner(
            width=icam.width,
            height=icam.height,
            fx=icam.fx,
            fy=icam.fy,
            cx=icam.cx,
            cy=icam.cy,
        )

        if not scanner.init():
            app_logger.warning("scanner init failed", force=True)
            return

        shared_state = SharedState()
        latest_frame_slot = LatestFrameSlot()
        mapping_queue = BoundedDropQueue(maxsize=cfg.mapping.queue_size)
        stop_event = threading.Event()

        # ------------------------------------------------------------
        # LocalMap
        # ------------------------------------------------------------
        local_map = LocalMap(
            voxel_size=cfg.tracking.local_map_voxel_size,
            max_points=cfg.tracking.local_map_max_points,
            local_radius=cfg.tracking.local_map_radius,
            min_points_for_tracking=cfg.tracking.local_map_min_points_for_tracking,
            logger=local_map_logger,
        )

        # ------------------------------------------------------------
        # Input
        # ------------------------------------------------------------
        input_adapter = InputAdapter(
            scanner=scanner,
            timestamp_unit=cfg.input.device_timestamp_unit,
            validate=cfg.input.validate_input_frame,
            print_timestamp_debug_once=cfg.input.print_timestamp_debug_once,
        )

        # ------------------------------------------------------------
        # Core modules
        # ------------------------------------------------------------
        tracker = build_tracker(
            cfg,
            local_map=local_map,
            logger=tracker_logger,
            app_logger=app_logger,
        )

        mapper = build_mapper(
            cfg,
            local_map=local_map,
            logger=mapper_logger,
        )

        renderer = None
        if cfg.render.enabled:
            renderer = build_renderer(cfg)

        # ------------------------------------------------------------
        # Workers
        # ------------------------------------------------------------
        tracking_worker = TrackingWorker(
            latest_frame_slot=latest_frame_slot,
            tracker=tracker,
            shared_state=shared_state,
            mapping_queue=mapping_queue,
            stop_event=stop_event,
            sleep_ms=cfg.tracking.sleep_ms,

            # worker 旧接口如果只有 enable_log，就用统一 status 开关映射。
            enable_log=cfg.logging.status.enabled,

            mapping_stride=cfg.tracking.output_mapping_stride,
        )

        mapping_worker = MappingWorker(
            mapping_queue=mapping_queue,
            mapper=mapper,
            shared_state=shared_state,
            stop_event=stop_event,

            # worker 旧接口如果只有 enable_log，就用统一 status 开关映射。
            enable_log=cfg.logging.status.enabled,
        )

        if cfg.render.enabled and renderer is not None:
            render_worker = RenderWorker(
                shared_state=shared_state,
                renderer=renderer,
                stop_event=stop_event,
                render_fps=cfg.render.fps,

                # RenderConfig 里的 enable_status_log / status_log_interval 已合并到 LoggingConfig。
                enable_status_log=cfg.logging.status.enabled,
                status_log_interval=cfg.logging.status.interval,
            )

        tracking_worker.start()
        mapping_worker.start()

        if render_worker is not None:
            app_logger.status("Render is enabled: rendering on MAIN thread", force=True)

        app_logger.status("Pipeline started. Press Ctrl+C to stop.", force=True)

        prev_ts = None
        ts_debug_count = 0

        while not stop_event.is_set():
            # 关键：主线程驱动 Open3D 事件循环
            if render_worker is not None:
                ok = render_worker.run_once()
                if not ok:
                    app_logger.status("render_worker.run_once returned False", force=True)
                    break

            frame = input_adapter.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # 前几帧打印一下时间差，确认 timestamp 单位是否正常
            if prev_ts is not None and ts_debug_count < 20:
                dt = frame.device_timestamp - prev_ts
                app_logger.debug(
                    f"timestamp: frame={frame.frame_id}, dt={dt:.6f}s",
                    frame_id=frame.frame_id,
                    force=True,
                )
                ts_debug_count += 1

            prev_ts = frame.device_timestamp
            latest_frame_slot.put(frame)

    except KeyboardInterrupt:
        app_logger.status("Stopping pipeline by KeyboardInterrupt...", force=True)

    except Exception as e:
        app_logger.warning(f"Unhandled exception in main: {repr(e)}", force=True)
        raise

    finally:
        # ------------------------------------------------------------
        # Stop workers
        # ------------------------------------------------------------
        try:
            if "stop_event" in locals() and stop_event is not None:
                stop_event.set()
        except Exception:
            pass

        if tracking_worker is not None:
            tracking_worker.join(timeout=1.0)

        if mapping_worker is not None:
            mapping_worker.join(timeout=1.0)

        if render_worker is not None:
            try:
                if render_worker.is_alive():
                    render_worker.join(timeout=1.0)
                else:
                    render_worker.renderer.close()
            except Exception as e:
                app_logger.warning(f"render_worker close failed: {repr(e)}", force=True)

        if scanner is not None:
            try:
                scanner.close()
            except Exception as e:
                app_logger.warning(f"scanner close failed: {repr(e)}", force=True)

        # ------------------------------------------------------------
        # Print stats
        # ------------------------------------------------------------
        if input_adapter is not None:
            try:
                app_logger.status(
                    f"Input timestamp info: {input_adapter.debug_timestamp_info()}",
                    force=True,
                )
            except Exception as e:
                app_logger.warning(f"input timestamp info failed: {repr(e)}", force=True)

        if tracking_worker is not None:
            try:
                app_logger.status(
                    f"Tracking stats: {tracking_worker.get_stats()}",
                    force=True,
                )
            except Exception as e:
                app_logger.warning(f"tracking stats failed: {repr(e)}", force=True)

        if mapping_worker is not None:
            try:
                app_logger.status(
                    f"Mapping stats: {mapping_worker.get_stats()}",
                    force=True,
                )
            except Exception as e:
                app_logger.warning(f"mapping stats failed: {repr(e)}", force=True)

        # ------------------------------------------------------------
        # Save final mesh
        # ------------------------------------------------------------
        if mapper is not None:
            try:
                mapper.save_final_mesh(prefix="fusion_model_final")
            except Exception as e:
                app_logger.warning(f"save final mesh failed: {repr(e)}", force=True)

        if log_path is not None:
            app_logger.status(f"Log file: {log_path}", force=True)

        log_manager.stop()

if __name__ == "__main__":
    main()