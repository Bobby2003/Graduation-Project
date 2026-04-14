import threading
from pathlib import Path

from config import PipelineConfig
from common.shared_state import SharedState
from common.queue_utils import LatestFrameSlot, BoundedDropQueue

from input.input_adapter import InputAdapter

from tracking.tracker import Tracker
from tracking.tracking_worker import TrackingWorker

from mapping.mapper import Mapper
from mapping.mapping_worker import MappingWorker

from rendering.renderer import Renderer
from rendering.render_worker import RenderWorker

from imu.imu_manager import IMUManager

from utils.log_redirect import LogRedirectManager

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

def build_tracker(cfg: PipelineConfig, imu_manager):
    return Tracker(
        imu_manager=imu_manager,

        width=cfg.width,
        height=cfg.height,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy,

        input_color_is_bgr=cfg.input_color_is_bgr,
        depth_scale=cfg.depth_scale,
        depth_trunc=cfg.depth_trunc,
        min_valid_pixels=cfg.min_valid_pixels,

        max_rotation_deg_per_frame=cfg.max_rotation_deg_per_frame,
        rotation_dominant_angle_deg=cfg.rotation_dominant_angle_deg,

        max_translation_when_rotating=cfg.max_translation_when_rotating,
        min_info_trace=cfg.min_info_trace,
        min_translation_per_frame=cfg.min_translation_per_frame,
        max_translation_per_frame=cfg.max_translation_per_frame,
        trans_smooth_alpha=cfg.trans_smooth_alpha,

        min_imu_rot_deg_per_frame=cfg.min_imu_rot_deg_per_frame,
        imu_as_vo_init_only=cfg.imu_as_vo_init_only,
        allow_imu_fallback_when_vo_fails=cfg.allow_imu_fallback_when_vo_fails,

        debug_print_odom=cfg.debug_print_odom,
    )

def build_mapper(cfg: PipelineConfig):
    return Mapper(
        voxel_length=cfg.voxel_length,
        sdf_trunc=cfg.sdf_trunc,

        width=cfg.width,
        height=cfg.height,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy,

        input_color_is_bgr=cfg.input_color_is_bgr,
        depth_scale=cfg.depth_scale,
        depth_trunc=cfg.depth_trunc,

        integrate_only_when_motion=cfg.integrate_only_when_motion,
        min_integrate_rot_deg=cfg.min_integrate_rot_deg,
        max_integrate_rot_deg=cfg.max_integrate_rot_deg,
        min_integrate_trans_m=cfg.min_integrate_trans_m,

        mesh_update_interval=cfg.mesh_update_interval,
        mesh_min_vertices_to_show=cfg.mesh_min_vertices_to_show,

        model_dir=cfg.model_dir,
    )

def build_renderer(cfg: PipelineConfig):
    return Renderer(
        window_name=cfg.render_window_name,
        width=cfg.render_width,
        height=cfg.render_height,
        enable_log=cfg.enable_console_log,
    )

def main():
    cfg = PipelineConfig()
    base_dir = Path(__file__).resolve().parent

    # 统一修正输出目录，避免相对路径落到“执行目录”
    if hasattr(cfg, "model_dir"):
        cfg.model_dir = resolve_output_dir(cfg.model_dir, base_dir)

    if hasattr(cfg, "log_dir"):
        cfg.log_dir = resolve_output_dir(cfg.log_dir, base_dir)

    # 启动日志重定向：终端 + 文件双写
    log_manager = LogRedirectManager()
    log_path = None
    if getattr(cfg, "save_log", False):
        log_path = log_manager.start(
            log_dir=cfg.log_dir,
            log_prefix=getattr(cfg, "log_prefix", "log"),
            enabled=cfg.save_log,
        )

    print(f"[Config] save_log={getattr(cfg, 'save_log', False)}")
    if hasattr(cfg, "log_dir"):
        print(f"[Config] log_dir={cfg.log_dir}")
    if hasattr(cfg, "model_dir"):
        print(f"[Config] model_dir={cfg.model_dir}")

    # 建议首次联调时先这样：
    # cfg.use_imu = False
    # cfg.enable_render = False

    scanner = UnifiedDepthScanner(
        width=cfg.width,
        height=cfg.height,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy,
    )

    if not scanner.init():
        print("[Main] scanner init failed")
        if getattr(cfg, "save_log", False):
            print(f"[Main] Log file: {log_path}")
            log_manager.stop()
        return

    shared_state = SharedState()
    latest_frame_slot = LatestFrameSlot()
    mapping_queue = BoundedDropQueue(maxsize=cfg.mapping_queue_size)
    stop_event = threading.Event()

    imu_manager = IMUManager() if cfg.use_imu else None

    input_adapter = InputAdapter(
        scanner=scanner,
        timestamp_unit=cfg.device_timestamp_unit,
        validate=cfg.validate_input_frame,
        print_timestamp_debug_once=True,
    )

    tracker = build_tracker(cfg, imu_manager)
    mapper = build_mapper(cfg)
    renderer = build_renderer(cfg)

    tracking_worker = TrackingWorker(
        latest_frame_slot=latest_frame_slot,
        tracker=tracker,
        shared_state=shared_state,
        mapping_queue=mapping_queue,
        stop_event=stop_event,
        sleep_ms=cfg.tracking_sleep_ms,
        enable_log=cfg.enable_console_log,
        mapping_stride=cfg.mapping_stride,
    )

    mapping_worker = MappingWorker(
        mapping_queue=mapping_queue,
        mapper=mapper,
        shared_state=shared_state,
        stop_event=stop_event,
        enable_log=cfg.enable_console_log,
    )

    render_worker = None
    if cfg.enable_render:
        render_worker = RenderWorker(
            shared_state=shared_state,
            renderer=renderer,
            stop_event=stop_event,
            render_fps=cfg.render_fps,
            enable_status_log=cfg.render_enable_status_log,
            status_log_interval=cfg.render_status_log_interval,
        )

    tracking_worker.start()
    mapping_worker.start()
    if render_worker is not None:
        render_worker.start()

    print("[Main] Pipeline started. Press Ctrl+C to stop.")

    prev_ts = None
    ts_debug_count = 0

    try:
        while not stop_event.is_set():
            frame = input_adapter.get_frame()
            if frame is None:
                continue

            # 前几帧打印一下时间差，确认 timestamp 单位是否正常
            if prev_ts is not None and ts_debug_count < 20:
                dt = frame.device_timestamp - prev_ts
                print(f"[Main][TS] frame={frame.frame_id}, dt={dt:.6f}s")
                ts_debug_count += 1
            prev_ts = frame.device_timestamp

            latest_frame_slot.put(frame)

    except KeyboardInterrupt:
        print("[Main] Stopping pipeline...")

    finally:
        stop_event.set()

        if tracking_worker is not None:
            tracking_worker.join(timeout=1.0)

        if mapping_worker is not None:
            mapping_worker.join(timeout=1.0)

        if render_worker is not None:
            render_worker.join(timeout=1.0)

        if scanner is not None:
            scanner.close()

        print("[Main] Input timestamp info:", input_adapter.debug_timestamp_info())
        print("[Main] Tracking stats:", tracking_worker.get_stats())
        print("[Main] Mapping stats:", mapping_worker.get_stats())

        mapper.save_final_mesh(prefix="imu_fusion_model_final")

        if getattr(cfg, "save_log", False):
            print(f"[Main] Log file: {log_path}")
            log_manager.stop()

if __name__ == "__main__":
    main()