import time
import traceback
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))

from config import PipelineConfig
from pipeline_api import RealtimeMappingPipeline

def print_status(title, pipeline: RealtimeMappingPipeline):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    status = pipeline.get_status()

    print("running:", status.get("running"))
    print("input_mode:", status.get("input_mode"))
    print("render_enabled:", status.get("render_enabled"))

    print("\ntracking_stats:")
    print(status.get("tracking_stats"))

    print("\nmapping_stats:")
    print(status.get("mapping_stats"))

    tracking = status.get("tracking")
    if tracking is not None:
        print("\nlatest_tracking:")
        print({
            "frame_id": tracking.get("frame_id"),
            "success": tracking.get("success"),
            "mode": tracking.get("mode"),
            "score": tracking.get("score"),
        })
    else:
        print("\nlatest_tracking: None")

    mapping = status.get("mapping")
    if mapping is not None:
        map_data = mapping.get("map_data")
        print("\nlatest_map:")
        print({
            "frame_id": mapping.get("frame_id"),
            "map_data": map_data,
        })
    else:
        print("\nlatest_map: None")

    print("last_error:", status.get("last_error"))

def main():
    cfg = PipelineConfig()

    # reset 测试建议先关闭渲染，避免 Open3D 窗口干扰自动测试
    cfg.render.enabled = False

    # 测试时可以关闭日志重定向，终端输出更直观
    cfg.logging.save_log = False

    # 可选：减少日志量
    cfg.logging.enable_console_log = True

    pipeline = RealtimeMappingPipeline(cfg)

    try:
        print("[TEST] starting pipeline in background mode...")
        ok = pipeline.start_background()
        if not ok:
            print("[TEST] pipeline start failed")
            return

        print("[TEST] pipeline started")

        # 记录 reset 前对象 id，用于确认 reset 后确实换了新的 Mapper/Tracker/LocalMap
        old_mapper_id = id(pipeline.mapper)
        old_tracker_id = id(pipeline.tracker)
        old_local_map_id = id(pipeline.local_map)
        old_shared_state_id = id(pipeline.shared_state)

        print("[TEST] old object ids:")
        print("mapper:", old_mapper_id)
        print("tracker:", old_tracker_id)
        print("local_map:", old_local_map_id)
        print("shared_state:", old_shared_state_id)

        # 跑一段时间，让 tracking/mapping 产生一些状态
        print("[TEST] running for 8 seconds before reset...")
        time.sleep(8.0)

        print_status("[BEFORE RESET]", pipeline)

        before_mapping_stats = pipeline.get_mapping_stats()
        before_tracking_stats = pipeline.get_tracking_stats()

        print("\n[TEST] before reset mapping_stats:", before_mapping_stats)
        print("[TEST] before reset tracking_stats:", before_tracking_stats)

        print("\n[TEST] calling reset_reconstruction()...")
        reset_result = pipeline.reset_reconstruction(
            reset_scanner_temporal_state=True,
            restart_workers=True,
            join_timeout=2.0,
            rebuild_renderer=True,
        )

        print("\n[TEST] reset_result:")
        print(reset_result)

        new_mapper_id = id(pipeline.mapper)
        new_tracker_id = id(pipeline.tracker)
        new_local_map_id = id(pipeline.local_map)
        new_shared_state_id = id(pipeline.shared_state)

        print("\n[TEST] new object ids:")
        print("mapper:", new_mapper_id)
        print("tracker:", new_tracker_id)
        print("local_map:", new_local_map_id)
        print("shared_state:", new_shared_state_id)

        print("\n[TEST] object id changed:")
        print("mapper changed:", old_mapper_id != new_mapper_id)
        print("tracker changed:", old_tracker_id != new_tracker_id)
        print("local_map changed:", old_local_map_id != new_local_map_id)
        print("shared_state changed:", old_shared_state_id != new_shared_state_id)

        print_status("[IMMEDIATELY AFTER RESET]", pipeline)

        # reset 后继续跑，确认 worker 和后台采集循环恢复
        print("\n[TEST] running for 8 seconds after reset...")
        time.sleep(8.0)

        print_status("[AFTER RESET RUNNING]", pipeline)

        after_mapping_stats = pipeline.get_mapping_stats()
        after_tracking_stats = pipeline.get_tracking_stats()

        print("\n[TEST] after reset mapping_stats:", after_mapping_stats)
        print("[TEST] after reset tracking_stats:", after_tracking_stats)

        print("\n[TEST] checking reset expectations...")

        if not reset_result.get("success"):
            print("[CHECK] FAIL: reset_result.success is False")
        else:
            print("[CHECK] OK: reset_result.success is True")

        if not pipeline.is_running():
            print("[CHECK] FAIL: pipeline is not running after reset")
        else:
            print("[CHECK] OK: pipeline is running after reset")

        if old_mapper_id == new_mapper_id:
            print("[CHECK] FAIL: mapper object was not replaced")
        else:
            print("[CHECK] OK: mapper object replaced")

        if old_tracker_id == new_tracker_id:
            print("[CHECK] FAIL: tracker object was not replaced")
        else:
            print("[CHECK] OK: tracker object replaced")

        if old_local_map_id == new_local_map_id:
            print("[CHECK] FAIL: local_map object was not replaced")
        else:
            print("[CHECK] OK: local_map object replaced")

        # 注意：
        # reset 后刚开始几秒不一定已经 integrated，
        # 因为 tracking 质量、mapping_stride、运动门限都会影响是否进入 mapping。
        # 所以这里不强制要求 integrated_frames > 0。
        print("\n[TEST] reset test finished")

    except KeyboardInterrupt:
        print("[TEST] interrupted by user")

    except Exception:
        print("[TEST] exception:")
        traceback.print_exc()

    finally:
        print("\n[TEST] stopping pipeline...")
        result = pipeline.stop(save_mesh=False)
        print("[TEST] stop result:")
        print(result)

if __name__ == "__main__":
    main()