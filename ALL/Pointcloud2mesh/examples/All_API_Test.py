import os
import sys
import time
import tempfile
from pathlib import Path

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))

import numpy as np

from pipeline_api import RealtimeMappingPipeline
from config import PipelineConfig
from common.types import RGBDFrame

# ============================================================
# 静默所有第三方输出，只保留本脚本的 [OK]/[FAIL]
# ============================================================

class OutputSilencer:
    def __init__(self):
        self.devnull_fd = None
        self.old_stdout_fd = None
        self.old_stderr_fd = None

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()

        self.old_stdout_fd = os.dup(1)
        self.old_stderr_fd = os.dup(2)

        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def restore(self):
        try:
            if self.old_stdout_fd is not None:
                os.dup2(self.old_stdout_fd, 1)
                os.close(self.old_stdout_fd)
                self.old_stdout_fd = None

            if self.old_stderr_fd is not None:
                os.dup2(self.old_stderr_fd, 2)
                os.close(self.old_stderr_fd)
                self.old_stderr_fd = None

            if self.devnull_fd is not None:
                os.close(self.devnull_fd)
                self.devnull_fd = None
        except Exception:
            pass

    def print_real(self, text: str):
        if self.old_stdout_fd is not None:
            os.write(
                self.old_stdout_fd,
                (text + "\n").encode("utf-8", errors="replace"),
            )

def call_api(silencer: OutputSilencer, name: str, func):
    try:
        func()
        silencer.print_real(f"[OK] {name}")
        return True
    except Exception as e:
        silencer.print_real(f"[FAIL] {name}: {repr(e)}")
        return False

# ============================================================
# 配置
# ============================================================

def make_test_cfg(base_dir: Path):
    cfg = PipelineConfig()

    # 关闭渲染，避免 Open3D 窗口 / 主线程问题
    if hasattr(cfg, "render"):
        cfg.render.enabled = False

    # 关闭或减少日志
    if hasattr(cfg, "logging"):
        if hasattr(cfg.logging, "save_log"):
            cfg.logging.save_log = False

        if hasattr(cfg.logging, "status") and hasattr(cfg.logging.status, "enabled"):
            cfg.logging.status.enabled = False

        if hasattr(cfg.logging, "debug") and hasattr(cfg.logging.debug, "enabled"):
            cfg.logging.debug.enabled = False

        if hasattr(cfg.logging, "profile") and hasattr(cfg.logging.profile, "enabled"):
            cfg.logging.profile.enabled = False

        if hasattr(cfg.logging, "log_dir"):
            cfg.logging.log_dir = str(base_dir / "logs")

    # 输出目录放到纯 ASCII 临时目录，避免 Windows 中文路径问题
    if hasattr(cfg, "mapping") and hasattr(cfg.mapping, "model_dir"):
        cfg.mapping.model_dir = str(base_dir / "models")

    return cfg

def get_frame_size(cfg):
    cam = getattr(cfg, "input_camera", None)

    if cam is None:
        cam = getattr(cfg, "tracking_camera", None)

    if cam is None:
        cam = getattr(cfg, "mapping_camera", None)

    width = int(getattr(cam, "width", 640))
    height = int(getattr(cam, "height", 480))

    return width, height

# ============================================================
# 构造假 RGBD 数据
# ============================================================

def make_dummy_rgbd(cfg, frame_id=0):
    width, height = get_frame_size(cfg)

    yy, xx = np.mgrid[0:height, 0:width]

    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[..., 0] = ((xx * 255) / max(width - 1, 1)).astype(np.uint8)
    color[..., 1] = ((yy * 255) / max(height - 1, 1)).astype(np.uint8)
    color[..., 2] = ((xx + yy + frame_id * 17) % 255).astype(np.uint8)

    depth_f = (
        900.0
        + 60.0 * np.sin(xx / 30.0 + frame_id * 0.1)
        + 50.0 * np.cos(yy / 40.0 + frame_id * 0.07)
        + 20.0 * np.sin((xx + yy) / 45.0)
    )

    depth_f = np.clip(depth_f, 500.0, 1500.0)
    depth = depth_f.astype(np.uint16)

    mask = np.ones((height, width), dtype=np.uint8)

    return color, depth, mask

def make_dummy_frame(cfg, frame_id=1):
    color, depth, mask = make_dummy_rgbd(cfg, frame_id=frame_id)
    now = time.perf_counter()

    return RGBDFrame(
        frame_id=frame_id,
        device_timestamp=now,
        host_timestamp=now,
        color=color,
        depth=depth,
        mask=mask,
        extras={
            "source": "api_callable_test",
        },
    )

def feed_some_frames(pipeline, cfg, start_id=1, count=5, sleep_s=0.05):
    for i in range(count):
        frame_id = start_id + i
        color, depth, mask = make_dummy_rgbd(cfg, frame_id=frame_id)

        pipeline.submit_rgbd(
            color=color,
            depth=depth,
            mask=mask,
            frame_id=frame_id,
            timestamp=time.perf_counter(),
            extras={
                "source": "warmup",
            },
        )

        time.sleep(sleep_s)

# ============================================================
# 关键：mock 保存 mesh
# ============================================================

def patch_save_final_mesh(pipeline):
    """
    严格测试 API 是否可调用时，不测试 Open3D 写 mesh 文件。

    这样可以避免：
    - 当前没有有效 mesh
    - Open3D write_triangle_mesh 失败
    - Windows 路径编码问题
    - 文件权限问题

    注意：
    reset_reconstruction() 会重建 mapper，所以 reset 后需要重新 patch。
    """
    if pipeline is not None and getattr(pipeline, "mapper", None) is not None:
        pipeline.mapper.save_final_mesh = lambda prefix="fusion_model_final": (
            False,
            None,
            None,
        )

# ============================================================
# 主测试
# ============================================================

def main():
    with tempfile.TemporaryDirectory(prefix="pipeline_api_callable_") as tmpdir:
        base_dir = Path(tmpdir).resolve()

        with OutputSilencer() as silencer:
            cfg = make_test_cfg(base_dir)
            pipeline = None

            # 1. 创建 pipeline
            def api_create_pipeline():
                nonlocal pipeline
                pipeline = RealtimeMappingPipeline(
                    cfg=cfg,
                    base_dir=base_dir,
                )

            call_api(
                silencer,
                "RealtimeMappingPipeline(cfg)",
                api_create_pipeline,
            )

            if pipeline is None:
                return

            # 2. 启动外部喂帧模式
            call_api(
                silencer,
                "pipeline.start_external()",
                lambda: pipeline.start_external(),
            )

            # 启动后 patch 保存逻辑
            patch_save_final_mesh(pipeline)

            # 3. 提交修补后的 RGBD
            color, depth, mask = make_dummy_rgbd(cfg, frame_id=1)

            call_api(
                silencer,
                "pipeline.submit_rgbd(color, depth, mask, frame_id, timestamp, extras)",
                lambda: pipeline.submit_rgbd(
                    color=color,
                    depth=depth,
                    mask=mask,
                    frame_id=1,
                    timestamp=time.perf_counter(),
                    extras={
                        "source": "submit_rgbd_test",
                    },
                ),
            )

            time.sleep(0.1)

            # 4. 提交完整 RGBDFrame
            call_api(
                silencer,
                "pipeline.submit_frame(frame)",
                lambda: pipeline.submit_frame(
                    make_dummy_frame(cfg, frame_id=2),
                ),
            )

            time.sleep(0.1)

            # 多喂几帧，只是让内部状态更完整；失败不算 API 测试失败
            try:
                feed_some_frames(
                    pipeline=pipeline,
                    cfg=cfg,
                    start_id=3,
                    count=5,
                    sleep_s=0.05,
                )
            except Exception:
                pass

            time.sleep(0.2)

            # 5. 查询整体状态
            call_api(
                silencer,
                "pipeline.get_status()",
                lambda: pipeline.get_status(),
            )

            # 6. 查询 tracking 统计
            call_api(
                silencer,
                "pipeline.get_tracking_stats()",
                lambda: pipeline.get_tracking_stats(),
            )

            # 7. 查询 mapping 统计
            call_api(
                silencer,
                "pipeline.get_mapping_stats()",
                lambda: pipeline.get_mapping_stats(),
            )

            # 8. 获取最新 tracking
            call_api(
                silencer,
                "pipeline.get_latest_tracking()",
                lambda: pipeline.get_latest_tracking(),
            )

            # 9. 获取最新 map snapshot
            call_api(
                silencer,
                "pipeline.get_latest_map()",
                lambda: pipeline.get_latest_map(),
            )

            # 10. 获取最新 mesh
            call_api(
                silencer,
                "pipeline.get_latest_mesh()",
                lambda: pipeline.get_latest_mesh(),
            )

            # 11. 获取 world 到 camera 外参
            call_api(
                silencer,
                "pipeline.get_current_extrinsic_world_to_camera()",
                lambda: pipeline.get_current_extrinsic_world_to_camera(),
            )

            # 12. 获取 camera 到 world 位姿
            call_api(
                silencer,
                "pipeline.get_current_camera_pose_world()",
                lambda: pipeline.get_current_camera_pose_world(),
            )

            # 13. 保存 mesh
            # 注意：这里已经 patch 了 mapper.save_final_mesh，所以只测 API 是否可调用
            patch_save_final_mesh(pipeline)

            call_api(
                silencer,
                "pipeline.save_mesh(prefix)",
                lambda: pipeline.save_mesh(prefix="api_test_mesh"),
            )

            # 14. 重置当前建图
            call_api(
                silencer,
                "pipeline.reset_reconstruction()",
                lambda: pipeline.reset_reconstruction(),
            )

            time.sleep(0.2)

            # reset_reconstruction 会重建 mapper，所以需要重新 patch
            patch_save_final_mesh(pipeline)

            # reset 后再喂几帧
            try:
                feed_some_frames(
                    pipeline=pipeline,
                    cfg=cfg,
                    start_id=100,
                    count=5,
                    sleep_s=0.05,
                )
            except Exception:
                pass

            time.sleep(0.2)

            # 15. 保存后重置建图
            # 注意：这里同样只测 API 是否可调用，不测试真实 mesh 保存
            patch_save_final_mesh(pipeline)

            call_api(
                silencer,
                "pipeline.reset_reconstruction_and_save(prefix)",
                lambda: pipeline.reset_reconstruction_and_save(
                    prefix="api_test_before_reset",
                ),
            )

            time.sleep(0.2)

            # reset_reconstruction_and_save 内部也会重建 mapper，所以再次 patch
            patch_save_final_mesh(pipeline)

            # 16. 清空 local map
            call_api(
                silencer,
                "pipeline.clear_local_map()",
                lambda: pipeline.clear_local_map(),
            )

            # 17. 重置 tracker
            call_api(
                silencer,
                "pipeline.reset_tracker()",
                lambda: pipeline.reset_tracker(),
            )

            # stop 前再 patch 一次，避免 stop(save_mesh=True) 真的写 mesh
            patch_save_final_mesh(pipeline)

            # 18. 停止并保存
            # 注意：这里 save_mesh=True 仍然传入，但底层保存函数已 mock
            call_api(
                silencer,
                "pipeline.stop(save_mesh=True)",
                lambda: pipeline.stop(
                    save_mesh=True,
                    final_mesh_prefix="api_test_final",
                ),
            )

            # 19. 关闭不保存
            pipeline2 = None

            def api_close():
                nonlocal pipeline2

                close_base_dir = base_dir / "close_case"
                close_base_dir.mkdir(parents=True, exist_ok=True)

                cfg2 = make_test_cfg(close_base_dir)

                pipeline2 = RealtimeMappingPipeline(
                    cfg=cfg2,
                    base_dir=close_base_dir,
                )

                pipeline2.start_external()
                pipeline2.close()

            call_api(
                silencer,
                "pipeline.close()",
                api_close,
            )

if __name__ == "__main__":
    main()