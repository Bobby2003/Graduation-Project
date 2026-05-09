import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# 让脚本可以从子目录导入上一级的 config.py / pipeline_api.py
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import PipelineConfig
from pipeline_api import RealtimeMappingPipeline

# ============================================================
# 模拟你的修补算法
# ============================================================

def repair_algorithm(color, depth, mask):
    """
    这里模拟你的修补算法。

    实际使用时，你可以在这里做：
    - 深度补洞
    - mask 修补
    - 噪声过滤
    - 边缘修复
    - 语义区域剔除
    - 深度平滑
    - inpainting
    """

    repaired_color = color.copy()
    repaired_depth = depth.copy()
    repaired_mask = mask.copy()

    # 示例 1：把 mask 中的无效区域简单补成邻近平均值
    #这里只是演示，不是高质量修补算法
    invalid = repaired_mask == 0

    if np.any(invalid):
        valid_depth = repaired_depth[repaired_mask > 0]

        if valid_depth.size > 0:
            median_depth = np.median(valid_depth)
            repaired_depth[invalid] = median_depth
            repaired_mask[invalid] = 1

    # 示例 2：简单限制深度范围，单位 mm
    repaired_mask = (
        (repaired_depth > 200.0) &
        (repaired_depth < 3000.0)
    ).astype(np.uint8)

    return repaired_color, repaired_depth.astype(np.float32), repaired_mask

# ============================================================
# 构造测试 RGBD 数据
# ============================================================

def fake_rgbd_source(width=640, height=480):
    """
    构造一个比纯平面更适合测试 tracking 的假 RGBD 数据。

    注意：
    这只是 API 测试数据，不代表真实建图质量。
    真正建图建议接入真实 RGBD 数据。
    """

    frame_id = 0

    yy, xx = np.mgrid[0:height, 0:width]

    cx = width / 2.0
    cy = height / 2.0

    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    while True:
        frame_id += 1

        # 彩色图，uint8, RGB 或 BGR 取决于 cfg.input.input_color_is_bgr
        color = np.zeros((height, width, 3), dtype=np.uint8)

        color[..., 0] = 80
        color[..., 1] = 120
        color[..., 2] = 160

        # 画一个移动的亮块，模拟纹理变化
        block_x = int((frame_id * 2) % (width - 100))
        color[150:250, block_x:block_x + 100, :] = np.array([200, 80, 80], dtype=np.uint8)

        # 深度单位：mm
        depth = np.ones((height, width), dtype=np.float32) * 1000.0

        # 添加中心凸起，避免完全平面导致 ICP 奇异
        bump = 250.0 * np.exp(-r2 / (2.0 * 90.0 ** 2))
        depth -= bump.astype(np.float32)

        # 添加斜坡结构
        depth += ((xx - cx) * 0.15).astype(np.float32)

        # 添加轻微随时间变化
        depth += np.sin(frame_id * 0.05) * 5.0

        # 模拟部分缺失区域，交给 repair_algorithm 修补
        mask = ((depth > 200.0) & (depth < 3000.0)).astype(np.uint8)

        if frame_id % 30 < 15:
            mask[200:260, 280:360] = 0
            depth[200:260, 280:360] = 0.0

        yield frame_id, color, depth, mask

# ============================================================
# 打印 API 状态
# ============================================================

def print_status(pipeline, tag=""):
    status = pipeline.get_status()

    print("\n================ STATUS", tag, "================")
    print("running:", status.get("running"))
    print("input_mode:", status.get("input_mode"))
    print("mapping_queue_size:", status.get("mapping_queue_size"))
    print("last_error:", status.get("last_error"))

    tracking = status.get("tracking")
    mapping = status.get("mapping")
    tracking_stats = status.get("tracking_stats")
    mapping_stats = status.get("mapping_stats")

    print("tracking:", tracking)
    print("mapping:", mapping)
    print("tracking_stats:", tracking_stats)
    print("mapping_stats:", mapping_stats)

# ============================================================
# 主测试流程
# ============================================================

def main():
    cfg = PipelineConfig()

    # 外部算法测试时，建议关闭渲染
    cfg.render.enabled = False

    # 输出模型目录
    cfg.mapping.model_dir = "./model"

    # 如果你不希望 console 太吵，可以关闭部分日志
    # cfg.logging.enable_console_log = False

    # 根据你的数据单位设置。
    # 如果 depth 单位是 mm，通常 depth_scale 是 1000.0 或 None。
    # 你的 pipeline_api.py 注释说 None 可由预处理器自动判断。
    # 这里不强制改，沿用 config.py 默认值。
    #
    # cfg.input.depth_scale = 1000.0
    # cfg.input.depth_trunc = 3.0

    pipeline = RealtimeMappingPipeline(cfg)

    print("[TEST] start_external()")
    ok = pipeline.start_external()

    if not ok:
        print("[TEST] pipeline start failed")
        return

    source = fake_rgbd_source(
        width=cfg.input_camera.width,
        height=cfg.input_camera.height,
    )

    try:
        # ========================================================
        # 第一阶段：测试 submit_rgbd / get_status / pose / mesh
        # ========================================================

        print("[TEST] feed repaired RGBD frames")

        for frame_id, color, depth, mask in source:
            repaired_color, repaired_depth, repaired_mask = repair_algorithm(
                color=color,
                depth=depth,
                mask=mask,
            )

            pipeline.submit_rgbd(
                color=repaired_color,
                depth=repaired_depth,
                mask=repaired_mask,
                frame_id=frame_id,
                timestamp=time.perf_counter(),
                extras={
                    "source": "test_repair_api",
                    "repair_algorithm": "dummy_repair_v1",
                },
            )

            # 模拟 30 FPS
            time.sleep(1.0 / 30.0)

            if frame_id % 30 == 0:
                print_status(pipeline, tag=f"frame={frame_id}")

                T_cw = pipeline.get_current_camera_pose_world()
                if T_cw is not None:
                    print("[TEST] current camera pose camera_to_world:")
                    print(T_cw)
                else:
                    print("[TEST] current camera pose is None")

                mesh = pipeline.get_latest_mesh()
                if mesh is not None:
                    try:
                        print("[TEST] latest mesh vertices:", len(mesh.vertices))
                        print("[TEST] latest mesh triangles:", len(mesh.triangles))
                    except Exception as e:
                        print("[TEST] failed to inspect mesh:", repr(e))
                else:
                    print("[TEST] latest mesh is None")

            if frame_id >= 120:
                break

        # ========================================================
        # 第二阶段：测试运行中保存 mesh
        # ========================================================

        print("\n[TEST] save_mesh() while running")
        save_ok, hist_path, latest_path = pipeline.save_mesh(prefix="repair_api_mid")

        print("[TEST] save_ok:", save_ok)
        print("[TEST] hist_path:", hist_path)
        print("[TEST] latest_path:", latest_path)

        # ========================================================
        # 第三阶段：测试 reset_reconstruction()
        # ========================================================

        print("\n[TEST] reset_reconstruction()")
        reset_result = pipeline.reset_reconstruction(
            reset_scanner_temporal_state=False,
            restart_workers=True,
            rebuild_renderer=True,
        )
        print("[TEST] reset_result:", reset_result)

        # reset 后继续喂一些帧，验证 pipeline 还能工作
        print("\n[TEST] feed frames after reset")

        for i, (frame_id, color, depth, mask) in enumerate(source):
            new_frame_id = i + 1

            repaired_color, repaired_depth, repaired_mask = repair_algorithm(
                color=color,
                depth=depth,
                mask=mask,
            )

            pipeline.submit_rgbd(
                color=repaired_color,
                depth=repaired_depth,
                mask=repaired_mask,
                frame_id=new_frame_id,
                timestamp=time.perf_counter(),
                extras={
                    "source": "test_repair_api_after_reset",
                    "repair_algorithm": "dummy_repair_v1",
                },
            )

            time.sleep(1.0 / 30.0)

            if new_frame_id % 30 == 0:
                print_status(pipeline, tag=f"after_reset_frame={new_frame_id}")

            if new_frame_id >= 90:
                break

    finally:
        # ========================================================
        # 第四阶段：测试 stop(save_mesh=True)
        # ========================================================

        print("\n[TEST] stop(save_mesh=True)")
        result = pipeline.stop(
            save_mesh=True,
            final_mesh_prefix="repair_api_final",
        )

        print("\n================ STOP RESULT ================")
        print(result)

if __name__ == "__main__":
    main()