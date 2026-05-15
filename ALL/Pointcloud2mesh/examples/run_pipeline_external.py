import time
import numpy as np
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))

from config import PipelineConfig
from pipeline_api import RealtimeMappingPipeline

def fake_rgbd_source():
    """
    示例数据源。
    实际使用时替换成你的外部 RGBD / depth 输入。
    """
    frame_id = 0
    while True:
        frame_id += 1

        color = np.zeros((480, 640, 3), dtype=np.uint8)

        # 假深度：单位 mm
        depth = np.ones((480, 640), dtype=np.float32) * 1000.0

        mask = (depth > 0).astype(np.uint8)

        yield frame_id, color, depth, mask

def main():
    cfg = PipelineConfig()
    cfg.render.enabled = False

    pipeline = RealtimeMappingPipeline(cfg)

    if not pipeline.start_external():
        print("pipeline start failed")
        return

    try:
        for frame_id, color, depth, mask in fake_rgbd_source():
            pipeline.submit_rgbd(
                color=color,
                depth=depth,
                mask=mask,
                frame_id=frame_id,
                timestamp=time.perf_counter(),
            )

            time.sleep(1.0 / 30.0)

            if frame_id >= 300:
                break

    finally:
        result = pipeline.stop(save_mesh=True)
        print(result)

if __name__ == "__main__":
    main()