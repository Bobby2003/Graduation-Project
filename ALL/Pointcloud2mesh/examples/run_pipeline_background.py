import time
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))

from config import PipelineConfig
from pipeline_api import RealtimeMappingPipeline

def main():
    cfg = PipelineConfig()

    # 后台模式必须关渲染
    cfg.render.enabled = False

    # 根据需要修改输出目录
    cfg.mapping.model_dir = "./model"

    pipeline = RealtimeMappingPipeline(cfg)

    if not pipeline.start_background():
        print("pipeline start failed")
        return

    try:
        for _ in range(10):
            time.sleep(1.0)
            status = pipeline.get_status()
            print(status)

    finally:
        result = pipeline.stop(save_mesh=True)
        print("stop result:", result)

if __name__ == "__main__":
    main()