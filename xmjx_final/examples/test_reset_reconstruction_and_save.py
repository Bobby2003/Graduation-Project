import time
import traceback
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))

from config import PipelineConfig
from pipeline_api import RealtimeMappingPipeline

def main():
    cfg = PipelineConfig()
    cfg.render.enabled = False
    cfg.logging.save_log = False

    pipeline = RealtimeMappingPipeline(cfg)

    try:
        if not pipeline.start_background():
            print("pipeline start failed")
            return

        print("[TEST] running before save+reset...")
        time.sleep(15.0)

        print("[TEST] calling reset_reconstruction_and_save()...")
        result = pipeline.reset_reconstruction_and_save(
            prefix="before_reset_test",
            reset_scanner_temporal_state=True,
            restart_workers=True,
            join_timeout=2.0,
            rebuild_renderer=True,
        )

        print("[TEST] result:")
        print(result)

        print("[TEST] running after reset...")
        time.sleep(8.0)

        print("[TEST] status after reset:")
        print(pipeline.get_status())

    except Exception:
        traceback.print_exc()

    finally:
        print("[TEST] stopping...")
        print(pipeline.stop(save_mesh=False))

if __name__ == "__main__":
    main()