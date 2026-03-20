import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import rerun as rr  # 替换 open3d

from slam_worker import worker_main

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    MAX_PTS = 1_000_000
    shm_pts = shared_memory.SharedMemory(create=True, size=MAX_PTS * 3 * 4)
    shm_col = shared_memory.SharedMemory(create=True, size=MAX_PTS * 3 * 4)

    buf_pts = np.ndarray((MAX_PTS, 3), dtype=np.float32, buffer=shm_pts.buf)
    buf_col = np.ndarray((MAX_PTS, 3), dtype=np.float32, buffer=shm_col.buf)

    n_pts_val = mp.Value('i', 0)
    ready_flag = mp.Event()
    stop_flag = mp.Value('i', 0)
    reset_flag = mp.Value('i', 0)

    worker = mp.Process(
        target=worker_main,
        args=(shm_pts.name, shm_col.name, n_pts_val, ready_flag, stop_flag, reset_flag)
    )
    worker.start()
    print("[Main] Worker 已启动，等待第一帧... \n'q' 退出   's' 保存点云   'r' 重置 VO")

    # 初始化 Rerun
    rr.init("SLAM Point Cloud", spawn=True)

    try:
        while True:
            if ready_flag.is_set():
                n = n_pts_val.value
                if n > 0:
                    pts = np.copy(buf_pts[:n])
                    col = np.copy(buf_col[:n])

                    # Rerun 记录点云（自动处理大规模数据）
                    rr.log(
                        "world/points",
                        rr.Points3D(
                            positions=pts,
                            colors=(col * 255).astype(np.uint8),  # Rerun 用 0-255
                            radii=0.005  # 点的大小
                        )
                    )

                    # 可选：记录相机位姿（如果你能从 worker 传出来）
                    # rr.log("world/camera", rr.Transform3D(translation=[x, y, z]))

            time.sleep(0.03)  # 30 FPS

            # 键盘控制（Rerun 窗口不支持直接键盘输入，用终端）
            # 你可以在终端输入命令或者用 Rerun 的 UI 控制

    except KeyboardInterrupt:
        print("\n[Main] 正在退出...")

    finally:
        stop_flag.value = 1
        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()

        shm_pts.close()
        shm_pts.unlink()
        shm_col.close()
        shm_col.unlink()
        print("[Main] 清理完成")