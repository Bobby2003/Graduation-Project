import os
import sys
import numpy as np
from multiprocessing import shared_memory

MAX_PTS = 1_000_000


def worker_main(shm_pts_name, shm_col_name, n_pts_val, ready_flag, stop_flag, reset_flag):
    import os
    import sys

    project_dir = r"C:/Users/Bobby2003/Desktop/end/Graduation-Project/奥比中光单相机点云处理"
    opencv_bin = r"C:\opencv\opencv\build\x64\vc16\bin"

    # 方法1：add_dll_directory
    os.add_dll_directory(opencv_bin)
    os.add_dll_directory(project_dir)

    # 方法2：加到 PATH（兼容性更好）
    os.environ['PATH'] = opencv_bin + os.pathsep + project_dir + os.pathsep + os.environ['PATH']

    sys.path.insert(0, project_dir)

    import slam_core
    from main import Depth3DScanner

    scanner = Depth3DScanner()
    if not scanner.init_device():
        print("[Worker] 设备初始化失败")
        return

    vo = slam_core.VisualOdometry(scanner.fx, scanner.fy, scanner.cx, scanner.cy)

    shm_pts = shared_memory.SharedMemory(name=shm_pts_name)
    shm_col = shared_memory.SharedMemory(name=shm_col_name)
    buf_pts = np.ndarray((MAX_PTS, 3), dtype=np.float32, buffer=shm_pts.buf)
    buf_col = np.ndarray((MAX_PTS, 3), dtype=np.float32, buffer=shm_col.buf)

    map_pts = np.zeros((0, 3), dtype=np.float32)
    map_col = np.zeros((0, 3), dtype=np.float32)

    print("[Worker] 开始采集（C++ 加速）...")

    while not stop_flag.value:
        if reset_flag.value:
            vo.reset()
            map_pts = np.zeros((0, 3), dtype=np.float32)
            map_col = np.zeros((0, 3), dtype=np.float32)
            reset_flag.value = 0
            print("[Worker] VO 已重置")

        res = scanner.sdk.capture_depth_frame()
        if res is None:
            continue
        raw, _ = res

        depth, vmask = scanner.process(raw)
        T_world = vo.track(depth)

        pts, col = slam_core.depth_to_world(
            depth, vmask, T_world,
            scanner.fx, scanner.fy,
            scanner.cx, scanner.cy
        )

        if len(pts) > 0:
            map_pts = np.vstack([map_pts, pts])
            map_col = np.vstack([map_col, col])

            if len(map_pts) > MAX_PTS:
                map_pts = map_pts[-MAX_PTS:]
                map_col = map_col[-MAX_PTS:]

            n = len(map_pts)
            buf_pts[:n] = map_pts
            buf_col[:n] = map_col
            n_pts_val.value = n
            ready_flag.set()

    scanner.close()
    shm_pts.close()
    shm_col.close()
    print("[Worker] 已退出")