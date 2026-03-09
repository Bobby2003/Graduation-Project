import os
import time
import copy
import threading
from queue import Queue, Empty
from collections import deque

import numpy as np
import cv2
import open3d as o3d

# 请根据你的环境替换或bbec_sdk 的导入，如果没有 orbbec_sdk 可以注释相关部分并用其他深度源
try:
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
    ORBBEC_AVAILABLE = True
except Exception:
    ORBBEC_AVAILABLE = False
    # 占位类（若没有 Orbbec SDK，则需要替换 capture_depth_frame 的实现）
    class OrbbecCameraSDK:
        ONI_SENSOR_DEPTH = 1
        def __init__(self, path=None): pass
        def initialize(self): return False
        def open_device(self): return False
        def create_stream(self, sensor): return False
        def start_stream(self): return False
        def capture_depth_frame(self): return None
        def cleanup(self): pass
    def get_default_sdk_path(): return ""

class RobustNonBlockingMapper:
    def __init__(self,
                 width=640, height=480,
                 fx=570.3, fy=570.3, cx=319.5, cy=239.5,
                 voxel_size=0.03,
                 icp_threshold=0.12,
                 save_interval_sec=5.0,
                 save_point_delta=200,
                 stable_save_delay=8.0,
                 max_queue_size=6):
        # Camera intrinsics & image size
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Precompute uv grid for depth->pointcloud mapping
        xu = np.arange(self.width, dtype=np.float32)
        yu = np.arange(self.height, dtype=np.float32)
        self.u_grid, self.v_grid = np.meshgrid(xu, yu)

        # Algorithm parameters
        self.voxel_size = voxel_size  # m
        self.icp_threshold = icp_threshold  # m
        self.save_interval_sec = save_interval_sec
        self.save_point_delta = save_point_delta
        self.stable_save_delay = stable_save_delay

        # State
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.latest_color = None
        self.latest_color_lock = threading.Lock()
        self.global_pcd = None
        self.global_lock = threading.Lock()
        self.depth_buffer = deque(maxlen=3)
        self.stop_event = threading.Event()

        # Pose/frames bookkeeping
        self.last_pose = np.identity(4)   # cumulative pose (world <- frame)
        self.last_down = None             # previous frame downsampled pcd for frame-to-frame ICP
        self.frame_idx = 0
        self.last_merge_time = 0.0
        self.dirty = False
        self.last_saved_point_count = 0
        self.last_save_time = 0.0

        # Setup camera SDK & color capture
        self.sdk_path = get_default_sdk_path() if ORBBEC_AVAILABLE else None
        self.camera_sdk = OrbbecCameraSDK(self.sdk_path)
        self.cv_camera = cv2.VideoCapture(0)
        self.cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Output
        os.makedirs("output", exist_ok=True)
        self.output_global_path = os.path.join("output", "global_cloud.ply")

        # ICP acceptance thresholds (may need tuning)
        self.MIN_FITNESS = 0.12
        self.MAX_RMSE = 0.06
        self.MIN_CORRESP = 30

    def initialize(self):
        # Initialize camera SDK streams (if available) and color capture
        ok_cam = self.cv_camera.isOpened()
        ok_sdk = False
        try:
            if ORBBEC_AVAILABLE:
                ok_sdk = all([
                    self.camera_sdk.initialize(),
                    self.camera_sdk.open_device(),
                    self.camera_sdk.create_stream(self.camera_sdk.ONI_SENSOR_DEPTH),
                    self.camera_sdk.start_stream()
                ])
        except Exception as e:
            print("相机 SDK 初始化异常:", e)
            ok_sdk = False

        if not ok_cam:
            print("警告：彩色摄像头未打开（cv2.VideoCapture），请检查设备。")
        if ORBBEC_AVAILABLE and not ok_sdk:
            print("警告：Orbbec SDK 初始化失败或未连接 Orbbec 设备。")
        if (not ORBBEC_AVAILABLE) and (not ok_cam):
            print("错误：没有可用的输入设备。")
            return False
        return ok_cam or ok_sdk

    def stop(self):
        self.stop_event.set()

    def filter_depth(self, depth_raw):
        """时域平滑 + 双边滤波（深度单位：毫米）"""
        depth = depth_raw.astype(np.float32)
        depth[(depth < 200) | (depth > 5000)] = 0  # 裁剪范围
        if len(self.depth_buffer) >= 1:
            stacked = np.stack(list(self.depth_buffer), axis=0)
            depth = (np.mean(stacked, axis=0) + depth) / 2.0
        # 双边滤波（映射到 0-255 做滤波）
        mapped = np.clip((depth / 5000.0 * 255.0), 0, 255).astype(np.uint8)
        bf = cv2.bilateralFilter(mapped, d=5, sigmaColor=20, sigmaSpace=20).astype(np.float32)
        depth_smooth = bf / 255.0 * 5000.0
        self.depth_buffer.append(depth_smooth)
        return depth_smooth

    def depth_to_pointcloud(self, depth, color):
        """depth（mm）+ color -> open3d.geometry.PointCloud（单位：米）"""
        mask = depth > 0
        if np.count_nonzero(mask) < 500:
            return None
        z = depth[mask] / 1000.0  # mm -> m
        u = self.u_grid[mask]
        v = self.v_grid[mask]

        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy  # y negative to match previous conventions
        points = np.stack([x, y, z], axis=-1)

        # color: BGR->RGB
        colors = color[v.astype(int), u.astype(int)][:, ::-1] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def capture_loop(self):
        """采集线程：从深度 SDK + 摄像头读取数据并推入队列"""
        while not self.stop_event.is_set():
            try:
                depth_res = None
                if ORBBEC_AVAILABLE:
                    depth_res = self.camera_sdk.capture_depth_frame()
                ret, color = self.cv_camera.read()
                if not ret:
                    # 等待下一帧
                    time.sleep(0.01)
                    continue

                # depth_res 兼容处理：支持 None / ndarray / (array, ts)
                if depth_res is None:
                    # 如果没有深度来源，继续（或你也可以用模拟深度）
                    time.sleep(0.01)
                    continue

                if isinstance(depth_res, (tuple, list)):
                    depth = depth_res[0]
                else:
                    depth = depth_res

                # resize color 与深度对齐（若需要）
                if (color.shape[1], color.shape[0]) != (self.width, self.height):
                    color = cv2.resize(color, (self.width, self.height))

                # 非阻塞放入队列（队列满则丢帧）
                try:
                    self.frame_queue.put_nowait((depth.copy(), color.copy()))
                except:
                    pass

                with self.latest_color_lock:
                    self.latest_color = color.copy()

            except Exception as e:
                print("采集线程异常:", e)
                time.sleep(0.05)

        print("采集线程退出。")

    def align_merge(self, src_pcd, src_down):
        """
        帧间配准并合并到全局：
        - src_pcd: 原始点云（未下采样）
        - src_down: 下采样点云（用于配准）
        返回 True 表示合并（accepted），False 表示跳过
        """
        self.frame_idx += 1
        frame_id = self.frame_idx

        # 若没有上一帧，则初始化 global 与 last_down
        if self.last_down is None:
            pcd_copy = copy.deepcopy(src_pcd)
            pcd_copy.transform(self.last_pose)  # 初始 pose 为 I
            # 去噪
            try:
                cl, ind = pcd_copy.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                pcd_copy = pcd_copy.select_by_index(ind)
            except Exception:
                pass
            pcd_copy = pcd_copy.voxel_down_sample(self.voxel_size)
            with self.global_lock:
                self.global_pcd = pcd_copy
                self.last_down = src_down
            self.dirty = True
            self.last_merge_time = time.time()
            print(f"[Frame {frame_id}] 初始化 global_pcd, points={len(self.global_pcd.points)}")
            # 保存初始帧变换结果便于调试
            try:
                o3d.io.write_point_cloud(f"output/frame_{frame_id:04d}_transformed.ply", copy.deepcopy(pcd_copy), write_ascii=False)
            except Exception as e:
                print(f"[Frame {frame_id}] 保存初始帧失败: {e}")
            return True

        # 帧间 ICP：将当前帧对齐到上一帧
        try:
            icp = o3d.pipelines.registration.registration_icp(
                src_down, self.last_down,
                max_correspondence_distance=self.icp_threshold,
                init=np.identity(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
            )
        except Exception as e:
            print(f"[Frame {frame_id}] ICP 异常: {e}")
            # 更新 last_down 以减少长时间失败导致漂移
            self.last_down = src_down
            return False

        fitness = getattr(icp, "fitness", 0.0)
        rmse = getattr(icp, "inlier_rmse", 1e9)
        corr_count = int(fitness * len(np.asarray(src_down.points))) if len(np.asarray(src_down.points)) > 0 else 0

        print(f"[Frame {frame_id}] ICP -> fitness={fitness:.4f}, rmse={rmse:.4f}, corr_count={corr_count}")

        # 质量检查
        if fitness < self.MIN_FITNESS or rmse > self.MAX_RMSE or corr_count < self.MIN_CORRESP:
            print(f"[Frame {frame_id}] 配准未通过（丢弃）。阈值 => fitness>={self.MIN_FITNESS}, rmse<={self.MAX_RMSE}, corr>={self.MIN_CORRESP}")
            # 仍更新 last_down 以保持帧间参考连续性
            self.last_down = src_down
            return False

        # 接受配准：累积位姿并将原始 pcd 变换到全局坐标系
        transform_f2prev = icp.transformation
        new_pose = self.last_pose @ transform_f2prev
        self.last_pose = new_pose

        pcd_to_merge = copy.deepcopy(src_pcd)
        pcd_to_merge.transform(new_pose)

        # 可选：裁剪非常远的点（根据场景调整）
        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-5,-5,-1), max_bound=(5,5,5))
        # pcd_to_merge = pcd_to_merge.crop(bbox)

        # 去噪
        try:
            cl, ind = pcd_to_merge.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd_to_merge = pcd_to_merge.select_by_index(ind)
        except Exception:
            pass

        # 合并（线程安全）
        with self.global_lock:
            if self.global_pcd is None:
                merged = pcd_to_merge
            else:
                merged = self.global_pcd + pcd_to_merge
            merged = merged.voxel_down_sample(self.voxel_size)
            self.global_pcd = merged

        # 更新 last_down（用于下一帧）
        self.last_down = src_down
        self.dirty = True
        self.last_merge_time = time.time()

        # 保存该帧变换结果，便于逐帧排查问题
        out_single = f"output/frame_{frame_id:04d}_transformed.ply"
        try:
            o3d.io.write_point_cloud(out_single, copy.deepcopy(pcd_to_merge), write_ascii=False)
            print(f"[Frame {frame_id}] 已保存该帧变换结果: {out_single} (points:{len(pcd_to_merge.points)})")
        except Exception as e:
            print(f"[Frame {frame_id}] 保存帧失败: {e}")

        return True

    def processing_loop(self):
        """处理线程：读取队列，生成点云，调用 align_merge，并按策略保存 global"""
        while not self.stop_event.is_set():
            try:
                depth_raw, color = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                depth = self.filter_depth(depth_raw)
                pcd = self.depth_to_pointcloud(depth, color)
                if pcd is None:
                    continue

                # 下采样用于配准
                src_down = pcd.voxel_down_sample(self.voxel_size)
                if len(src_down.points) < 50:
                    print(f"[Frame {self.frame_idx+1}] 下采样后点太少，跳过 (points={len(src_down.points)})")
                    continue

                merged_ok = self.align_merge(pcd, src_down)

                # 保存 global: 采用“显著点数变化或稳定延时”策略
                now = time.time()
                with self.global_lock:
                    global_pts = len(self.global_pcd.points) if self.global_pcd is not None else 0

                # 如果全局被修改并且一定时间内没有新的合并（stable），则保存；
                # 或者如果点数较上次保存变化显著也保存
                if self.dirty and (now - self.last_merge_time) >= self.stable_save_delay and (now - self.last_save_time) >= self.save_interval_sec:
                    # 保存全局
                    with self.global_lock:
                        if self.global_pcd is not None and len(self.global_pcd.points) > 0:
                            try:
                                o3d.io.write_point_cloud(self.output_global_path, self.global_pcd, write_ascii=False)
                                self.last_saved_point_count = len(self.global_pcd.points)
                                self.last_save_time = now
                                self.dirty = False
                                print(f"[{time.strftime('%H:%M:%S')}] 稳定后已保存 global 点云: {self.output_global_path} (pts={self.last_saved_point_count})")
                            except Exception as e:
                                print("保存 global 失败:", e)

                # 另一个保存策略：定期检查点数差异显著则保存（防止长时间无稳定时也保存）
                if (now - self.last_save_time) >= self.save_interval_sec and global_pts > 0:
                    if abs(global_pts - self.last_saved_point_count) >= self.save_point_delta:
                        with self.global_lock:
                            try:
                                o3d.io.write_point_cloud(self.output_global_path, self.global_pcd, write_ascii=False)
                                self.last_saved_point_count = global_pts
                                self.last_save_time = now
                                self.dirty = False
                                print(f"[{time.strftime('%H:%M:%S')}] 点数显著变化，已保存 global (pts={global_pts})")
                            except Exception as e:
                                print("保存 global 失败:", e)
                    else:
                        # 非显著变化则跳过
                        pass

            except Exception as e:
                print("处理线程异常:", e)
                time.sleep(0.01)

        print("处理线程退出。")

    def run(self):
        if not self.initialize():
            return

        cap_thread = threading.Thread(target=self.capture_loop, daemon=True)
        proc_thread = threading.Thread(target=self.processing_loop, daemon=True)
        cap_thread.start()
        proc_thread.start()

        print("运行中。按 'q' 退出。 调试输出保存在 output/ 目录。")
        try:
            while not self.stop_event.is_set():
                with self.latest_color_lock:
                    frame = None if self.latest_color is None else self.latest_color.copy()

                if frame is not None:
                    with self.global_lock:
                        pts = len(self.global_pcd.points) if self.global_pcd is not None else 0
                    cv2.putText(frame, f"Frame:{self.frame_idx} Points:{pts}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("RGB Preview (press q to quit)", frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("收到退出命令，正在停止...")
                    self.stop_event.set()
                    break

            cap_thread.join(timeout=2.0)
            proc_thread.join(timeout=2.0)

        except KeyboardInterrupt:
            print("KeyboardInterrupt，正在停止...")
            self.stop_event.set()
        finally:
            # cleanup
            try:
                if ORBBEC_AVAILABLE:
                    self.camera_sdk.cleanup()
            except Exception:
                pass
            try:
                self.cv_camera.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

            # 退出前保存一次 final global
            with self.global_lock:
                if self.global_pcd is not None and len(self.global_pcd.points) > 0:
                    try:
                        o3d.io.write_point_cloud(self.output_global_path, self.global_pcd, write_ascii=False)
                        print(f"退出前已保存最终 global 点云：{self.output_global_path} (pts={len(self.global_pcd.points)})")
                    except Exception as e:
                        print("退出时保存 global 失败:", e)

            print("已安全退出。")

if __name__ == "__main__":
    mapper = RobustNonBlockingMapper(
        width=640, height=480,
        fx=570.3, fy=570.3, cx=319.5, cy=239.5,
        voxel_size=0.03,
        icp_threshold=0.12,
        save_interval_sec=5.0,
        save_point_delta=200,
        stable_save_delay=8.0,
        max_queue_size=6
    )
    mapper.run()