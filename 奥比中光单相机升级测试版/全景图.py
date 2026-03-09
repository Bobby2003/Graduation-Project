import os
import time
import copy
import threading
from queue import Queue, Empty
from collections import deque

import numpy as np
import cv2
import open3d as o3d

# -------------- Orbbec SDK 占位（与 main3.py 相同） --------------
try:
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
    ORBBEC_AVAILABLE = True
except Exception:
    ORBBEC_AVAILABLE = False
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

# -------------- 主类 --------------
class RobustPanoramaMapper:
    def __init__(self,
                 width=640, height=480,
                 fx=570.3, fy=570.3, cx=319.5, cy=239.5,
                 voxel_size=0.03,
                 icp_threshold=0.12,
                 save_interval_sec=5.0,
                 stable_save_delay=8.0,
                 max_queue_size=6):
        # 相机内参与尺寸
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # 预计算 uv 网格（用于反投影）
        xu = np.arange(self.width, dtype=np.float32)
        yu = np.arange(self.height, dtype=np.float32)
        self.u_grid, self.v_grid = np.meshgrid(xu, yu)

        # 参数
        self.voxel_size = voxel_size
        self.icp_threshold = icp_threshold
        self.save_interval_sec = save_interval_sec
        self.stable_save_delay = stable_save_delay

        # 状态
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.latest_color = None
        self.latest_color_lock = threading.Lock()
        self.global_pcd = None
        self.global_lock = threading.Lock()
        self.depth_buffer = deque(maxlen=3)
        self.stop_event = threading.Event()

        # 位姿与帧记录
        self.last_pose = np.identity(4)   # world <- frame 累积
        self.last_down = None
        self.frame_idx = 0
        self.last_merge_time = 0.0
        self.dirty = False

        # 保存被接受帧用于全景合成：每项为 dict { 'pose':4x4, 'depth':ndarray(mm), 'color':ndarray(BGR) }
        self.saved_frames = []

        # 摄像头（color）和 SDK 深度接口
        self.sdk_path = get_default_sdk_path() if ORBBEC_AVAILABLE else None
        self.camera_sdk = OrbbecCameraSDK(self.sdk_path)
        self.cv_camera = cv2.VideoCapture(0)
        self.cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 输出
        os.makedirs("output", exist_ok=True)
        self.output_global_path = os.path.join("output", "global_cloud.ply")
        self.output_panorama_path = os.path.join("output", "panorama_4096x2048.jpg")

        # ICP 阈值（可调）
        self.MIN_FITNESS = 0.12
        self.MAX_RMSE = 0.06
        self.MIN_CORRESP = 30

    def initialize(self):
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
            print("警告：彩色摄像头未打开，请检查设备。")
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
        depth[(depth < 200) | (depth > 5000)] = 0
        if len(self.depth_buffer) >= 1:
            stacked = np.stack(list(self.depth_buffer), axis=0)
            depth = (np.mean(stacked, axis=0) + depth) / 2.0
        mapped = np.clip((depth / 5000.0 * 255.0), 0, 255).astype(np.uint8)
        bf = cv2.bilateralFilter(mapped, d=5, sigmaColor=20, sigmaSpace=20).astype(np.float32)
        depth_smooth = bf / 255.0 * 5000.0
        self.depth_buffer.append(depth_smooth)
        return depth_smooth

    def depth_to_pointcloud(self, depth, color):
        mask = depth > 0
        if np.count_nonzero(mask) < 500:
            return None
        z = (depth[mask] / 1000.0).astype(np.float32)  # m
        u = self.u_grid[mask]
        v = self.v_grid[mask]
        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy
        points = np.stack([x, y, z], axis=-1)
        colors = color[v.astype(int), u.astype(int)][:, ::-1] / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def capture_loop(self):
        while not self.stop_event.is_set():
            try:
                depth_res = None
                if ORBBEC_AVAILABLE:
                    depth_res = self.camera_sdk.capture_depth_frame()
                ret, color = self.cv_camera.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if depth_res is None:
                    time.sleep(0.01)
                    continue

                if isinstance(depth_res, (tuple, list)):
                    depth = depth_res[0]
                else:
                    depth = depth_res

                if (color.shape[1], color.shape[0]) != (self.width, self.height):
                    color = cv2.resize(color, (self.width, self.height))

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

    def align_merge(self, src_pcd, src_down, depth_mm, color_bgr):
        """
        与 main3.py 基本一致，但在接受配准时保存 depth/color/pose 供全景拼接使用
        """
        self.frame_idx += 1
        frame_id = self.frame_idx

        if self.last_down is None:
            pcd_copy = copy.deepcopy(src_pcd)
            pcd_copy.transform(self.last_pose)
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
            try:
                o3d.io.write_point_cloud(f"output/frame_{frame_id:04d}_transformed.ply", copy.deepcopy(pcd_copy), write_ascii=False)
            except Exception as e:
                print(f"[Frame {frame_id}] 保存初始帧失败: {e}")

            # 保存首帧的 pose/color/depth
            self.saved_frames.append({
                'pose': copy.deepcopy(self.last_pose),
                'depth': depth_mm.copy(),
                'color': color_bgr.copy()
            })
            return True

        try:
            icp = o3d.pipelines.registration.registration_icp(
                src_down, self.last_down,
                max_correspondence_distance=self.icp_threshold,
                init=np.identity(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
            )
        except Exception as e:
            print(f"[Frame {frame_id}] ICP 异常:", e)
            self.last_down = src_down
            return False

        fitness = getattr(icp, "fitness", 0.0)
        rmse = getattr(icp, "inlier_rmse", 1e9)
        corr_count = int(fitness * len(np.asarray(src_down.points))) if len(np.asarray(src_down.points)) > 0 else 0

        print(f"[Frame {frame_id}] ICP -> fitness={fitness:.4f}, rmse={rmse:.4f}, corr_count={corr_count}")

        if fitness < self.MIN_FITNESS or rmse > self.MAX_RMSE or corr_count < self.MIN_CORRESP:
            print(f"[Frame {frame_id}] 配准未通过（丢弃）。")
            self.last_down = src_down
            return False

        transform_f2prev = icp.transformation
        new_pose = self.last_pose @ transform_f2prev
        self.last_pose = new_pose

        pcd_to_merge = copy.deepcopy(src_pcd)
        pcd_to_merge.transform(new_pose)

        try:
            cl, ind = pcd_to_merge.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd_to_merge = pcd_to_merge.select_by_index(ind)
        except Exception:
            pass

        with self.global_lock:
            if self.global_pcd is None:
                merged = pcd_to_merge
            else:
                merged = self.global_pcd + pcd_to_merge
            merged = merged.voxel_down_sample(self.voxel_size)
            self.global_pcd = merged

        self.last_down = src_down
        self.dirty = True
        self.last_merge_time = time.time()

        out_single = f"output/frame_{frame_id:04d}_transformed.ply"
        try:
            o3d.io.write_point_cloud(out_single, copy.deepcopy(pcd_to_merge), write_ascii=False)
            print(f"[Frame {frame_id}] 已保存该帧变换结果: {out_single} (points:{len(pcd_to_merge.points)})")
        except Exception as e:
            print(f"[Frame {frame_id}] 保存帧失败: {e}")

        # 保存被接受帧（pose = world <- frame）
        self.saved_frames.append({
            'pose': copy.deepcopy(new_pose),
            'depth': depth_mm.copy(),
            'color': color_bgr.copy()
        })
        return True

    def processing_loop(self):
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

                src_down = pcd.voxel_down_sample(self.voxel_size)
                if len(src_down.points) < 50:
                    print(f"[Frame {self.frame_idx+1}] 下采样后点太少，跳过 (points={len(src_down.points)})")
                    continue

                merged_ok = self.align_merge(pcd, src_down, depth_mm=depth, color_bgr=color)

                now = time.time()
                with self.global_lock:
                    global_pts = len(self.global_pcd.points) if self.global_pcd is not None else 0

                if self.dirty and (now - self.last_merge_time) >= self.stable_save_delay:
                    with self.global_lock:
                        if self.global_pcd is not None and len(self.global_pcd.points) > 0:
                            try:
                                o3d.io.write_point_cloud(self.output_global_path, self.global_pcd, write_ascii=False)
                                self.dirty = False
                                print(f"[{time.strftime('%H:%M:%S')}] 稳定后已保存 global 点云: {self.output_global_path} (pts={len(self.global_pcd.points)})")
                            except Exception as e:
                                print("保存 global 失败:", e)
            except Exception as e:
                print("处理线程异常:", e)
                time.sleep(0.01)
        print("处理线程退出。")

    def generate_panorama(self, pano_w=4096, pano_h=2048, fill_inpaint=True):
        """
        根据保存的 frames 生成 equirectangular 全景图
        - pano_w, pano_h: 输出全景分辨率（典型 2:1 比例）
        - fill_inpaint: 若为 True，则对未被覆盖的区域做简单 inpaint 填充
        """
        if len(self.saved_frames) == 0:
            print("没有可用帧用于生成全景。")
            return False

        print(f"开始生成全景，使用 {len(self.saved_frames)} 帧，输出尺寸 {pano_w}x{pano_h} ...")

        # 计算球心：使用相机位置的平均值（pose 是 world <- frame，摄像机在 frame 原点 -> 在 world 中的位置为 pose[:3,3]）
        cam_positions = np.array([f['pose'][:3, 3] for f in self.saved_frames])
        center = np.mean(cam_positions, axis=0)

        # 初始化累积数组
        acc_rgb = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        acc_weight = np.zeros((pano_h, pano_w), dtype=np.float32)

        # 为了加速，每帧向量化处理：获取有效深度掩码, 反投影到相机坐标, 变换到世界坐标, 计算方向到 center 并映射到经纬度
        for idx, f in enumerate(self.saved_frames):
            depth_mm = f['depth']  # float mm 数组
            color_bgr = f['color']
            pose = f['pose']  # 4x4 (world <- frame)

            mask = depth_mm > 0
            if np.count_nonzero(mask) < 50:
                continue

            # 向量化取出像素坐标
            us = self.u_grid[mask].astype(np.float32)
            vs = self.v_grid[mask].astype(np.float32)
            zs = (depth_mm[mask] / 1000.0).astype(np.float32)  # m

            xs = (us - self.cx) * zs / self.fx
            ys = -(vs - self.cy) * zs / self.fy
            pts_cam = np.stack([xs, ys, zs, np.ones_like(xs)], axis=1)  # N x 4

            # 变换到 world
            pts_world = (pose @ pts_cam.T).T[:, :3]  # N x 3

            # 计算方向向量从球心指向点
            dirs = pts_world - center[None, :]
            norms = np.linalg.norm(dirs, axis=1)
            valid = norms > 1e-6
            if np.count_nonzero(valid) == 0:
                continue
            dirs = dirs[valid]
            norms = norms[valid]
            pts_world = pts_world[valid]
            us_idx = us[valid].astype(np.int32)
            vs_idx = vs[valid].astype(np.int32)

            # 球面坐标（lon, lat）
            # lon: [-pi, pi], lat: [-pi/2, pi/2]
            dx = dirs[:, 0]
            dy = dirs[:, 1]
            dz = dirs[:, 2]
            lon = np.arctan2(dx, dz)  # 注意坐标系选择（x 沿右，z 沿前）
            lat = np.arcsin(np.clip(dy / norms, -1.0, 1.0))

            u = ((lon + np.pi) / (2 * np.pi) * pano_w).astype(np.int32) % pano_w
            v = ((np.pi / 2 - lat) / np.pi * pano_h).astype(np.int32)
            v = np.clip(v, 0, pano_h - 1)

            colors = color_bgr[vs_idx, us_idx][:, ::-1].astype(np.float32) / 255.0  # to RGB float

            # 累加（简单平均）——也可以用距离权重/视角权重
            for i in range(len(u)):
                ui = u[i]; vi = v[i]
                acc_rgb[vi, ui, :] += colors[i]
                acc_weight[vi, ui] += 1.0

            if (idx + 1) % 10 == 0:
                print(f"处理帧 {idx+1}/{len(self.saved_frames)} ...")

        # 归一化
        mask_nonzero = acc_weight > 0
        pano = np.zeros_like(acc_rgb, dtype=np.uint8)
        pano[mask_nonzero] = (acc_rgb[mask_nonzero] / acc_weight[mask_nonzero, None] * 255.0).astype(np.uint8)
        pano_bgr = pano[:, :, ::-1]  # RGB->BGR for saving

        # 对空白区域进行简单 inpaint 填充（先生成掩码）
        if fill_inpaint:
            hole_mask = (acc_weight == 0).astype(np.uint8) * 255
            # OpenCV inpaint 要求 8-bit 3ch 输入 & 非零为修补区域
            if np.any(hole_mask):
                try:
                    pano_bgr = cv2.inpaint(pano_bgr, hole_mask, 3, cv2.INPAINT_TELEA)
                except Exception as e:
                    print("inpaint 失败:", e)

        # 保存结果
        cv2.imwrite(self.output_panorama_path, pano_bgr)
        print(f"全景已保存：{self.output_panorama_path}")
        return True

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

            # 退出前保存一次 final global（点云）
            with self.global_lock:
                if self.global_pcd is not None and len(self.global_pcd.points) > 0:
                    try:
                        o3d.io.write_point_cloud(self.output_global_path, self.global_pcd, write_ascii=False)
                        print(f"退出前已保存最终 global 点云：{self.output_global_path} (pts={len(self.global_pcd.points)})")
                    except Exception as e:
                        print("退出时保存 global 失败:", e)

            # 生成全景（使用累计的 accepted frames）
            try:
                self.generate_panorama(pano_w=4096, pano_h=2048, fill_inpaint=True)
            except Exception as e:
                print("生成全景失败:", e)

            print("已安全退出。")

if __name__ == "__main__":
    mapper = RobustPanoramaMapper(
        width=640, height=480,
        fx=570.3, fy=570.3, cx=319.5, cy=239.5,
        voxel_size=0.03,
        icp_threshold=0.12,
        save_interval_sec=5.0,
        stable_save_delay=8.0,
        max_queue_size=6
    )
    mapper.run()