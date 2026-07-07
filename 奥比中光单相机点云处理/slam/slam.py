import cv2
import numpy as np
import open3d as o3d
import threading
import queue
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from common import K, is_motion_valid
from frontend import FrontEnd
from map_builder import MapBuilder
from backend import BagOfWords, PoseGraphOptimizer, LoopDetector, BackEnd
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path


class SLAM2:
    KF_TRANS_THRESH = 0.10
    KF_ROT_THRESH   = 0.15

    def __init__(self):
        self.frontend    = FrontEnd()
        self.map_builder = MapBuilder(voxel_size=0.02)
        self.bow         = BagOfWords(vocab_size=256)
        self.pose_graph  = PoseGraphOptimizer()
        self.loop_det    = LoopDetector(self.bow, self.frontend)
        self.backend     = BackEnd(self.pose_graph, self.bow,
                                   self.loop_det, self.map_builder)

        self.cur_pose    = np.eye(4)
        self.prev_kf     = None
        self.kf_id       = 0
        self.frame_id    = 0
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event  = threading.Event()
        self._executor   = ThreadPoolExecutor(max_workers=2)

        self.sdk     = None   # OrbbecCameraSDK 实例
        self.cv_cam  = None   # 回退用

    # ── 初始化相机 ────────────────────────────
    def init(self):
        try:
            sdk = OrbbecCameraSDK(get_default_sdk_path())
            if not sdk.initialize():
                raise RuntimeError("oniInitialize 失败")
            devices = sdk.get_device_list()
            if not devices:
                raise RuntimeError("未检测到设备")
            if not sdk.open_device(device_index=0):
                raise RuntimeError("open_device 失败")
            if not sdk.create_stream(OrbbecCameraSDK.ONI_SENSOR_DEPTH):
                raise RuntimeError("创建深度流失败")
            if not sdk.start_stream():  # 只启动深度流
                self.sdk.set_registration_mode(1)  # 开启 DEPTH_TO_COLOR 对齐
                raise RuntimeError("启动深度流失败")
            self.sdk = sdk

            # 彩色用 OpenCV 摄像头，和 color_3d_reconstructor.py 一样
            self.cv_cam = cv2.VideoCapture(0)
            if not self.cv_cam.isOpened():
                for idx in [1, 2, 3]:
                    self.cv_cam = cv2.VideoCapture(idx)
                    if self.cv_cam.isOpened():
                        break
            if self.cv_cam.isOpened():
                self.cv_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cv_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cv_cam.set(cv2.CAP_PROP_FPS, 30)
                print("[SLAM2] RGB摄像头就绪")
            else:
                print("[SLAM2] 未找到RGB摄像头，将用深度图代替彩色")
                self.cv_cam = None

            print("[SLAM2] Orbbec SDK 已启动")

        except Exception as e:
            print(f"[SLAM2] SDK 不可用({e})，全部回退到 cv2.VideoCapture")
            if self.sdk:
                self.sdk.cleanup()
                self.sdk = None
            self.cv_cam = cv2.VideoCapture(0)

    # ── 取一帧 ────────────────────────────────
    def _grab_frame(self):
        if self.sdk:
            result = self.sdk.capture_depth_frame(timeout=1000)
            if result is None:
                return None, None
            depth_arr, _ = result
            depth_m = depth_arr.astype(np.float64) * 0.001
            depth_m = np.fliplr(depth_m)  # ← 强制水平翻转，对齐 RGB 坐标系
            valid_pixels = np.sum((depth_m > 0.3) & (depth_m < 8.0))
            print(f"[Depth] valid_pixels={valid_pixels} max={depth_m.max():.2f}m")

            # 彩色从 OpenCV 读
            if self.cv_cam and self.cv_cam.isOpened():
                ret, color = self.cv_cam.read()
                if ret and color is not None:
                    if color.shape[:2] != (480, 640):
                        color = cv2.resize(color, (640, 480))
                    return color, depth_m

            # 没有彩色摄像头就用深度图伪造
            depth_vis = (depth_arr / depth_arr.max() * 255).astype(np.uint8) if depth_arr.max() > 0 else np.zeros(
                (480, 640), dtype=np.uint8)
            return cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR), depth_m

        else:
            ret, color = self.cv_cam.read()
            if not ret:
                return None, None
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            return color, gray.astype(np.float64) / 255.0 * 3.0

        # 临时加到 _grab_frame() 末尾，运行一次就够了
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        axes[0].set_title('RGB')
        axes[1].imshow(depth_m, cmap='jet', vmin=0.3, vmax=3.0)
        axes[1].set_title('Depth (fliplr)')
        # 叠加：把深度图 resize 到 RGB 尺寸后叠加
        depth_vis = (np.clip(depth_m, 0, 3) / 3 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color, 0.5, depth_color, 0.5, 0)
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay')
        plt.savefig('depth_rgb_check.png', dpi=150)
        plt.close()
        print("[Debug] 已保存 depth_rgb_check.png")

    # ── 是否需要新关键帧 ──────────────────────
    def _need_keyframe(self, T_rel):
        t = np.linalg.norm(T_rel[:3, 3])
        r = np.arccos(np.clip((np.trace(T_rel[:3,:3])-1)/2, -1, 1))
        return t > self.KF_TRANS_THRESH or r > self.KF_ROT_THRESH

    # ── SLAM 主循环（子线程）─────────────────
    def _slam_loop(self):
        from common import KeyFrame
        while not self.stop_event.is_set():
            color, depth_m = self._grab_frame()
            if color is None:
                time.sleep(0.01)
                continue

            gray       = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            kps, descs = self.frontend.detect(gray)
            self.frame_id += 1

            if self.prev_kf is None:
                kf = KeyFrame(self.kf_id, self.cur_pose,
                              depth_m, color, kps, descs)
                self.pose_graph.add_vertex(self.kf_id, self.cur_pose, fixed=True)
                self.map_builder.add_keyframe(kf)
                self.prev_kf = kf
                self.kf_id  += 1
                continue

            matches = self.frontend.match(self.prev_kf.descs, descs)
            ok, T_rel = self.frontend.pnp(
                self.prev_kf.kps, kps, matches, self.prev_kf.depth_m)

            if ok and is_motion_valid(T_rel):
                self.cur_pose = self.prev_kf.pose @ T_rel
            else:
                T_rel = np.eye(4)

            disp = color.copy()
            for kp in kps:
                cv2.circle(disp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0,255,0), -1)
            status = "OK" if ok else "LOST"
            cv2.putText(disp, f"F:{self.frame_id} KF:{self.kf_id} [{status}]",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            if self._need_keyframe(T_rel):
                kf = KeyFrame(self.kf_id, self.cur_pose,
                              depth_m, color, kps, descs)
                self.pose_graph.add_vertex(self.kf_id, self.cur_pose)
                self.pose_graph.add_edge(self.prev_kf.kf_id, self.kf_id, T_rel)
                self.map_builder.add_keyframe(kf)
                self.backend.add_keyframe(kf, T_rel, self.prev_kf.kf_id)
                self.prev_kf = kf
                self.kf_id  += 1

            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put(disp)

    # ── 主线程：Open3D + cv2 显示 ─────────────
    def run(self):
        self.backend.start()

        vis = o3d.visualization.Visualizer()
        vis.create_window("SLAM2 Map", width=1280, height=800)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size       = 1.5
        opt.background_color = np.array([0.05, 0.05, 0.05])

        t_slam = threading.Thread(target=self._slam_loop, daemon=True)
        t_slam.start()

        print("[SLAM2] 运行中  |  's' 保存  |  'q' 退出")
        last_vis_update = time.time()

        try:
            while True:
                if not vis.poll_events():
                    break
                vis.update_renderer()

                now = time.time()
                if self.map_builder.is_dirty() or now - last_vis_update > 0.5:
                    snap = self.map_builder.get_snapshot()
                    pcd.points = snap.points
                    pcd.colors = snap.colors
                    vis.update_geometry(pcd)
                    last_vis_update = now

                try:
                    disp = self.frame_queue.get_nowait()
                    cv2.imshow("SLAM2 Front", disp)
                except queue.Empty:
                    pass

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.map_builder.save(f"slam2_map_{ts}.pcd")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self.backend.stop()
            self.backend.join(timeout=3)
            self._executor.shutdown(wait=False)
            vis.destroy_window()
            cv2.destroyAllWindows()
            if self.sdk:
                self.sdk.cleanup()
            if self.cv_cam:
                self.cv_cam.release()
            print("[SLAM2] 已退出")


if __name__ == "__main__":
    slam = SLAM2()
    slam.init()
    slam.run()