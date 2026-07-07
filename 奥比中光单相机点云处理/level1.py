"""
Level 2 RGB-D SLAM
前端: ORB特征 + PnP + ICP精对齐
后端: g2o位姿图优化 (VertexSE3 + EdgeSE3)
回环: ORB词袋 + ICP验证
注意: Open3D Visualizer 必须在主线程运行
"""

import cv2
import numpy as np
import open3d as o3d
import g2o
import threading
import queue
import time
import os
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

# ─────────────────────────────────────────────
#  相机参数
# ─────────────────────────────────────────────
FX, FY        = 525.0, 525.0
CX, CY        = 319.5, 239.5
WIDTH, HEIGHT = 640, 480
K             = np.array([[FX,0,CX],[0,FY,CY],[0,0,1]], dtype=np.float64)
DEPTH_SCALE   = 0.001
MIN_DEPTH, MAX_DEPTH = 0.3, 3.0

# ─────────────────────────────────────────────
#  关键帧
# ─────────────────────────────────────────────
class KeyFrame:
    def __init__(self, kf_id, pose, depth_m, color_bgr, kps, descs):
        self.kf_id     = kf_id
        self.pose      = pose.copy()
        self.depth_m   = depth_m
        self.color_bgr = color_bgr
        self.kps       = kps
        self.descs     = descs
        self.bow_vec   = None

    def get_pcd(self, stride=3):
        rows, cols = self.depth_m.shape
        u, v = np.meshgrid(np.arange(0,cols,stride), np.arange(0,rows,stride))
        d    = self.depth_m[::stride, ::stride]
        valid = (d > MIN_DEPTH) & (d < MAX_DEPTH)
        z = d[valid]; x = (u[valid]-CX)*z/FX; y = (v[valid]-CY)*z/FY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack([x,y,z], axis=-1))
        return pcd


# ─────────────────────────────────────────────
#  轻量 BoW
# ─────────────────────────────────────────────
class BagOfWords:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab      = None
        self.is_trained = False
        self.min_train  = 20

    def train(self, all_descs_list):
        all_descs = np.vstack(all_descs_list).astype(np.float32)
        if len(all_descs) < self.vocab_size:
            return False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, _, centers = cv2.kmeans(all_descs, self.vocab_size, None, criteria,
                                   attempts=3, flags=cv2.KMEANS_PP_CENTERS)
        self.vocab = centers.astype(np.uint8); self.is_trained = True
        return True

    def encode(self, descs):
        if not self.is_trained or descs is None or len(descs) == 0:
            return None
        dists = np.sum((descs.astype(np.float32)[:,None,:] -
                        self.vocab.astype(np.float32)[None,:,:])**2, axis=-1)
        words = np.argmin(dists, axis=-1)
        vec   = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
        norm  = np.linalg.norm(vec)
        return vec/norm if norm > 0 else vec

    def similarity(self, v1, v2):
        if v1 is None or v2 is None: return 0.0
        return float(np.dot(v1, v2))


# ─────────────────────────────────────────────
#  g2o 位姿图优化器
# ─────────────────────────────────────────────
class PoseGraphOptimizer:
    def __init__(self):
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        self.optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))
        self.optimizer.set_verbose(False)
        self.lock = threading.Lock()

    def add_vertex(self, kf_id, pose_4x4, fixed=False):
        v = g2o.VertexSE3()
        v.set_id(kf_id)
        v.set_estimate(g2o.Isometry3d(pose_4x4[:3,:3], pose_4x4[:3,3]))
        v.set_fixed(fixed)
        with self.lock: self.optimizer.add_vertex(v)

    def add_edge(self, id_from, id_to, T_rel, info=None, robust_kernel=False):
        e = g2o.EdgeSE3()
        e.set_vertex(0, self.optimizer.vertex(id_from))
        e.set_vertex(1, self.optimizer.vertex(id_to))
        e.set_measurement(g2o.Isometry3d(T_rel[:3,:3], T_rel[:3,3]))
        if info is None:
            info = np.eye(6)*500.0; info[3:,3:] *= 10.0
        e.set_information(info)
        if robust_kernel: e.set_robust_kernel(g2o.RobustKernelHuber(0.1))
        with self.lock: self.optimizer.add_edge(e)

    def optimize(self, iterations=20):
        with self.lock:
            self.optimizer.initialize_optimization()
            self.optimizer.optimize(iterations)

    def get_pose(self, kf_id):
        with self.lock:
            v = self.optimizer.vertex(kf_id)
            if v is None: return None
            iso = v.estimate()
            T = np.eye(4)
            T[:3,:3] = iso.rotation().matrix()
            T[:3, 3] = iso.translation()
            return T


# ─────────────────────────────────────────────
#  ICP 精对齐
# ─────────────────────────────────────────────
def refine_icp(pcd_ref, pcd_cur, T_init, max_dist=0.05, max_iter=30):
    for pcd in [pcd_ref, pcd_cur]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20))
    result = o3d.pipelines.registration.registration_icp(
        pcd_cur, pcd_ref, max_dist, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return result.transformation, result.fitness > 0.3, result.fitness


# ─────────────────────────────────────────────
#  前端
# ─────────────────────────────────────────────
class FrontEnd:
    def __init__(self):
        self.orb     = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, gray):
        return self.orb.detectAndCompute(gray, None)

    def match(self, d1, d2):
        if d1 is None or d2 is None or len(d1)<2 or len(d2)<2: return []
        pairs = self.matcher.knnMatch(d1, d2, k=2)
        return [m for p in pairs if len(p)==2
                for m,n in [p] if m.distance < 0.75*n.distance]

    def pnp(self, kps_ref, kps_cur, matches, depth_ref):
        if len(matches) < 8: return False, None
        pts3d, pts2d = [], []
        for m in matches:
            u, v = kps_ref[m.queryIdx].pt
            ui, vi = int(round(u)), int(round(v))
            if not (0<=vi<HEIGHT and 0<=ui<WIDTH): continue
            z = depth_ref[vi, ui]
            if z < MIN_DEPTH or z > MAX_DEPTH: continue
            pts3d.append([(u-CX)*z/FX, (v-CY)*z/FY, z])
            pts2d.append(kps_cur[m.trainIdx].pt)
        if len(pts3d) < 6: return False, None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d, dtype=np.float64),
            np.array(pts2d, dtype=np.float64),
            K, None, iterationsCount=100, reprojectionError=2.0, confidence=0.99)
        if not ok or inliers is None or len(inliers) < 6: return False, None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
        return True, T


# ─────────────────────────────────────────────
#  回环检测
# ─────────────────────────────────────────────
class LoopDetector:
    MIN_GAP    = 20
    BOW_THRESH = 0.15

    def __init__(self, bow, frontend):
        self.bow = bow; self.frontend = frontend

    def detect(self, cur_kf, all_keyframes):
        if not self.bow.is_trained: return None
        if cur_kf.bow_vec is None:
            cur_kf.bow_vec = self.bow.encode(cur_kf.descs)
        if cur_kf.bow_vec is None: return None

        candidates = []
        for kf in all_keyframes:
            if cur_kf.kf_id - kf.kf_id < self.MIN_GAP: continue
            if kf.bow_vec is None: kf.bow_vec = self.bow.encode(kf.descs)
            score = self.bow.similarity(cur_kf.bow_vec, kf.bow_vec)
            if score > self.BOW_THRESH: candidates.append((score, kf))
        if not candidates: return None

        best_score, best_kf = max(candidates, key=lambda x: x[0])
        matches = self.frontend.match(best_kf.descs, cur_kf.descs)
        if len(matches) < 15: return None
        ok, T_pnp = self.frontend.pnp(best_kf.kps, cur_kf.kps, matches, best_kf.depth_m)
        if not ok: return None
        T_icp, icp_ok, fitness = refine_icp(best_kf.get_pcd(), cur_kf.get_pcd(), T_pnp)
        if not icp_ok or fitness < 0.4: return None

        print(f"[Loop] KF{cur_kf.kf_id} ↔ KF{best_kf.kf_id}  "
              f"BoW:{best_score:.2f}  ICP:{fitness:.2f}")
        return best_kf, T_icp


# ─────────────────────────────────────────────
#  点云地图
# ─────────────────────────────────────────────
class MapBuilder:
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size
        self.global_pcd = o3d.geometry.PointCloud()
        self.lock       = threading.Lock()
        self._dirty     = False

    def add_keyframe(self, kf):
        pcd = kf.get_pcd(stride=2)
        if len(pcd.points) == 0: return
        rgb    = cv2.cvtColor(kf.color_bgr, cv2.COLOR_BGR2RGB)
        colors = rgb[::2,::2].reshape(-1,3).astype(np.float64)/255.0
        n = len(pcd.points)
        if len(colors) > n:   colors = colors[:n]
        elif len(colors) < n: colors = np.vstack([colors, np.zeros((n-len(colors),3))])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform(kf.pose)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        with self.lock:
            self.global_pcd += pcd
            self._dirty = True

    def rebuild(self, keyframes):
        new_pcd = o3d.geometry.PointCloud()
        for kf in keyframes:
            pcd = kf.get_pcd(stride=3)
            if len(pcd.points) == 0: continue
            pcd.transform(kf.pose)
            new_pcd += pcd
        new_pcd = new_pcd.voxel_down_sample(self.voxel_size)
        with self.lock:
            self.global_pcd = new_pcd
            self._dirty     = True
        print(f"[Map] 重建完成: {len(new_pcd.points)} 点")

    def get_snapshot(self):
        with self.lock:
            self._dirty = False
            snap = o3d.geometry.PointCloud()
            snap.points = o3d.utility.Vector3dVector(
                np.asarray(self.global_pcd.points).copy())
            snap.colors = o3d.utility.Vector3dVector(
                np.asarray(self.global_pcd.colors).copy())
            return snap

    def is_dirty(self): return self._dirty

    def save(self, path):
        with self.lock:
            o3d.io.write_point_cloud(path, self.global_pcd)
        print(f"[Map] 已保存: {path}  ({len(self.global_pcd.points)} 点)")


# ─────────────────────────────────────────────
#  后端线程
# ─────────────────────────────────────────────
class BackEnd(threading.Thread):
    def __init__(self, pose_graph, bow, loop_detector, map_builder):
        super().__init__(daemon=True)
        self.pose_graph    = pose_graph
        self.bow           = bow
        self.loop_detector = loop_detector
        self.map_builder   = map_builder
        self.kf_queue      = queue.Queue()
        self.keyframes     = []
        self.stop_event    = threading.Event()
        self._bow_trained  = False

    def add_keyframe(self, kf, T_rel, prev_kf_id):
        self.kf_queue.put((kf, T_rel, prev_kf_id))

    def run(self):
        while not self.stop_event.is_set():
            try:
                kf, T_rel, prev_kf_id = self.kf_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.keyframes.append(kf)
            if not self._bow_trained and len(self.keyframes) >= self.bow.min_train:
                descs_list = [k.descs for k in self.keyframes if k.descs is not None]
                if self.bow.train(descs_list):
                    self._bow_trained = True
                    print(f"[BoW] 词典训练完成，词汇量:{self.bow.vocab_size}")
            if self._bow_trained and len(self.keyframes) > self.loop_detector.MIN_GAP:
                result = self.loop_detector.detect(
                    kf, self.keyframes[:-self.loop_detector.MIN_GAP])
                if result is not None:
                    loop_kf, T_loop = result
                    self.pose_graph.add_edge(
                        kf.kf_id, loop_kf.kf_id, T_loop, robust_kernel=True)
                    self.pose_graph.optimize(iterations=20)
                    for k in self.keyframes:
                        new_pose = self.pose_graph.get_pose(k.kf_id)
                        if new_pose is not None: k.pose = new_pose
                    self.map_builder.rebuild(self.keyframes)
                    print("[Backend] 全局优化完成，地图已更新")

    def stop(self): self.stop_event.set()


# ─────────────────────────────────────────────
#  主 SLAM 系统
# ─────────────────────────────────────────────
class SLAM2:
    KF_TRANS_THRESH = 0.15
    KF_ROT_THRESH   = 0.17

    def __init__(self):
        self.sdk         = OrbbecCameraSDK(get_default_sdk_path())
        self.cv_cam      = None
        self.frontend    = FrontEnd()
        self.map_builder = MapBuilder(voxel_size=0.02)
        self.pose_graph  = PoseGraphOptimizer()
        self.bow         = BagOfWords(vocab_size=256)
        self.loop_det    = LoopDetector(self.bow, self.frontend)
        self.backend     = BackEnd(
            self.pose_graph, self.bow, self.loop_det, self.map_builder)

        self.cur_pose   = np.eye(4)
        self.ref_kf     = None
        self.kf_id      = 0
        self.frame_id   = 0
        self.lost_count = 0

        self.stop_event  = threading.Event()
        self.frame_queue = queue.Queue(maxsize=2)

        # 主线程可视化状态
        self._vis_pcd    = None
        self._vis        = None

    # ── 初始化 ────────────────────────────────
    def init(self):
        print("[SLAM2] 初始化相机...")
        if not self.sdk.initialize():   raise RuntimeError("SDK 初始化失败")
        if not self.sdk.open_device():  raise RuntimeError("打开设备失败")
        if not self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH):
            raise RuntimeError("创建深度流失败")
        if not self.sdk.start_stream(): raise RuntimeError("启动深度流失败")

        for idx in range(4):
            cam = cv2.VideoCapture(idx)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cam.set(cv2.CAP_PROP_FPS, 30)
                ret, _ = cam.read()
                if ret:
                    self.cv_cam = cam
                    print(f"[SLAM2] RGB 摄像头索引: {idx}")
                    break
                cam.release()
        if self.cv_cam is None:
            raise RuntimeError("未找到 RGB 摄像头")
        print("[SLAM2] 初始化完成")

    # ── 关键帧判断 ────────────────────────────
    def _is_keyframe(self, T_rel):
        t = np.linalg.norm(T_rel[:3,3])
        r = np.arccos(np.clip((np.trace(T_rel[:3,:3])-1)/2, -1, 1))
        return t > self.KF_TRANS_THRESH or r > self.KF_ROT_THRESH

    # ── 处理一帧 ─────────────────────────────
    def process_frame(self, depth_m, color_bgr):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self.frontend.detect(gray)
        disp = color_bgr.copy()

        if self.ref_kf is None:
            kf = KeyFrame(self.kf_id, self.cur_pose,
                          depth_m, color_bgr, kps, descs)
            self._register_keyframe(kf, np.eye(4), -1)
            self.map_builder.add_keyframe(kf)
            self.ref_kf = kf
            print("[KF 0] 初始化")
            return disp

        matches = self.frontend.match(self.ref_kf.descs, descs)
        ok, T_pnp = self.frontend.pnp(
            self.ref_kf.kps, kps, matches, self.ref_kf.depth_m)

        T_rel, status = None, ""
        cur_pcd = KeyFrame(0, np.eye(4), depth_m, color_bgr, kps, descs).get_pcd()

        if ok and len(cur_pcd.points) > 100:
            T_icp, icp_ok, fitness = refine_icp(self.ref_kf.get_pcd(), cur_pcd, T_pnp)
            if icp_ok: T_rel, status = T_icp, f"PnP+ICP fit:{fitness:.2f}"
            else:      T_rel, status = T_pnp, "PnP only"
        elif ok:
            T_rel, status = T_pnp, "PnP only"

        if T_rel is None:
            self.lost_count += 1
            if self.lost_count >= 5:
                print("[SLAM2] 跟踪丢失，重置参考帧")
                kf = KeyFrame(self.kf_id, self.cur_pose,
                              depth_m, color_bgr, kps, descs)
                self._register_keyframe(kf, np.eye(4), self.ref_kf.kf_id)
                self.ref_kf = kf
                self.lost_count = 0
            self.frame_id += 1
            return disp

        self.lost_count = 0
        self.cur_pose   = self.ref_kf.pose @ np.linalg.inv(T_rel)

        if self._is_keyframe(T_rel):
            kf = KeyFrame(self.kf_id, self.cur_pose,
                          depth_m, color_bgr, kps, descs)
            self._register_keyframe(kf, T_rel, self.ref_kf.kf_id)
            self.map_builder.add_keyframe(kf)
            self.ref_kf = kf
            print(f"[KF {self.kf_id-1}] {status}  "
                  f"t:{np.linalg.norm(T_rel[:3,3])*100:.1f}cm  "
                  f"地图:{len(self.map_builder.global_pcd.points)}pts")

        for m in matches[:40]:
            cv2.circle(disp, tuple(map(int, kps[m.trainIdx].pt)), 3, (0,255,0), -1)

        self.frame_id += 1
        return disp

    def _register_keyframe(self, kf, T_rel, prev_id):
        self.pose_graph.add_vertex(kf.kf_id, kf.pose, fixed=(self.kf_id==0))
        if prev_id >= 0:
            self.pose_graph.add_edge(prev_id, kf.kf_id, T_rel)
        self.backend.add_keyframe(kf, T_rel, prev_id)
        self.kf_id += 1

    # ── 采集 + 处理线程（子线程）────────────────
    def _slam_loop(self):
        while not self.stop_event.is_set():
            depth_result, color_result = [None], [None]

            def get_depth():
                r = self.sdk.capture_depth_frame(timeout=500)
                if r:
                    d = r[0].astype(np.float32) * DEPTH_SCALE
                    d[(d < MIN_DEPTH) | (d > MAX_DEPTH)] = 0.0
                    depth_result[0] = d

            def get_color():
                ret, frame = self.cv_cam.read()
                if ret and frame is not None:
                    if frame.shape[:2] != (HEIGHT, WIDTH):
                        frame = cv2.resize(frame, (WIDTH, HEIGHT))
                    color_result[0] = frame

            t_d = threading.Thread(target=get_depth)
            t_c = threading.Thread(target=get_color)
            t_d.start(); t_c.start()
            t_d.join();  t_c.join()

            if depth_result[0] is None or color_result[0] is None:
                continue

            disp = self.process_frame(depth_result[0], color_result[0])

            # HUD
            bow_status = ("BoW:ready" if self.bow.is_trained
                          else f"BoW:collecting({self.kf_id}/{self.bow.min_train})")
            cv2.putText(disp, f"Frame:{self.frame_id}  KF:{self.kf_id}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.putText(disp,
                        f"t:[{self.cur_pose[0,3]:.2f},"
                        f"{self.cur_pose[1,3]:.2f},"
                        f"{self.cur_pose[2,3]:.2f}]",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
            cv2.putText(disp, bow_status,
                        (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

            # 把显示帧放进队列给主线程 imshow
            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put(disp)

    # ── 主线程：Open3D 可视化 + cv2.imshow ────
    def run(self):
        self.backend.start()

        # Open3D 窗口必须在主线程创建
        vis = o3d.visualization.Visualizer()
        vis.create_window("SLAM2 Map", width=1280, height=800)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size       = 1.5
        opt.background_color = np.array([0.05, 0.05, 0.05])

        # SLAM 处理放子线程
        t_slam = threading.Thread(target=self._slam_loop, daemon=True)
        t_slam.start()

        print("[SLAM2] 运行中  |  's' 保存  |  'q' 退出")
        last_vis_update = time.time()

        try:
            while True:
                # ── Open3D 事件循环（必须主线程）──
                if not vis.poll_events():
                    break
                vis.update_renderer()

                # 每 0.5s 刷新一次点云
                now = time.time()
                if self.map_builder.is_dirty() or now - last_vis_update > 0.5:
                    snap = self.map_builder.get_snapshot()
                    pcd.points = snap.points
                    pcd.colors = snap.colors
                    vis.update_geometry(pcd)
                    last_vis_update = now

                # ── cv2 前端画面 ──
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
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.sdk.stop_stream()
            self.sdk.close_device()
            if self.cv_cam: self.cv_cam.release()
            print("[SLAM2] 已退出")


# ─────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    slam = SLAM2()
    slam.init()
    slam.run()