import threading
import queue
import numpy as np
import cv2
import g2o
from concurrent.futures import ThreadPoolExecutor
from common import refine_icp


# ── 轻量 BoW ──────────────────────────────────
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
        self.vocab      = centers.astype(np.uint8)
        self.is_trained = True
        return True

    def encode(self, descs):
        if not self.is_trained or descs is None or len(descs) == 0:
            return None
        dists = np.sum((descs.astype(np.float32)[:,None,:] -
                        self.vocab.astype(np.float32)[None,:,:])**2, axis=-1)
        words = np.argmin(dists, axis=-1)
        vec   = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
        norm  = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def similarity(self, v1, v2):
        if v1 is None or v2 is None:
            return 0.0
        return float(np.dot(v1, v2))


# ── 位姿图优化器 ──────────────────────────────
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
        with self.lock:
            self.optimizer.add_vertex(v)

    def add_edge(self, id_from, id_to, T_rel, info=None, robust_kernel=False):
        e = g2o.EdgeSE3()
        e.set_vertex(0, self.optimizer.vertex(id_from))
        e.set_vertex(1, self.optimizer.vertex(id_to))
        e.set_measurement(g2o.Isometry3d(T_rel[:3,:3], T_rel[:3,3]))
        if info is None:
            info = np.eye(6) * 500.0
            info[3:, 3:] *= 10.0
        e.set_information(info)
        if robust_kernel:
            e.set_robust_kernel(g2o.RobustKernelHuber(0.1))
        with self.lock:
            self.optimizer.add_edge(e)

    def optimize(self, iterations=20):
        with self.lock:
            self.optimizer.initialize_optimization()
            self.optimizer.optimize(iterations)

    def get_pose(self, kf_id):
        with self.lock:
            v = self.optimizer.vertex(kf_id)
            if v is None:
                return None
            iso = v.estimate()
            T = np.eye(4)
            T[:3,:3] = iso.rotation().matrix()
            T[:3, 3] = iso.translation()
            return T


# ── 回环检测 ──────────────────────────────────
class LoopDetector:
    MIN_GAP    = 20
    BOW_THRESH = 0.15

    def __init__(self, bow, frontend):
        self.bow      = bow
        self.frontend = frontend

    def detect(self, cur_kf, all_keyframes):
        if not self.bow.is_trained:
            return None
        if cur_kf.bow_vec is None:
            cur_kf.bow_vec = self.bow.encode(cur_kf.descs)
        if cur_kf.bow_vec is None:
            return None

        candidates = []
        for kf in all_keyframes:
            if cur_kf.kf_id - kf.kf_id < self.MIN_GAP:
                continue
            if kf.bow_vec is None:
                kf.bow_vec = self.bow.encode(kf.descs)
            score = self.bow.similarity(cur_kf.bow_vec, kf.bow_vec)
            if score > self.BOW_THRESH:
                candidates.append((score, kf))
        if not candidates:
            return None

        best_score, best_kf = max(candidates, key=lambda x: x[0])
        matches = self.frontend.match(best_kf.descs, cur_kf.descs)
        if len(matches) < 15:
            return None
        ok, T_pnp = self.frontend.pnp(
            best_kf.kps, cur_kf.kps, matches, best_kf.depth_m)
        if not ok:
            return None
        T_icp, icp_ok, fitness = refine_icp(
            best_kf.get_pcd(stride=3), cur_kf.get_pcd(stride=3),
            T_pnp, max_dist=0.05, max_iter=50)
        if not icp_ok or fitness < 0.4:
            return None

        print(f"[Loop] KF{cur_kf.kf_id} ↔ KF{best_kf.kf_id}  "
              f"BoW:{best_score:.2f}  ICP:{fitness:.2f}")
        return best_kf, T_icp


# ── 后端线程 ──────────────────────────────────
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
        self._executor     = ThreadPoolExecutor(max_workers=2)

    def add_keyframe(self, kf, T_rel, prev_kf_id):
        self.kf_queue.put((kf, T_rel, prev_kf_id))

    def run(self):
        while not self.stop_event.is_set():
            try:
                kf, T_rel, prev_kf_id = self.kf_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.keyframes.append(kf)

            # BoW 词典训练
            if not self._bow_trained and len(self.keyframes) >= self.bow.min_train:
                descs_list = [k.descs for k in self.keyframes if k.descs is not None]
                if self.bow.train(descs_list):
                    self._bow_trained = True
                    print(f"[BoW] 词典训练完成，词汇量:{self.bow.vocab_size}")
                    futures = {self._executor.submit(self.bow.encode, k.descs): k
                               for k in self.keyframes if k.descs is not None}
                    for fut, k in futures.items():
                        k.bow_vec = fut.result()

            # 回环检测 + 全局优化
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
                        if new_pose is not None:
                            k.pose = new_pose
                    self.map_builder.rebuild(self.keyframes)
                    print("[Backend] 全局优化完成，地图已更新")

    def stop(self):
        self.stop_event.set()
        self._executor.shutdown(wait=False)
