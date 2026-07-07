"""
DMTet Point Cloud → Mesh Reconstruction (v7.1)
==============================================
修复内容：
  v7:   精确 GT SDF（预计算 + trimesh）
  v7.1: 修复 compute_gt_sdf_from_points 中 tensor 维度错误
"""

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import trimesh

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip install tqdm")
    from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─── 检测 torch.compile ─────────────────────────────────────

def check_torch_compile_available():
    if not hasattr(torch, 'compile'):
        return False, "PyTorch < 2.0"
    if not torch.cuda.is_available():
        return False, "无 CUDA"
    if sys.platform == 'win32':
        try:
            import triton
            return True, "Triton 可用"
        except ImportError:
            return False, "Windows 无 Triton"
    try:
        import triton
        return True, "Triton 可用"
    except ImportError:
        return False, "未安装 Triton"

TORCH_COMPILE_OK, COMPILE_REASON = check_torch_compile_available()


# ─── 自动检测显存 ─────────────────────────────────────────────

def auto_detect_vram():
    if not torch.cuda.is_available():
        return 8, 8192, 2048
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 16:
        return 32, 16384, 4096
    elif total_gb >= 10:
        return 16, 16384, 2048
    elif total_gb >= 7:
        return 12, 8192, 1024
    elif total_gb >= 5:
        return 8, 4096, 1024
    else:
        return 4, 2048, 512

AUTO_BATCH, AUTO_CHUNK, AUTO_CD_CHUNK = auto_detect_vram()


# ─── 配置 ────────────────────────────────────────────────────

class Config:
    PCD_DIR         = "dataset/pointcloud"
    STL_DIR         = "dataset/stl"
    SAVE_DIR        = "checkpoints_mesh"
    NUM_INPUT_PTS   = 4096
    NUM_GT_PTS      = 4096
    TET_GRID_RES    = 48
    BATCH_SIZE      = AUTO_BATCH
    EPOCHS          = 200
    LR              = 5e-4
    LR_DECAY_STEP   = 80
    LR_DECAY_RATE   = 0.5
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_INTERVAL   = 20
    CHUNK_SIZE      = AUTO_CHUNK
    USE_AMP         = True
    NUM_WORKERS     = 0

    W_CHAMFER       = 3.0
    W_NORMAL        = 0.05
    W_SDF_REG       = 0.01
    W_SDF_GT        = 1.0
    W_LAP_SDF       = 0.1
    W_DEFORM_REG    = 0.3
    W_SIGN_CLARITY  = 0.02

    PHASE1_EPOCHS   = 60

    PRECOMPUTE_SDF  = True
    CD_SUBSAMPLE    = 4096
    CD_CHUNK        = AUTO_CD_CHUNK
    PIN_MEMORY      = True
    PREFETCH_FACTOR = 2

    SURFACE_BAND    = 0.15

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def get_gpu_info():
    if not torch.cuda.is_available():
        return "CPU mode"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    return f"GPU: {allocated:.1f}/{total:.1f}GB (res:{reserved:.1f}GB)"


def print_system_info():
    print("=" * 60)
    print("系统信息:")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存: {total:.1f} GB")
    else:
        print("  ⚠️  未检测到 GPU")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"  torch.compile: {'✅' if TORCH_COMPILE_OK else f'❌ {COMPILE_REASON}'}")
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"  内存: {mem.total / 1024**3:.1f} GB")
    print(f"\n  自动配置:")
    print(f"    BATCH_SIZE   = {cfg.BATCH_SIZE}")
    print(f"    CHUNK_SIZE   = {cfg.CHUNK_SIZE}")
    print(f"    SURFACE_BAND = {cfg.SURFACE_BAND}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 精确 GT SDF 预计算
# ═══════════════════════════════════════════════════════════════

def compute_precise_sdf_for_mesh(mesh, query_points):
    """用 trimesh 精确计算 SDF"""
    closest_pts, distances, _ = trimesh.proximity.closest_point(mesh, query_points)

    try:
        inside = mesh.contains(query_points)
    except Exception:
        inside = _simple_inside_check(mesh, query_points)

    sdf = distances.astype(np.float32)
    sdf[inside] *= -1.0
    return sdf


def _simple_inside_check(mesh, points):
    """ray casting fallback"""
    try:
        ray_origins = points.copy()
        ray_directions = np.tile([1.0, 0.0, 0.0], (len(points), 1))
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        hits = intersector.hits_id(ray_origins, ray_directions)
        inside = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            mask = hits[:, 0] == i if len(hits) > 0 else np.array([], dtype=bool)
            inside[i] = mask.sum() % 2 == 1
        return inside
    except Exception:
        return np.zeros(len(points), dtype=bool)


def precompute_all_gt_sdf(stl_paths, tet_verts_np):
    """预计算所有训练样本的 GT SDF"""
    cache_dir = os.path.join(cfg.SAVE_DIR, "sdf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    gt_sdf_dict = {}
    print(f"\n  预计算精确 GT SDF ({len(stl_paths)} 个 mesh)...")

    for stl_path in tqdm(stl_paths, desc="  计算 GT SDF"):
        basename = os.path.splitext(os.path.basename(stl_path))[0]
        cache_path = os.path.join(cache_dir, f"{basename}_sdf.npy")

        if os.path.exists(cache_path):
            sdf = np.load(cache_path)
            if sdf.shape[0] == tet_verts_np.shape[0]:
                gt_sdf_dict[stl_path] = torch.from_numpy(sdf)
                continue

        mesh = trimesh.load(stl_path)
        verts = mesh.vertices
        center = verts.mean(axis=0)
        verts = verts - center
        scale = np.abs(verts).max()
        if scale > 1e-6:
            verts = verts / scale * 0.85
        mesh.vertices = verts

        sdf = compute_precise_sdf_for_mesh(mesh, tet_verts_np)
        np.save(cache_path, sdf)
        gt_sdf_dict[stl_path] = torch.from_numpy(sdf)

    print(f"  GT SDF 预计算完成！")

    sample_sdf = list(gt_sdf_dict.values())[0]
    n_inside = (sample_sdf < 0).sum().item()
    n_outside = (sample_sdf >= 0).sum().item()
    print(f"  样本 SDF 分布: 内部 {n_inside} ({100*n_inside/len(sample_sdf):.1f}%), "
          f"外部 {n_outside} ({100*n_outside/len(sample_sdf):.1f}%)")

    return gt_sdf_dict


# ═══════════════════════════════════════════════════════════════
# 运行时 GT SDF（增强后的数据）—— ★ 修复维度错误 ★
# ═══════════════════════════════════════════════════════════════

def compute_gt_sdf_from_points(tet_verts, gt_points, chunk_size=None):
    """
    训练时用的 GT SDF 计算（针对增强后的点云）
    使用 KNN + PCA 法向量估计符号

    tet_verts: (V, 3)
    gt_points: (B, N, 3)
    returns: (B, V) SDF
    """
    if chunk_size is None:
        chunk_size = cfg.CHUNK_SIZE

    B, N, _ = gt_points.shape
    V = tet_verts.shape[0]
    device = gt_points.device

    # 子采样 GT 点以节省显存
    n_sub = min(N, 2048)
    if N > n_sub:
        idx = torch.randperm(N, device=device)[:n_sub]
        gt_sub = gt_points[:, idx]  # (B, n_sub, 3)
    else:
        gt_sub = gt_points

    K = 8
    sdf_list = []

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        tv = tet_verts[start:end]  # (C, 3)
        C = tv.shape[0]

        # tv_exp: (B, C, 3)
        tv_exp = tv.unsqueeze(0).expand(B, C, 3)

        # 计算距离矩阵 (B, C, n_sub)
        dist = torch.cdist(tv_exp, gt_sub)

        # 最近点距离和索引
        unsigned_dist, nearest_idx = dist.min(dim=2)  # (B, C)

        # K 近邻索引
        topk_dist, topk_idx = dist.topk(K, dim=2, largest=False)  # (B, C, K)

        # 获取 K 近邻点坐标
        # gt_sub: (B, n_sub, 3) → 用 gather 取出 (B, C, K, 3)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(B, C, K, 3)  # (B, C, K, 3)
        gt_sub_exp = gt_sub.unsqueeze(1).expand(B, C, n_sub, 3)    # (B, C, n_sub, 3)
        knn_pts = torch.gather(gt_sub_exp, 2, topk_idx_exp)        # (B, C, K, 3)

        # 局部重心
        knn_center = knn_pts.mean(dim=2)  # (B, C, 3)

        # 获取最近点坐标
        nearest_idx_exp = nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(B, C, 1, 3)  # (B, C, 1, 3)
        nearest_pts = torch.gather(gt_sub_exp, 2, nearest_idx_exp).squeeze(2)  # (B, C, 3)

        # PCA 估计法向量
        centered = knn_pts - knn_center.unsqueeze(2)  # (B, C, K, 3)
        # 协方差矩阵 (B, C, 3, 3)
        cov = torch.matmul(centered.transpose(2, 3), centered)

        try:
            _, _, Vh = torch.linalg.svd(cov)
            normal = Vh[:, :, 2, :]  # 最小奇异值方向 (B, C, 3)
        except Exception:
            normal = F.normalize(tv_exp - knn_center, dim=-1)

        # 法向量定向：指向远离重心的方向
        to_query = tv_exp - knn_center
        dot = (normal * to_query).sum(dim=-1)  # (B, C)
        normal = torch.where(dot.unsqueeze(-1) < 0, -normal, normal)

        # 符号判断：查询点到最近表面点的向量 · 法向量
        to_nearest = tv_exp - nearest_pts  # (B, C, 3)
        sign_dot = (to_nearest * normal).sum(dim=-1)  # (B, C)
        sign = torch.where(sign_dot >= 0,
                           torch.ones_like(sign_dot),
                           -torch.ones_like(sign_dot))

        sdf_list.append(sign * unsigned_dist)

    return torch.cat(sdf_list, dim=1)


# ═══════════════════════════════════════════════════════════════
# 四面体网格生成
# ═══════════════════════════════════════════════════════════════

def create_tet_grid(res=64):
    n = res + 1
    x = torch.linspace(-1, 1, n)
    grid = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1)
    verts = grid.reshape(-1, 3)

    tet_local = torch.tensor([
        [0, 5, 1, 7], [0, 5, 7, 4], [0, 7, 3, 1],
        [0, 7, 2, 3], [0, 4, 7, 6], [0, 7, 6, 2],
    ], dtype=torch.long)

    i_idx = torch.arange(res)
    j_idx = torch.arange(res)
    k_idx = torch.arange(res)
    ii, jj, kk = torch.meshgrid(i_idx, j_idx, k_idx, indexing='ij')
    ii, jj, kk = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)

    v0 = ii * n * n + jj * n + kk
    v1 = (ii+1) * n * n + jj * n + kk
    v2 = ii * n * n + (jj+1) * n + kk
    v3 = (ii+1) * n * n + (jj+1) * n + kk
    v4 = ii * n * n + jj * n + (kk+1)
    v5 = (ii+1) * n * n + jj * n + (kk+1)
    v6 = ii * n * n + (jj+1) * n + (kk+1)
    v7 = (ii+1) * n * n + (jj+1) * n + (kk+1)

    cube_verts = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1)
    tets = cube_verts[:, tet_local.reshape(-1)].reshape(-1, 4)

    print(f"  四面体网格: {verts.shape[0]} 顶点, {tets.shape[0]} 四面体")
    return verts, tets


# ═══════════════════════════════════════════════════════════════
# 向量化边索引
# ═══════════════════════════════════════════════════════════════

def build_tet_edge_list_fast(tets):
    edge_pairs = torch.tensor([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
    ], dtype=torch.long)

    all_edges = tets[:, edge_pairs.reshape(-1)].reshape(-1, 2)
    all_edges_sorted, _ = all_edges.sort(dim=-1)

    V = tets.max().item() + 1
    edge_ids = all_edges_sorted[:, 0].long() * V + all_edges_sorted[:, 1].long()
    unique_ids = torch.unique(edge_ids)

    edge_indices = torch.stack([unique_ids // V, unique_ids % V], dim=0)
    print(f"  正则化边数: {edge_indices.shape[1]}")
    return edge_indices


# ═══════════════════════════════════════════════════════════════
# 分块 Chamfer Distance
# ═══════════════════════════════════════════════════════════════

def chamfer_distance_chunked(pred_verts, gt_pts, chunk_size=None):
    if chunk_size is None:
        chunk_size = cfg.CD_CHUNK

    if pred_verts.shape[0] == 0:
        return gt_pts.new_tensor(10.0, requires_grad=True)

    max_pts = cfg.CD_SUBSAMPLE
    if pred_verts.shape[0] > max_pts:
        idx = torch.randperm(pred_verts.shape[0], device=pred_verts.device)[:max_pts]
        pred_verts = pred_verts[idx]
    if gt_pts.shape[0] > max_pts:
        idx = torch.randperm(gt_pts.shape[0], device=gt_pts.device)[:max_pts]
        gt_pts = gt_pts[idx]

    N = pred_verts.shape[0]
    M = gt_pts.shape[0]

    min_p2g = torch.full((N,), float('inf'), device=pred_verts.device)
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        pred_chunk = pred_verts[i:end_i]
        for j in range(0, M, chunk_size):
            end_j = min(j + chunk_size, M)
            dist = torch.cdist(pred_chunk.unsqueeze(0),
                               gt_pts[j:end_j].unsqueeze(0)).squeeze(0)
            min_vals = dist.min(dim=1)[0]
            min_p2g[i:end_i] = torch.minimum(min_p2g[i:end_i], min_vals)

    min_g2p = torch.full((M,), float('inf'), device=pred_verts.device)
    for j in range(0, M, chunk_size):
        end_j = min(j + chunk_size, M)
        gt_chunk = gt_pts[j:end_j]
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            dist = torch.cdist(gt_chunk.unsqueeze(0),
                               pred_verts[i:end_i].unsqueeze(0)).squeeze(0)
            min_vals = dist.min(dim=1)[0]
            min_g2p[j:end_j] = torch.minimum(min_g2p[j:end_j], min_vals)

    return min_p2g.mean() + min_g2p.mean()


# ═══════════════════════════════════════════════════════════════
# GPU 端数据增强
# ═══════════════════════════════════════════════════════════════

def augment_batch_gpu(input_pts, gt_pts):
    B = input_pts.shape[0]
    device = input_pts.device

    angles = torch.rand(B, device=device) * 2 * 3.14159265
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)

    rot = torch.stack([
        torch.stack([cos_a, zeros, sin_a], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_a, zeros, cos_a], dim=-1),
    ], dim=1)

    input_pts = torch.bmm(input_pts, rot.transpose(1, 2))
    gt_pts = torch.bmm(gt_pts, rot.transpose(1, 2))

    scales = 0.9 + 0.2 * torch.rand(B, 1, 1, device=device)
    input_pts = input_pts * scales
    gt_pts = gt_pts * scales

    input_pts = input_pts + torch.randn_like(input_pts) * 0.005

    return input_pts, gt_pts


# ═══════════════════════════════════════════════════════════════
# 向量化可微 Marching Tetrahedra
# ═══════════════════════════════════════════════════════════════

class DMTetMesh(nn.Module):
    def __init__(self, tet_verts, tets):
        super().__init__()
        self.register_buffer('tet_verts', tet_verts)
        self.register_buffer('tets', tets)

        edge_pairs = torch.tensor([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
        ], dtype=torch.long)
        self.register_buffer('edge_pairs', edge_pairs)

        self._precompute_edges()
        self._build_triangle_table()

    def _precompute_edges(self):
        T_count = self.tets.shape[0]
        V = self.tet_verts.shape[0]

        tet_edges = self.tets[:, self.edge_pairs.reshape(-1)].reshape(T_count, 6, 2)
        tet_edges_sorted, _ = tet_edges.sort(dim=-1)

        edge_ids = tet_edges_sorted[:, :, 0].long() * V + tet_edges_sorted[:, :, 1].long()
        unique_edge_ids, inverse_idx = torch.unique(edge_ids.reshape(-1), return_inverse=True)
        inverse_idx = inverse_idx.reshape(T_count, 6)

        unique_v0 = unique_edge_ids // V
        unique_v1 = unique_edge_ids % V

        self.register_buffer('unique_edges', torch.stack([unique_v0, unique_v1], dim=1))
        self.register_buffer('tet_edge_indices', inverse_idx)
        print(f"  唯一边数: {self.unique_edges.shape[0]}")

    def _build_triangle_table(self):
        tri_table_dict = {
            0: [], 1: [(0,1,2)], 2: [(0,4,3)],
            3: [(1,3,2),(1,4,3)], 4: [(3,5,1)],
            5: [(0,5,2),(2,5,3)], 6: [(0,4,1),(1,4,5)],
            7: [(2,5,4)], 8: [(2,4,5)],
            9: [(0,1,4),(1,5,4)], 10: [(0,2,5),(0,5,3)],
            11: [(1,5,3)], 12: [(1,2,4),(2,5,4)],
            13: [(0,2,4)], 14: [(0,3,1)], 15: [],
        }

        table = torch.full((16, 2, 3), -1, dtype=torch.long)
        num_tris = torch.zeros(16, dtype=torch.long)

        for case_id, tris in tri_table_dict.items():
            num_tris[case_id] = len(tris)
            for ti, (a, b, c) in enumerate(tris):
                table[case_id, ti] = torch.tensor([a, b, c])

        self.register_buffer('tri_table', table)
        self.register_buffer('num_tris_table', num_tris)

    def forward(self, sdf, deformation):
        device = sdf.device
        verts = self.tet_verts + deformation

        e0_idx = self.unique_edges[:, 0]
        e1_idx = self.unique_edges[:, 1]
        s0 = sdf[e0_idx]
        s1 = sdf[e1_idx]

        denom = s0 - s1
        safe_denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        w = (s0 / safe_denom).clamp(0.001, 0.999)

        edge_verts = verts[e0_idx] * (1 - w.unsqueeze(1)) + verts[e1_idx] * w.unsqueeze(1)

        tet_sdf = sdf[self.tets]
        signs = (tet_sdf > 0).long()
        case_ids = signs[:, 0] + signs[:, 1] * 2 + signs[:, 2] * 4 + signs[:, 3] * 8

        active_mask = (case_ids != 0) & (case_ids != 15)
        if active_mask.sum() == 0:
            return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

        active_cases = case_ids[active_mask]
        active_edge_idx = self.tet_edge_indices[active_mask]

        tri_edges_local = self.tri_table[active_cases]
        tri_counts = self.num_tris_table[active_cases]

        A = active_cases.shape[0]
        tri_edges_flat = tri_edges_local.reshape(A, -1).clamp(min=0)
        global_edge_idx = torch.gather(active_edge_idx, 1, tri_edges_flat).reshape(A, 2, 3)

        faces_list = []
        for ti in range(2):
            valid = tri_counts > ti
            if valid.sum() == 0:
                continue
            faces_list.append(global_edge_idx[valid, ti, :])

        if not faces_list:
            return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

        all_faces = torch.cat(faces_list, dim=0)
        used_edges = all_faces.unique()
        remap = torch.full((self.unique_edges.shape[0],), -1, dtype=torch.long, device=device)
        remap[used_edges] = torch.arange(used_edges.shape[0], device=device)

        return edge_verts[used_edges], remap[all_faces]


# ═══════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3,    64,  1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,   128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,  256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,  512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512,  latent_dim, 1), nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).max(dim=-1)[0]


# ═══════════════════════════════════════════════════════════════
# SDF Decoder（粗筛加速 + AMP 修复）
# ═══════════════════════════════════════════════════════════════

class SDFDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        in_dim = latent_dim + 3

        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512 + in_dim, 512)
        self.fc4 = nn.Linear(512, 256)

        self.sdf_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.def_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.sphere_weight = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.def_head[-1].weight)
        nn.init.zeros_(self.def_head[-1].bias)
        nn.init.xavier_uniform_(self.sdf_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.sdf_head[-1].bias)

    def _run_mlp(self, feat_exp, pts_exp):
        x_in = torch.cat([feat_exp, pts_exp], dim=-1)
        h = F.relu(self.fc1(x_in))
        h = F.relu(self.fc2(h))
        h = torch.cat([h, x_in], dim=-1)
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        sdf = self.sdf_head(h).squeeze(-1)
        deformation = self.def_head(h)
        return sdf, deformation

    def forward(self, feat, query_pts, chunk_size=None, use_accel=True,
                pcd=None, surface_band=None):
        if chunk_size is None:
            chunk_size = cfg.CHUNK_SIZE
        if surface_band is None:
            surface_band = cfg.SURFACE_BAND

        B = feat.shape[0]
        V = query_pts.shape[0]
        device = feat.device

        sphere_w = torch.sigmoid(self.sphere_weight)
        grid_spacing = 2.0 / cfg.TET_GRID_RES

        if not use_accel or pcd is None:
            return self._forward_full(feat, query_pts, chunk_size, sphere_w, grid_spacing)

        # ── 粗筛加速路径 ──
        with torch.no_grad():
            n_sub = min(pcd.shape[1], 256)
            idx = torch.randperm(pcd.shape[1], device=device)[:n_sub]
            pcd_sub = pcd[:, idx]

            coarse_dist = torch.cdist(
                query_pts.unsqueeze(0).expand(B, V, 3),
                pcd_sub
            ).min(dim=2)[0]

        sphere_sdf_all = query_pts.norm(dim=-1).unsqueeze(0).expand(B, V) - 0.6
        analytic_sdf = coarse_dist.clone()

        sdf_out = (sphere_w * sphere_sdf_all * 0.3 + analytic_sdf * 0.1).to(torch.float32)
        def_out = torch.zeros(B, V, 3, device=device, dtype=torch.float32)

        for b in range(B):
            near_mask = coarse_dist[b] < surface_band
            near_idx = near_mask.nonzero(as_tuple=True)[0]

            if near_idx.shape[0] == 0:
                continue

            pts_near = query_pts[near_idx]
            feat_b = feat[b:b+1].expand(near_idx.shape[0], -1)

            mlp_sdf_list = []
            mlp_def_list = []
            for start in range(0, near_idx.shape[0], chunk_size):
                end = min(start + chunk_size, near_idx.shape[0])
                s, d = self._run_mlp(feat_b[start:end], pts_near[start:end])
                mlp_sdf_list.append(s)
                mlp_def_list.append(d)

            mlp_sdf = torch.cat(mlp_sdf_list, dim=0)
            mlp_def = torch.cat(mlp_def_list, dim=0)

            sphere_sdf = pts_near.norm(dim=-1) - 0.6
            final_sdf = mlp_sdf + sphere_w * sphere_sdf.to(mlp_sdf.dtype) * 0.3

            sdf_out[b, near_idx] = final_sdf.to(sdf_out.dtype)
            def_out[b, near_idx] = mlp_def.to(def_out.dtype)

        def_out = torch.tanh(def_out) * grid_spacing * 0.5
        return sdf_out, def_out

    def _forward_full(self, feat, query_pts, chunk_size, sphere_w, grid_spacing):
        B = feat.shape[0]
        V = query_pts.shape[0]

        all_sdf = []
        all_def = []

        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            pts_chunk = query_pts[start:end]
            C = pts_chunk.shape[0]

            feat_exp = feat.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
            pts_exp = pts_chunk.unsqueeze(0).expand(B, C, 3).reshape(B * C, 3)

            sdf_pred, def_pred = self._run_mlp(feat_exp, pts_exp)
            sdf_pred = sdf_pred.reshape(B, C)
            def_pred = def_pred.reshape(B, C, 3)

            sphere_sdf = pts_chunk.norm(dim=-1).unsqueeze(0).expand(B, C) - 0.6
            sdf_chunk = sdf_pred + sphere_w * sphere_sdf.to(sdf_pred.dtype) * 0.3

            all_sdf.append(sdf_chunk)
            all_def.append(def_pred)

        sdf = torch.cat(all_sdf, dim=1)
        deformation = torch.cat(all_def, dim=1)
        deformation = torch.tanh(deformation) * grid_spacing * 0.5

        return sdf, deformation


# ═══════════════════════════════════════════════════════════════
# 完整模型
# ═══════════════════════════════════════════════════════════════

class DMTetModel(nn.Module):
    def __init__(self, tet_res=64, latent_dim=512):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = SDFDecoder(latent_dim)
        tet_verts, tets = create_tet_grid(tet_res)
        self.dmtet = DMTetMesh(tet_verts, tets)

    @property
    def tet_verts(self):
        return self.dmtet.tet_verts

    @property
    def tets(self):
        return self.dmtet.tets

    def forward(self, pcd, return_deformation=False, use_accel=True):
        feat = self.encoder(pcd)
        sdf, deformation = self.decoder(
            feat, self.dmtet.tet_verts, chunk_size=cfg.CHUNK_SIZE,
            use_accel=use_accel, pcd=pcd
        )

        results = []
        for b in range(pcd.shape[0]):
            v, f = self.dmtet(sdf[b], deformation[b])
            results.append((v, f))

        if return_deformation:
            return results, sdf, deformation
        return results, sdf


# ═══════════════════════════════════════════════════════════════
# 正则化 Loss
# ═══════════════════════════════════════════════════════════════

def compute_sdf_laplacian_loss(sdf, edge_indices):
    sdf_i = sdf[:, edge_indices[0]]
    sdf_j = sdf[:, edge_indices[1]]
    return ((sdf_i - sdf_j) ** 2).mean()


def compute_deformation_reg(deformation, edge_indices):
    l2_loss = (deformation ** 2).sum(dim=-1).mean()
    def_i = deformation[:, edge_indices[0]]
    def_j = deformation[:, edge_indices[1]]
    smooth_loss = ((def_i - def_j) ** 2).sum(dim=-1).mean()
    return l2_loss + smooth_loss


def compute_sign_clarity_loss(sdf):
    return torch.exp(-sdf.abs() * 5.0).mean()


def normal_consistency_loss(verts, faces):
    if faces.shape[0] < 2 or verts.shape[0] == 0:
        return verts.new_tensor(0.0) if verts.shape[0] > 0 else torch.tensor(0.0)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    if faces.shape[0] > 2000:
        normals = normals[torch.randperm(faces.shape[0], device=verts.device)[:2000]]

    return (1 - (normals[:-1] * normals[1:]).sum(dim=-1)).mean()


# ═══════════════════════════════════════════════════════════════
# 总 Loss
# ═══════════════════════════════════════════════════════════════

def compute_loss(results, sdf, deformation, gt_points, gt_sdf, edge_indices, epoch):
    B = len(results)
    loss_sdf_gt = F.l1_loss(sdf, gt_sdf)
    loss_sdf_reg = (sdf ** 2).mean() * 0.001

    loss_lap = compute_sdf_laplacian_loss(sdf, edge_indices)
    loss_deform = compute_deformation_reg(deformation, edge_indices)
    loss_sign = compute_sign_clarity_loss(sdf)

    if epoch <= cfg.PHASE1_EPOCHS:
        total = (cfg.W_SDF_GT * loss_sdf_gt
                 + loss_sdf_reg
                 + cfg.W_LAP_SDF * loss_lap
                 + cfg.W_DEFORM_REG * loss_deform
                 + cfg.W_SIGN_CLARITY * loss_sign)
        return total, loss_sdf_gt, sdf.new_tensor(0.0), loss_lap

    # Phase 2
    loss_chamfer = sdf.new_tensor(0.0)
    loss_normal = sdf.new_tensor(0.0)
    valid = 0

    for b in range(B):
        v, f = results[b]
        if v.shape[0] == 0 or f.shape[0] == 0:
            loss_chamfer = loss_chamfer + 5.0
            continue
        valid += 1
        loss_chamfer = loss_chamfer + chamfer_distance_chunked(v, gt_points[b], cfg.CD_CHUNK)
        loss_normal = loss_normal + normal_consistency_loss(v, f)

    loss_chamfer /= max(B, 1)
    loss_normal /= max(valid, 1)

    progress = min((epoch - cfg.PHASE1_EPOCHS) / 50.0, 1.0)
    w_sdf = cfg.W_SDF_GT * (1 - 0.8 * progress)
    w_cd = cfg.W_CHAMFER * progress

    total = (w_cd * loss_chamfer
             + cfg.W_NORMAL * loss_normal
             + w_sdf * loss_sdf_gt
             + loss_sdf_reg
             + cfg.W_LAP_SDF * loss_lap
             + cfg.W_DEFORM_REG * loss_deform
             + cfg.W_SIGN_CLARITY * loss_sign)

    return total, loss_sdf_gt, loss_chamfer, loss_lap


# ═══════════════════════════════════════════════════════════════
# 数据集（支持预计算 GT SDF）
# ═══════════════════════════════════════════════════════════════

class MeshReconDataset(Dataset):
    def __init__(self, pcd_dir, stl_dir, n_input=4096, n_gt=4096, gt_sdf_dict=None):
        self.n_input = n_input
        self.n_gt = n_gt
        self.gt_sdf_dict = gt_sdf_dict

        pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))
        stl_files = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))

        pcd_names = {os.path.splitext(os.path.basename(f))[0]: f for f in pcd_files}
        stl_names = {os.path.splitext(os.path.basename(f))[0]: f for f in stl_files}

        common = sorted(set(pcd_names.keys()) & set(stl_names.keys()))
        self.pairs = [(pcd_names[n], stl_names[n]) for n in common]
        print(f"  数据集: {len(self.pairs)} 对")

        self.cached_data = []
        print("  预加载数据...")
        t0 = time.time()
        for pcd_path, stl_path in tqdm(self.pairs, desc="  加载", leave=False):
            pc = trimesh.load(pcd_path)
            pts = np.array(pc.vertices, dtype=np.float32)

            mesh = trimesh.load(stl_path)
            gt_pts, _ = trimesh.sample.sample_surface(mesh, self.n_gt)
            gt_pts = gt_pts.astype(np.float32)

            # 归一化到 [-0.85, 0.85]
            center = gt_pts.mean(axis=0)
            gt_pts = gt_pts - center
            pts = pts - center
            scale = np.abs(gt_pts).max()
            if scale > 1e-6:
                gt_pts = gt_pts / scale * 0.85
                pts = pts / scale * 0.85

            self.cached_data.append((pts, gt_pts, stl_path))
        print(f"  预加载完成: {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pts, gt_pts, stl_path = self.cached_data[idx]

        if len(pts) >= self.n_input:
            idx_sel = np.random.choice(len(pts), self.n_input, replace=False)
        else:
            idx_sel = np.random.choice(len(pts), self.n_input, replace=True)
        input_pts = pts[idx_sel]

        if self.gt_sdf_dict is not None and stl_path in self.gt_sdf_dict:
            gt_sdf = self.gt_sdf_dict[stl_path]
            return torch.from_numpy(input_pts), torch.from_numpy(gt_pts.copy()), gt_sdf
        else:
            return torch.from_numpy(input_pts), torch.from_numpy(gt_pts.copy()), torch.tensor(0.0)


# ═══════════════════════════════════════════════════════════════
# 训练主循环
# ═══════════════════════════════════════════════════════════════

def train():
    print_system_info()
    device = cfg.DEVICE

    # 先创建四面体网格以获取顶点坐标
    tet_verts_temp, _ = create_tet_grid(cfg.TET_GRID_RES)
    tet_verts_np = tet_verts_temp.numpy()

    # 预计算 GT SDF
    stl_files = sorted(glob.glob(os.path.join(cfg.STL_DIR, "*.stl")))
    gt_sdf_dict = precompute_all_gt_sdf(stl_files, tet_verts_np)

    dataset = MeshReconDataset(
        cfg.PCD_DIR, cfg.STL_DIR, cfg.NUM_INPUT_PTS, cfg.NUM_GT_PTS,
        gt_sdf_dict=gt_sdf_dict
    )
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    print(f"  训练集: {n_train}, 验证集: {n_val}")
    print(f"  Batch: {cfg.BATCH_SIZE}, 每 epoch {len(train_loader)} batches")

    model = DMTetModel(tet_res=cfg.TET_GRID_RES, latent_dim=512).to(device)

    if TORCH_COMPILE_OK and device == 'cuda':
        try:
            model.encoder = torch.compile(model.encoder, mode='reduce-overhead')
            print("  ✅ torch.compile (encoder)")
        except Exception as e:
            print(f"  ⚠️  torch.compile 失败: {e}")
    else:
        print(f"  ℹ️  torch.compile 跳过: {COMPILE_REASON}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_RATE)

    use_amp = cfg.USE_AMP and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    edge_indices = build_tet_edge_list_fast(model.tets).to(device)

    best_val_loss = float('inf')
    print(f"\n开始训练 ({cfg.EPOCHS} epochs)...")
    print(f"  Phase 1 (SDF): epoch 1~{cfg.PHASE1_EPOCHS}")
    print(f"  Phase 2 (CD):  epoch {cfg.PHASE1_EPOCHS+1}~{cfg.EPOCHS}")
    if use_amp:
        print(f"  AMP: ✅")

    # warm-up
    if device == 'cuda':
        print("  GPU warm-up...")
        dummy = torch.randn(2, cfg.NUM_INPUT_PTS, 3, device=device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    _ = model(dummy, use_accel=False)
            else:
                _ = model(dummy, use_accel=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"  warm-up 完成, {get_gpu_info()}")

    epoch_times = []

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_sdf = 0.0
        epoch_cd = 0.0
        epoch_lap = 0.0
        n_mesh_ok = 0
        n_total = 0
        n_logged = 0

        phase = "P1-SDF" if epoch <= cfg.PHASE1_EPOCHS else "P2-CD"
        t_epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"[{phase}] E{epoch}/{cfg.EPOCHS}",
                    leave=False, ncols=140)

        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                pcd_batch, gt_batch, precomp_sdf = batch_data
            else:
                pcd_batch, gt_batch = batch_data
                precomp_sdf = None

            pcd_batch = pcd_batch.to(device, non_blocking=True)
            gt_batch = gt_batch.to(device, non_blocking=True)

            # 数据增强
            pcd_batch, gt_batch = augment_batch_gpu(pcd_batch, gt_batch)

            # GT SDF：增强后重新计算
            gt_sdf = compute_gt_sdf_from_points(model.tet_verts, gt_batch, chunk_size=cfg.CHUNK_SIZE)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    results, sdf, deformation = model(
                        pcd_batch, return_deformation=True, use_accel=True
                    )
                    loss, l_sdf, l_cd, l_lap = compute_loss(
                        results, sdf, deformation, gt_batch, gt_sdf, edge_indices, epoch
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                results, sdf, deformation = model(
                    pcd_batch, return_deformation=True, use_accel=True
                )
                loss, l_sdf, l_cd, l_lap = compute_loss(
                    results, sdf, deformation, gt_batch, gt_sdf, edge_indices, epoch
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if batch_idx % 4 == 0:
                epoch_loss += loss.item()
                epoch_sdf += l_sdf.item()
                epoch_cd += l_cd.item()
                epoch_lap += l_lap.item()
                n_logged += 1

            for v, f in results:
                n_total += 1
                if v.shape[0] > 0 and f.shape[0] > 0:
                    n_mesh_ok += 1

            if batch_idx % 4 == 0:
                sw = torch.sigmoid(model.decoder.sphere_weight).item()
                pbar.set_postfix({
                    'L': f'{loss.item():.4f}',
                    'sdf': f'{l_sdf.item():.3f}',
                    'cd': f'{l_cd.item():.3f}',
                    'sw': f'{sw:.2f}',
                    'mesh': f'{n_mesh_ok}/{n_total}',
                })

        scheduler.step()
        t_epoch_end = time.time()
        epoch_time = t_epoch_end - t_epoch_start
        epoch_times.append(epoch_time)

        avg_loss = epoch_loss / max(n_logged, 1)
        avg_sdf = epoch_sdf / max(n_logged, 1)
        avg_cd = epoch_cd / max(n_logged, 1)
        avg_lap = epoch_lap / max(n_logged, 1)

        # 验证
        avg_val = float('inf')
        if epoch % 2 == 0 or epoch == 1 or epoch == cfg.EPOCHS:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3:
                        pcd_batch, gt_batch, _ = batch_data
                    else:
                        pcd_batch, gt_batch = batch_data

                    pcd_batch = pcd_batch.to(device, non_blocking=True)
                    gt_batch = gt_batch.to(device, non_blocking=True)
                    gt_sdf = compute_gt_sdf_from_points(
                        model.tet_verts, gt_batch, chunk_size=cfg.CHUNK_SIZE
                    )

                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            results, sdf, deformation = model(
                                pcd_batch, return_deformation=True, use_accel=True
                            )
                            loss, _, _, _ = compute_loss(
                                results, sdf, deformation, gt_batch, gt_sdf, edge_indices, epoch
                            )
                    else:
                        results, sdf, deformation = model(
                            pcd_batch, return_deformation=True, use_accel=True
                        )
                        loss, _, _, _ = compute_loss(
                            results, sdf, deformation, gt_batch, gt_sdf, edge_indices, epoch
                        )
                    val_loss_sum += loss.item()
                    val_count += 1

            avg_val = val_loss_sum / max(val_count, 1)

        gpu_info = get_gpu_info()
        lr_now = optimizer.param_groups[0]['lr']
        sw = torch.sigmoid(model.decoder.sphere_weight).item()
        avg_epoch_time = np.mean(epoch_times[-5:])
        eta_min = avg_epoch_time * (cfg.EPOCHS - epoch) / 60

        print(f"[{phase}] E{epoch}/{cfg.EPOCHS}  "
              f"Train:{avg_loss:.5f} (SDF:{avg_sdf:.3f} CD:{avg_cd:.3f} LAP:{avg_lap:.3f})  "
              f"Val:{avg_val:.5f}  LR:{lr_now:.1e}  "
              f"SW:{sw:.2f}  Mesh:{n_mesh_ok}/{n_total}  "
              f"⏱{epoch_time:.1f}s  ETA:{eta_min:.0f}min  {gpu_info}")

        if avg_val < best_val_loss and avg_val < float('inf'):
            best_val_loss = avg_val
            save_path = os.path.join(cfg.SAVE_DIR, "best_mesh_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  💾 最优模型 (Val={avg_val:.6f})")

        if epoch % cfg.SAVE_INTERVAL == 0:
            save_path = os.path.join(cfg.SAVE_DIR, f"mesh_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"训练完毕！最优 Val Loss = {best_val_loss:.6f}")
    print(f"总耗时: {sum(epoch_times)/60:.1f} 分钟")
    print(f"平均每 epoch: {np.mean(epoch_times):.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()