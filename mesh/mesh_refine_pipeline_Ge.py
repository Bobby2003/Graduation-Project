import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import trimesh

try:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as sp_connected_components

    SCIPY_AVAILABLE = True
except Exception:
    sp = None
    sp_connected_components = None
    SCIPY_AVAILABLE = False


# =============================
# Basic
# =============================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_vec3(s):
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError("sensor_origin 应为 x,y,z")
    return np.array(vals, dtype=np.float64)


def robust_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return float(1.4826 * mad)


def mesh_report(mesh, include_components=False):
    comps = None
    if include_components:
        try:
            comps = len(list(mesh.split(only_watertight=False)))
        except Exception:
            comps = None

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": comps,
        "watertight": bool(getattr(mesh, "is_watertight", False)),
        "bbox_extent": mesh.bounding_box.extents.tolist() if len(mesh.vertices) else None,
        "surface_area": float(mesh.area) if len(mesh.faces) else 0.0,
    }


def load_mesh(path):
    obj = trimesh.load(path, force="mesh", process=False, maintain_order=True)

    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene 中没有可用网格")
        obj = trimesh.util.concatenate(meshes)

    if not isinstance(obj, trimesh.Trimesh) or len(obj.vertices) == 0 or len(obj.faces) == 0:
        raise ValueError("无效或空网格")

    return obj


def clean_mesh_light(mesh, fix_normals=False):
    mesh = mesh.copy()

    for fn in [
        "remove_duplicate_faces",
        "remove_degenerate_faces",
        "remove_unreferenced_vertices",
        "remove_infinite_values",
    ]:
        try:
            getattr(mesh, fn)()
        except Exception:
            pass

    if fix_normals:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    return mesh


def clean_mesh_heavy(mesh, fix_normals=True):
    mesh = clean_mesh_light(mesh, fix_normals=False)
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    if fix_normals:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass
    return mesh


def split_components(mesh):
    try:
        return list(mesh.split(only_watertight=False))
    except TypeError:
        return list(mesh.split())


def abs_normal_dot(normals, normal):
    return np.clip(
        np.abs(np.asarray(normals, dtype=np.float64) @ np.asarray(normal, dtype=np.float64)),
        0.0,
        1.0,
    )


# =============================
# PCA / Plane
# =============================
def weighted_pca(points, weights=None):
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3)

    w = np.ones(len(p), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sum(w) + 1e-12
    c = np.sum(p * w[:, None], axis=0) / sw
    x = (p - c) * np.sqrt(w[:, None])
    cov = (x.T @ x) / sw
    vals, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n, vals


def robust_plane(points, weights=None, iters=6, huber_k=1.5):
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros(3), np.array([0.0, 0.0, 1.0], dtype=np.float64)

    w = np.ones(len(p), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    c = p.mean(axis=0)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for _ in range(iters):
        c, n, _ = weighted_pca(p, w)
        d = np.dot(p - c, n)
        s = robust_sigma(d) + 1e-12
        r = np.abs(d) / (huber_k * s + 1e-12)
        w = np.where(r <= 1.0, w, w / (r + 1e-12))
    return c, n


def dist_cfg(plane_dist, bbox_diag, near_ratio=0.25, far_ratio=0.50):
    r = plane_dist / (bbox_diag + 1e-12)
    if r <= near_ratio:
        return dict(stage="near", snap=1.00, inlier=0.85, outlier=1.00, shrink=0.70, smooth=1.20, noise=1.00,
                    detect_dist=1.00, detect_ang=1.00)
    if r <= far_ratio:
        return dict(stage="mid", snap=1.10, inlier=1.05, outlier=1.45, shrink=0.42, smooth=1.80, noise=1.35,
                    detect_dist=1.20, detect_ang=1.12)
    return dict(stage="far", snap=1.22, inlier=1.35, outlier=2.40, shrink=0.12, smooth=2.80, noise=2.10,
                detect_dist=1.65, detect_ang=1.35)


# =============================
# Sparse graph
# =============================
def make_sparse_graph_from_pairs(num_nodes, pairs):
    if not SCIPY_AVAILABLE:
        return None
    if pairs is None or len(pairs) == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.uint8)

    pairs = np.asarray(pairs, dtype=np.int32)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    g = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.uint8)
    g.sum_duplicates()
    g.data[:] = 1
    return g


def build_vertex_adjacency_sparse(num_vertices, faces):
    if not SCIPY_AVAILABLE:
        return None

    f = np.asarray(faces, dtype=np.int32)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 0], f[:, 2], f[:, 1], f[:, 0], f[:, 2]])
    data = np.ones(len(rows), dtype=np.uint8)
    g = sp.csr_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices), dtype=np.uint8)
    g.sum_duplicates()
    g.data[:] = 1
    return g


def boundary_vertices(mesh):
    try:
        e = np.sort(mesh.edges_sorted, axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        return ue[cnt == 1].reshape(-1)
    except Exception:
        return np.empty(0, dtype=np.int32)


# =============================
# Fast components / noise removal
# =============================
def connected_face_labels(mesh):
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.empty(0, dtype=np.int32), 0

    if not SCIPY_AVAILABLE:
        comps = split_components(mesh)
        labels = np.empty(n_faces, dtype=np.int32)
        offset = 0
        for i, c in enumerate(comps):
            n = len(c.faces)
            labels[offset:offset + n] = i
            offset += n
        return labels, len(comps)

    face_adj = np.asarray(mesh.face_adjacency, dtype=np.int32)
    if face_adj.size == 0:
        labels = np.arange(n_faces, dtype=np.int32)
        return labels, n_faces

    rows = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
    cols = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces), dtype=np.uint8)
    n_comp, labels = sp_connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32), int(n_comp)


def submesh_from_face_mask(mesh, face_mask):
    face_mask = np.asarray(face_mask, dtype=bool)
    faces_old = np.asarray(mesh.faces, dtype=np.int32)
    verts_old = np.asarray(mesh.vertices, dtype=np.float64)

    faces_kept = faces_old[face_mask]
    if len(faces_kept) == 0:
        return mesh.copy()

    used_vids = np.unique(faces_kept.reshape(-1))
    new_index = np.full(len(verts_old), -1, dtype=np.int32)
    new_index[used_vids] = np.arange(len(used_vids), dtype=np.int32)

    verts_new = verts_old[used_vids]
    faces_new = new_index[faces_kept]

    out = trimesh.Trimesh(vertices=verts_new, faces=faces_new, process=False)
    try:
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out


def component_stats(mesh):
    labels, n_comp = connected_face_labels(mesh)
    if n_comp == 0:
        return labels, n_comp, np.empty(0), np.empty(0), np.empty(0)

    faces = np.asarray(mesh.faces, dtype=np.int32)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    area_faces = np.asarray(mesh.area_faces, dtype=np.float64)

    face_counts = np.bincount(labels, minlength=n_comp)
    area_sums = np.bincount(labels, weights=area_faces, minlength=n_comp)
    max_extents = np.zeros(n_comp, dtype=np.float64)

    for cid in range(n_comp):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        vids = np.unique(faces[idx].reshape(-1))
        pts = verts[vids]
        ext = pts.max(axis=0) - pts.min(axis=0)
        max_extents[cid] = float(np.max(ext))

    return labels, n_comp, face_counts, area_sums, max_extents


def filter_components_fast(mesh, min_faces=80, min_area=0.0015, min_extent=0.04, keep_top_k=20):
    labels, n_comp, face_counts, area_sums, max_extents = component_stats(mesh)
    if n_comp <= 1:
        return mesh

    keep = set(np.where(
        (face_counts >= min_faces) &
        (area_sums >= min_area) &
        (max_extents >= min_extent)
    )[0].tolist())

    top_ids = np.argsort(-face_counts)[:keep_top_k]
    keep.update(top_ids.tolist())

    if not keep:
        keep = {int(np.argmax(face_counts))}

    face_mask = np.isin(labels, list(keep))
    return submesh_from_face_mask(mesh, face_mask)


def remove_floating_noise_fast(mesh, min_faces=20, min_area=0.0005, min_extent=0.02, keep_top_k=8):
    labels, n_comp, face_counts, area_sums, max_extents = component_stats(mesh)
    if n_comp <= 1:
        return mesh

    keep = set(np.where(
        (face_counts >= min_faces) &
        (area_sums >= min_area) &
        (max_extents >= min_extent)
    )[0].tolist())

    top_ids = np.argsort(-face_counts)[:keep_top_k]
    keep.update(top_ids.tolist())

    if not keep:
        keep = {int(np.argmax(face_counts))}

    face_mask = np.isin(labels, list(keep))
    return submesh_from_face_mask(mesh, face_mask)


# =============================
# Advanced Hole Filling (Dynamic Programming Triangulation)
# =============================

def get_boundary_edges(mesh):
    try:
        e = np.sort(mesh.edges_sorted, axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        return ue[cnt == 1]
    except Exception:
        return np.empty((0, 2), dtype=np.int32)


def extract_small_boundary_loops(mesh, max_hole_edges=60, max_loops=800):
    boundary = get_boundary_edges(mesh)
    if len(boundary) == 0: return []

    flat_b = boundary.flatten()
    if len(flat_b) == 0: return []
    max_vid = flat_b.max()
    deg = np.bincount(flat_b, minlength=max_vid + 1)

    mask = (deg[boundary[:, 0]] == 2) & (deg[boundary[:, 1]] == 2)
    valid_edges = boundary[mask]
    if len(valid_edges) == 0: return []

    if SCIPY_AVAILABLE:
        rows, cols = valid_edges[:, 0], valid_edges[:, 1]
        data = np.ones(len(rows), dtype=np.uint8)
        n_v = max_vid + 1
        g = sp.csr_matrix((data, (rows, cols)), shape=(n_v, n_v))
        g = g + g.T
        n_comp, labels = sp_connected_components(g, directed=False)

        comp_sizes = np.bincount(labels)
        valid_comps = np.where((comp_sizes >= 3) & (comp_sizes <= max_hole_edges))[0]

        loops = []
        adj_dict = {}
        for r, c in zip(rows, cols):
            adj_dict.setdefault(r, []).append(c)
            adj_dict.setdefault(c, []).append(r)

        for comp in valid_comps:
            nodes = np.where(labels == comp)[0]
            if len(nodes) == 0: continue

            start = nodes[0]
            ns_start = adj_dict.get(start, [])
            if len(ns_start) != 2:
                continue

            loop = [start]
            prev, cur = start, ns_start[0]

            ok = True
            while cur != start and len(loop) <= max_hole_edges:
                loop.append(cur)
                ns = adj_dict.get(cur, [])
                if len(ns) != 2:
                    ok = False
                    break
                nxt = ns[0] if ns[1] == prev else ns[1]
                prev, cur = cur, nxt

            if ok and cur == start:
                loops.append(np.array(loop, dtype=np.int32))
                if len(loops) >= max_loops: break
        return loops

    # 无 Scipy 时修复后的纯 Python 回退逻辑
    adj = {}
    for a, b in valid_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited = set()
    loops = []

    def ek(a, b):
        return (a, b) if a < b else (b, a)

    for start in list(adj.keys()):
        if len(adj[start]) != 2: continue
        nxt = adj[start][0]
        if ek(start, nxt) in visited: continue

        loop = [start]
        prev, cur = start, nxt
        ok = True

        while cur != start:
            loop.append(cur)
            if len(loop) > max_hole_edges:
                ok = False
                break
            visited.add(ek(prev, cur))
            ns = adj.get(cur, [])
            if len(ns) != 2:
                ok = False
                break
            nxt = ns[0] if ns[1] == prev else ns[1]
            prev, cur = cur, nxt

        if ok and 3 <= len(loop) <= max_hole_edges:
            visited.add(ek(prev, cur))
            loops.append(np.array(loop, dtype=np.int32))
            if len(loops) >= max_loops: break
    return loops


def newell_normal(pts):
    """使用 Newell 方法计算 3D 多边形的鲁棒法线"""
    n = np.zeros(3)
    for i in range(len(pts)):
        v1 = pts[i]
        v2 = pts[(i + 1) % len(pts)]
        n[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
        n[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
        n[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def triangulate_polygon_3d_dp(loop_pts):
    """
    使用动态规划进行最小权重三角化 (Minimum Weight Triangulation)
    适用于非平面、凹多边形的优质孔洞填充，不会像质心法那样产生尖刺。
    """
    n = len(loop_pts)
    if n < 3:
        return []
    if n == 3:
        return [[0, 1, 2]]

    # 估算孔洞的整体法线
    poly_normal = newell_normal(loop_pts)

    # DP table 和 选择记录
    dp = np.full((n, n), np.inf)
    choice = np.zeros((n, n), dtype=int)

    # 边界条件：相邻顶点的代价为0
    for i in range(n - 1):
        dp[i, i + 1] = 0.0

    def tri_weight(p1, p2, p3):
        cross = np.cross(p2 - p1, p3 - p1)
        area = 0.5 * np.linalg.norm(cross)
        if area < 1e-8:
            return 1e6  # 惩罚退化三角形

        tri_norm = cross / (np.linalg.norm(cross) + 1e-12)
        dot_val = np.dot(tri_norm, poly_normal)

        # 严重惩罚法线翻转（自交/内扣）的三角形
        penalty = 1.0
        if dot_val < 0.2:
            penalty = 10.0 + (1.0 - dot_val) * 20.0

        return area * penalty

    # DP 填表
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                weight = tri_weight(loop_pts[i], loop_pts[k], loop_pts[j])
                cost = dp[i, k] + dp[k, j] + weight
                if cost < dp[i, j]:
                    dp[i, j] = cost
                    choice[i, j] = k

    # 回溯构建面
    faces = []

    def build_faces(i, j):
        if j - i < 2: return
        k = choice[i, j]
        faces.append([i, k, j])
        build_faces(i, k)
        build_faces(k, j)

    build_faces(0, n - 1)
    return faces


def fill_small_holes_early(
        mesh,
        enable=True,
        max_hole_edges=60,
        max_hole_diameter=0.40,
        max_hole_area=0.05,
        max_plane_residual=0.05,
        max_candidate_loops=1500,
        try_trimesh_fill=True,
        verbose=False,
):
    stats = {
        "enabled": bool(enable),
        "trimesh_fill_applied": False,
        "candidate_loops": 0,
        "filled_holes": 0,
        "added_faces": 0,
    }

    if not enable:
        return mesh.copy(), stats

    out = mesh.copy()

    if try_trimesh_fill:
        try:
            before_faces = len(out.faces)
            trimesh.repair.fill_holes(out)
            stats["trimesh_fill_applied"] = len(out.faces) > before_faces
        except Exception:
            pass

    # 迭代修补，因为补上一个洞可能会产生/闭合新的洞轮廓
    filled_total = 0
    added_faces_total = 0

    for iteration in range(2):
        loops = extract_small_boundary_loops(
            out,
            max_hole_edges=max_hole_edges,
            max_loops=max_candidate_loops
        )
        if iteration == 0:
            stats["candidate_loops"] = int(len(loops))

        if len(loops) == 0:
            break

        V = np.asarray(out.vertices, dtype=np.float64)
        F = list(out.faces)

        filled_this_round = 0

        for loop in loops:
            pts = V[loop]
            if len(pts) < 3:
                continue

            bb = pts.max(axis=0) - pts.min(axis=0)
            diam = float(np.linalg.norm(bb))
            if diam > max_hole_diameter:
                continue

            # 使用 PCA 检查粗略的平整度，放宽限制允许一定的曲面
            c, n, _ = weighted_pca(pts)
            residual = np.abs(np.dot(pts - c, n))
            if residual.max() > max_plane_residual:
                continue

            # DP 最小权重三角化，直接利用已有顶点，避免产生中心突刺
            local_faces = triangulate_polygon_3d_dp(pts)
            if not local_faces:
                continue

            # 将局部索引映射回全局顶点索引
            mapped_faces = []
            for face in local_faces:
                mapped_faces.append([loop[face[0]], loop[face[1]], loop[face[2]]])

            F.extend(mapped_faces)
            added_faces_total += len(mapped_faces)
            filled_this_round += 1

        filled_total += filled_this_round

        if filled_this_round > 0:
            out = trimesh.Trimesh(vertices=V, faces=F, process=False)
            out = clean_mesh_light(out, fix_normals=True)
        else:
            break

    stats["filled_holes"] = int(filled_total)
    stats["added_faces"] = int(added_faces_total)

    if verbose:
        print(
            f"[2/8] 空洞修补 (DP-Triangulation)... "
            f"candidates={stats['candidate_loops']} "
            f"filled={stats['filled_holes']} "
            f"added_faces={stats['added_faces']}"
        )

    return out, stats


# =============================
# Cache
# =============================
def make_cache(mesh):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    FN = np.asarray(mesh.face_normals, dtype=np.float64)
    TC = np.asarray(mesh.triangles_center, dtype=np.float64)
    FA = np.asarray(mesh.area_faces, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(mesh.bounding_box.extents)) + 1e-12

    face_adj_pairs = np.asarray(mesh.face_adjacency, dtype=np.int32) if mesh.face_adjacency is not None else np.empty(
        (0, 2), dtype=np.int32)
    face_adj_csr = make_sparse_graph_from_pairs(len(F), face_adj_pairs) if SCIPY_AVAILABLE else None
    vert_adj_csr = build_vertex_adjacency_sparse(len(V), F) if SCIPY_AVAILABLE else None

    is_boundary = np.zeros(len(V), dtype=bool)
    b = boundary_vertices(mesh)
    if len(b):
        is_boundary[b] = True

    return {
        "vertices": V,
        "faces": F,
        "face_normals": FN,
        "tri_centers": TC,
        "face_areas": FA,
        "bbox_diag": bbox_diag,
        "face_adj_pairs": face_adj_pairs,
        "face_adj_csr": face_adj_csr,
        "vert_adj_csr": vert_adj_csr,
        "is_boundary": is_boundary,
    }


# =============================
# Plane detection
# =============================
def connected_groups_sparse(ids, csr_graph):
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) == 0:
        return []
    if len(ids) == 1:
        return [ids]

    if SCIPY_AVAILABLE and csr_graph is not None:
        sub = csr_graph[ids][:, ids]
        n_comp, labels = sp_connected_components(sub, directed=False, return_labels=True)
        return [ids[labels == i] for i in range(n_comp)]

    return [ids]


def fit_plane_faces(face_ids, cache):
    face_ids = np.asarray(face_ids, dtype=np.int32)
    centers = cache["tri_centers"][face_ids]
    areas = cache["face_areas"][face_ids]
    normals = cache["face_normals"][face_ids]

    c, n = robust_plane(centers, areas)
    dist = np.dot(centers - c, n)
    _, _, vals = weighted_pca(centers, areas)
    dots = abs_normal_dot(normals, n)

    return {
        "face_ids": face_ids,
        "centroid": c,
        "normal": n,
        "sigma": robust_sigma(dist),
        "area": float(np.sum(areas)),
        "planarity_ratio": float(vals[0] / (np.sum(vals) + 1e-12)),
        "mean_normal_angle": float(np.degrees(np.mean(np.arccos(np.clip(dots, 0.0, 1.0))))) if len(dots) else 0.0,
    }


def initial_patch_labels(cache, angle_thr=22.0):
    num_faces = len(cache["faces"])
    face_adj_pairs = cache["face_adj_pairs"]

    if num_faces == 0:
        return np.empty(0, dtype=np.int32), 0

    if not SCIPY_AVAILABLE or face_adj_pairs is None or len(face_adj_pairs) == 0:
        labels = np.arange(num_faces, dtype=np.int32)
        return labels, num_faces

    fn = cache["face_normals"]
    cos_thr = np.cos(np.radians(angle_thr))

    dots = np.abs(np.sum(fn[face_adj_pairs[:, 0]] * fn[face_adj_pairs[:, 1]], axis=1))
    sel = face_adj_pairs[dots >= cos_thr]

    if len(sel) == 0:
        labels = np.arange(num_faces, dtype=np.int32)
        return labels, num_faces

    rows = np.concatenate([sel[:, 0], sel[:, 1]])
    cols = np.concatenate([sel[:, 1], sel[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    g = sp.csr_matrix((data, (rows, cols)), shape=(num_faces, num_faces), dtype=np.uint8)

    n_comp, labels = sp_connected_components(g, directed=False, return_labels=True)
    return labels.astype(np.int32), int(n_comp)


def split_planes(face_ids, cache, cfg, depth=0):
    face_ids = np.asarray(face_ids, dtype=np.int32)
    if len(face_ids) < cfg["min_plane_faces"]:
        return []

    plane = fit_plane_faces(face_ids, cache)
    centers = cache["tri_centers"][face_ids]
    normals = cache["face_normals"][face_ids]

    d = np.dot(centers - plane["centroid"], plane["normal"])

    dc = dict(stage="none", detect_dist=1.0, detect_ang=1.0)
    if cfg["sensor_origin"] is not None and cfg["use_distance_strategy"]:
        dc = dist_cfg(
            np.linalg.norm(plane["centroid"] - cfg["sensor_origin"]),
            cache["bbox_diag"],
            cfg["near_ratio"],
            cfg["far_ratio"],
        )

    dist_thr = max(
        cfg["base_face_plane_dist"] * dc["detect_dist"],
        cfg["sigma_dist_mult"] * plane["sigma"] * dc["detect_dist"],
    )
    cos_thr = np.cos(np.radians(cfg["plane_normal_angle_deg"] * dc["detect_ang"]))
    dots = abs_normal_dot(normals, plane["normal"])
    inlier_mask = (np.abs(d) <= dist_thr) & (dots >= cos_thr)

    inlier = face_ids[inlier_mask]
    outlier = face_ids[~inlier_mask]
    res = []

    if len(inlier) >= cfg["min_plane_faces"]:
        for g in connected_groups_sparse(inlier, cache["face_adj_csr"]):
            if len(g) < cfg["min_plane_faces"]:
                continue
            if float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            p = fit_plane_faces(g, cache)
            if p["area"] >= cfg["min_plane_area"]:
                res.append(p)

    if depth < cfg["max_split_depth"] and len(outlier) >= cfg["min_plane_faces"]:
        for g in connected_groups_sparse(outlier, cache["face_adj_csr"]):
            if len(g) < cfg["min_plane_faces"]:
                continue
            if float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            res.extend(split_planes(g, cache, cfg, depth + 1))

    return res


def dedup_planes(planes, overlap=0.85):
    kept = []
    kept_ids = []

    for p in sorted(planes, key=lambda x: (len(x["face_ids"]), x["area"]), reverse=True):
        ids = np.sort(np.asarray(p["face_ids"], dtype=np.int32))
        ok = True
        for k_ids in kept_ids:
            inter = np.intersect1d(ids, k_ids, assume_unique=False).size
            denom = max(1, min(len(ids), len(k_ids)))
            if inter / denom >= overlap:
                ok = False
                break
        if ok:
            kept.append(p)
            kept_ids.append(ids)
    return kept


def detect_planes(mesh, cfg, verbose=False):
    if verbose:
        print("[4/8] 多平面识别...")

    cache = make_cache(mesh)
    planes = []

    patch_labels, patch_count = initial_patch_labels(cache, cfg["patch_normal_angle_deg"])
    face_counts = np.bincount(patch_labels, minlength=patch_count)
    area_sums = np.bincount(patch_labels, weights=cache["face_areas"], minlength=patch_count)

    valid_patch_ids = np.where(
        (face_counts >= cfg["min_patch_faces"]) &
        (area_sums >= cfg["min_plane_area"])
    )[0]

    for pid in valid_patch_ids:
        patch = np.where(patch_labels == pid)[0].astype(np.int32)
        planes.extend(split_planes(patch, cache, cfg, 0))

    planes = dedup_planes(planes)

    if verbose:
        print(f"  初始 patch 数: {patch_count}")
        print(f"  有效 patch 数: {len(valid_patch_ids)}")
        print(f"  识别平面数: {len(planes)}")

    return planes, cache


# =============================
# Plane refinement
# =============================
def expand_ring_sparse(ids, vert_adj_csr, rings=1):
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) == 0 or rings <= 0 or vert_adj_csr is None or not SCIPY_AVAILABLE:
        return np.unique(ids)

    mask = np.zeros(vert_adj_csr.shape[0], dtype=bool)
    mask[ids] = True
    cur = mask.copy()

    for _ in range(rings):
        nb = vert_adj_csr @ cur.astype(np.uint8)
        cur = cur | (nb > 0)

    return np.where(cur)[0].astype(np.int32)


def smooth_values_python(vals, valid, local_adj_csr, iters=8, self_w=0.25):
    x = vals.copy()
    nb_w = 1.0 - self_w

    indptr = local_adj_csr.indptr
    indices = local_adj_csr.indices
    valid_idx = np.where(valid)[0]

    for _ in range(iters):
        y = x.copy()
        for i in valid_idx:
            s, e = indptr[i], indptr[i + 1]
            ns = indices[s:e]
            if len(ns) == 0:
                continue
            vv = x[ns][valid[ns]]
            if vv.size:
                y[i] = self_w * x[i] + nb_w * float(np.mean(vv))
        x = y
    return x


def smooth_values_sparse(vals, valid, A, iters=8, self_w=0.25):
    if A is None or A.nnz == 0:
        return vals.astype(np.float64, copy=True)

    x = vals.astype(np.float64, copy=True)
    v = valid.astype(np.float64)
    nb_w = 1.0 - self_w

    denom = A @ v
    active = (v > 0.5) & (denom > 0)

    for _ in range(iters):
        avg = np.zeros_like(x)
        Ax = A @ x
        avg[active] = Ax[active] / denom[active]
        x[active] = self_w * x[active] + nb_w * avg[active]

    return x


def adaptive_smooth_iters(base_iters, sigma_v, eff_noise, dist_boost, stage_smooth_mul, pass_smooth_mul):
    n = int(round(base_iters * pass_smooth_mul * (1.0 + 0.35 * eff_noise + 0.20 * dist_boost) * stage_smooth_mul))
    if sigma_v < 0.0015:
        return max(4, min(16, n))
    if sigma_v < 0.003:
        return max(5, min(24, n))
    return max(6, min(40, n))


def refine_one_plane(task):
    i = task["plane_idx"]
    p = task["plane"]
    cfg = task["cfg"]
    pass_mul = task["pass_mul"]
    only_stage = task["only_stage"]
    state = task["state"]

    V = state["V"]
    F = state["F"]
    vert_adj_csr = state["vert_adj_csr"]
    is_boundary = state["is_boundary"]
    bbox_diag = state["bbox_diag"]

    face_ids = p["face_ids"]
    base_vids = np.unique(F[face_ids].reshape(-1))

    dc = dict(stage="none", snap=1, inlier=1, outlier=1, shrink=1, smooth=1, noise=1)
    plane_dist = None
    dist_boost = 0.0

    if cfg["sensor_origin"] is not None:
        plane_dist = float(np.linalg.norm(p["centroid"] - cfg["sensor_origin"]))
        dist_boost = cfg["distance_gain"] * (plane_dist / bbox_diag)
        if cfg["use_distance_strategy"]:
            dc = dist_cfg(plane_dist, bbox_diag, cfg["near_ratio"], cfg["far_ratio"])

    if only_stage is not None and dc["stage"] != only_stage:
        return None

    vids = expand_ring_sparse(base_vids, vert_adj_csr, cfg["expand_rings"] + int(pass_mul.get("expand_rings_add", 0)))
    pts = V[vids]

    c, n = robust_plane(pts, None, 6, 1.4)
    sd = np.dot(pts - c, n)
    sigma_v = robust_sigma(sd)

    noise_ratio = sigma_v / max(cfg["vertex_inlier_dist"], 1e-12)
    noise_boost = min(1.8, cfg["noise_adapt_gain"] * pass_mul.get("noise", 1.0) * noise_ratio)
    eff_noise = noise_boost * dc["noise"]

    smooth_iters = adaptive_smooth_iters(
        cfg["base_normal_smooth_iterations"],
        sigma_v,
        eff_noise,
        dist_boost,
        dc["smooth"],
        pass_mul.get("smooth", 1.0),
    )

    residual_shrink = (
            cfg["base_residual_shrink"]
            * pass_mul.get("shrink", 1.0)
            / (1.0 + 0.8 * eff_noise + 0.3 * dist_boost)
            * dc["shrink"]
    )
    residual_shrink = max(0.001, min(0.08, residual_shrink))

    inlier_thr = max(cfg["vertex_inlier_dist"] * dc["inlier"], 2.0 * sigma_v)
    outlier_thr = max(cfg["vertex_outlier_dist"] * pass_mul.get("outlier", 1.0) * dc["outlier"], 3.5 * sigma_v)
    snap_strength = cfg["base_snap_strength"] * pass_mul.get("snap", 1.0) * dc["snap"]

    valid = np.abs(sd) <= outlier_thr
    vals = np.where(valid, sd, 0.0)

    A_local = None
    if SCIPY_AVAILABLE and vert_adj_csr is not None and len(vids) > 0:
        A_local = vert_adj_csr[vids][:, vids]

    if SCIPY_AVAILABLE and A_local is not None:
        sm = smooth_values_sparse(vals, valid, A_local, smooth_iters, 0.25)
    else:
        if A_local is None:
            sm = vals.copy()
        else:
            sm = smooth_values_python(vals, valid, A_local, smooth_iters, 0.25)

    moved_vids = vids[valid]
    if len(moved_vids) == 0:
        return {
            "vids": np.empty(0, dtype=np.int32),
            "acc": np.empty((0, 3), dtype=np.float64),
            "w": np.empty(0, dtype=np.float64),
            "report": {
                "plane_id": int(i),
                "stage": dc["stage"],
                "plane_dist_sensor": plane_dist,
                "faces": int(len(face_ids)),
                "vertex_sigma": float(sigma_v),
                "inlier_thr": float(inlier_thr),
                "outlier_thr": float(outlier_thr),
                "smooth_iters": int(smooth_iters),
                "residual_shrink": float(residual_shrink),
            }
        }

    d = vals[valid]
    ds = sm[valid]
    bmul = np.where(is_boundary[moved_vids], cfg["boundary_scale"], 1.0)

    target = V[moved_vids].copy()
    inlier = np.abs(d) <= inlier_thr
    target[inlier] -= d[inlier, None] * n
    target[~inlier] -= (d[~inlier] - residual_shrink * ds[~inlier])[:, None] * n

    non_inlier_weight = min(1.0, 0.72 + 0.30 * eff_noise + 0.22 * dist_boost)
    w = np.where(inlier, snap_strength, non_inlier_weight) * bmul

    return {
        "vids": moved_vids.astype(np.int32),
        "acc": w[:, None] * target,
        "w": w,
        "report": {
            "plane_id": int(i),
            "stage": dc["stage"],
            "plane_dist_sensor": plane_dist,
            "faces": int(len(face_ids)),
            "vertex_sigma": float(sigma_v),
            "inlier_thr": float(inlier_thr),
            "outlier_thr": float(outlier_thr),
            "smooth_iters": int(smooth_iters),
            "residual_shrink": float(residual_shrink),
        }
    }


def refine_planes_parallel(mesh, planes, cfg, shared_cache, only_stage=None, pass_mul=None, verbose=False, tag="[5/8]",
                           num_workers=1):
    if not planes:
        return mesh.copy(), []

    pass_mul = pass_mul or {}
    mesh2 = mesh.copy()

    state = {
        "V": np.asarray(mesh.vertices, dtype=np.float64),
        "F": np.asarray(mesh.faces, dtype=np.int32),
        "vert_adj_csr": shared_cache["vert_adj_csr"],
        "is_boundary": shared_cache["is_boundary"] if cfg["boundary_protect"] else np.zeros(len(mesh.vertices),
                                                                                            dtype=bool),
        "bbox_diag": shared_cache["bbox_diag"],
    }

    tasks = [{
        "plane_idx": i,
        "plane": p,
        "cfg": cfg,
        "pass_mul": pass_mul,
        "only_stage": only_stage,
        "state": state,
    } for i, p in enumerate(planes)]

    if verbose:
        backend = "scipy.sparse" if SCIPY_AVAILABLE else "fallback"
        print(f"{tag} 平面精修... mode=thread, workers={num_workers}, smooth_backend={backend}")

    if num_workers is None or num_workers <= 1:
        results = [refine_one_plane(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(refine_one_plane, tasks))

    V = state["V"]
    acc = np.zeros_like(V, dtype=np.float64)
    wsum = np.zeros((len(V), 1), dtype=np.float64)
    reports = []

    for r in results:
        if r is None:
            continue
        if len(r["vids"]):
            acc[r["vids"]] += r["acc"]
            wsum[r["vids"], 0] += r["w"]
        reports.append(r["report"])

        if verbose:
            rr = r["report"]
            print(
                f"  plane={rr['plane_id']:03d} stage={rr['stage']:<4} faces={rr['faces']:5d} sigma={rr['vertex_sigma']:.6f} out={rr['outlier_thr']:.5f} shrink={rr['residual_shrink']:.5f} smooth={rr['smooth_iters']}")

    moved = wsum[:, 0] > 1e-12
    V2 = V.copy()
    V2[moved] = acc[moved] / wsum[moved]

    out = trimesh.Trimesh(vertices=V2, faces=mesh2.faces.copy(), process=False)
    try:
        trimesh.repair.fix_normals(out)
    except Exception:
        pass

    return out, sorted(reports, key=lambda x: x["plane_id"])


def select_far_second_pass_planes(planes, reports, min_faces=60, min_sigma=0.003):
    ids = [
        int(r["plane_id"])
        for r in reports
        if r.get("stage") == "far"
           and r.get("faces", 0) >= min_faces
           and r.get("vertex_sigma", 0.0) >= min_sigma
    ]
    return [planes[i] for i in ids]


# =============================
# Global smooth
# =============================
def global_smooth(mesh, iters=1, method="taubin"):
    if iters <= 0:
        return mesh.copy()

    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    m.compute_vertex_normals()

    if method == "taubin":
        m = m.filter_smooth_taubin(number_of_iterations=int(iters))
    elif method == "laplacian":
        m = m.filter_smooth_laplacian(number_of_iterations=int(iters))
    elif method == "simple":
        m = m.filter_smooth_simple(number_of_iterations=int(iters))
    else:
        raise ValueError("不支持的平滑方法")

    out = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles), process=False)
    try:
        trimesh.repair.fix_normals(out)
    except Exception:
        pass
    return out


# =============================
# Save
# =============================
def save_outputs(mesh, planes, reports1, reports2, before, after, out_dir, input_path, num_workers, timings,
                 hole_stats):
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(input_path))[0]

    obj_path = os.path.join(out_dir, f"{base}_refined.obj")
    ply_path = os.path.join(out_dir, f"{base}_refined.ply")
    planes_path = os.path.join(out_dir, f"{base}_planes.json")
    report_path = os.path.join(out_dir, f"{base}_quality_report.json")

    mesh.export(obj_path)
    mesh.export(ply_path)

    with open(planes_path, "w", encoding="utf-8") as f:
        json.dump({
            "detected_plane_count": len(planes),
            "planes": [
                {
                    "plane_id": i,
                    "faces": int(len(p["face_ids"])),
                    "area": float(p["area"]),
                    "sigma": float(p["sigma"]),
                    "centroid": [float(x) for x in p["centroid"]],
                    "normal": [float(x) for x in p["normal"]],
                } for i, p in enumerate(planes)
            ],
            "first_pass_reports": reports1,
            "far_second_pass_reports": reports2,
        }, f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "input": input_path,
            "output_obj": obj_path,
            "output_ply": ply_path,
            "planes_json": planes_path,
            "before": before,
            "after": after,
            "early_hole_fill": hole_stats,
            "scipy_sparse_enabled": SCIPY_AVAILABLE,
            "num_workers": int(num_workers),
            "timings_sec": timings,
        }, f, indent=2, ensure_ascii=False)

    return obj_path, ply_path, planes_path, report_path


# =============================
# Pipeline
# =============================
def process_mesh(args):
    total_t0 = time.perf_counter()

    sensor_origin = parse_vec3(args.sensor_origin)
    num_workers = args.num_workers if args.num_workers and args.num_workers > 0 else max(1, cpu_count() - 1)

    detect_cfg = {
        "patch_normal_angle_deg": args.patch_normal_angle_deg,
        "min_patch_faces": args.min_patch_faces,
        "min_plane_faces": args.min_plane_faces,
        "min_plane_area": args.min_plane_area,
        "base_face_plane_dist": args.base_face_plane_dist,
        "plane_normal_angle_deg": args.plane_normal_angle_deg,
        "sigma_dist_mult": args.sigma_dist_mult,
        "max_split_depth": args.max_split_depth,
        "sensor_origin": sensor_origin,
        "use_distance_strategy": args.use_distance_strategy,
        "near_ratio": args.near_ratio,
        "far_ratio": args.far_ratio,
    }

    refine_cfg = {
        "vertex_inlier_dist": args.vertex_inlier_dist,
        "vertex_outlier_dist": args.vertex_outlier_dist,
        "base_snap_strength": args.base_snap_strength,
        "base_residual_shrink": args.base_residual_shrink,
        "base_normal_smooth_iterations": args.base_normal_smooth_iterations,
        "noise_adapt_gain": args.noise_adapt_gain,
        "boundary_protect": not args.no_boundary_protect,
        "boundary_scale": args.boundary_scale,
        "expand_rings": args.expand_rings,
        "sensor_origin": sensor_origin,
        "distance_gain": args.distance_gain,
        "use_distance_strategy": args.use_distance_strategy,
        "near_ratio": args.near_ratio,
        "far_ratio": args.far_ratio,
    }

    timings = {}

    if args.verbose:
        print("========== PIPELINE START ==========")
        print(f"scipy sparse available: {SCIPY_AVAILABLE}")
        print(f"num_workers: {num_workers}")
        print("[0/8] 读取网格...")

    t0 = time.perf_counter()
    mesh0 = load_mesh(args.input)
    timings["load_mesh"] = round(time.perf_counter() - t0, 4)

    before = mesh_report(mesh0, include_components=False)

    if args.verbose:
        print(f"  load_mesh: {timings['load_mesh']:.3f}s")
        print("[1/8] 基础清理...")

    t0 = time.perf_counter()
    mesh = clean_mesh_light(mesh0, fix_normals=False)
    timings["clean_mesh_1"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    mesh, hole_stats = fill_small_holes_early(
        mesh,
        enable=not args.disable_early_hole_fill,
        max_hole_edges=args.max_hole_edges,
        max_hole_diameter=args.max_hole_diameter,
        max_hole_area=args.max_hole_area,
        max_plane_residual=args.max_hole_plane_residual,
        max_candidate_loops=args.max_hole_candidate_loops,
        try_trimesh_fill=not args.disable_trimesh_fill_holes,
        verbose=args.verbose
    )
    timings["early_hole_fill"] = round(time.perf_counter() - t0, 4) - timings["clean_mesh_1"]

    if args.verbose:
        print(f"  clean_mesh_1: {timings['clean_mesh_1']:.3f}s")
        print(f"  early_hole_fill: {timings['early_hole_fill']:.3f}s")
        print("[3/8] 连通分量过滤...")

    t0 = time.perf_counter()
    mesh = filter_components_fast(
        mesh,
        args.min_component_faces,
        args.min_component_area,
        args.min_component_max_extent,
        args.keep_top_k_faces
    )
    timings["filter_components"] = round(time.perf_counter() - t0, 4)

    if args.verbose:
        print(f"  filter_components: {timings['filter_components']:.3f}s")

    t0 = time.perf_counter()
    planes, shared_cache = detect_planes(mesh, detect_cfg, args.verbose)
    timings["detect_planes"] = round(time.perf_counter() - t0, 4)

    if args.verbose:
        print(f"  detect_planes: {timings['detect_planes']:.3f}s")

    t0 = time.perf_counter()
    mesh, reports1 = refine_planes_parallel(
        mesh,
        planes,
        refine_cfg,
        shared_cache=shared_cache,
        only_stage=None,
        pass_mul=None,
        verbose=args.verbose,
        tag="[5/8]",
        num_workers=num_workers
    )
    timings["refine_pass_1"] = round(time.perf_counter() - t0, 4)

    reports2 = []
    timings["refine_pass_2_far"] = 0.0
    if args.enable_far_second_pass:
        if args.verbose:
            print("[5.5/8] far second pass 选择平面...")
        selected = select_far_second_pass_planes(
            planes,
            reports1,
            args.far_second_pass_min_faces,
            args.far_second_pass_min_sigma
        )
        if args.verbose:
            print(f"  选中 far 平面数: {len(selected)}")

        if selected:
            pass_mul = {
                "outlier": args.far_second_pass_outlier_mul,
                "shrink": args.far_second_pass_shrink_mul,
                "smooth": args.far_second_pass_smooth_mul,
                "snap": args.far_second_pass_snap_mul,
                "noise": args.far_second_pass_noise_mul,
                "expand_rings_add": args.far_second_pass_expand_rings,
            }

            t0 = time.perf_counter()
            mesh, reports2 = refine_planes_parallel(
                mesh,
                selected,
                refine_cfg,
                shared_cache=shared_cache,
                only_stage=None,
                pass_mul=pass_mul,
                verbose=args.verbose,
                tag="[5.5/8]",
                num_workers=num_workers
            )
            timings["refine_pass_2_far"] = round(time.perf_counter() - t0, 4)

    if args.global_smooth_iter > 0:
        if args.verbose:
            print("[6/8] 全局平滑...")
        t0 = time.perf_counter()
        mesh = global_smooth(mesh, args.global_smooth_iter, args.global_smooth_method)
        timings["global_smooth"] = round(time.perf_counter() - t0, 4)
    else:
        timings["global_smooth"] = 0.0

    if args.verbose:
        print("[7/8] 去除游离噪点碎块...")
    t0 = time.perf_counter()
    mesh = remove_floating_noise_fast(
        mesh,
        min_faces=args.noise_min_faces,
        min_area=args.noise_min_area,
        min_extent=args.noise_min_extent,
        keep_top_k=args.noise_keep_top_k
    )
    timings["remove_floating_noise"] = round(time.perf_counter() - t0, 4)

    if args.verbose:
        print("[8/8] 最终轻量清理...")
    t0 = time.perf_counter()
    mesh = clean_mesh_light(mesh, fix_normals=True)
    timings["clean_mesh_final"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    after = mesh_report(mesh, include_components=True)
    timings["final_report"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    obj_path, ply_path, planes_path, report_path = save_outputs(
        mesh, planes, reports1, reports2, before, after,
        args.output_dir, args.input, num_workers, timings, hole_stats
    )
    timings["save_outputs"] = round(time.perf_counter() - t0, 4)

    timings["total"] = round(time.perf_counter() - total_t0, 4)

    print("\n========== PIPELINE DONE ==========")
    print(f"Output OBJ:     {obj_path}")
    print(f"Output PLY:     {ply_path}")
    print(f"Planes JSON:    {planes_path}")
    print(f"Quality Report: {report_path}")
    print("Timings (sec):")
    for k, v in timings.items():
        print(f"  {k}: {v:.3f}")
    print("===================================")


# =============================
# CLI
# =============================
def build_parser():
    p = argparse.ArgumentParser(description="Mesh refine pipeline (fast CPU version + DP hole filling)")

    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)

    # initial component filtering
    p.add_argument("--min_component_faces", type=int, default=80)
    p.add_argument("--min_component_area", type=float, default=0.0015)
    p.add_argument("--min_component_max_extent", type=float, default=0.04)
    p.add_argument("--keep_top_k_faces", type=int, default=20)

    # early hole fill - 显著放宽了限制
    p.add_argument("--disable_early_hole_fill", action="store_true")
    p.add_argument("--disable_trimesh_fill_holes", action="store_true")
    p.add_argument("--max_hole_edges", type=int, default=60, help="大幅提升边数限制以处理真实扫描空洞")
    p.add_argument("--max_hole_diameter", type=float, default=0.40, help="允许修补直径达40cm的洞")
    p.add_argument("--max_hole_area", type=float, default=0.05)
    p.add_argument("--max_hole_plane_residual", type=float, default=0.05, help="允许曲面、墙角上的孔洞")
    p.add_argument("--max_hole_candidate_loops", type=int, default=1500)

    # plane detection
    p.add_argument("--patch_normal_angle_deg", type=float, default=24.0)
    p.add_argument("--min_patch_faces", type=int, default=25)
    p.add_argument("--min_plane_faces", type=int, default=40)
    p.add_argument("--min_plane_area", type=float, default=0.0008)
    p.add_argument("--base_face_plane_dist", type=float, default=0.014)
    p.add_argument("--plane_normal_angle_deg", type=float, default=18.0)
    p.add_argument("--sigma_dist_mult", type=float, default=3.6)
    p.add_argument("--max_split_depth", type=int, default=7)

    # plane refine
    p.add_argument("--vertex_inlier_dist", type=float, default=0.008)
    p.add_argument("--vertex_outlier_dist", type=float, default=0.085)
    p.add_argument("--base_snap_strength", type=float, default=1.0)
    p.add_argument("--base_residual_shrink", type=float, default=0.006)
    p.add_argument("--base_normal_smooth_iterations", type=int, default=22)
    p.add_argument("--noise_adapt_gain", type=float, default=1.8)
    p.add_argument("--no_boundary_protect", action="store_true")
    p.add_argument("--boundary_scale", type=float, default=0.10)
    p.add_argument("--expand_rings", type=int, default=1)

    # distance strategy
    p.add_argument("--sensor_origin", type=str, default=None)
    p.add_argument("--distance_gain", type=float, default=0.45)
    p.add_argument("--use_distance_strategy", action="store_true")
    p.add_argument("--near_ratio", type=float, default=0.25)
    p.add_argument("--far_ratio", type=float, default=0.50)

    # far second pass
    p.add_argument("--enable_far_second_pass", action="store_true")
    p.add_argument("--far_second_pass_outlier_mul", type=float, default=1.60)
    p.add_argument("--far_second_pass_shrink_mul", type=float, default=0.45)
    p.add_argument("--far_second_pass_smooth_mul", type=float, default=1.80)
    p.add_argument("--far_second_pass_snap_mul", type=float, default=1.20)
    p.add_argument("--far_second_pass_noise_mul", type=float, default=1.30)
    p.add_argument("--far_second_pass_expand_rings", type=int, default=1)
    p.add_argument("--far_second_pass_min_sigma", type=float, default=0.003)
    p.add_argument("--far_second_pass_min_faces", type=int, default=60)

    # smooth
    p.add_argument("--global_smooth_iter", type=int, default=0)
    p.add_argument("--global_smooth_method", choices=["taubin", "laplacian", "simple"], default="taubin")

    # noise remove
    p.add_argument("--noise_min_faces", type=int, default=18)
    p.add_argument("--noise_min_area", type=float, default=0.00035)
    p.add_argument("--noise_min_extent", type=float, default=0.018)
    p.add_argument("--noise_keep_top_k", type=int, default=10)

    p.add_argument("--num_workers", type=int, default=0, help="<=0 自动使用 cpu_count()-1；建议 6~12")
    p.add_argument("--verbose", action="store_true")

    return p


def main():
    process_mesh(build_parser().parse_args())


if __name__ == "__main__":
    main()