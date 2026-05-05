import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import trimesh
# =============================
# Default run config / presets
# =============================
DEFAULT_INPUT = r".\input\imu_fusion_model_final_20260413_193242.ply"
DEFAULT_OUTPUT_DIR = r".\outputs_default"
DEFAULT_SENSOR_ORIGIN = "0,0,0"
DEFAULT_NUM_WORKERS = 8
DEFAULT_PRESET = "balanced"

PRESET_CONFIGS = {
    # 正式平衡版：优先推荐
    "balanced": {
        "patch_normal_angle_deg": 24.0,
        "min_patch_faces": 25,
        "min_plane_faces": 80,
        "min_plane_area": 0.0008,
        "base_face_plane_dist": 0.014,
        "plane_normal_angle_deg": 18.0,
        "sigma_dist_mult": 3.6,
        "max_split_depth": 3,
    },

    # 快速版：批处理/预览更合适
    "fast": {
        "patch_normal_angle_deg": 24.0,
        "min_patch_faces": 25,
        "min_plane_faces": 100,
        "min_plane_area": 0.0008,
        "base_face_plane_dist": 0.014,
        "plane_normal_angle_deg": 18.0,
        "sigma_dist_mult": 3.6,
        "max_split_depth": 4,
    },

    # 细节版：保留更多 plane
    "detail": {
        "patch_normal_angle_deg": 24.0,
        "min_patch_faces": 25,
        "min_plane_faces": 70,
        "min_plane_area": 0.0008,
        "base_face_plane_dist": 0.014,
        "plane_normal_angle_deg": 18.0,
        "sigma_dist_mult": 3.6,
        "max_split_depth": 4,
    },
}
try:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as sp_connected_components
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    sp = None
    sp_connected_components = None
    cKDTree = None
    SCIPY_AVAILABLE = False

# =============================
# CUDA / CuPy
# =============================
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

def cuda_enabled(args):
    return bool(getattr(args, "use_cuda", False) and CUPY_AVAILABLE)

def to_cpu(x):
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

def xp_module(use_gpu=False):
    return cp if (use_gpu and CUPY_AVAILABLE) else np

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

def robust_sigma(x, use_gpu=False):
    xp = xp_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)
    if x.size == 0:
        return 0.0
    med = xp.median(x)
    mad = xp.median(xp.abs(x - med)) + 1e-12
    out = 1.4826 * mad
    if use_gpu and CUPY_AVAILABLE:
        return float(cp.asnumpy(out))
    return float(out)

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

def abs_normal_dot(normals, normal, use_gpu=False):
    xp = xp_module(use_gpu)
    arr_n = xp.asarray(normals, dtype=xp.float64)
    n = xp.asarray(normal, dtype=xp.float64)
    out = xp.clip(xp.abs(arr_n @ n), 0.0, 1.0)
    return to_cpu(out) if use_gpu else out

# =============================
# PCA / Plane
# =============================
def weighted_pca(points, weights=None, use_gpu=False):
    xp = xp_module(use_gpu)
    p = xp.asarray(points, dtype=xp.float64)

    if p.shape[0] == 0:
        return np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3)

    w = xp.ones(len(p), dtype=xp.float64) if weights is None else xp.asarray(weights, dtype=xp.float64)
    sw = xp.sum(w) + 1e-12
    c = xp.sum(p * w[:, None], axis=0) / sw
    x = (p - c) * xp.sqrt(w[:, None])
    cov = (x.T @ x) / sw
    vals, vecs = xp.linalg.eigh(cov)
    n = vecs[:, 0]
    n = n / (xp.linalg.norm(n) + 1e-12)

    if use_gpu and CUPY_AVAILABLE:
        return cp.asnumpy(c), cp.asnumpy(n), cp.asnumpy(vals)
    return np.asarray(c), np.asarray(n), np.asarray(vals)

def robust_plane(points, weights=None, iters=6, huber_k=1.5, use_gpu=False):
    xp = xp_module(use_gpu)
    p = xp.asarray(points, dtype=xp.float64)

    if p.shape[0] == 0:
        return np.zeros(3), np.array([0.0, 0.0, 1.0], dtype=np.float64)

    w = xp.ones(len(p), dtype=xp.float64) if weights is None else xp.asarray(weights, dtype=xp.float64).copy()
    c = xp.mean(p, axis=0)
    n = xp.asarray([0.0, 0.0, 1.0], dtype=xp.float64)

    for _ in range(iters):
        c_cpu, n_cpu, _ = weighted_pca(to_cpu(p), to_cpu(w), use_gpu=use_gpu)
        c = xp.asarray(c_cpu, dtype=xp.float64)
        n = xp.asarray(n_cpu, dtype=xp.float64)
        d = (p - c) @ n
        s = robust_sigma(d, use_gpu=use_gpu) + 1e-12
        r = xp.abs(d) / (huber_k * s + 1e-12)
        w = xp.where(r <= 1.0, w, w / (r + 1e-12))

    if use_gpu and CUPY_AVAILABLE:
        return cp.asnumpy(c), cp.asnumpy(n)
    return np.asarray(c), np.asarray(n)

def dist_cfg(plane_dist, bbox_diag, near_ratio=0.25, far_ratio=0.50):
    r = plane_dist / (bbox_diag + 1e-12)
    if r <= near_ratio:
        return dict(stage="near", snap=1.00, inlier=0.85, outlier=1.00, shrink=0.70, smooth=1.20, noise=1.00, detect_dist=1.00, detect_ang=1.00)
    if r <= far_ratio:
        return dict(stage="mid", snap=1.10, inlier=1.05, outlier=1.45, shrink=0.42, smooth=1.80, noise=1.35, detect_dist=1.20, detect_ang=1.12)
    return dict(stage="far", snap=1.22, inlier=1.35, outlier=2.40, shrink=0.12, smooth=2.80, noise=2.10, detect_dist=1.65, detect_ang=1.35)

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
        be = get_boundary_edges(mesh)
        if len(be) == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(be.reshape(-1)).astype(np.int32, copy=False)
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
        comps = list(mesh.split(only_watertight=False))
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
# Hole filling and gap stitching
# =============================
def plane_basis_from_normal(normal):
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    if abs(n[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v

def polygon_area_2d(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def point_in_triangle_2d(p, a, b, c, eps=1e-12):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)

def is_convex_corner(a, b, c, ccw=True, eps=1e-12):
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return cross > eps if ccw else cross < -eps

def ear_clip_triangulation_2d(poly):
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    area = polygon_area_2d(poly)
    if abs(area) < 1e-12:
        return []

    ccw = area > 0
    idx = list(range(n))
    tris = []
    guard = 0
    max_guard = n * n

    while len(idx) > 3 and guard < max_guard:
        ear_found = False
        m = len(idx)

        for i in range(m):
            i_prev = idx[(i - 1) % m]
            i_curr = idx[i]
            i_next = idx[(i + 1) % m]

            a = poly[i_prev]
            b = poly[i_curr]
            c = poly[i_next]

            if not is_convex_corner(a, b, c, ccw=ccw):
                continue

            has_inside = False
            for j in idx:
                if j in (i_prev, i_curr, i_next):
                    continue
                if point_in_triangle_2d(poly[j], a, b, c):
                    has_inside = True
                    break
            if has_inside:
                continue

            tris.append((i_prev, i_curr, i_next))
            del idx[i]
            ear_found = True
            break

        if not ear_found:
            return []

        guard += 1

    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))

    return tris

def get_boundary_edges(mesh):
    try:
        edges_u = np.asarray(mesh.edges_unique, dtype=np.int32)
        if len(edges_u) == 0:
            return np.empty((0, 2), dtype=np.int32)

        inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64).reshape(-1)
        counts = np.bincount(inv, minlength=len(edges_u))

        return edges_u[counts == 1]
    except Exception:
        return np.empty((0, 2), dtype=np.int32)

def build_boundary_data(mesh):
    boundary = get_boundary_edges(mesh)
    adj = {}
    edge_set = set()

    for a, b in boundary:
        a = int(a)
        b = int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        edge_set.add((a, b) if a < b else (b, a))

    return boundary, adj, edge_set

def build_boundary_cache(mesh):
    boundary, adj, edge_set = build_boundary_data(mesh)
    if len(boundary) == 0:
        boundary_vs = np.empty(0, dtype=np.int32)
    else:
        boundary_vs = np.unique(boundary.reshape(-1)).astype(np.int32, copy=False)
    return {
        "boundary": boundary,
        "adj": adj,
        "edge_set": edge_set,
        "boundary_vs": boundary_vs,
    }

def extract_boundary_loops_ordered(mesh, max_loop_edges=128, max_loops=4000, boundary_cache=None):
    if boundary_cache is None:
        boundary, adj, _ = build_boundary_data(mesh)
    else:
        boundary = boundary_cache["boundary"]
        adj = boundary_cache["adj"]

    if len(boundary) == 0:
        return []

    valid_vertices = {v for v, ns in adj.items() if len(ns) == 2}
    if not valid_vertices:
        return []

    valid_edges = set()
    for a, b in boundary:
        a = int(a)
        b = int(b)
        if a in valid_vertices and b in valid_vertices:
            valid_edges.add((a, b) if a < b else (b, a))

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    visited = set()
    loops = []

    for a, b in list(valid_edges):
        if (a, b) in visited:
            continue

        start = a
        prev = a
        cur = b
        loop = [start]
        visited.add((a, b))

        ok = True
        while True:
            loop.append(cur)
            if len(loop) > max_loop_edges:
                ok = False
                break

            ns = adj.get(cur, [])
            if len(ns) != 2:
                ok = False
                break

            nxt = ns[0] if ns[1] == prev else ns[1]

            if nxt == start:
                break

            ek = edge_key(cur, nxt)
            if ek in visited:
                ok = False
                break

            visited.add(ek)
            prev, cur = cur, nxt

        if ok and 3 <= len(loop) <= max_loop_edges:
            loops.append(np.array(loop, dtype=np.int32))
            if len(loops) >= max_loops:
                break

    uniq = []
    seen = set()
    for loop in loops:
        key = min(tuple(loop.tolist()), tuple(loop[::-1].tolist()))
        if key not in seen:
            seen.add(key)
            uniq.append(loop)

    return uniq

def extract_open_boundary_chains(mesh, max_chain_edges=800, max_chains=12000, boundary_cache=None):
    if boundary_cache is None:
        boundary, adj, _ = build_boundary_data(mesh)
    else:
        boundary = boundary_cache["boundary"]
        adj = boundary_cache["adj"]

    if len(boundary) == 0:
        return []

    endpoints = [v for v, ns in adj.items() if len(ns) == 1]
    if not endpoints:
        return []

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    visited = set()
    chains = []

    for start in endpoints:
        ns = adj.get(start, [])
        if len(ns) != 1:
            continue

        nxt = ns[0]
        if edge_key(start, nxt) in visited:
            continue

        chain = [start]
        prev = start
        cur = nxt
        visited.add(edge_key(start, nxt))

        while True:
            chain.append(cur)
            if len(chain) > max_chain_edges:
                break

            ns_cur = adj.get(cur, [])
            if len(ns_cur) == 1:
                break
            if len(ns_cur) != 2:
                break

            nxt2 = ns_cur[0] if ns_cur[1] == prev else ns_cur[1]
            ek = edge_key(cur, nxt2)
            if ek in visited:
                break

            visited.add(ek)
            prev, cur = cur, nxt2

        if len(chain) >= 2:
            chains.append(np.array(chain, dtype=np.int32))
            if len(chains) >= max_chains:
                break

    uniq = []
    seen = set()
    for ch in chains:
        key = min(tuple(ch.tolist()), tuple(ch[::-1].tolist()))
        if key not in seen:
            seen.add(key)
            uniq.append(ch)

    return uniq

def chain_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))

def endpoint_tangent(points, at_start=True):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if at_start:
        t = pts[1] - pts[0]
    else:
        t = pts[-1] - pts[-2]
    n = np.linalg.norm(t) + 1e-12
    return t / n

def mean_vertex_normal(vn, ids):
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.mean(vn[ids], axis=0)
    n = np.linalg.norm(x) + 1e-12
    return x / n

def chain_plane_residual(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        c = pts.mean(axis=0) if len(pts) > 0 else np.zeros(3, dtype=np.float64)
        return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float64), c
    c, n, _ = weighted_pca(pts)
    d = np.abs(np.dot(pts - c, n))
    return float(np.max(d)), n, c

def orient_chain_pair_for_min_gap(A_ids, B_ids, V):
    cands = [
        (A_ids, B_ids),
        (A_ids[::-1], B_ids),
        (A_ids, B_ids[::-1]),
        (A_ids[::-1], B_ids[::-1]),
    ]

    best = None
    best_score = None

    for a_ids, b_ids in cands:
        a0 = V[int(a_ids[0])]
        a1 = V[int(a_ids[-1])]
        b0 = V[int(b_ids[0])]
        b1 = V[int(b_ids[-1])]
        score = np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)
        if best_score is None or score < best_score:
            best_score = score
            best = (np.asarray(a_ids, dtype=np.int32), np.asarray(b_ids, dtype=np.int32))

    return best

def is_polygon_convex_2d(poly, eps=1e-12):
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 4:
        return True

    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) <= eps:
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif sign != cur:
            return False
    return True

def fan_triangulation_2d(poly):
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return []
    ccw = polygon_area_2d(poly) > 0
    tris = []
    for i in range(1, n - 1):
        if ccw:
            tris.append((0, i, i + 1))
        else:
            tris.append((0, i + 1, i))
    return tris
def gpu_boundary_candidate_pairs(boundary_vs, V, VN, max_bridge_dist, normal_dot_min, block_size=1024):
    if not CUPY_AVAILABLE:
        return np.empty((0, 2), dtype=np.int32)

    ids = np.asarray(boundary_vs, dtype=np.int32)
    if len(ids) < 2:
        return np.empty((0, 2), dtype=np.int32)

    pts = cp.asarray(V[ids], dtype=cp.float64)
    nrm = cp.asarray(VN[ids], dtype=cp.float64)
    r2 = float(max_bridge_dist * max_bridge_dist)

    pairs_all = []

    for i0 in range(0, len(ids), block_size):
        i1 = min(i0 + block_size, len(ids))
        A = pts[i0:i1]
        AN = nrm[i0:i1]

        d2 = cp.sum((A[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        nd = cp.abs(AN @ nrm.T)

        mask = (d2 > 1e-24) & (d2 <= r2) & (nd >= normal_dot_min)

        ii, jj = cp.where(mask)
        if ii.size == 0:
            continue

        ii = cp.asnumpy(ii) + i0
        jj = cp.asnumpy(jj)

        pairs = np.column_stack([ids[ii], ids[jj]]).astype(np.int32)
        pairs_all.append(pairs)

    if not pairs_all:
        return np.empty((0, 2), dtype=np.int32)

    return np.vstack(pairs_all)

def gpu_chain_endpoint_candidates(chains, V, radius, block_size=2048):
    candidate_map = [set() for _ in range(len(chains))]
    if not CUPY_AVAILABLE:
        return candidate_map

    valid_chain_ids = [i for i, ch in enumerate(chains) if len(ch) >= 2]
    if len(valid_chain_ids) < 2:
        return candidate_map

    endpoint_pts = []
    endpoint_owner = []

    for i in valid_chain_ids:
        ch = chains[i]
        endpoint_pts.append(V[int(ch[0])])
        endpoint_pts.append(V[int(ch[-1])])
        endpoint_owner.append(i)
        endpoint_owner.append(i)

    endpoint_pts = np.asarray(endpoint_pts, dtype=np.float64)
    owner = np.asarray(endpoint_owner, dtype=np.int32)

    P = cp.asarray(endpoint_pts, dtype=cp.float64)
    r2 = float(radius * radius)

    for i0 in range(0, len(endpoint_pts), block_size):
        i1 = min(i0 + block_size, len(endpoint_pts))
        A = P[i0:i1]

        d2 = cp.sum((A[:, None, :] - P[None, :, :]) ** 2, axis=2)
        mask = (d2 > 1e-24) & (d2 <= r2)

        ii, jj = cp.where(mask)
        if ii.size == 0:
            continue

        ii = cp.asnumpy(ii) + i0
        jj = cp.asnumpy(jj)

        for a_idx, b_idx in zip(ii, jj):
            ca = int(owner[a_idx])
            cb = int(owner[b_idx])
            if ca != cb:
                candidate_map[ca].add(cb)

    return candidate_map

def best_match_from_candidate_pairs(candidate_pairs, V, adj, edge_set, VN, normal_dot_min, max_bridge_dist):
    if len(candidate_pairs) == 0:
        return {}

    V = np.asarray(V, dtype=np.float64)
    VN = np.asarray(VN, dtype=np.float64)
    pairs = np.asarray(candidate_pairs, dtype=np.int32)

    gi = pairs[:, 0]
    gj = pairs[:, 1]

    Pi = V[gi]
    Pj = V[gj]

    d = np.linalg.norm(Pi - Pj, axis=1)
    nd = np.abs(np.sum(VN[gi] * VN[gj], axis=1))

    mask = (d > 1e-12) & (d <= float(max_bridge_dist)) & (nd >= float(normal_dot_min))
    if not np.any(mask):
        return {}

    gi = gi[mask]
    gj = gj[mask]
    d = d[mask]

    # 预构建 set，避免在内层重复构造
    adj_set = {int(k): set(v) for k, v in adj.items()}

    best_match = {}
    best_dist = {}

    def consider(a, b, dist_ab):
        if a == b:
            return
        if edge_exists(edge_set, a, b):
            return
        if len(adj_set.get(a, set()) & adj_set.get(b, set())) > 0:
            return

        old = best_dist.get(a, None)
        if old is None or dist_ab < old:
            best_dist[a] = float(dist_ab)
            best_match[a] = int(b)

    for a, b, dist_ab in zip(gi, gj, d):
        a = int(a)
        b = int(b)
        consider(a, b, dist_ab)

    return best_match
def knn_candidate_pairs_boundary(boundary_vs, V, k_neighbors=8, max_dist=np.inf):
    ids = np.asarray(boundary_vs, dtype=np.int32)
    if len(ids) < 2:
        return np.empty((0, 2), dtype=np.int32)

    pts = np.asarray(V[ids], dtype=np.float64)

    if SCIPY_AVAILABLE and cKDTree is not None:
        tree = cKDTree(pts)
        k = min(int(k_neighbors) + 1, len(ids))
        dists, idxs = tree.query(pts, k=k)

        if k == 1:
            return np.empty((0, 2), dtype=np.int32)

        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

        src = np.repeat(np.arange(len(ids), dtype=np.int32), k - 1)
        dst = idxs[:, 1:].reshape(-1).astype(np.int32)
        dd = dists[:, 1:].reshape(-1)

        mask = np.isfinite(dd)
        if np.isfinite(max_dist):
            mask &= (dd <= float(max_dist))

        if not np.any(mask):
            return np.empty((0, 2), dtype=np.int32)

        src = src[mask]
        dst = dst[mask]

        return np.column_stack([ids[src], ids[dst]]).astype(np.int32)

    # fallback
    pairs = []
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        order = np.argsort(d)
        used = 0
        for j in order:
            if j == i:
                continue
            if np.isfinite(max_dist) and d[j] > max_dist:
                continue
            pairs.append([ids[i], ids[j]])
            used += 1
            if used >= k_neighbors:
                break

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)

    return np.asarray(pairs, dtype=np.int32)
def knn_candidate_map_for_chains(chains, V, k_neighbors=8, max_dist=np.inf):
    n = len(chains)
    candidate_map = [set() for _ in range(n)]

    valid_chain_ids = [i for i, ch in enumerate(chains) if len(ch) >= 2]
    if len(valid_chain_ids) < 2:
        return candidate_map

    endpoint_pts = []
    endpoint_owner = []

    for i in valid_chain_ids:
        ch = chains[i]
        endpoint_pts.append(V[int(ch[0])])
        endpoint_owner.append(i)
        endpoint_pts.append(V[int(ch[-1])])
        endpoint_owner.append(i)

    endpoint_pts = np.asarray(endpoint_pts, dtype=np.float64)
    endpoint_owner = np.asarray(endpoint_owner, dtype=np.int32)

    if SCIPY_AVAILABLE and cKDTree is not None:
        tree = cKDTree(endpoint_pts)
        k = min(int(k_neighbors) + 1, len(endpoint_pts))
        dists, idxs = tree.query(endpoint_pts, k=k)

        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

        for ep_idx in range(len(endpoint_pts)):
            src_chain = int(endpoint_owner[ep_idx])
            for d, j_ep in zip(dists[ep_idx, 1:], idxs[ep_idx, 1:]):
                if not np.isfinite(d):
                    continue
                if np.isfinite(max_dist) and d > max_dist:
                    continue
                dst_chain = int(endpoint_owner[j_ep])
                if dst_chain != src_chain:
                    candidate_map[src_chain].add(dst_chain)
    else:
        for ii in range(len(valid_chain_ids)):
            i = valid_chain_ids[ii]
            ai0 = V[int(chains[i][0])]
            ai1 = V[int(chains[i][-1])]
            ranked = []
            for jj in range(len(valid_chain_ids)):
                if ii == jj:
                    continue
                j = valid_chain_ids[jj]
                bj0 = V[int(chains[j][0])]
                bj1 = V[int(chains[j][-1])]
                d = min(
                    np.linalg.norm(ai0 - bj0),
                    np.linalg.norm(ai0 - bj1),
                    np.linalg.norm(ai1 - bj0),
                    np.linalg.norm(ai1 - bj1),
                )
                if np.isfinite(max_dist) and d > max_dist:
                    continue
                ranked.append((d, j))
            ranked.sort(key=lambda x: x[0])
            for _, j in ranked[:k_neighbors]:
                candidate_map[i].add(j)

    return candidate_map
def endpoint_candidate_map_for_chains(chains, V, radius, max_neighbors_per_chain=64, use_gpu=False):
    n = len(chains)
    candidate_map = [set() for _ in range(n)]

    if n < 2:
        return candidate_map

    valid_chain_ids = [i for i, ch in enumerate(chains) if len(ch) >= 2]
    if len(valid_chain_ids) < 2:
        return candidate_map

    if use_gpu and CUPY_AVAILABLE:
        candidate_map = gpu_chain_endpoint_candidates(
            chains,
            V,
            radius=radius,
            block_size=2048,
        )
    elif not SCIPY_AVAILABLE or cKDTree is None:
        for ii in range(len(valid_chain_ids)):
            i = valid_chain_ids[ii]
            ai0 = V[int(chains[i][0])]
            ai1 = V[int(chains[i][-1])]
            for jj in range(ii + 1, len(valid_chain_ids)):
                j = valid_chain_ids[jj]
                bj0 = V[int(chains[j][0])]
                bj1 = V[int(chains[j][-1])]
                d = min(
                    np.linalg.norm(ai0 - bj0),
                    np.linalg.norm(ai0 - bj1),
                    np.linalg.norm(ai1 - bj0),
                    np.linalg.norm(ai1 - bj1),
                )
                if d <= radius:
                    candidate_map[i].add(j)
                    candidate_map[j].add(i)
        return candidate_map
    else:
        endpoint_pts = []
        endpoint_owner = []
        for i in valid_chain_ids:
            ch = chains[i]
            endpoint_pts.append(V[int(ch[0])])
            endpoint_owner.append(i)
            endpoint_pts.append(V[int(ch[-1])])
            endpoint_owner.append(i)

        endpoint_pts = np.asarray(endpoint_pts, dtype=np.float64)
        tree = cKDTree(endpoint_pts)
        neighs = tree.query_ball_point(endpoint_pts, r=radius)

        for ep_idx, js in enumerate(neighs):
            i = endpoint_owner[ep_idx]
            for j_ep in js:
                j = endpoint_owner[j_ep]
                if j == i:
                    continue
                candidate_map[i].add(j)

    for i in valid_chain_ids:
        if len(candidate_map[i]) <= max_neighbors_per_chain:
            continue

        ai0 = V[int(chains[i][0])]
        ai1 = V[int(chains[i][-1])]
        ranked = []
        for j in candidate_map[i]:
            bj0 = V[int(chains[j][0])]
            bj1 = V[int(chains[j][-1])]
            d = min(
                np.linalg.norm(ai0 - bj0),
                np.linalg.norm(ai0 - bj1),
                np.linalg.norm(ai1 - bj0),
                np.linalg.norm(ai1 - bj1),
            )
            ranked.append((d, j))
        ranked.sort(key=lambda x: x[0])
        candidate_map[i] = set(j for _, j in ranked[:max_neighbors_per_chain])

    return candidate_map

def fill_small_holes_from_loops(
    mesh,
    max_hole_edges=160,
    max_hole_diameter=1.5,
    max_hole_area=1.2,
    max_plane_residual=0.10,
    max_candidate_loops=8000,
    boundary_cache=None,
):
    stats = {
        "candidate_loops": 0,
        "accepted_loops": 0,
        "filled_holes": 0,
        "added_faces": 0,
    }

    if boundary_cache is None:
        boundary_cache = build_boundary_cache(mesh)

    loops = extract_boundary_loops_ordered(
        mesh,
        max_loop_edges=max_hole_edges,
        max_loops=max_candidate_loops,
        boundary_cache=boundary_cache,
    )
    stats["candidate_loops"] = int(len(loops))
    if len(loops) == 0:
        return mesh.copy(), stats

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    new_faces = []

    for loop in loops:
        pts = V[loop]
        if len(pts) < 3:
            continue

        bb = pts.max(axis=0) - pts.min(axis=0)
        diam = float(np.linalg.norm(bb))
        if diam > max_hole_diameter:
            continue

        c, n, _ = weighted_pca(pts)
        residual = np.abs(np.dot(pts - c, n))
        if residual.max() > max_plane_residual:
            continue

        u, v = plane_basis_from_normal(n)
        poly2d = np.column_stack(((pts - c) @ u, (pts - c) @ v))
        area = abs(polygon_area_2d(poly2d))
        if area <= 1e-12 or area > max_hole_area:
            continue

        if is_polygon_convex_2d(poly2d):
            tris_local = fan_triangulation_2d(poly2d)
        else:
            tris_local = ear_clip_triangulation_2d(poly2d)

        if len(tris_local) == 0:
            continue

        stats["accepted_loops"] += 1
        stats["filled_holes"] += 1

        for a, b, cidx in tris_local:
            ia = int(loop[a])
            ib = int(loop[b])
            ic = int(loop[cidx])
            if ia == ib or ib == ic or ia == ic:
                continue
            new_faces.append([ia, ib, ic])

    if len(new_faces) == 0:
        return mesh.copy(), stats

    F2 = np.vstack([F, np.asarray(new_faces, dtype=np.int32)])
    out = trimesh.Trimesh(vertices=V.copy(), faces=F2, process=False)
    out = clean_mesh_light(out, fix_normals=False)
    stats["added_faces"] = int(len(new_faces))
    return out, stats

def estimate_vertex_normals_fast(mesh):
    try:
        return np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception:
        v = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
        return v

def edge_exists(edge_set, a, b):
    k = (a, b) if a < b else (b, a)
    return k in edge_set

def triangle_area_3d(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def zipper_triangulate_between_chains(A_ids, B_ids, V, min_triangle_area=1e-8):
    A_ids = np.asarray(A_ids, dtype=np.int32)
    B_ids = np.asarray(B_ids, dtype=np.int32)

    if len(A_ids) < 2 or len(B_ids) < 2:
        return []

    i = 0
    j = 0
    tris = []

    while i < len(A_ids) - 1 or j < len(B_ids) - 1:
        if i == len(A_ids) - 1:
            a = int(A_ids[i])
            b = int(B_ids[j])
            c = int(B_ids[j + 1])
            if len({a, b, c}) == 3 and triangle_area_3d(V[a], V[b], V[c]) > min_triangle_area:
                tris.append([a, b, c])
            j += 1
            continue

        if j == len(B_ids) - 1:
            a = int(A_ids[i])
            b = int(A_ids[i + 1])
            c = int(B_ids[j])
            if len({a, b, c}) == 3 and triangle_area_3d(V[a], V[b], V[c]) > min_triangle_area:
                tris.append([a, b, c])
            i += 1
            continue

        a0 = int(A_ids[i])
        a1 = int(A_ids[i + 1])
        b0 = int(B_ids[j])
        b1 = int(B_ids[j + 1])

        cost_a = np.linalg.norm(V[a1] - V[b0])
        cost_b = np.linalg.norm(V[a0] - V[b1])

        if cost_a <= cost_b:
            if len({a0, a1, b0}) == 3 and triangle_area_3d(V[a0], V[a1], V[b0]) > min_triangle_area:
                tris.append([a0, a1, b0])
            i += 1
        else:
            if len({a0, b0, b1}) == 3 and triangle_area_3d(V[a0], V[b0], V[b1]) > min_triangle_area:
                tris.append([a0, b0, b1])
            j += 1

    return tris

def stitch_boundary_gaps(
    mesh,
    max_bridge_dist=0.12,
    normal_dot_min=0.55,
    max_bridge_pairs=6000,
    min_triangle_area=1e-8,
    boundary_cache=None,
    use_gpu_prefilter=False,
    gpu_min_boundary_vertices=12000,
    knn_neighbors=8,
    use_knn_candidates=True,
):
    stats = {
        "boundary_vertices": 0,
        "candidate_pairs": 0,
        "mutual_pairs": 0,
        "bridge_quads": 0,
        "added_faces": 0,
        "gpu_prefilter_used": False,
        "candidate_mode": "none",
    }

    if boundary_cache is None:
        boundary_cache = build_boundary_cache(mesh)

    boundary = boundary_cache["boundary"]
    adj = boundary_cache["adj"]
    edge_set = boundary_cache["edge_set"]
    boundary_vs = boundary_cache["boundary_vs"]

    if len(boundary) == 0:
        return mesh.copy(), stats

    stats["boundary_vertices"] = int(len(boundary_vs))
    if len(boundary_vs) < 4:
        return mesh.copy(), stats

    V = np.asarray(mesh.vertices, dtype=np.float64)
    VN = estimate_vertex_normals_fast(mesh)

    best_match = {}

    use_gpu_now = bool(
        use_gpu_prefilter
        and CUPY_AVAILABLE
        and len(boundary_vs) >= int(gpu_min_boundary_vertices)
        and not use_knn_candidates
    )

    if use_knn_candidates:
        candidate_pairs = knn_candidate_pairs_boundary(
            boundary_vs,
            V,
            k_neighbors=knn_neighbors,
            max_dist=max_bridge_dist,
        )
        stats["candidate_mode"] = "knn"

        best_match = best_match_from_candidate_pairs(
            candidate_pairs,
            V=V,
            adj=adj,
            edge_set=edge_set,
            VN=VN,
            normal_dot_min=normal_dot_min,
            max_bridge_dist=max_bridge_dist,
        )

    elif use_gpu_now:
        candidate_pairs = gpu_boundary_candidate_pairs(
            boundary_vs,
            V,
            VN,
            max_bridge_dist=max_bridge_dist,
            normal_dot_min=normal_dot_min,
            block_size=1024,
        )
        stats["gpu_prefilter_used"] = True
        stats["candidate_mode"] = "gpu_radius"

        best_match = best_match_from_candidate_pairs(
            candidate_pairs,
            V=V,
            adj=adj,
            edge_set=edge_set,
            VN=VN,
            normal_dot_min=normal_dot_min,
            max_bridge_dist=max_bridge_dist,
        )

    else:
        pts = V[boundary_vs]

        if SCIPY_AVAILABLE and cKDTree is not None:
            tree = cKDTree(pts)
            try:
                local_pairs = tree.query_pairs(r=max_bridge_dist, output_type="ndarray")
            except TypeError:
                local_pairs = np.array(list(tree.query_pairs(r=max_bridge_dist)), dtype=np.int32)
        else:
            local_pairs_list = []
            for i in range(len(pts)):
                d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
                js = np.where(d <= max_bridge_dist)[0]
                for j in js:
                    local_pairs_list.append([i, i + 1 + int(j)])
            local_pairs = np.asarray(local_pairs_list, dtype=np.int32) if local_pairs_list else np.empty((0, 2), dtype=np.int32)

        if len(local_pairs) > 0:
            candidate_pairs = np.column_stack([
                boundary_vs[local_pairs[:, 0]],
                boundary_vs[local_pairs[:, 1]],
            ]).astype(np.int32)

            candidate_pairs = np.vstack([
                candidate_pairs,
                candidate_pairs[:, ::-1]
            ])
            stats["candidate_mode"] = "cpu_radius"

            best_match = best_match_from_candidate_pairs(
                candidate_pairs,
                V=V,
                adj=adj,
                edge_set=edge_set,
                VN=VN,
                normal_dot_min=normal_dot_min,
                max_bridge_dist=max_bridge_dist,
            )

    stats["candidate_pairs"] = int(len(best_match))
    if not best_match:
        return mesh.copy(), stats

    mutual_pairs = []
    seen = set()
    for a, b in best_match.items():
        if best_match.get(b, None) == a:
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                mutual_pairs.append(key)

    stats["mutual_pairs"] = int(len(mutual_pairs))
    if len(mutual_pairs) < 2:
        return mesh.copy(), stats

    partner = {}
    for a, b in mutual_pairs[:max_bridge_pairs]:
        partner[a] = b
        partner[b] = a

    new_faces = []
    used_quads = set()

    for a, b in boundary:
        a = int(a)
        b = int(b)

        pa = partner.get(a, None)
        pb = partner.get(b, None)
        if pa is None or pb is None:
            continue

        if pa == pb or pa in (a, b) or pb in (a, b):
            continue
        if pa == b or pb == a:
            continue

        quad_key = tuple(sorted([a, b, pa, pb]))
        if quad_key in used_quads:
            continue

        A = V[a]
        B = V[b]
        C = V[pb]
        D = V[pa]

        area_ABC = triangle_area_3d(A, B, C)
        area_ACD = triangle_area_3d(A, C, D)
        area_ABD = triangle_area_3d(A, B, D)
        area_BCD = triangle_area_3d(B, C, D)

        area1 = area_ABC + area_ACD
        area2 = area_ABD + area_BCD

        faces_candidate = None
        if area1 >= area2:
            if area_ABC > min_triangle_area and area_ACD > min_triangle_area:
                faces_candidate = [[a, b, pb], [a, pb, pa]]
        else:
            if area_ABD > min_triangle_area and area_BCD > min_triangle_area:
                faces_candidate = [[a, b, pa], [b, pb, pa]]

        if faces_candidate is None:
            continue

        new_faces.extend(faces_candidate)
        used_quads.add(quad_key)

    stats["bridge_quads"] = int(len(used_quads))

    if len(new_faces) == 0:
        return mesh.copy(), stats

    F = np.asarray(mesh.faces, dtype=np.int32)
    F2 = np.vstack([F, np.asarray(new_faces, dtype=np.int32)])
    out = trimesh.Trimesh(vertices=V.copy(), faces=F2, process=False)
    out = clean_mesh_light(out, fix_normals=False)

    stats["added_faces"] = int(len(new_faces))
    return out, stats
def stitch_open_boundary_chains(
    mesh,
    max_chain_endpoint_dist=0.55,
    max_chain_avg_gap=0.40,
    max_chain_plane_residual=0.12,
    tangent_dot_min=0.15,
    normal_dot_min=0.20,
    max_chain_pairs=256,
    min_chain_vertices=6,
    min_triangle_area=1e-8,
    max_chain_neighbor_candidates=64,
    boundary_cache=None,
):
    stats = {
        "chain_count": 0,
        "candidate_pairs": 0,
        "accepted_pairs": 0,
        "added_faces": 0,
    }

    chains = extract_open_boundary_chains(
        mesh,
        max_chain_edges=800,
        max_chains=12000,
        boundary_cache=boundary_cache,
    )
    stats["chain_count"] = int(len(chains))
    if len(chains) < 2:
        return mesh.copy(), stats

    V = np.asarray(mesh.vertices, dtype=np.float64)
    try:
        VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception:
        VN = np.zeros((len(V), 3), dtype=np.float64)

    chain_infos = []
    for ch in chains:
        if len(ch) < min_chain_vertices:
            chain_infos.append(None)
            continue

        ids = np.asarray(ch, dtype=np.int32)
        pts = V[ids]
        clen = chain_length(pts)
        if clen <= 1e-8:
            chain_infos.append(None)
            continue

        sample_n = min(len(ids), 24)
        if sample_n < 2:
            chain_infos.append(None)
            continue

        sample_idx = np.linspace(0, len(ids) - 1, sample_n).astype(np.int32)

        chain_infos.append({
            "ids": ids,
            "pts": pts,
            "len": clen,
            "p0": pts[0],
            "p1": pts[-1],
            "t0": endpoint_tangent(pts, at_start=True),
            "t1": endpoint_tangent(pts, at_start=False),
            "nmean": mean_vertex_normal(VN, ids),
            "sample_idx": sample_idx,
        })

    candidate_map = knn_candidate_map_for_chains(
        chains,
        V,
        k_neighbors=max_chain_neighbor_candidates,
        max_dist=max_chain_endpoint_dist,
    )

    candidates = []
    max_ep_dist2 = float(max_chain_endpoint_dist * max_chain_endpoint_dist)

    for i in range(len(chains)):
        infoA = chain_infos[i]
        if infoA is None:
            continue

        js = sorted(j for j in candidate_map[i] if j > i)
        if not js and (not SCIPY_AVAILABLE or cKDTree is None):
            js = list(range(i + 1, len(chains)))

        for j in js:
            infoB = chain_infos[j]
            if infoB is None:
                continue

            A = infoA["ids"]
            B = infoB["ids"]

            A2, B2 = orient_chain_pair_for_min_gap(A, B, V)
            ptsA2 = V[A2]
            ptsB2 = V[B2]

            d0 = np.sum((ptsA2[0] - ptsB2[0]) ** 2)
            d1 = np.sum((ptsA2[-1] - ptsB2[-1]) ** 2)
            ep_dist = float(np.sqrt(max(d0, d1)))
            if d0 > max_ep_dist2 or d1 > max_ep_dist2:
                continue

            lenA = infoA["len"]
            lenB = infoB["len"]
            ratio = max(lenA, lenB) / (min(lenA, lenB) + 1e-12)
            if ratio > 4.0:
                continue

            sample_n = min(len(A2), len(B2), 24)
            if sample_n < 2:
                continue

            idxA = np.linspace(0, len(A2) - 1, sample_n).astype(np.int32)
            idxB = np.linspace(0, len(B2) - 1, sample_n).astype(np.int32)

            avg_gap = float(np.mean(np.linalg.norm(V[A2[idxA]] - V[B2[idxB]], axis=1)))
            if avg_gap > max_chain_avg_gap:
                continue

            ta0 = endpoint_tangent(ptsA2, at_start=True)
            ta1 = endpoint_tangent(ptsA2, at_start=False)
            tb0 = endpoint_tangent(ptsB2, at_start=True)
            tb1 = endpoint_tangent(ptsB2, at_start=False)

            tscore = 0.5 * (abs(np.dot(ta0, tb0)) + abs(np.dot(ta1, tb1)))
            if tscore < tangent_dot_min:
                continue

            na = mean_vertex_normal(VN, A2)
            nb = mean_vertex_normal(VN, B2)
            nd = abs(float(np.dot(na, nb)))
            if nd < normal_dot_min:
                continue

            ptsAB = np.vstack([ptsA2, ptsB2])
            resid, _, _ = chain_plane_residual(ptsAB)
            if resid > max_chain_plane_residual:
                continue

            score = ep_dist + 0.6 * avg_gap - 0.15 * tscore - 0.10 * nd
            candidates.append((score, A2, B2))

    stats["candidate_pairs"] = int(len(candidates))
    if len(candidates) == 0:
        return mesh.copy(), stats

    candidates.sort(key=lambda x: x[0])

    used_vertices = set()
    new_faces = []
    accepted = 0

    for _, A2, B2 in candidates:
        setA = set(A2.tolist())
        setB = set(B2.tolist())

        if len(setA & used_vertices) > 0 or len(setB & used_vertices) > 0:
            continue

        tris = zipper_triangulate_between_chains(A2, B2, V, min_triangle_area=min_triangle_area)
        if len(tris) == 0:
            continue

        new_faces.extend(tris)
        used_vertices.update(setA)
        used_vertices.update(setB)
        accepted += 1

        if accepted >= max_chain_pairs:
            break

    stats["accepted_pairs"] = int(accepted)
    if len(new_faces) == 0:
        return mesh.copy(), stats

    F = np.asarray(mesh.faces, dtype=np.int32)
    F2 = np.vstack([F, np.asarray(new_faces, dtype=np.int32)])
    out = trimesh.Trimesh(vertices=V.copy(), faces=F2, process=False)
    out = clean_mesh_light(out, fix_normals=False)

    stats["added_faces"] = int(len(new_faces))
    return out, stats

def run_early_hole_repair(
    mesh,
    enable=True,
    try_trimesh_fill=True,
    max_hole_edges=160,
    max_hole_diameter=1.5,
    max_hole_area=1.2,
    max_hole_plane_residual=0.10,
    max_hole_candidate_loops=8000,
    enable_gap_stitch=True,
    max_bridge_dist=0.30,
    bridge_normal_dot_min=0.15,
    max_bridge_pairs=20000,
    enable_chain_stitch=True,
    max_chain_endpoint_dist=0.55,
    max_chain_avg_gap=0.40,
    max_chain_plane_residual=0.12,
    chain_tangent_dot_min=0.15,
    chain_normal_dot_min=0.20,
    max_chain_pairs=256,
    max_chain_neighbor_candidates=64,
    use_gpu_gap_prefilter=False,
    use_gpu_chain=False,
    verbose=False,
):
    stats = {
        "enabled": bool(enable),
        "trimesh_fill_applied": False,
        "loop_fill": {},
        "gap_stitch": {},
        "chain_stitch": {},
        "timings_sec": {},
    }

    if not enable:
        return mesh.copy(), stats

    out = mesh.copy()

    if try_trimesh_fill:
        t0 = time.perf_counter()
        try:
            before_faces = len(out.faces)
            trimesh.repair.fill_holes(out)
            stats["trimesh_fill_applied"] = len(out.faces) > before_faces
        except Exception:
            pass
        stats["timings_sec"]["trimesh_fill"] = round(time.perf_counter() - t0, 4)
    else:
        stats["timings_sec"]["trimesh_fill"] = 0.0

    t0 = time.perf_counter()
    boundary_cache = build_boundary_cache(out)
    out, loop_stats = fill_small_holes_from_loops(
        out,
        max_hole_edges=max_hole_edges,
        max_hole_diameter=max_hole_diameter,
        max_hole_area=max_hole_area,
        max_plane_residual=max_hole_plane_residual,
        max_candidate_loops=max_hole_candidate_loops,
        boundary_cache=boundary_cache,
    )
    stats["loop_fill"] = loop_stats
    stats["timings_sec"]["loop_fill"] = round(time.perf_counter() - t0, 4)

    if enable_gap_stitch:
        t0 = time.perf_counter()
        boundary_cache = build_boundary_cache(out)
        out, gap_stats = stitch_boundary_gaps(
            out,
            max_bridge_dist=max_bridge_dist,
            normal_dot_min=bridge_normal_dot_min,
            max_bridge_pairs=max_bridge_pairs,
            boundary_cache=boundary_cache,
            use_gpu_prefilter=use_gpu_gap_prefilter,
            gpu_min_boundary_vertices=12000,
            knn_neighbors=8,
            use_knn_candidates=True,
        )
        stats["timings_sec"]["gap_stitch"] = round(time.perf_counter() - t0, 4)
    else:
        gap_stats = {
            "boundary_vertices": 0,
            "candidate_pairs": 0,
            "mutual_pairs": 0,
            "bridge_quads": 0,
            "added_faces": 0,
            "gpu_prefilter_used": False,
        }
        stats["timings_sec"]["gap_stitch"] = 0.0
    stats["gap_stitch"] = gap_stats

    if enable_chain_stitch:
        t0 = time.perf_counter()
        boundary_cache = build_boundary_cache(out)
        out, chain_stats = stitch_open_boundary_chains(
            out,
            max_chain_endpoint_dist=max_chain_endpoint_dist,
            max_chain_avg_gap=max_chain_avg_gap,
            max_chain_plane_residual=max_chain_plane_residual,
            tangent_dot_min=chain_tangent_dot_min,
            normal_dot_min=chain_normal_dot_min,
            max_chain_pairs=max_chain_pairs,
            min_chain_vertices=6,
            max_chain_neighbor_candidates=max_chain_neighbor_candidates,
            boundary_cache=boundary_cache,
        )
        stats["timings_sec"]["chain_stitch"] = round(time.perf_counter() - t0, 4)
    else:
        chain_stats = {
            "chain_count": 0,
            "candidate_pairs": 0,
            "accepted_pairs": 0,
            "added_faces": 0,
        }
        stats["timings_sec"]["chain_stitch"] = 0.0
    stats["chain_stitch"] = chain_stats

    t0 = time.perf_counter()
    out = clean_mesh_light(out, fix_normals=False)
    stats["timings_sec"]["post_clean"] = round(time.perf_counter() - t0, 4)

    if verbose:
        total_added = (
            loop_stats.get("added_faces", 0)
            + gap_stats.get("added_faces", 0)
            + chain_stats.get("added_faces", 0)
        )
        print(
            f"[2/8] 早期空洞修补... "
            f"loop_filled={loop_stats.get('filled_holes', 0)} "
            f"gap_mode={gap_stats.get('candidate_mode', 'na')} "
            f"gap_boundary_v={gap_stats.get('boundary_vertices', 0)} "
            f"gap_candidates={gap_stats.get('candidate_pairs', 0)} "
            f"gap_mutual={gap_stats.get('mutual_pairs', 0)} "
            f"gap_quads={gap_stats.get('bridge_quads', 0)} "
            f"chain_count={chain_stats.get('chain_count', 0)} "
            f"chain_candidates={chain_stats.get('candidate_pairs', 0)} "
            f"chain_pairs={chain_stats.get('accepted_pairs', 0)} "
            f"added_faces={total_added} "
            f"time_loop={stats['timings_sec']['loop_fill']}s "
            f"time_gap={stats['timings_sec']['gap_stitch']}s "
            f"time_chain={stats['timings_sec']['chain_stitch']}s"
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

    face_adj_pairs = np.asarray(mesh.face_adjacency, dtype=np.int32) if mesh.face_adjacency is not None else np.empty((0, 2), dtype=np.int32)
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

def fit_plane_faces(face_ids, cache, use_gpu=False):
    face_ids = np.asarray(face_ids, dtype=np.int32)
    centers = cache["tri_centers"][face_ids]
    areas = cache["face_areas"][face_ids]
    normals = cache["face_normals"][face_ids]

    c, n = robust_plane(centers, areas, use_gpu=use_gpu)

    xp = xp_module(use_gpu)
    centers_x = xp.asarray(centers, dtype=xp.float64)
    c_x = xp.asarray(c, dtype=xp.float64)
    n_x = xp.asarray(n, dtype=xp.float64)
    dist = (centers_x - c_x) @ n_x

    _, _, vals = weighted_pca(centers, areas, use_gpu=use_gpu)
    dots = abs_normal_dot(normals, n, use_gpu=use_gpu)
    mean_angle = float(np.degrees(np.mean(np.arccos(np.clip(to_cpu(dots), 0.0, 1.0))))) if len(dots) else 0.0

    return {
        "face_ids": face_ids,
        "centroid": np.asarray(c, dtype=np.float64),
        "normal": np.asarray(n, dtype=np.float64),
        "sigma": robust_sigma(dist, use_gpu=use_gpu),
        "area": float(np.sum(areas)),
        "planarity_ratio": float(vals[0] / (np.sum(vals) + 1e-12)),
        "mean_normal_angle": mean_angle,
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

    use_gpu = bool(cfg.get("use_cuda", False) and CUPY_AVAILABLE)

    plane = fit_plane_faces(face_ids, cache, use_gpu=use_gpu)
    centers = cache["tri_centers"][face_ids]
    normals = cache["face_normals"][face_ids]

    xp = xp_module(use_gpu)
    centers_x = xp.asarray(centers, dtype=xp.float64)
    c_x = xp.asarray(plane["centroid"], dtype=xp.float64)
    n_x = xp.asarray(plane["normal"], dtype=xp.float64)
    d = (centers_x - c_x) @ n_x

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
    dots = abs_normal_dot(normals, plane["normal"], use_gpu=use_gpu)

    inlier_mask = (to_cpu(xp.abs(d)) <= dist_thr) & (to_cpu(dots) >= cos_thr)

    inlier = face_ids[inlier_mask]
    outlier = face_ids[~inlier_mask]
    res = []

    if len(inlier) >= cfg["min_plane_faces"]:
        for g in connected_groups_sparse(inlier, cache["face_adj_csr"]):
            if len(g) < cfg["min_plane_faces"]:
                continue
            if float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            p = fit_plane_faces(g, cache, use_gpu=use_gpu)
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
    # 先把每个 plane 的 face_ids 统一变成“已排序且唯一”的 int32 数组
    prepared = []
    for p in planes:
        ids = np.unique(np.asarray(p["face_ids"], dtype=np.int32))
        q = dict(p)
        q["face_ids"] = ids
        prepared.append(q)

    kept = []
    kept_ids = []

    # 仍然优先保留更大的 plane
    for p in sorted(prepared, key=lambda x: (len(x["face_ids"]), x["area"]), reverse=True):
        ids = p["face_ids"]
        n_ids = len(ids)
        ok = True

        for k_ids in kept_ids:
            min_len = min(n_ids, len(k_ids))
            if min_len == 0:
                continue

            # overlap 阈值对应的最少交集数
            need = int(np.ceil(overlap * min_len))

            # 因为 ids 和 k_ids 都已经 unique + sorted
            inter = np.intersect1d(ids, k_ids, assume_unique=True).size
            if inter >= need:
                ok = False
                break

        if ok:
            kept.append(p)
            kept_ids.append(ids)

    return kept

def detect_planes(mesh, cfg, verbose=False):
    if verbose:
        print("[4/8] 多平面识别...")

    dp_timings = {}

    t0 = time.perf_counter()
    cache = make_cache(mesh)
    dp_timings["make_cache"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    patch_labels, patch_count = initial_patch_labels(cache, cfg["patch_normal_angle_deg"])
    dp_timings["initial_patch_labels"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    face_counts = np.bincount(patch_labels, minlength=patch_count)
    area_sums = np.bincount(patch_labels, weights=cache["face_areas"], minlength=patch_count)

    valid_patch_mask = (
        (face_counts >= cfg["min_patch_faces"]) &
        (area_sums >= cfg["min_plane_area"])
    )
    valid_patch_ids = np.where(valid_patch_mask)[0]
    dp_timings["patch_filter"] = round(time.perf_counter() - t0, 4)

    planes = []
    t0 = time.perf_counter()
    if len(valid_patch_ids) > 0:
        order = np.argsort(patch_labels, kind="mergesort")
        labels_sorted = patch_labels[order]

        starts = np.flatnonzero(
            np.r_[True, labels_sorted[1:] != labels_sorted[:-1]]
        )
        ends = np.r_[starts[1:], len(order)]

        for s, e in zip(starts, ends):
            pid = int(labels_sorted[s])
            if not valid_patch_mask[pid]:
                continue
            patch = order[s:e].astype(np.int32)
            planes.extend(split_planes(patch, cache, cfg, 0))
    dp_timings["split_planes_total"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    planes = dedup_planes(planes)
    dp_timings["dedup_planes"] = round(time.perf_counter() - t0, 4)

    if verbose:
        print(f"  初始 patch 数: {patch_count}")
        print(f"  有效 patch 数: {len(valid_patch_ids)}")
        print(f"  识别平面数: {len(planes)}")
        print(f"  detect_planes breakdown: {dp_timings}")

    return planes, cache, dp_timings

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
    use_gpu = bool(state.get("use_cuda", False) and CUPY_AVAILABLE)

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

    c, n = robust_plane(pts, None, 6, 1.4, use_gpu=use_gpu)

    xp = xp_module(use_gpu)
    pts_x = xp.asarray(pts, dtype=xp.float64)
    c_x = xp.asarray(c, dtype=xp.float64)
    n_x = xp.asarray(n, dtype=xp.float64)

    sd_x = (pts_x - c_x) @ n_x
    sigma_v = robust_sigma(sd_x, use_gpu=use_gpu)

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

    sd = to_cpu(sd_x)
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

    if use_gpu:
        moved_vids_x = cp.asarray(moved_vids, dtype=cp.int32)
        Vx = cp.asarray(V, dtype=cp.float64)
        target_x = Vx[moved_vids_x].copy()
        d_x = cp.asarray(d, dtype=cp.float64)
        ds_x = cp.asarray(ds, dtype=cp.float64)
        inlier_x = cp.asarray(np.abs(d) <= inlier_thr)

        target_x[inlier_x] -= d_x[inlier_x, None] * n_x
        target_x[~inlier_x] -= (d_x[~inlier_x] - residual_shrink * ds_x[~inlier_x])[:, None] * n_x

        non_inlier_weight = min(1.0, 0.72 + 0.30 * eff_noise + 0.22 * dist_boost)
        w_x = cp.where(inlier_x, snap_strength, non_inlier_weight) * cp.asarray(bmul, dtype=cp.float64)

        acc = cp.asnumpy(w_x[:, None] * target_x)
        w = cp.asnumpy(w_x)
    else:
        target = V[moved_vids].copy()
        inlier = np.abs(d) <= inlier_thr
        target[inlier] -= d[inlier, None] * n
        target[~inlier] -= (d[~inlier] - residual_shrink * ds[~inlier])[:, None] * n

        non_inlier_weight = min(1.0, 0.72 + 0.30 * eff_noise + 0.22 * dist_boost)
        w = np.where(inlier, snap_strength, non_inlier_weight) * bmul
        acc = w[:, None] * target

    return {
        "vids": moved_vids.astype(np.int32),
        "acc": acc,
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
            "use_cuda": bool(use_gpu),
        }
    }

def refine_planes_parallel(mesh, planes, cfg, shared_cache, only_stage=None, pass_mul=None, verbose=False, tag="[5/8]", num_workers=1):
    if not planes:
        return mesh.copy(), []

    pass_mul = pass_mul or {}
    mesh2 = mesh.copy()

    state = {
        "V": np.asarray(mesh.vertices, dtype=np.float64),
        "F": np.asarray(mesh.faces, dtype=np.int32),
        "vert_adj_csr": shared_cache["vert_adj_csr"],
        "is_boundary": shared_cache["is_boundary"] if cfg["boundary_protect"] else np.zeros(len(mesh.vertices), dtype=bool),
        "bbox_diag": shared_cache["bbox_diag"],
        "use_cuda": bool(cfg.get("use_cuda", False) and CUPY_AVAILABLE),
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
        backend = "cuda(cupy)" if state["use_cuda"] else ("scipy.sparse" if SCIPY_AVAILABLE else "fallback")
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

    moved = wsum[:, 0] > 1e-12
    V2 = V.copy()
    V2[moved] = acc[moved] / wsum[moved]

    out = trimesh.Trimesh(vertices=V2, faces=mesh2.faces.copy(), process=False)
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
def save_outputs(mesh, planes, reports1, reports2, before, after, out_dir, input_path, num_workers, timings, hole_stats, use_cuda):
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
            "cupy_available": CUPY_AVAILABLE,
            "cuda_enabled": bool(use_cuda),
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
    use_cuda = cuda_enabled(args)

    # 细分 GPU 使用范围：
    # early hole fill 保留 GPU 预筛
    # plane detect / refine 默认强制走 CPU，避免整体变慢
    use_cuda_early = use_cuda
    use_cuda_planes = False
    use_cuda_chain = False
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
        "use_cuda": use_cuda_planes,
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
        "use_cuda": use_cuda_planes,
    }

    timings = {}

    if args.verbose:
        print("========== PIPELINE START ==========")
        print(f"scipy sparse available: {SCIPY_AVAILABLE}")
        print(f"cupy available: {CUPY_AVAILABLE}")
        print(f"cuda enabled: {use_cuda}")
        print(f"cuda early-hole: {use_cuda_early}")
        print(f"cuda planes: {use_cuda_planes}")
        print(f"cuda chains: {use_cuda_chain}")
        print(f"num_workers: {num_workers}")
        print("[0/8] 读取网格...")

    t0 = time.perf_counter()
    mesh0 = load_mesh(args.input)
    timings["load_mesh"] = round(time.perf_counter() - t0, 4)

    before = mesh_report(mesh0, include_components=False)

    if args.verbose:
        print("[1/8] 基础清理...")

    t0 = time.perf_counter()
    mesh = clean_mesh_light(mesh0, fix_normals=False)
    timings["clean_mesh_1"] = round(time.perf_counter() - t0, 4)

    if args.verbose:
        print("[2/8] 早期空洞修补 / 裂缝桥接...")

    t0 = time.perf_counter()
    mesh, hole_stats = run_early_hole_repair(
        mesh,
        enable=not args.disable_early_hole_fill,
        try_trimesh_fill=not args.disable_trimesh_fill_holes,
        max_hole_edges=args.max_hole_edges,
        max_hole_diameter=args.max_hole_diameter,
        max_hole_area=args.max_hole_area,
        max_hole_plane_residual=args.max_hole_plane_residual,
        max_hole_candidate_loops=args.max_hole_candidate_loops,
        enable_gap_stitch=not args.disable_gap_stitch,
        max_bridge_dist=args.max_bridge_dist,
        bridge_normal_dot_min=args.bridge_normal_dot_min,
        max_bridge_pairs=args.max_bridge_pairs,
        enable_chain_stitch=not args.disable_chain_stitch,
        max_chain_endpoint_dist=args.max_chain_endpoint_dist,
        max_chain_avg_gap=args.max_chain_avg_gap,
        max_chain_plane_residual=args.max_chain_plane_residual,
        chain_tangent_dot_min=args.chain_tangent_dot_min,
        chain_normal_dot_min=args.chain_normal_dot_min,
        max_chain_pairs=args.max_chain_pairs,
        max_chain_neighbor_candidates=args.max_chain_neighbor_candidates,
        use_gpu_gap_prefilter=use_cuda_early,
        use_gpu_chain=use_cuda_chain,
        verbose=args.verbose
    )
    timings["early_hole_fill"] = round(time.perf_counter() - t0, 4)

    if args.verbose:
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

    t0 = time.perf_counter()
    planes, shared_cache, detect_breakdown = detect_planes(mesh, detect_cfg, args.verbose)
    timings["detect_planes"] = round(time.perf_counter() - t0, 4)
    timings["detect_planes_make_cache"] = detect_breakdown["make_cache"]
    timings["detect_planes_initial_patch_labels"] = detect_breakdown["initial_patch_labels"]
    timings["detect_planes_patch_filter"] = detect_breakdown["patch_filter"]
    timings["detect_planes_split_planes_total"] = detect_breakdown["split_planes_total"]
    timings["detect_planes_dedup_planes"] = detect_breakdown["dedup_planes"]

    t0 = time.perf_counter()
    mesh, reports1 = refine_planes_parallel(
        mesh,
        planes,
        refine_cfg,
        shared_cache=shared_cache,
        only_stage=None,
        pass_mul=None,
        verbose=False,
        tag="[5/8]",
        num_workers=num_workers
    )
    timings["refine_pass_1"] = round(time.perf_counter() - t0, 4)

    reports2 = []
    timings["refine_pass_2_far"] = 0.0
    if args.enable_far_second_pass:
        selected = select_far_second_pass_planes(
            planes,
            reports1,
            args.far_second_pass_min_faces,
            args.far_second_pass_min_sigma
        )
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
                verbose=False,
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
    mesh = clean_mesh_light(mesh, fix_normals=not args.skip_final_fix_normals)
    timings["clean_mesh_final"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    after = mesh_report(
        mesh,
        include_components=not args.skip_component_count_in_report
    )
    timings["final_report"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    if args.profile_fast_mode:
        ensure_dir(args.output_dir)
        obj_path, ply_path, planes_path, report_path = None, None, None, None
    else:
        obj_path, ply_path, planes_path, report_path = save_outputs(
            mesh, planes, reports1, reports2, before, after,
            args.output_dir, args.input, num_workers, timings, hole_stats, use_cuda
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
def apply_preset(args):
    preset_name = getattr(args, "preset", DEFAULT_PRESET)
    cfg = PRESET_CONFIGS.get(preset_name, PRESET_CONFIGS[DEFAULT_PRESET])

    # 只覆盖 plane detection 相关参数
    args.patch_normal_angle_deg = cfg["patch_normal_angle_deg"]
    args.min_patch_faces = cfg["min_patch_faces"]
    args.min_plane_faces = cfg["min_plane_faces"]
    args.min_plane_area = cfg["min_plane_area"]
    args.base_face_plane_dist = cfg["base_face_plane_dist"]
    args.plane_normal_angle_deg = cfg["plane_normal_angle_deg"]
    args.sigma_dist_mult = cfg["sigma_dist_mult"]
    args.max_split_depth = cfg["max_split_depth"]

    return args
# =============================
# CLI
# =============================
def build_parser():
    p = argparse.ArgumentParser(description="Mesh refine pipeline GPU version (CPU-GPU hybrid)")

    p.add_argument("--input", default=DEFAULT_INPUT, help="输入网格路径，默认使用预设 PLY")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    p.add_argument("--preset", choices=["balanced", "fast", "detail"], default=DEFAULT_PRESET, help="参数预设")

    p.add_argument("--min_component_faces", type=int, default=80)
    p.add_argument("--min_component_area", type=float, default=0.0015)
    p.add_argument("--min_component_max_extent", type=float, default=0.04)
    p.add_argument("--keep_top_k_faces", type=int, default=20)

    # early hole fill
    p.add_argument("--disable_early_hole_fill", action="store_true")
    p.add_argument("--disable_trimesh_fill_holes", action="store_true")
    p.add_argument("--max_hole_edges", type=int, default=160)
    p.add_argument("--max_hole_diameter", type=float, default=1.5)
    p.add_argument("--max_hole_area", type=float, default=1.2)
    p.add_argument("--max_hole_plane_residual", type=float, default=0.10)
    p.add_argument("--max_hole_candidate_loops", type=int, default=8000)

    # point gap stitch
    p.add_argument("--disable_gap_stitch", action="store_true")
    p.add_argument("--max_bridge_dist", type=float, default=0.30, help="边界顶点最大桥接距离")
    p.add_argument("--bridge_normal_dot_min", type=float, default=0.15, help="边界顶点法向相容阈值")
    p.add_argument("--max_bridge_pairs", type=int, default=20000)

    # chain stitch
    p.add_argument("--disable_chain_stitch", action="store_true")
    p.add_argument("--max_chain_endpoint_dist", type=float, default=0.55)
    p.add_argument("--max_chain_avg_gap", type=float, default=0.40)
    p.add_argument("--max_chain_plane_residual", type=float, default=0.12)
    p.add_argument("--chain_tangent_dot_min", type=float, default=0.15)
    p.add_argument("--chain_normal_dot_min", type=float, default=0.20)
    p.add_argument("--max_chain_pairs", type=int, default=256)
    p.add_argument("--max_chain_neighbor_candidates", type=int, default=64)

    # plane detection
    p.add_argument("--patch_normal_angle_deg", type=float, default=24.0)
    p.add_argument("--min_patch_faces", type=int, default=25)
    p.add_argument("--min_plane_faces", type=int, default=80)
    p.add_argument("--min_plane_area", type=float, default=0.0008)
    p.add_argument("--base_face_plane_dist", type=float, default=0.014)
    p.add_argument("--plane_normal_angle_deg", type=float, default=18.0)
    p.add_argument("--sigma_dist_mult", type=float, default=3.6)
    p.add_argument("--max_split_depth", type=int, default=3)

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
    p.add_argument("--sensor_origin", type=str, default=DEFAULT_SENSOR_ORIGIN)
    p.add_argument("--distance_gain", type=float, default=0.45)
    p.add_argument("--use_distance_strategy", dest="use_distance_strategy", action="store_true", help="启用距离策略")
    p.add_argument("--no_use_distance_strategy", dest="use_distance_strategy", action="store_false",
                   help="关闭距离策略")
    p.set_defaults(use_distance_strategy=True)
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

    p.add_argument("--global_smooth_iter", type=int, default=0)
    p.add_argument("--global_smooth_method", choices=["taubin", "laplacian", "simple"], default="taubin")

    p.add_argument("--noise_min_faces", type=int, default=18)
    p.add_argument("--noise_min_area", type=float, default=0.00035)
    p.add_argument("--noise_min_extent", type=float, default=0.018)
    p.add_argument("--noise_keep_top_k", type=int, default=10)

    p.add_argument("--skip_final_fix_normals", dest="skip_final_fix_normals", action="store_true",
                   help="跳过最终法线修复")
    p.add_argument("--enable_final_fix_normals", dest="skip_final_fix_normals", action="store_false",
                   help="启用最终法线修复")
    p.set_defaults(skip_final_fix_normals=True)
    p.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)

    # GPU
    p.add_argument("--use_cuda", dest="use_cuda", action="store_true", help="启用 CUDA / CuPy 混合加速（需已安装 cupy）")
    p.add_argument("--no_use_cuda", dest="use_cuda", action="store_false", help="关闭 CUDA / CuPy")
    p.set_defaults(use_cuda=True)

    p.add_argument("--verbose", dest="verbose", action="store_true", help="显示详细日志")
    p.add_argument("--quiet", dest="verbose", action="store_false", help="静默模式")
    p.set_defaults(verbose=True)
    p.add_argument("--skip_component_count_in_report", action="store_true", help="调参时跳过最终组件统计，加快测试")
    p.add_argument("--profile_fast_mode", action="store_true", help="调参时跳过OBJ/PLY/JSON导出，加快测试")
    return p

def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        print("[INFO] 未提供命令行参数，使用代码内默认 preset 直接运行。")

    args = parser.parse_args()

    # 应用 preset，覆盖 plane detection 相关参数
    args = apply_preset(args)

    # 如果用户没有手工指定 output_dir，则按 preset 自动命名输出目录
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = rf".\outputs_{args.preset}"

    print("========== DEFAULT RUN CONFIG ==========")
    print(f"preset: {args.preset}")
    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    print(f"sensor_origin: {args.sensor_origin}")
    print(f"use_cuda: {getattr(args, 'use_cuda', False)}")
    print(f"use_distance_strategy: {args.use_distance_strategy}")
    print(f"num_workers: {args.num_workers}")
    print(f"patch_normal_angle_deg: {args.patch_normal_angle_deg}")
    print(f"min_patch_faces: {args.min_patch_faces}")
    print(f"min_plane_faces: {args.min_plane_faces}")
    print(f"min_plane_area: {args.min_plane_area}")
    print(f"base_face_plane_dist: {args.base_face_plane_dist}")
    print(f"plane_normal_angle_deg: {args.plane_normal_angle_deg}")
    print(f"sigma_dist_mult: {args.sigma_dist_mult}")
    print(f"max_split_depth: {args.max_split_depth}")
    print("========================================")

    process_mesh(args)

if __name__ == "__main__":
    main()