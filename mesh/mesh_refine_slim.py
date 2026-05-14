# mesh_refine_slim.py
# 瘦身版：删除 chain stitch / far second pass / global smooth /
#         trimesh fill_holes 分支 / profile_fast_mode
# 默认 verbose=True，PLY 导出，OBJ/JSON 不导出

# ----- [0] BLAS 线程数（必须最先调用）-----
from config_slim import apply_env_threads
apply_env_threads()

# ----- [1] 标准库 -----
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# ----- [2] 第三方 -----
import numpy as np
import trimesh
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.spatial import cKDTree

# ----- [3] 项目配置 -----
from config_slim import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PRESET,
    PRESET_CONFIGS,
    COMPONENT_FILTER_CONFIG,
    HOLE_FILL_CONFIG,
    PLANE_DETECT_CONFIG,
    PLANE_REFINE_CONFIG,
    DISTANCE_STRATEGY_CONFIG,
    NOISE_FILTER_CONFIG,
    RUNTIME_CONFIG,
    apply_preset_to_args,
    enable_log,
    log,
)

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

def mesh_report(mesh):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_extent = (V.max(axis=0) - V.min(axis=0)).tolist() if len(V) > 0 else None
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_extent": bbox_extent,
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

def abs_normal_dot(normals, normal):
    return np.clip(
        np.abs(np.asarray(normals, dtype=np.float64) @ np.asarray(normal, dtype=np.float64)),
        0.0, 1.0,
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

def robust_plane(points, weights=None, iters=3, huber_k=1.5):
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
        return dict(stage="near", snap=1.00, inlier=0.85, outlier=1.00, shrink=0.70, smooth=1.20, noise=1.00, detect_dist=1.00, detect_ang=1.00)
    if r <= far_ratio:
        return dict(stage="mid", snap=1.10, inlier=1.05, outlier=1.45, shrink=0.42, smooth=1.80, noise=1.35, detect_dist=1.20, detect_ang=1.12)
    return dict(stage="far", snap=1.22, inlier=1.35, outlier=2.40, shrink=0.12, smooth=2.80, noise=2.10, detect_dist=1.65, detect_ang=1.35)

# =============================
# Sparse graph
# =============================
def make_sparse_graph_from_pairs(num_nodes, pairs):
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
    f = np.asarray(faces, dtype=np.int32)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 0], f[:, 2], f[:, 1], f[:, 0], f[:, 2]])
    data = np.ones(len(rows), dtype=np.uint8)
    g = sp.csr_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices), dtype=np.uint8)
    g.sum_duplicates()
    g.data[:] = 1
    return g

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

def boundary_vertices(mesh):
    be = get_boundary_edges(mesh)
    if len(be) == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(be.reshape(-1)).astype(np.int32, copy=False)

# =============================
# Components / noise removal
# =============================
def connected_face_labels(mesh):
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.empty(0, dtype=np.int32), 0
    face_adj = np.asarray(mesh.face_adjacency, dtype=np.int32)
    if face_adj.size == 0:
        return np.arange(n_faces, dtype=np.int32), n_faces
    rows = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
    cols = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces), dtype=np.uint8)
    n_comp, labels = sp_connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32), int(n_comp)

def submesh_from_face_mask(mesh, face_mask):
    face_mask = np.asarray(face_mask, dtype=bool)
    if face_mask.all():
        return mesh
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

    vlabels = np.full(len(verts), -1, dtype=np.int32)
    vlabels[faces[:, 0]] = labels
    vlabels[faces[:, 1]] = labels
    vlabels[faces[:, 2]] = labels

    valid = vlabels >= 0
    vl = vlabels[valid]
    vp = verts[valid]

    cmin = np.full((n_comp, 3),  np.inf, dtype=np.float64)
    cmax = np.full((n_comp, 3), -np.inf, dtype=np.float64)
    np.minimum.at(cmin, vl, vp)
    np.maximum.at(cmax, vl, vp)

    extents = cmax - cmin
    extents[~np.isfinite(extents)] = 0.0
    max_extents = extents.max(axis=1)

    return labels, n_comp, face_counts, area_sums, max_extents

def _keep_components(mesh, min_faces, min_area, min_extent, keep_top_k):
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

def filter_components_fast(mesh, min_faces=80, min_area=0.0015, min_extent=0.04, keep_top_k=20):
    return _keep_components(mesh, min_faces, min_area, min_extent, keep_top_k)

def remove_floating_noise_fast(mesh, min_faces=20, min_area=0.0005, min_extent=0.02, keep_top_k=8):
    return _keep_components(mesh, min_faces, min_area, min_extent, keep_top_k)

# =============================
# Hole filling helpers (2D)
# =============================
def plane_basis_from_normal(normal):
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
    return u, v

def polygon_area_2d(poly):
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def point_in_triangle_2d(p, a, b, c, eps=1e-12):
    v0 = c - a; v1 = b - a; v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)

def is_convex_corner(a, b, c, ccw=True, eps=1e-12):
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return cross > eps if ccw else cross < -eps

def is_polygon_convex_2d(poly, eps=1e-12):
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 4:
        return True
    sign = 0
    for i in range(n):
        a = poly[i]; b = poly[(i + 1) % n]; c = poly[(i + 2) % n]
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
        tris.append((0, i, i + 1) if ccw else (0, i + 1, i))
    return tris

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
            i_prev = idx[(i - 1) % m]; i_curr = idx[i]; i_next = idx[(i + 1) % m]
            a = poly[i_prev]; b = poly[i_curr]; c = poly[i_next]
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

# =============================
# Boundary data
# =============================
def build_boundary_data(mesh):
    boundary = get_boundary_edges(mesh)
    if len(boundary) == 0:
        return boundary, {}, set()
    a = boundary[:, 0].astype(np.int64)
    b = boundary[:, 1].astype(np.int64)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.argsort(src, kind="stable")
    src_s = src[order]; dst_s = dst[order]
    uniq_v, first_idx = np.unique(src_s, return_index=True)
    next_idx = np.r_[first_idx[1:], len(src_s)]
    adj = {}
    for v, s, e in zip(uniq_v.tolist(), first_idx.tolist(), next_idx.tolist()):
        adj[int(v)] = dst_s[s:e].tolist()
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    edge_set = set(zip(lo.tolist(), hi.tolist()))
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
        a = int(a); b = int(b)
        if a in valid_vertices and b in valid_vertices:
            valid_edges.add((a, b) if a < b else (b, a))

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    visited = set()
    loops = []

    for a, b in list(valid_edges):
        if (a, b) in visited:
            continue
        start = a; prev = a; cur = b
        loop = [start]
        visited.add((a, b))
        ok = True
        while True:
            loop.append(cur)
            if len(loop) > max_loop_edges:
                ok = False; break
            ns = adj.get(cur, [])
            if len(ns) != 2:
                ok = False; break
            nxt = ns[0] if ns[1] == prev else ns[1]
            if nxt == start:
                break
            ek = edge_key(cur, nxt)
            if ek in visited:
                ok = False; break
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

# =============================
# Fill small holes from closed loops
# =============================
def fill_small_holes_from_loops(
    mesh,
    max_hole_edges=160,
    max_hole_diameter=1.5,
    max_hole_area=1.2,
    max_plane_residual=0.10,
    max_candidate_loops=8000,
    boundary_cache=None,
):
    stats = {"candidate_loops": 0, "accepted_loops": 0, "filled_holes": 0, "added_faces": 0}

    if boundary_cache is None:
        boundary_cache = build_boundary_cache(mesh)

    loops = extract_boundary_loops_ordered(
        mesh, max_loop_edges=max_hole_edges, max_loops=max_candidate_loops,
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
        if float(np.linalg.norm(bb)) > max_hole_diameter:
            continue
        c, n, _ = weighted_pca(pts)
        if np.abs(np.dot(pts - c, n)).max() > max_plane_residual:
            continue
        u, v = plane_basis_from_normal(n)
        poly2d = np.column_stack(((pts - c) @ u, (pts - c) @ v))
        area = abs(polygon_area_2d(poly2d))
        if area <= 1e-12 or area > max_hole_area:
            continue
        tris_local = fan_triangulation_2d(poly2d) if is_polygon_convex_2d(poly2d) else ear_clip_triangulation_2d(poly2d)
        if len(tris_local) == 0:
            continue
        stats["accepted_loops"] += 1
        stats["filled_holes"] += 1
        for a, b, cidx in tris_local:
            ia = int(loop[a]); ib = int(loop[b]); ic = int(loop[cidx])
            if ia == ib or ib == ic or ia == ic:
                continue
            new_faces.append([ia, ib, ic])

    if len(new_faces) == 0:
        return mesh.copy(), stats

    F2 = np.vstack([F, np.asarray(new_faces, dtype=np.int32)])
    out = trimesh.Trimesh(vertices=V.copy(), faces=F2, process=False)
    stats["added_faces"] = int(len(new_faces))
    return out, stats

# =============================
# Gap stitch (point-to-point bridging)
# =============================
def estimate_vertex_normals_fast(mesh):
    try:
        return np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception:
        return np.zeros((len(mesh.vertices), 3), dtype=np.float64)

def edge_exists(edge_set, a, b):
    return ((a, b) if a < b else (b, a)) in edge_set

def triangle_area_3d(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def knn_candidate_pairs_boundary(boundary_vs, V, k_neighbors=8, max_dist=np.inf):
    ids = np.asarray(boundary_vs, dtype=np.int32)
    if len(ids) < 2:
        return np.empty((0, 2), dtype=np.int32)
    pts = np.asarray(V[ids], dtype=np.float64)
    tree = cKDTree(pts)
    k = min(int(k_neighbors) + 1, len(ids))
    dists, idxs = tree.query(pts, k=k)
    if k == 1:
        return np.empty((0, 2), dtype=np.int32)
    dists = np.atleast_2d(dists); idxs = np.atleast_2d(idxs)
    src = np.repeat(np.arange(len(ids), dtype=np.int32), k - 1)
    dst = idxs[:, 1:].reshape(-1).astype(np.int32)
    dd = dists[:, 1:].reshape(-1)
    mask = np.isfinite(dd)
    if np.isfinite(max_dist):
        mask &= (dd <= float(max_dist))
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([ids[src[mask]], ids[dst[mask]]]).astype(np.int32)

def best_match_from_candidate_pairs(candidate_pairs, V, adj, edge_set, VN, normal_dot_min, max_bridge_dist):
    if len(candidate_pairs) == 0:
        return {}
    V = np.asarray(V, dtype=np.float64)
    VN = np.asarray(VN, dtype=np.float64)
    pairs = np.asarray(candidate_pairs, dtype=np.int32)
    gi = pairs[:, 0]; gj = pairs[:, 1]
    Pi = V[gi]; Pj = V[gj]
    d = np.linalg.norm(Pi - Pj, axis=1)
    nd = np.abs(np.sum(VN[gi] * VN[gj], axis=1))
    mask = (d > 1e-12) & (d <= float(max_bridge_dist)) & (nd >= float(normal_dot_min))
    if not np.any(mask):
        return {}
    gi = gi[mask]; gj = gj[mask]; d = d[mask]
    adj_set = {int(k): set(v) for k, v in adj.items()}

    best_match = {}
    best_dist = {}
    for a, b, dist_ab in zip(gi, gj, d):
        a = int(a); b = int(b)
        if a == b:
            continue
        if edge_exists(edge_set, a, b):
            continue
        if len(adj_set.get(a, set()) & adj_set.get(b, set())) > 0:
            continue
        old = best_dist.get(a, None)
        if old is None or dist_ab < old:
            best_dist[a] = float(dist_ab)
            best_match[a] = b
    return best_match

def stitch_boundary_gaps(
    mesh,
    max_bridge_dist=0.12,
    normal_dot_min=0.55,
    max_bridge_pairs=6000,
    min_triangle_area=1e-8,
    boundary_cache=None,
    knn_neighbors=8,
):
    stats = {"boundary_vertices": 0, "candidate_pairs": 0, "mutual_pairs": 0, "bridge_quads": 0, "added_faces": 0}

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

    candidate_pairs = knn_candidate_pairs_boundary(boundary_vs, V, k_neighbors=knn_neighbors, max_dist=max_bridge_dist)
    best_match = best_match_from_candidate_pairs(
        candidate_pairs, V=V, adj=adj, edge_set=edge_set, VN=VN,
        normal_dot_min=normal_dot_min, max_bridge_dist=max_bridge_dist,
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
        a = int(a); b = int(b)
        pa = partner.get(a, None); pb = partner.get(b, None)
        if pa is None or pb is None:
            continue
        if pa == pb or pa in (a, b) or pb in (a, b):
            continue
        if pa == b or pb == a:
            continue
        quad_key = tuple(sorted([a, b, pa, pb]))
        if quad_key in used_quads:
            continue

        A = V[a]; B = V[b]; C = V[pb]; D = V[pa]
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
    stats["added_faces"] = int(len(new_faces))
    return out, stats

# =============================
# Early hole repair pipeline
# =============================
def run_early_hole_repair(
    mesh,
    enable=True,
    max_hole_edges=160,
    max_hole_diameter=1.5,
    max_hole_area=1.2,
    max_hole_plane_residual=0.10,
    max_hole_candidate_loops=8000,
    enable_gap_stitch=True,
    max_bridge_dist=0.30,
    bridge_normal_dot_min=0.15,
    max_bridge_pairs=20000,
    verbose=False,
):
    stats = {"enabled": bool(enable), "loop_fill": {}, "gap_stitch": {}}

    if not enable:
        return mesh.copy(), stats

    out = mesh.copy()

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

    if enable_gap_stitch:
        boundary_cache = build_boundary_cache(out)
        out, gap_stats = stitch_boundary_gaps(
            out,
            max_bridge_dist=max_bridge_dist,
            normal_dot_min=bridge_normal_dot_min,
            max_bridge_pairs=max_bridge_pairs,
            boundary_cache=boundary_cache,
            knn_neighbors=8,
        )
    else:
        gap_stats = {"boundary_vertices": 0, "candidate_pairs": 0, "mutual_pairs": 0, "bridge_quads": 0, "added_faces": 0}
    stats["gap_stitch"] = gap_stats

    if verbose:
        total_added = loop_stats.get("added_faces", 0) + gap_stats.get("added_faces", 0)
        log(f"[2/7] 早期空洞修补... "
            f"loop_filled={loop_stats.get('filled_holes', 0)} "
            f"gap_quads={gap_stats.get('bridge_quads', 0)} "
            f"added_faces={total_added}")

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
    face_adj_csr = make_sparse_graph_from_pairs(len(F), face_adj_pairs)
    vert_adj_csr = build_vertex_adjacency_sparse(len(V), F)

    is_boundary = np.zeros(len(V), dtype=bool)
    b = boundary_vertices(mesh)
    if len(b):
        is_boundary[b] = True

    return {
        "vertices": V, "faces": F,
        "face_normals": FN, "tri_centers": TC, "face_areas": FA,
        "bbox_diag": bbox_diag,
        "face_adj_pairs": face_adj_pairs,
        "face_adj_csr": face_adj_csr,
        "vert_adj_csr": vert_adj_csr,
        "is_boundary": is_boundary,
    }

# =============================
# Plane detection
# =============================
def connected_groups_sparse(ids, csr_graph, face_adj_pairs=None):
    ids = np.asarray(ids, dtype=np.int32)
    n = len(ids)
    if n == 0:
        return []
    if n == 1:
        return [ids]

    N = csr_graph.shape[0]
    use_pair_filter = (
        face_adj_pairs is not None
        and len(face_adj_pairs) > 0
        and n > max(5000, int(0.15 * N))
    )

    if use_pair_filter:
        in_set = np.zeros(N, dtype=bool)
        in_set[ids] = True
        mask = in_set[face_adj_pairs[:, 0]] & in_set[face_adj_pairs[:, 1]]
        sel = face_adj_pairs[mask]
        if len(sel) == 0:
            return [ids[i:i + 1] for i in range(n)]
        remap = np.full(N, -1, dtype=np.int32)
        remap[ids] = np.arange(n, dtype=np.int32)
        rs = remap[sel[:, 0]]; cs = remap[sel[:, 1]]
        rows = np.concatenate([rs, cs])
        cols = np.concatenate([cs, rs])
        data = np.ones(len(rows), dtype=np.uint8)
        small = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.uint8)
        n_comp, labels = sp_connected_components(small, directed=False, return_labels=True)
        return [ids[labels == i] for i in range(n_comp)]

    sub = csr_graph[ids][:, ids]
    n_comp, labels = sp_connected_components(sub, directed=False, return_labels=True)
    return [ids[labels == i] for i in range(n_comp)]

def fit_plane_faces(face_ids, cache):
    face_ids = np.asarray(face_ids, dtype=np.int32)
    centers = cache["tri_centers"][face_ids]
    areas = cache["face_areas"][face_ids]
    normals = cache["face_normals"][face_ids]
    c, n = robust_plane(centers, areas, iters=3, huber_k=1.5)
    dist = np.dot(centers - c, n)
    _, _, vals = weighted_pca(centers, areas)
    dots = abs_normal_dot(normals, n)
    mean_angle = float(np.degrees(np.mean(np.arccos(np.clip(dots, 0.0, 1.0))))) if len(dots) else 0.0
    return {
        "face_ids": face_ids,
        "centroid": c,
        "normal": n,
        "sigma": robust_sigma(dist),
        "area": float(np.sum(areas)),
        "planarity_ratio": float(vals[0] / (np.sum(vals) + 1e-12)),
        "mean_normal_angle": mean_angle,
    }

def initial_patch_labels(cache, angle_thr=22.0):
    num_faces = len(cache["faces"])
    face_adj_pairs = cache["face_adj_pairs"]
    if num_faces == 0:
        return np.empty(0, dtype=np.int32), 0
    if face_adj_pairs is None or len(face_adj_pairs) == 0:
        return np.arange(num_faces, dtype=np.int32), num_faces
    fn = cache["face_normals"]
    cos_thr = np.cos(np.radians(angle_thr))
    dots = np.abs(np.sum(fn[face_adj_pairs[:, 0]] * fn[face_adj_pairs[:, 1]], axis=1))
    sel = face_adj_pairs[dots >= cos_thr]
    if len(sel) == 0:
        return np.arange(num_faces, dtype=np.int32), num_faces
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
            cfg["near_ratio"], cfg["far_ratio"],
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
    face_adj_pairs = cache.get("face_adj_pairs", None)

    if len(inlier) >= cfg["min_plane_faces"]:
        for g in connected_groups_sparse(inlier, cache["face_adj_csr"], face_adj_pairs):
            if len(g) < cfg["min_plane_faces"]:
                continue
            if float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            p = fit_plane_faces(g, cache)
            if p["area"] >= cfg["min_plane_area"]:
                res.append(p)

    if depth < cfg["max_split_depth"] and len(outlier) >= cfg["min_plane_faces"]:
        for g in connected_groups_sparse(outlier, cache["face_adj_csr"], face_adj_pairs):
            if len(g) < cfg["min_plane_faces"]:
                continue
            if float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            res.extend(split_planes(g, cache, cfg, depth + 1))

    return res

def dedup_planes(planes, overlap=0.85, total_faces=None):
    prepared = []
    max_face_id = 0
    for p in planes:
        ids = np.unique(np.asarray(p["face_ids"], dtype=np.int32))
        q = dict(p); q["face_ids"] = ids
        prepared.append(q)
        if len(ids) > 0:
            mx = int(ids.max())
            if mx > max_face_id:
                max_face_id = mx
    if total_faces is None:
        total_faces = max_face_id + 1

    kept = []
    kept_ids = []
    kept_masks = []
    BIG = 2000

    for p in sorted(prepared, key=lambda x: (len(x["face_ids"]), x["area"]), reverse=True):
        ids = p["face_ids"]
        n_ids = len(ids)
        ok = True
        for k_ids, k_mask in zip(kept_ids, kept_masks):
            min_len = min(n_ids, len(k_ids))
            if min_len == 0:
                continue
            need = int(np.ceil(overlap * min_len))
            inter = int(k_mask[ids].sum()) if k_mask is not None else np.intersect1d(ids, k_ids, assume_unique=True).size
            if inter >= need:
                ok = False
                break
        if ok:
            kept.append(p)
            kept_ids.append(ids)
            if n_ids >= BIG:
                m = np.zeros(total_faces, dtype=bool)
                m[ids] = True
                kept_masks.append(m)
            else:
                kept_masks.append(None)

    return kept

def detect_planes(mesh, cfg, verbose=False):
    if verbose:
        log("[4/7] 多平面识别...")

    cache = make_cache(mesh)

    patch_labels, patch_count = initial_patch_labels(cache, cfg["patch_normal_angle_deg"])
    face_counts = np.bincount(patch_labels, minlength=patch_count)
    area_sums = np.bincount(patch_labels, weights=cache["face_areas"], minlength=patch_count)
    valid_patch_mask = (face_counts >= cfg["min_patch_faces"]) & (area_sums >= cfg["min_plane_area"])
    valid_patch_ids = np.where(valid_patch_mask)[0]

    planes = []
    if len(valid_patch_ids) > 0:
        order = np.argsort(patch_labels, kind="mergesort")
        labels_sorted = patch_labels[order]
        starts = np.flatnonzero(np.r_[True, labels_sorted[1:] != labels_sorted[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts, ends):
            pid = int(labels_sorted[s])
            if not valid_patch_mask[pid]:
                continue
            patch = order[s:e].astype(np.int32)
            planes.extend(split_planes(patch, cache, cfg, 0))

    planes = dedup_planes(planes, total_faces=len(cache["faces"]))

    if verbose:
        log(f"  初始 patch 数: {patch_count}")
        log(f"  有效 patch 数: {len(valid_patch_ids)}")
        log(f"  识别平面数: {len(planes)}")

    return planes, cache

# =============================
# Plane refinement
# =============================
def expand_ring_sparse(ids, vert_adj_csr, rings=1):
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) == 0 or rings <= 0 or vert_adj_csr is None:
        return np.unique(ids)
    n = vert_adj_csr.shape[0]
    indptr = vert_adj_csr.indptr
    indices = vert_adj_csr.indices
    mask = np.zeros(n, dtype=bool)
    mask[ids] = True
    frontier = np.unique(ids)
    for _ in range(rings):
        if len(frontier) == 0:
            break
        starts = indptr[frontier]
        ends = indptr[frontier + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            break
        all_neigh = np.empty(total, dtype=indices.dtype)
        pos = 0
        for s, e in zip(starts, ends):
            all_neigh[pos:pos + (e - s)] = indices[s:e]
            pos += (e - s)
        new_neigh = all_neigh[~mask[all_neigh]]
        if len(new_neigh) == 0:
            break
        new_neigh = np.unique(new_neigh)
        mask[new_neigh] = True
        frontier = new_neigh
    return np.where(mask)[0].astype(np.int32)

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

def adaptive_smooth_iters(base_iters, sigma_v, eff_noise, dist_boost, stage_smooth_mul):
    n = int(round(base_iters * (1.0 + 0.35 * eff_noise + 0.20 * dist_boost) * stage_smooth_mul))
    if sigma_v < 0.0015:
        return max(4, min(16, n))
    if sigma_v < 0.003:
        return max(5, min(24, n))
    return max(6, min(40, n))

def refine_one_plane(task):
    i = task["plane_idx"]
    p = task["plane"]
    cfg = task["cfg"]
    state = task["state"]

    V = state["V"]; F = state["F"]
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

    vids = expand_ring_sparse(base_vids, vert_adj_csr, cfg["expand_rings"])
    pts = V[vids]

    c, n = robust_plane(pts, None, 3, 1.4)
    sd = np.dot(pts - c, n)
    sigma_v = robust_sigma(sd)

    noise_ratio = sigma_v / max(cfg["vertex_inlier_dist"], 1e-12)
    noise_boost = min(1.8, cfg["noise_adapt_gain"] * noise_ratio)
    eff_noise = noise_boost * dc["noise"]

    smooth_iters = adaptive_smooth_iters(
        cfg["base_normal_smooth_iterations"], sigma_v, eff_noise, dist_boost, dc["smooth"]
    )

    residual_shrink = (
        cfg["base_residual_shrink"] / (1.0 + 0.8 * eff_noise + 0.3 * dist_boost) * dc["shrink"]
    )
    residual_shrink = max(0.001, min(0.08, residual_shrink))

    inlier_thr = max(cfg["vertex_inlier_dist"] * dc["inlier"], 2.0 * sigma_v)
    outlier_thr = max(cfg["vertex_outlier_dist"] * dc["outlier"], 3.5 * sigma_v)
    snap_strength = cfg["base_snap_strength"] * dc["snap"]

    valid = np.abs(sd) <= outlier_thr
    vals = np.where(valid, sd, 0.0)

    A_local = vert_adj_csr[vids][:, vids] if (vert_adj_csr is not None and len(vids) > 0) else None
    sm = smooth_values_sparse(vals, valid, A_local, smooth_iters, 0.25) if A_local is not None else vals.copy()

    moved_vids = vids[valid]
    if len(moved_vids) == 0:
        return {
            "vids": np.empty(0, dtype=np.int32),
            "acc": np.empty((0, 3), dtype=np.float64),
            "w": np.empty(0, dtype=np.float64),
            "report": {
                "plane_id": int(i), "stage": dc["stage"],
                "plane_dist_sensor": plane_dist, "faces": int(len(face_ids)),
                "vertex_sigma": float(sigma_v),
                "inlier_thr": float(inlier_thr), "outlier_thr": float(outlier_thr),
                "smooth_iters": int(smooth_iters), "residual_shrink": float(residual_shrink),
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
            "plane_id": int(i), "stage": dc["stage"],
            "plane_dist_sensor": plane_dist, "faces": int(len(face_ids)),
            "vertex_sigma": float(sigma_v),
            "inlier_thr": float(inlier_thr), "outlier_thr": float(outlier_thr),
            "smooth_iters": int(smooth_iters), "residual_shrink": float(residual_shrink),
        }
    }

def refine_planes_parallel(mesh, planes, cfg, shared_cache, verbose=False, tag="[5/7]", num_workers=1):
    if not planes:
        return mesh.copy(), []

    state = {
        "V": np.asarray(mesh.vertices, dtype=np.float64),
        "F": np.asarray(mesh.faces, dtype=np.int32),
        "vert_adj_csr": shared_cache["vert_adj_csr"],
        "is_boundary": shared_cache["is_boundary"] if cfg["boundary_protect"] else np.zeros(len(mesh.vertices), dtype=bool),
        "bbox_diag": shared_cache["bbox_diag"],
    }

    tasks = [{"plane_idx": i, "plane": p, "cfg": cfg, "state": state} for i, p in enumerate(planes)]

    if verbose:
        log(f"{tag} 平面精修... workers={num_workers}, planes={len(planes)}")

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

    out = trimesh.Trimesh(vertices=V2, faces=np.asarray(mesh.faces, dtype=np.int32).copy(), process=False)
    return out, sorted(reports, key=lambda x: x["plane_id"])

# =============================
# Save outputs
# =============================
def save_outputs(mesh, planes, reports1, before, after, out_dir, input_path,
                 num_workers, timings, hole_stats,
                 export_obj=False, export_ply=True,
                 export_planes_json=False, export_quality_report_json=False):
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(input_path))[0]

    obj_path = os.path.join(out_dir, f"{base}_refined.obj") if export_obj else None
    ply_path = os.path.join(out_dir, f"{base}_refined.ply") if export_ply else None
    planes_path = os.path.join(out_dir, f"{base}_planes.json") if export_planes_json else None
    report_path = os.path.join(out_dir, f"{base}_quality_report.json") if export_quality_report_json else None

    if export_obj:
        mesh.export(obj_path)
    if export_ply:
        try:
            mesh.export(ply_path, encoding='binary')
        except TypeError:
            mesh.export(ply_path)

    if export_planes_json:
        with open(planes_path, "w", encoding="utf-8") as f:
            json.dump({
                "detected_plane_count": len(planes),
                "planes": [{
                    "plane_id": i,
                    "faces": int(len(p["face_ids"])),
                    "area": float(p["area"]),
                    "sigma": float(p["sigma"]),
                    "centroid": [float(x) for x in p["centroid"]],
                    "normal": [float(x) for x in p["normal"]],
                } for i, p in enumerate(planes)],
                "first_pass_reports": reports1,
            }, f, indent=2, ensure_ascii=False)

    if export_quality_report_json:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "input": input_path,
                "output_obj": obj_path, "output_ply": ply_path,
                "planes_json": planes_path,
                "before": before, "after": after,
                "early_hole_fill": hole_stats,
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

    log("========== PIPELINE START ==========")
    log(f"num_workers: {num_workers}")
    log(f"export: PLY={not args.disable_export_ply}, OBJ={args.export_obj}, "
        f"planes.json={args.export_planes_json}, report.json={args.export_quality_report_json}")

    log("[0/7] 读取网格...")
    t0 = time.perf_counter()
    mesh0 = load_mesh(args.input)
    timings["load_mesh"] = round(time.perf_counter() - t0, 4)
    before = mesh_report(mesh0)

    log("[1/7] 基础清理...")
    t0 = time.perf_counter()
    mesh = clean_mesh_light(mesh0, fix_normals=False)
    timings["clean_mesh_1"] = round(time.perf_counter() - t0, 4)

    log("[2/7] 早期空洞修补 / 裂缝桥接...")
    t0 = time.perf_counter()
    mesh, hole_stats = run_early_hole_repair(
        mesh,
        enable=not args.disable_early_hole_fill,
        max_hole_edges=args.max_hole_edges,
        max_hole_diameter=args.max_hole_diameter,
        max_hole_area=args.max_hole_area,
        max_hole_plane_residual=args.max_hole_plane_residual,
        max_hole_candidate_loops=args.max_hole_candidate_loops,
        enable_gap_stitch=not args.disable_gap_stitch,
        max_bridge_dist=args.max_bridge_dist,
        bridge_normal_dot_min=args.bridge_normal_dot_min,
        max_bridge_pairs=args.max_bridge_pairs,
        verbose=args.verbose,
    )
    timings["early_hole_fill"] = round(time.perf_counter() - t0, 4)

    log("[3/7] 连通分量过滤...")
    t0 = time.perf_counter()
    mesh = filter_components_fast(
        mesh,
        args.min_component_faces, args.min_component_area,
        args.min_component_max_extent, args.keep_top_k_faces
    )
    timings["filter_components"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    planes, shared_cache = detect_planes(mesh, detect_cfg, args.verbose)
    timings["detect_planes"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    mesh, reports1 = refine_planes_parallel(
        mesh, planes, refine_cfg, shared_cache=shared_cache,
        verbose=args.verbose, tag="[5/7]", num_workers=num_workers,
    )
    timings["refine_planes"] = round(time.perf_counter() - t0, 4)

    log("[6/7] 去除游离噪点碎块...")
    t0 = time.perf_counter()
    mesh = remove_floating_noise_fast(
        mesh,
        min_faces=args.noise_min_faces, min_area=args.noise_min_area,
        min_extent=args.noise_min_extent, keep_top_k=args.noise_keep_top_k,
    )
    timings["remove_floating_noise"] = round(time.perf_counter() - t0, 4)

    log("[7/7] 最终轻量清理...")
    t0 = time.perf_counter()
    mesh = clean_mesh_light(mesh, fix_normals=not args.skip_final_fix_normals)
    timings["clean_mesh_final"] = round(time.perf_counter() - t0, 4)

    after = mesh_report(mesh)

    obj_path, ply_path, planes_path, report_path = save_outputs(
        mesh, planes, reports1, before, after,
        args.output_dir, args.input, num_workers, timings, hole_stats,
        export_obj=args.export_obj,
        export_ply=not args.disable_export_ply,
        export_planes_json=args.export_planes_json,
        export_quality_report_json=args.export_quality_report_json,
    )
    timings["total"] = round(time.perf_counter() - total_t0, 4)

    log("")
    log("========== PIPELINE DONE ==========")
    log(f"Output OBJ:     {obj_path}")
    log(f"Output PLY:     {ply_path}")
    log(f"Planes JSON:    {planes_path}")
    log(f"Quality Report: {report_path}")
    log("Timings (sec):")
    for k, v in timings.items():
        log(f"  {k}: {v:.3f}")
    log("===================================")

# =============================
# CLI
# =============================
def build_parser():
    p = argparse.ArgumentParser(description="Mesh refine pipeline (slim)")

    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default=DEFAULT_PRESET)

    # component filter
    p.add_argument("--min_component_faces", type=int, default=COMPONENT_FILTER_CONFIG["min_component_faces"])
    p.add_argument("--min_component_area", type=float, default=COMPONENT_FILTER_CONFIG["min_component_area"])
    p.add_argument("--min_component_max_extent", type=float, default=COMPONENT_FILTER_CONFIG["min_component_max_extent"])
    p.add_argument("--keep_top_k_faces", type=int, default=COMPONENT_FILTER_CONFIG["keep_top_k_faces"])

    # early hole fill
    p.add_argument("--disable_early_hole_fill", action="store_true", default=HOLE_FILL_CONFIG["disable_early_hole_fill"])
    p.add_argument("--max_hole_edges", type=int, default=HOLE_FILL_CONFIG["max_hole_edges"])
    p.add_argument("--max_hole_diameter", type=float, default=HOLE_FILL_CONFIG["max_hole_diameter"])
    p.add_argument("--max_hole_area", type=float, default=HOLE_FILL_CONFIG["max_hole_area"])
    p.add_argument("--max_hole_plane_residual", type=float, default=HOLE_FILL_CONFIG["max_hole_plane_residual"])
    p.add_argument("--max_hole_candidate_loops", type=int, default=HOLE_FILL_CONFIG["max_hole_candidate_loops"])

    # gap stitch
    p.add_argument("--disable_gap_stitch", action="store_true", default=HOLE_FILL_CONFIG["disable_gap_stitch"])
    p.add_argument("--max_bridge_dist", type=float, default=HOLE_FILL_CONFIG["max_bridge_dist"])
    p.add_argument("--bridge_normal_dot_min", type=float, default=HOLE_FILL_CONFIG["bridge_normal_dot_min"])
    p.add_argument("--max_bridge_pairs", type=int, default=HOLE_FILL_CONFIG["max_bridge_pairs"])

    # plane detection
    p.add_argument("--patch_normal_angle_deg", type=float, default=PLANE_DETECT_CONFIG["patch_normal_angle_deg"])
    p.add_argument("--min_patch_faces", type=int, default=PLANE_DETECT_CONFIG["min_patch_faces"])
    p.add_argument("--min_plane_faces", type=int, default=PLANE_DETECT_CONFIG["min_plane_faces"])
    p.add_argument("--min_plane_area", type=float, default=PLANE_DETECT_CONFIG["min_plane_area"])
    p.add_argument("--base_face_plane_dist", type=float, default=PLANE_DETECT_CONFIG["base_face_plane_dist"])
    p.add_argument("--plane_normal_angle_deg", type=float, default=PLANE_DETECT_CONFIG["plane_normal_angle_deg"])
    p.add_argument("--sigma_dist_mult", type=float, default=PLANE_DETECT_CONFIG["sigma_dist_mult"])
    p.add_argument("--max_split_depth", type=int, default=PLANE_DETECT_CONFIG["max_split_depth"])

    # plane refine
    p.add_argument("--vertex_inlier_dist", type=float, default=PLANE_REFINE_CONFIG["vertex_inlier_dist"])
    p.add_argument("--vertex_outlier_dist", type=float, default=PLANE_REFINE_CONFIG["vertex_outlier_dist"])
    p.add_argument("--base_snap_strength", type=float, default=PLANE_REFINE_CONFIG["base_snap_strength"])
    p.add_argument("--base_residual_shrink", type=float, default=PLANE_REFINE_CONFIG["base_residual_shrink"])
    p.add_argument("--base_normal_smooth_iterations", type=int, default=PLANE_REFINE_CONFIG["base_normal_smooth_iterations"])
    p.add_argument("--noise_adapt_gain", type=float, default=PLANE_REFINE_CONFIG["noise_adapt_gain"])
    p.add_argument("--no_boundary_protect", action="store_true", default=PLANE_REFINE_CONFIG["no_boundary_protect"])
    p.add_argument("--boundary_scale", type=float, default=PLANE_REFINE_CONFIG["boundary_scale"])
    p.add_argument("--expand_rings", type=int, default=PLANE_REFINE_CONFIG["expand_rings"])

    # distance strategy
    p.add_argument("--sensor_origin", type=str, default=DISTANCE_STRATEGY_CONFIG["sensor_origin"])
    p.add_argument("--distance_gain", type=float, default=DISTANCE_STRATEGY_CONFIG["distance_gain"])
    p.add_argument("--use_distance_strategy", dest="use_distance_strategy", action="store_true")
    p.add_argument("--no_use_distance_strategy", dest="use_distance_strategy", action="store_false")
    p.set_defaults(use_distance_strategy=DISTANCE_STRATEGY_CONFIG["use_distance_strategy"])
    p.add_argument("--near_ratio", type=float, default=DISTANCE_STRATEGY_CONFIG["near_ratio"])
    p.add_argument("--far_ratio", type=float, default=DISTANCE_STRATEGY_CONFIG["far_ratio"])

    # noise filter
    p.add_argument("--noise_min_faces", type=int, default=NOISE_FILTER_CONFIG["noise_min_faces"])
    p.add_argument("--noise_min_area", type=float, default=NOISE_FILTER_CONFIG["noise_min_area"])
    p.add_argument("--noise_min_extent", type=float, default=NOISE_FILTER_CONFIG["noise_min_extent"])
    p.add_argument("--noise_keep_top_k", type=int, default=NOISE_FILTER_CONFIG["noise_keep_top_k"])

    # runtime
    p.add_argument("--skip_final_fix_normals", dest="skip_final_fix_normals", action="store_true")
    p.add_argument("--enable_final_fix_normals", dest="skip_final_fix_normals", action="store_false")
    p.set_defaults(skip_final_fix_normals=RUNTIME_CONFIG["skip_final_fix_normals"])
    p.add_argument("--num_workers", type=int, default=RUNTIME_CONFIG["num_workers"])
    p.add_argument("--verbose", dest="verbose", action="store_true")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.set_defaults(verbose=RUNTIME_CONFIG["verbose"])

    # export control
    p.add_argument("--export_obj", action="store_true", default=RUNTIME_CONFIG["export_obj"])
    p.add_argument("--disable_export_ply", action="store_true", default=RUNTIME_CONFIG["disable_export_ply"])
    p.add_argument("--export_planes_json", action="store_true", default=RUNTIME_CONFIG["export_planes_json"])
    p.add_argument("--export_quality_report_json", action="store_true", default=RUNTIME_CONFIG["export_quality_report_json"])
    p.add_argument("--export_json", action="store_true",
                   help="一次性打开 planes.json + quality_report.json")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args = apply_preset_to_args(args)

    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = rf".\outputs_{args.preset}"

    if getattr(args, "export_json", False):
        args.export_planes_json = True
        args.export_quality_report_json = True

    enable_log(bool(args.verbose))

    log("========== RUN CONFIG ==========")
    log(f"preset: {args.preset}")
    log(f"input: {args.input}")
    log(f"output_dir: {args.output_dir}")
    log(f"sensor_origin: {args.sensor_origin}")
    log(f"num_workers: {args.num_workers}")
    log("================================")

    process_mesh(args)

if __name__ == "__main__":
    main()