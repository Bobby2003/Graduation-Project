import os, json, argparse
from collections import defaultdict, deque
from multiprocessing import get_context, cpu_count

import numpy as np
import trimesh

try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except Exception:
    sp = None
    SCIPY_AVAILABLE = False

_WORKER_STATE = {}

def _init_refine_worker(state):
    global _WORKER_STATE
    _WORKER_STATE = state

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
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return float(1.4826 * mad)

def mesh_report(mesh):
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
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene 中没有可用网格")
        obj = trimesh.util.concatenate(meshes)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.vertices) == 0 or len(obj.faces) == 0:
        raise ValueError("无效或空网格")
    return obj

def clean_mesh(mesh):
    mesh = mesh.copy()
    for fn in ["remove_duplicate_faces", "remove_degenerate_faces", "remove_unreferenced_vertices", "remove_infinite_values"]:
        try:
            getattr(mesh, fn)()
        except Exception:
            pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.process(validate=True)
    except Exception:
        pass
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

def filter_components(mesh, min_faces=80, min_area=0.0015, min_extent=0.04, keep_top_k=20):
    comps = split_components(mesh)
    if not comps:
        return mesh
    stats = []
    for i, c in enumerate(comps):
        ext = c.bounding_box.extents if len(c.vertices) else np.zeros(3)
        stats.append((i, len(c.faces), float(c.area), float(np.max(ext))))
    keep = {i for i, f, a, e in stats if f >= min_faces and a >= min_area and e >= min_extent}
    for i, *_ in sorted(stats, key=lambda x: x[1], reverse=True)[:keep_top_k]:
        keep.add(i)
    kept = [comps[i] for i in sorted(keep)] or [max(comps, key=lambda x: len(x.faces))]
    out = trimesh.util.concatenate(kept)
    try:
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out

def make_cache(mesh):
    return {
        "vertices": np.asarray(mesh.vertices, dtype=np.float64),
        "faces": np.asarray(mesh.faces, dtype=np.int32),
        "face_normals": np.asarray(mesh.face_normals, dtype=np.float64),
        "tri_centers": np.asarray(mesh.triangles_center, dtype=np.float64),
        "face_areas": np.asarray(mesh.area_faces, dtype=np.float64),
        "bbox_diag": float(np.linalg.norm(mesh.bounding_box.extents)) + 1e-12,
    }

def abs_normal_dot(normals, normal):
    return np.clip(np.abs(np.asarray(normals, dtype=np.float64) @ np.asarray(normal, dtype=np.float64)), 0.0, 1.0)

def face_graph(mesh):
    g = defaultdict(list)
    adj = mesh.face_adjacency
    if adj is not None:
        for a, b in adj:
            a, b = int(a), int(b)
            g[a].append(b)
            g[b].append(a)
    return g

def vertex_graph(mesh):
    vg = [set() for _ in range(len(mesh.vertices))]
    for a, b, c in np.asarray(mesh.faces):
        a, b, c = int(a), int(b), int(c)
        vg[a].update((b, c))
        vg[b].update((a, c))
        vg[c].update((a, b))
    return [np.fromiter(sorted(x), dtype=np.int32) if x else np.empty(0, dtype=np.int32) for x in vg]

def connected_groups(ids, graph):
    ids = set(ids.tolist() if isinstance(ids, np.ndarray) else ids)
    vis, groups = set(), []
    for s in ids:
        if s in vis:
            continue
        q = deque([s])
        vis.add(s)
        group = [s]
        while q:
            u = q.popleft()
            for v in graph.get(u, []):
                if v in ids and v not in vis:
                    vis.add(v)
                    q.append(v)
                    group.append(v)
        groups.append(np.array(group, dtype=np.int32))
    return groups

def weighted_pca(points, weights=None):
    p = np.asarray(points, dtype=np.float64)
    w = np.ones(len(p), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sum(w) + 1e-12
    c = np.sum(p * w[:, None], axis=0) / sw
    x = (p - c) * np.sqrt(w[:, None])
    cov = (x.T @ x) / sw
    vals, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    return c, n / (np.linalg.norm(n) + 1e-12), vals

def robust_plane(points, weights=None, iters=8, huber_k=1.5):
    p = np.asarray(points, dtype=np.float64)
    w = np.ones(len(p), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    c, n = p.mean(axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for _ in range(iters):
        c, n, _ = weighted_pca(p, w)
        d = np.dot(p - c, n)
        s = robust_sigma(d) + 1e-12
        r = np.abs(d) / (huber_k * s + 1e-12)
        w = np.where(r <= 1.0, w, w / (r + 1e-12))
    return c, n

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

def dist_cfg(plane_dist, bbox_diag, near_ratio=0.25, far_ratio=0.50):
    r = plane_dist / (bbox_diag + 1e-12)
    if r <= near_ratio:
        return dict(stage="near", snap=1.00, inlier=0.85, outlier=1.00, shrink=0.70, smooth=1.20, noise=1.00, detect_dist=1.00, detect_ang=1.00)
    if r <= far_ratio:
        return dict(stage="mid", snap=1.10, inlier=1.05, outlier=1.45, shrink=0.42, smooth=1.80, noise=1.35, detect_dist=1.20, detect_ang=1.12)
    return dict(stage="far", snap=1.22, inlier=1.35, outlier=2.40, shrink=0.12, smooth=2.80, noise=2.10, detect_dist=1.65, detect_ang=1.35)

def initial_patches(cache, adjacency, angle_thr=22.0):
    fn = cache["face_normals"]
    cos_thr = np.cos(np.radians(angle_thr))
    vis = np.zeros(len(fn), dtype=bool)
    patches = []
    for seed in range(len(fn)):
        if vis[seed]:
            continue
        vis[seed] = True
        q = deque([seed])
        group = [seed]
        while q:
            u = q.popleft()
            nu = fn[u]
            for v in adjacency.get(u, []):
                if not vis[v] and abs(np.dot(nu, fn[v])) >= cos_thr:
                    vis[v] = True
                    q.append(v)
                    group.append(v)
        patches.append(np.array(group, dtype=np.int32))
    return patches

def split_planes(face_ids, adjacency, cfg, cache, depth=0):
    face_ids = np.asarray(face_ids, dtype=np.int32)
    if len(face_ids) < cfg["min_plane_faces"]:
        return []
    plane = fit_plane_faces(face_ids, cache)
    centers = cache["tri_centers"][face_ids]
    normals = cache["face_normals"][face_ids]
    d = np.dot(centers - plane["centroid"], plane["normal"])

    dc = dict(stage="none", detect_dist=1.0, detect_ang=1.0)
    if cfg["sensor_origin"] is not None and cfg["use_distance_strategy"]:
        dc = dist_cfg(np.linalg.norm(plane["centroid"] - cfg["sensor_origin"]), cache["bbox_diag"], cfg["near_ratio"], cfg["far_ratio"])

    dist_thr = max(cfg["base_face_plane_dist"] * dc["detect_dist"], cfg["sigma_dist_mult"] * plane["sigma"] * dc["detect_dist"])
    cos_thr = np.cos(np.radians(cfg["plane_normal_angle_deg"] * dc["detect_ang"]))
    dots = abs_normal_dot(normals, plane["normal"])
    inlier_mask = (np.abs(d) <= dist_thr) & (dots >= cos_thr)

    inlier, outlier, res = face_ids[inlier_mask], face_ids[~inlier_mask], []
    if len(inlier) >= cfg["min_plane_faces"]:
        for g in connected_groups(inlier, adjacency):
            if len(g) < cfg["min_plane_faces"] or float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            p = fit_plane_faces(g, cache)
            if p["area"] >= cfg["min_plane_area"]:
                res.append(p)

    if depth < cfg["max_split_depth"] and len(outlier) >= cfg["min_plane_faces"]:
        for g in connected_groups(outlier, adjacency):
            if len(g) < cfg["min_plane_faces"] or float(np.sum(cache["face_areas"][g])) < cfg["min_plane_area"]:
                continue
            res.extend(split_planes(g, adjacency, cfg, cache, depth + 1))
    return res

def dedup_planes(planes, overlap=0.85):
    kept = []
    for p in sorted(planes, key=lambda x: (len(x["face_ids"]), x["area"]), reverse=True):
        ps, ok = set(p["face_ids"].tolist()), True
        for k in kept:
            ks = set(k["face_ids"].tolist())
            if len(ps & ks) / max(1, min(len(ps), len(ks))) >= overlap:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept

def detect_planes(mesh, cfg, verbose=False):
    if verbose:
        print("[3/7] 多平面识别...")
    cache, adj, planes = make_cache(mesh), face_graph(mesh), []
    patches = initial_patches(cache, adj, cfg["patch_normal_angle_deg"])
    for patch in patches:
        if len(patch) >= cfg["min_patch_faces"] and float(np.sum(cache["face_areas"][patch])) >= cfg["min_plane_area"]:
            planes.extend(split_planes(patch, adj, cfg, cache, 0))
    planes = dedup_planes(planes)
    if verbose:
        print(f"  初始 patch 数: {len(patches)}")
        print(f"  识别平面数: {len(planes)}")
    return planes

def boundary_vertices(mesh):
    try:
        e = np.sort(mesh.edges_sorted, axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        return ue[cnt == 1].reshape(-1)
    except Exception:
        return np.empty(0, dtype=np.int32)

def expand_ring(ids, vg, rings=1):
    cur = set(ids.tolist() if isinstance(ids, np.ndarray) else ids)
    for _ in range(rings):
        nxt = set(cur)
        for u in cur:
            nxt.update(vg[u].tolist())
        cur = nxt
    return np.array(sorted(cur), dtype=np.int32)

def build_local_sparse_graph(vids, vg):
    if not SCIPY_AVAILABLE or len(vids) == 0:
        return None
    pos = {int(v): i for i, v in enumerate(vids)}
    rows, cols = [], []
    for i, gv in enumerate(vids):
        for nb in vg[gv]:
            j = pos.get(int(nb))
            if j is not None:
                rows.append(i)
                cols.append(j)
    if not rows:
        return None
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vids), len(vids)), dtype=np.float64)

def smooth_values_python(vids, vg, vals, valid, iters=8, self_w=0.25):
    pos = {int(v): i for i, v in enumerate(vids)}
    neigh = [[pos[int(n)] for n in vg[v] if int(n) in pos] for v in vids]
    x, nb_w = vals.copy(), 1.0 - self_w
    for _ in range(iters):
        y = x.copy()
        for i, ns in enumerate(neigh):
            if not valid[i] or not ns:
                continue
            vv = [x[j] for j in ns if valid[j]]
            if vv:
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
        avg[active] = (A @ x)[active] / denom[active]
        x[active] = self_w * x[active] + nb_w * avg[active]
    return x

def adaptive_smooth_iters(base_iters, sigma_v, eff_noise, dist_boost, stage_smooth_mul, pass_smooth_mul):
    n = int(round(base_iters * pass_smooth_mul * (1.0 + 0.35 * eff_noise + 0.20 * dist_boost) * stage_smooth_mul))
    if sigma_v < 0.0015:
        return max(4, min(16, n))
    if sigma_v < 0.003:
        return max(5, min(24, n))
    return max(6, min(40, n))

def _refine_plane_worker(task):
    S = _WORKER_STATE
    i, p, cfg, pass_mul, only_stage = task["plane_idx"], task["plane"], task["cfg"], task["pass_mul"], task["only_stage"]
    V, F, VG, is_boundary, bbox_diag = S["V"], S["F"], S["VG"], S["is_boundary"], S["bbox_diag"]

    face_ids = p["face_ids"]
    base_vids = np.unique(F[face_ids].reshape(-1))

    dc, plane_dist, dist_boost = dict(stage="none", snap=1, inlier=1, outlier=1, shrink=1, smooth=1, noise=1), None, 0.0
    if cfg["sensor_origin"] is not None:
        plane_dist = float(np.linalg.norm(p["centroid"] - cfg["sensor_origin"]))
        dist_boost = cfg["distance_gain"] * (plane_dist / bbox_diag)
        if cfg["use_distance_strategy"]:
            dc = dist_cfg(plane_dist, bbox_diag, cfg["near_ratio"], cfg["far_ratio"])
    if only_stage is not None and dc["stage"] != only_stage:
        return None

    vids = expand_ring(base_vids, VG, cfg["expand_rings"] + int(pass_mul.get("expand_rings_add", 0)))
    pts = V[vids]
    c, n = robust_plane(pts, None, 10, 1.4)
    sd = np.dot(pts - c, n)
    sigma_v = robust_sigma(sd)

    noise_ratio = sigma_v / max(cfg["vertex_inlier_dist"], 1e-12)
    noise_boost = min(1.8, cfg["noise_adapt_gain"] * pass_mul.get("noise", 1.0) * noise_ratio)
    eff_noise = noise_boost * dc["noise"]

    smooth_iters = adaptive_smooth_iters(cfg["base_normal_smooth_iterations"], sigma_v, eff_noise, dist_boost, dc["smooth"], pass_mul.get("smooth", 1.0))
    residual_shrink = cfg["base_residual_shrink"] * pass_mul.get("shrink", 1.0) / (1.0 + 0.8 * eff_noise + 0.3 * dist_boost) * dc["shrink"]
    residual_shrink = max(0.001, min(0.08, residual_shrink))
    inlier_thr = max(cfg["vertex_inlier_dist"] * dc["inlier"], 2.0 * sigma_v)
    outlier_thr = max(cfg["vertex_outlier_dist"] * pass_mul.get("outlier", 1.0) * dc["outlier"], 3.5 * sigma_v)
    snap_strength = cfg["base_snap_strength"] * pass_mul.get("snap", 1.0) * dc["snap"]

    valid = np.abs(sd) <= outlier_thr
    vals = np.where(valid, sd, 0.0)
    A = build_local_sparse_graph(vids, VG)
    sm = smooth_values_sparse(vals, valid, A, smooth_iters, 0.25) if SCIPY_AVAILABLE else smooth_values_python(vids, VG, vals, valid, smooth_iters, 0.25)

    moved_vids = vids[valid]
    if len(moved_vids) == 0:
        return {
            "vids": np.empty(0, dtype=np.int32),
            "acc": np.empty((0, 3), dtype=np.float64),
            "w": np.empty(0, dtype=np.float64),
            "report": {
                "plane_id": int(i), "stage": dc["stage"], "plane_dist_sensor": plane_dist,
                "faces": int(len(face_ids)), "vertex_sigma": float(sigma_v), "inlier_thr": float(inlier_thr),
                "outlier_thr": float(outlier_thr), "smooth_iters": int(smooth_iters), "residual_shrink": float(residual_shrink)
            }
        }

    d, ds = vals[valid], sm[valid]
    bmul = np.where(is_boundary[moved_vids], cfg["boundary_scale"], 1.0)
    target = V[moved_vids].copy()
    inlier = np.abs(d) <= inlier_thr
    target[inlier] -= d[inlier, None] * n
    target[~inlier] -= (d[~inlier] - residual_shrink * ds[~inlier])[:, None] * n
    w = np.where(inlier, snap_strength, min(1.0, 0.72 + 0.30 * eff_noise + 0.22 * dist_boost)) * bmul

    return {
        "vids": moved_vids.astype(np.int32),
        "acc": w[:, None] * target,
        "w": w,
        "report": {
            "plane_id": int(i), "stage": dc["stage"], "plane_dist_sensor": plane_dist,
            "faces": int(len(face_ids)), "vertex_sigma": float(sigma_v), "inlier_thr": float(inlier_thr),
            "outlier_thr": float(outlier_thr), "smooth_iters": int(smooth_iters), "residual_shrink": float(residual_shrink)
        }
    }

def refine_planes_parallel(mesh, planes, cfg, only_stage=None, pass_mul=None, verbose=False, tag="[4/7]", num_workers=1):
    if not planes:
        return mesh.copy(), []
    pass_mul = pass_mul or {}

    mesh2 = mesh.copy()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    VG = vertex_graph(mesh)
    is_boundary = np.zeros(len(V), dtype=bool)
    if cfg["boundary_protect"]:
        b = boundary_vertices(mesh2)
        if len(b):
            is_boundary[b] = True

    state = {
        "V": V,
        "F": F,
        "VG": VG,
        "is_boundary": is_boundary,
        "bbox_diag": float(np.linalg.norm(mesh2.bounding_box.extents)) + 1e-12,
    }
    tasks = [{"plane_idx": i, "plane": p, "cfg": cfg, "pass_mul": pass_mul, "only_stage": only_stage} for i, p in enumerate(planes)]

    if verbose:
        mode = "parallel" if num_workers and num_workers > 1 else "single"
        backend = "scipy.sparse" if SCIPY_AVAILABLE else "python"
        print(f"{tag} 平面精修... mode={mode}, workers={num_workers}, smooth_backend={backend}")

    if num_workers is None or num_workers <= 1:
        _init_refine_worker(state)
        results = [_refine_plane_worker(t) for t in tasks]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_init_refine_worker, initargs=(state,)) as pool:
            results = pool.map(_refine_plane_worker, tasks)

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
            print(f"  plane={rr['plane_id']:03d} stage={rr['stage']:<4} faces={rr['faces']:5d} sigma={rr['vertex_sigma']:.6f} out={rr['outlier_thr']:.5f} shrink={rr['residual_shrink']:.5f} smooth={rr['smooth_iters']}")

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
    ids = [int(r["plane_id"]) for r in reports if r.get("stage") == "far" and r.get("faces", 0) >= min_faces and r.get("vertex_sigma", 0.0) >= min_sigma]
    return [planes[i] for i in ids]

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

def remove_small_clusters(mesh, min_faces=40):
    comps = split_components(mesh)
    kept = [c for c in comps if len(c.faces) >= min_faces] or comps
    out = trimesh.util.concatenate(kept)
    try:
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out

def save_outputs(mesh, planes, reports1, reports2, before, after, out_dir, input_path, num_workers):
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
            "scipy_sparse_enabled": SCIPY_AVAILABLE,
            "num_workers": int(num_workers),
        }, f, indent=2, ensure_ascii=False)

    return obj_path, ply_path, planes_path, report_path

def process_mesh(args):
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

    if args.verbose:
        print("========== PIPELINE START ==========")
        print(f"scipy sparse available: {SCIPY_AVAILABLE}")
        print(f"num_workers: {num_workers}")

    mesh0 = load_mesh(args.input)
    before = mesh_report(mesh0)

    if args.verbose:
        print("[1/7] 基础清理...")
    mesh = clean_mesh(mesh0)

    if args.verbose:
        print("[2/7] 连通分量过滤...")
    mesh = filter_components(mesh, args.min_component_faces, args.min_component_area, args.min_component_max_extent, args.keep_top_k_faces)

    planes = detect_planes(mesh, detect_cfg, args.verbose)

    mesh, reports1 = refine_planes_parallel(mesh, planes, refine_cfg, None, None, args.verbose, "[4/7]", num_workers)

    reports2 = []
    if args.enable_far_second_pass:
        if args.verbose:
            print("[4.5/7] far second pass 选择平面...")
        selected = select_far_second_pass_planes(planes, reports1, args.far_second_pass_min_faces, args.far_second_pass_min_sigma)
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
            mesh, reports2 = refine_planes_parallel(mesh, selected, refine_cfg, None, pass_mul, args.verbose, "[4.5/7]", num_workers)

    if args.global_smooth_iter > 0:
        if args.verbose:
            print("[5/7] 全局平滑...")
        mesh = global_smooth(mesh, args.global_smooth_iter, args.global_smooth_method)

    if args.verbose:
        print("[6/7] 二次碎片清理...")
    mesh = remove_small_clusters(mesh, args.second_pass_min_faces)

    if args.verbose:
        print("[7/7] 最终清理...")
    mesh = clean_mesh(mesh)

    after = mesh_report(mesh)
    obj_path, ply_path, planes_path, report_path = save_outputs(mesh, planes, reports1, reports2, before, after, args.output_dir, args.input, num_workers)

    print("\n========== PIPELINE DONE ==========")
    print(f"Output OBJ:     {obj_path}")
    print(f"Output PLY:     {ply_path}")
    print(f"Planes JSON:    {planes_path}")
    print(f"Quality Report: {report_path}")
    print("===================================")

def build_parser():
    p = argparse.ArgumentParser(description="Mesh refine pipeline (CPU-only, simplified, no wall-corner)")
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--min_component_faces", type=int, default=80)
    p.add_argument("--min_component_area", type=float, default=0.0015)
    p.add_argument("--min_component_max_extent", type=float, default=0.04)
    p.add_argument("--keep_top_k_faces", type=int, default=20)

    p.add_argument("--patch_normal_angle_deg", type=float, default=24.0)
    p.add_argument("--min_patch_faces", type=int, default=25)
    p.add_argument("--min_plane_faces", type=int, default=40)
    p.add_argument("--min_plane_area", type=float, default=0.0008)
    p.add_argument("--base_face_plane_dist", type=float, default=0.014)
    p.add_argument("--plane_normal_angle_deg", type=float, default=18.0)
    p.add_argument("--sigma_dist_mult", type=float, default=3.6)
    p.add_argument("--max_split_depth", type=int, default=7)

    p.add_argument("--vertex_inlier_dist", type=float, default=0.008)
    p.add_argument("--vertex_outlier_dist", type=float, default=0.085)
    p.add_argument("--base_snap_strength", type=float, default=1.0)
    p.add_argument("--base_residual_shrink", type=float, default=0.006)
    p.add_argument("--base_normal_smooth_iterations", type=int, default=22)
    p.add_argument("--noise_adapt_gain", type=float, default=1.8)
    p.add_argument("--no_boundary_protect", action="store_true")
    p.add_argument("--boundary_scale", type=float, default=0.10)
    p.add_argument("--expand_rings", type=int, default=1)

    p.add_argument("--sensor_origin", type=str, default=None)
    p.add_argument("--distance_gain", type=float, default=0.45)
    p.add_argument("--use_distance_strategy", action="store_true")
    p.add_argument("--near_ratio", type=float, default=0.25)
    p.add_argument("--far_ratio", type=float, default=0.50)

    p.add_argument("--enable_far_second_pass", action="store_true")
    p.add_argument("--far_second_pass_outlier_mul", type=float, default=1.60)
    p.add_argument("--far_second_pass_shrink_mul", type=float, default=0.45)
    p.add_argument("--far_second_pass_smooth_mul", type=float, default=1.80)
    p.add_argument("--far_second_pass_snap_mul", type=float, default=1.20)
    p.add_argument("--far_second_pass_noise_mul", type=float, default=1.30)
    p.add_argument("--far_second_pass_expand_rings", type=int, default=1)
    p.add_argument("--far_second_pass_min_sigma", type=float, default=0.003)
    p.add_argument("--far_second_pass_min_faces", type=int, default=60)

    p.add_argument("--global_smooth_iter", type=int, default=1)
    p.add_argument("--global_smooth_method", choices=["taubin", "laplacian", "simple"], default="taubin")
    p.add_argument("--second_pass_min_faces", type=int, default=40)

    p.add_argument("--num_workers", type=int, default=0, help="<=0 自动使用 cpu_count()-1；1 表示单进程")
    p.add_argument("--verbose", action="store_true")
    return p

def main():
    process_mesh(build_parser().parse_args())

if __name__ == "__main__":
    main()