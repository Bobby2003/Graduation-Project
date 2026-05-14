"""
内存级 mesh refine 适配器（slim 版本）
- 直接接收 Open3D mesh
- 复用 mesh_refine_slim.py 的所有处理函数
- 不修改任何上游代码
"""
import os
import time
import json
import numpy as np
import trimesh
import open3d as o3d
from datetime import datetime
from multiprocessing import cpu_count

import mesh_refine_slim as mrs

# =============================
# 类型转换
# =============================
def o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    V = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    F = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    if len(V) == 0 or len(F) == 0:
        raise ValueError(f"空 mesh: V={len(V)}, F={len(F)}")
    return trimesh.Trimesh(vertices=V, faces=F, process=False, maintain_order=True)

def trimesh_to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    out.compute_vertex_normals()
    return out

# =============================
# 参数构造
# =============================
def make_default_args(output_dir: str, **overrides):
    """复用 mesh_refine_slim 的 argparse 默认值 + preset"""
    parser = mrs.build_parser()
    args = parser.parse_args(["--input", "__memory__", "--output_dir", output_dir])
    # 应用 preset（slim 版 main() 里是自动调用的，这里手动调用一次）
    args = mrs.apply_preset_to_args(args)
    # 覆盖外部传入的参数
    for k, v in overrides.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            print(f"⚠️  unknown arg ignored: {k}={v}")
    return args

# =============================
# 内存版 pipeline（去掉文件 IO）
# =============================
def _run_pipeline_core(mesh: trimesh.Trimesh, args, verbose=True):
    """改自 mrs.process_mesh，去掉文件 IO，返回结果 mesh + meta"""
    timings = {}
    total_t0 = time.perf_counter()

    # 控制 slim 内部 log() 是否打印
    mrs.enable_log(bool(verbose))

    sensor_origin = mrs.parse_vec3(args.sensor_origin)
    num_workers = args.num_workers if args.num_workers > 0 else max(1, cpu_count() - 1)

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

    before = mrs.mesh_report(mesh)

    # [1/7] 基础清理
    if verbose: print("[1/7] 基础清理...")
    t0 = time.perf_counter()
    m = mrs.clean_mesh_light(mesh, fix_normals=False)
    timings["clean_1"] = round(time.perf_counter() - t0, 4)

    # [2/7] 早期空洞修补 / 裂缝桥接
    if verbose: print("[2/7] 早期空洞修补...")
    t0 = time.perf_counter()
    m, hole_stats = mrs.run_early_hole_repair(
        m,
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
        verbose=verbose,
    )
    timings["hole_repair"] = round(time.perf_counter() - t0, 4)

    # [3/7] 连通分量过滤
    if verbose: print("[3/7] 连通分量过滤...")
    t0 = time.perf_counter()
    m = mrs.filter_components_fast(
        m,
        args.min_component_faces,
        args.min_component_area,
        args.min_component_max_extent,
        args.keep_top_k_faces,
    )
    timings["filter_components"] = round(time.perf_counter() - t0, 4)

    # [4/7] 平面识别
    t0 = time.perf_counter()
    planes, shared_cache = mrs.detect_planes(m, detect_cfg, verbose)
    timings["detect_planes"] = round(time.perf_counter() - t0, 4)

    # [5/7] 平面精修
    t0 = time.perf_counter()
    m, reports1 = mrs.refine_planes_parallel(
        m, planes, refine_cfg,
        shared_cache=shared_cache,
        verbose=verbose, tag="[5/7]",
        num_workers=num_workers,
    )
    timings["refine_planes"] = round(time.perf_counter() - t0, 4)

    # [6/7] 去游离碎片
    if verbose: print("[6/7] 去游离噪点...")
    t0 = time.perf_counter()
    m = mrs.remove_floating_noise_fast(
        m,
        min_faces=args.noise_min_faces,
        min_area=args.noise_min_area,
        min_extent=args.noise_min_extent,
        keep_top_k=args.noise_keep_top_k,
    )
    timings["remove_noise"] = round(time.perf_counter() - t0, 4)

    # [7/7] 最终清理
    if verbose: print("[7/7] 最终清理...")
    t0 = time.perf_counter()
    m = mrs.clean_mesh_light(m, fix_normals=not args.skip_final_fix_normals)
    timings["clean_final"] = round(time.perf_counter() - t0, 4)

    after = mrs.mesh_report(m)
    timings["total"] = round(time.perf_counter() - total_t0, 4)

    return m, {
        "before": before,
        "after": after,
        "timings": timings,
        "planes_count": len(planes),
        "hole_stats": hole_stats,
    }

# =============================
# 主入口
# =============================
def refine_mesh_in_memory(
    o3d_mesh: o3d.geometry.TriangleMesh,
    output_dir: str = "./outputs_online",
    prefix: str = "fusion_refined",
    save_files: bool = True,
    save_raw: bool = True,
    verbose: bool = True,
    **refine_overrides,
):
    """
    主入口：内存级 mesh 优化（slim 版）

    Args:
        o3d_mesh: 来自 mapper.volume.extract_triangle_mesh() 的 mesh
        output_dir: 优化结果输出目录
        prefix: 文件名前缀
        save_files: 是否保存优化结果
        save_raw: 是否保存优化前的对比版
        refine_overrides: 覆盖默认参数

    Returns:
        (refined_o3d_mesh, meta_dict)
    """
    t_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 60)
        print("🔧 ONLINE MESH REFINE (slim) START")
        print(f"   Input: {len(o3d_mesh.vertices)} verts / {len(o3d_mesh.triangles)} faces")
        print("=" * 60)

    if len(o3d_mesh.vertices) < 50:
        print(f"⚠️  Mesh 太小，跳过优化")
        return o3d_mesh, {"skipped": True, "reason": "too_few_vertices"}

    tm = o3d_to_trimesh(o3d_mesh)
    args = make_default_args(output_dir, **refine_overrides)
    refined_tm, meta = _run_pipeline_core(tm, args, verbose=verbose)
    refined_o3d = trimesh_to_o3d(refined_tm)

    # 文件保存
    if save_files:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_raw:
            raw_path = os.path.join(output_dir, f"{prefix}_RAW_{ts}.obj")
            o3d.io.write_triangle_mesh(raw_path, o3d_mesh)
            meta["raw_path"] = raw_path

        out_path = os.path.join(output_dir, f"{prefix}_{ts}.obj")
        latest_path = os.path.join(output_dir, f"{prefix}_latest.obj")
        o3d.io.write_triangle_mesh(out_path, refined_o3d, write_vertex_normals=True)
        o3d.io.write_triangle_mesh(latest_path, refined_o3d, write_vertex_normals=True)
        meta["refined_path"] = out_path
        meta["refined_latest_path"] = latest_path

        meta_path = os.path.join(output_dir, f"{prefix}_meta_{ts}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.perf_counter() - t_start
    meta["total_elapsed_sec"] = round(elapsed, 3)

    if verbose:
        b, a = meta["before"], meta["after"]
        print("=" * 60)
        print(f"✅ REFINE DONE in {elapsed:.2f}s")
        print(f"   Vertices:  {b['vertices']:>7} → {a['vertices']:>7}  (Δ={a['vertices']-b['vertices']:+d})")
        print(f"   Faces:     {b['faces']:>7} → {a['faces']:>7}  (Δ={a['faces']-b['faces']:+d})")
        print(f"   Planes detected: {meta['planes_count']}")
        if save_files:
            print(f"   Saved: {meta.get('refined_latest_path')}")
        print("=" * 60 + "\n")

    return refined_o3d, meta