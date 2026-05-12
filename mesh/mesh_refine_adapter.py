"""
内存级 mesh refine 适配器
- 直接接收 Open3D mesh
- 复用 mesh_refine_pipeline.py 的所有处理函数
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

import mesh_refine_pipelineGPU as mrp

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

def make_default_args(output_dir: str, **overrides):
    """复用 mesh_refine_pipeline 的 argparse 默认值"""
    parser = mrp.build_parser()
    args = parser.parse_args(["--input", "__memory__", "--output_dir", output_dir])
    for k, v in overrides.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

def _run_pipeline_core(mesh: trimesh.Trimesh, args, verbose=True):
    """改自 mrp.process_mesh，去掉文件 IO"""
    timings = {}
    total_t0 = time.perf_counter()

    sensor_origin = mrp.parse_vec3(args.sensor_origin)
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

    before = mrp.mesh_report(mesh)

    # [1] 基础清理
    if verbose: print("[1/8] 基础清理...")
    t0 = time.perf_counter()
    m = mrp.clean_mesh_light(mesh, fix_normals=False)
    timings["clean_1"] = round(time.perf_counter() - t0, 4)

    # [2] 早期空洞修补
    t0 = time.perf_counter()
    m, hole_stats = mrp.run_early_hole_repair(
        m,
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
        verbose=verbose,
    )
    timings["hole_repair"] = round(time.perf_counter() - t0, 4)

    # [3] 连通分量过滤
    if verbose: print("[3/8] 连通分量过滤...")
    t0 = time.perf_counter()
    m = mrp.filter_components_fast(
        m, args.min_component_faces, args.min_component_area,
        args.min_component_max_extent, args.keep_top_k_faces,
    )
    timings["filter_components"] = round(time.perf_counter() - t0, 4)

    # [4] 平面识别
    t0 = time.perf_counter()
    planes, shared_cache, detect_breakdown = mrp.detect_planes(
        m, detect_cfg, verbose, return_timings=True
    )
    timings["detect_planes"] = round(time.perf_counter() - t0, 4)
    timings["detect_planes_make_cache"] = detect_breakdown["make_cache"]
    timings["detect_planes_initial_patch_labels"] = detect_breakdown["initial_patch_labels"]
    timings["detect_planes_patch_filter"] = detect_breakdown["patch_filter"]
    timings["detect_planes_split_planes_total"] = detect_breakdown["split_planes_total"]
    timings["detect_planes_dedup_planes"] = detect_breakdown["dedup_planes"]

    # [5] 平面精修
    t0 = time.perf_counter()
    m, reports1 = mrp.refine_planes_parallel(
        m, planes, refine_cfg, shared_cache=shared_cache,
        only_stage=None, pass_mul=None, verbose=False,
        tag="[5/8]", num_workers=num_workers,
    )
    timings["refine_pass_1"] = round(time.perf_counter() - t0, 4)

    # [5.5] 远场二次精修
    timings["refine_pass_2"] = 0.0
    if args.enable_far_second_pass:
        selected = mrp.select_far_second_pass_planes(
            planes, reports1,
            args.far_second_pass_min_faces,
            args.far_second_pass_min_sigma,
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
            m, _ = mrp.refine_planes_parallel(
                m, selected, refine_cfg, shared_cache=shared_cache,
                only_stage=None, pass_mul=pass_mul, verbose=False,
                tag="[5.5/8]", num_workers=num_workers,
            )
            timings["refine_pass_2"] = round(time.perf_counter() - t0, 4)

    # [6] 全局平滑
    if args.global_smooth_iter > 0:
        if verbose: print("[6/8] 全局平滑...")
        t0 = time.perf_counter()
        m = mrp.global_smooth(m, args.global_smooth_iter, args.global_smooth_method)
        timings["global_smooth"] = round(time.perf_counter() - t0, 4)
    else:
        timings["global_smooth"] = 0.0

    # [7] 去游离碎片
    if verbose: print("[7/8] 去游离噪点...")
    t0 = time.perf_counter()
    m = mrp.remove_floating_noise_fast(
        m, min_faces=args.noise_min_faces, min_area=args.noise_min_area,
        min_extent=args.noise_min_extent, keep_top_k=args.noise_keep_top_k,
    )
    timings["remove_noise"] = round(time.perf_counter() - t0, 4)

    # [8] 最终清理
    if verbose: print("[8/8] 最终清理...")
    t0 = time.perf_counter()
    m = mrp.clean_mesh_light(m, fix_normals=not args.skip_final_fix_normals)
    timings["clean_final"] = round(time.perf_counter() - t0, 4)

    after = mrp.mesh_report(m)

    timings["total"] = round(time.perf_counter() - total_t0, 4)

    return m, {
        "before": before,
        "after": after,
        "timings": timings,
        "planes_count": len(planes),
        "hole_stats": hole_stats,
    }

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
    主入口：内存级 mesh 优化

    Args:
        o3d_mesh: 来自 mapper.volume.extract_triangle_mesh() 的 mesh
        output_dir: 优化结果输出目录
        prefix: 文件名前缀
        save_files: 是否保存优化结果
        save_raw: 是否保存优化前的对比版
        refine_overrides: 覆盖默认参数（如 global_smooth_iter=2）

    Returns:
        (refined_o3d_mesh, meta_dict)
    """
    t_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 60)
        print("🔧 ONLINE MESH REFINE START")
        print(f"   Input: {len(o3d_mesh.vertices)} verts / {len(o3d_mesh.triangles)} faces")
        print("=" * 60)

    if len(o3d_mesh.vertices) < 50:
        print(f"⚠️  Mesh 太小，跳过优化")
        return o3d_mesh, {"skipped": True, "reason": "too_few_vertices"}

    tm = o3d_to_trimesh(o3d_mesh)
    args = make_default_args(output_dir, verbose=verbose, **refine_overrides)
    refined_tm, meta = _run_pipeline_core(tm, args, verbose=verbose)
    refined_o3d = trimesh_to_o3d(refined_tm)

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