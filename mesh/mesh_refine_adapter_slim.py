"""
内存级 mesh refine 适配器（slim 版本）
- 直接接收 Open3D mesh
- 内部委托给 MeshRefinePipeline
- 保持原有函数签名 100% 兼容
"""
import os
import time
import json
import numpy as np
import trimesh
import open3d as o3d
from datetime import datetime

from mesh_refine_api import MeshRefinePipeline

# =============================
# 类型转换（保留给外部调用者用）
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
# Pipeline 实例缓存（避免每次重建 args）
# =============================
_PIPELINE_CACHE = {}

def _get_pipeline(preset: str = "balanced", verbose: bool = True) -> MeshRefinePipeline:
    """按 preset 缓存 pipeline 实例（args 复用，节省 ~10ms / 次）"""
    key = preset
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = MeshRefinePipeline(preset=preset, verbose=verbose)
    return _PIPELINE_CACHE[key]

# =============================
# 主入口（签名 100% 兼容旧版）
# =============================
def refine_mesh_in_memory(
    o3d_mesh: o3d.geometry.TriangleMesh,
    output_dir: str = "./outputs_online",
    prefix: str = "fusion_refined",
    save_files: bool = True,
    save_raw: bool = True,
    verbose: bool = True,
    preset: str = "balanced",          # 新增（旧调用不传也兼容）
    **refine_overrides,
):
    """
    主入口：内存级 mesh 优化（slim API 版）

    Returns:
        (refined_o3d_mesh, meta_dict)

    meta_dict 字段（保持与旧版兼容）：
        before / after / timings / planes_count / hole_stats
        raw_path (可选) / refined_path / refined_latest_path / total_elapsed_sec
    """
    t_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 60)
        print("🔧 ONLINE MESH REFINE (slim API) START")
        print(f"   Input: {len(o3d_mesh.vertices)} verts / {len(o3d_mesh.triangles)} faces")
        print("=" * 60)

    # 太小跳过
    if len(o3d_mesh.vertices) < 50:
        print("⚠️  Mesh 太小，跳过优化")
        return o3d_mesh, {"skipped": True, "reason": "too_few_vertices"}

    # 调用新 API（return_details=True 拿全部数据）
    pipe = _get_pipeline(preset=preset, verbose=verbose)
    result = pipe.process(
        o3d_mesh,                    # 自动识别 o3d → trimesh
        return_details=True,
        verbose=verbose,
        **refine_overrides,
    )

    # trimesh → open3d
    refined_o3d = trimesh_to_o3d(result.mesh)

    # 构造旧版 meta 格式（保证向后兼容）
    meta = {
        "before": result.before,
        "after": result.after,
        "timings": result.timings,
        "planes_count": result.num_planes,
        "hole_stats": result.hole_stats,
    }

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

# =============================
# 新增：直接拿 PipelineResult 的接口（给愿意用新 API 的调用方）
# =============================
def refine_mesh_with_details(
    o3d_mesh: o3d.geometry.TriangleMesh,
    preset: str = "balanced",
    save_intermediate: bool = False,
    verbose: bool = True,
    **refine_overrides,
):
    """
    新接口：直接返回 PipelineResult（含 trimesh mesh 和全部工作流数据）

    注意：返回的 result.mesh 是 trimesh.Trimesh，不是 open3d。
    如果需要 open3d，自己用 trimesh_to_o3d() 转换。

    Examples:
        result = refine_mesh_with_details(o3d_mesh, save_intermediate=True)
        print(result.num_planes, result.timings)
        for p in result.planes_summary():
            print(p)
        result.save_all("./out_dir", name_prefix="online")
    """
    pipe = _get_pipeline(preset=preset, verbose=verbose)
    return pipe.process(
        o3d_mesh,
        return_details=True,
        save_intermediate=save_intermediate,
        verbose=verbose,
        **refine_overrides,
    )