# mesh_refine_api.py
"""
Mesh refine 高层 API 接口

用法:
    # 1) 简单模式 - 只要 mesh
    from mesh_refine_api import MeshRefinePipeline
    pipe = MeshRefinePipeline(preset="balanced")
    refined = pipe.process("input.ply")              # → trimesh.Trimesh
    refined.export("out.ply")

    # 2) 详细模式 - 要工作流数据
    result = pipe.process("input.ply", return_details=True)
    print(result.num_planes, result.timings, result.hole_stats)
    for p in result.planes_summary():
        print(p)

    # 3) 内存输入（自动识别 trimesh / open3d / (V,F)）
    result = pipe.process(o3d_mesh, return_details=True)

    # 4) 一次性入口（不需要构造类）
    from mesh_refine_api import refine_mesh
    mesh = refine_mesh("input.ply")
    result = refine_mesh("input.ply", return_details=True)

    # 5) 处理 + 保存到目录
    pipe.process_and_save("input.ply", "out_dir/", export_json=True)
"""
import os
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from multiprocessing import cpu_count

import numpy as np
import trimesh

import mesh_refine_slim as mrs
from config_slim import apply_preset_to_args, enable_log, log

# =============================================================================
#                              PipelineResult
# =============================================================================
@dataclass
class PipelineResult:
    """
    Pipeline 详细输出。

    字段:
        mesh                 : 最终 trimesh.Trimesh
        before / after       : 前后状态字典 (vertices/faces/bbox_extent)
        timings              : 各阶段耗时 (秒)
        planes               : 检测到的平面 (含 face_ids/centroid/normal/sigma/...)
        refine_reports       : 每个平面的精修报告
        hole_stats           : 空洞修补统计
        intermediate_meshes  : 各阶段中间 mesh (仅 save_intermediate=True)
        config               : 实际使用的配置摘要

    便利属性:
        num_planes / total_time / vertices / faces

    便利方法:
        summary()         : 简短摘要 dict
        planes_summary()  : 平面信息列表 (轻量, 不含 face_ids)
        to_json_dict()    : 全部可序列化的 dict (不含 mesh)
        save_mesh(path)   : 保存 mesh
        save_all(dir)     : 保存 mesh + JSON + 中间 mesh
    """
    mesh: Any = None
    before: Dict = field(default_factory=dict)
    after: Dict = field(default_factory=dict)
    timings: Dict = field(default_factory=dict)
    planes: List = field(default_factory=list)
    refine_reports: List = field(default_factory=list)
    hole_stats: Dict = field(default_factory=dict)
    intermediate_meshes: Optional[Dict[str, Any]] = None
    config: Optional[Dict] = None

    @property
    def num_planes(self) -> int:
        return len(self.planes)

    @property
    def total_time(self) -> float:
        return float(self.timings.get("total", 0.0))

    @property
    def vertices(self) -> int:
        return int(self.after.get("vertices", 0))

    @property
    def faces(self) -> int:
        return int(self.after.get("faces", 0))

    def planes_summary(self) -> List[Dict]:
        """轻量平面信息（去掉 face_ids 数组）"""
        return [{
            "plane_id": i,
            "faces": int(len(p["face_ids"])),
            "area": float(p["area"]),
            "sigma": float(p["sigma"]),
            "planarity_ratio": float(p.get("planarity_ratio", 0.0)),
            "mean_normal_angle": float(p.get("mean_normal_angle", 0.0)),
            "centroid": [float(x) for x in p["centroid"]],
            "normal": [float(x) for x in p["normal"]],
        } for i, p in enumerate(self.planes)]

    def summary(self) -> Dict:
        """简短摘要"""
        return {
            "before": self.before,
            "after": self.after,
            "num_planes": self.num_planes,
            "total_time": self.total_time,
            "timings": self.timings,
            "hole_stats": self.hole_stats,
        }

    def to_json_dict(self) -> Dict:
        """完整可序列化 dict（不含 mesh / numpy 数组）"""
        return {
            "before": self.before,
            "after": self.after,
            "timings": self.timings,
            "hole_stats": self.hole_stats,
            "config": self.config,
            "num_planes": self.num_planes,
            "planes": self.planes_summary(),
            "refine_reports": self.refine_reports,
        }

    def save_mesh(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if ext == ".ply":
            try:
                self.mesh.export(path, encoding="binary")
            except TypeError:
                self.mesh.export(path)
        else:
            self.mesh.export(path)

    def save_all(self, out_dir: str, name_prefix: str = "result"):
        os.makedirs(out_dir, exist_ok=True)
        ply_path = os.path.join(out_dir, f"{name_prefix}.ply")
        json_path = os.path.join(out_dir, f"{name_prefix}.json")
        self.save_mesh(ply_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False, default=str)
        if self.intermediate_meshes:
            for stage, m in self.intermediate_meshes.items():
                p = os.path.join(out_dir, f"{name_prefix}_stage_{stage}.ply")
                try:
                    m.export(p, encoding="binary")
                except TypeError:
                    m.export(p)
        return {"mesh_path": ply_path, "json_path": json_path}

# =============================================================================
#                              输入归一化
# =============================================================================
def to_trimesh(mesh_input) -> trimesh.Trimesh:
    """统一把任意输入转成 trimesh.Trimesh

    支持：
        - 文件路径 str
        - trimesh.Trimesh
        - open3d.geometry.TriangleMesh
        - (vertices, faces) 元组
    """
    if isinstance(mesh_input, trimesh.Trimesh):
        return mesh_input
    if isinstance(mesh_input, str):
        return mrs.load_mesh(mesh_input)
    # open3d (lazy import)
    try:
        import open3d as o3d
        if isinstance(mesh_input, o3d.geometry.TriangleMesh):
            V = np.asarray(mesh_input.vertices, dtype=np.float64)
            F = np.asarray(mesh_input.triangles, dtype=np.int32)
            if len(V) == 0 or len(F) == 0:
                raise ValueError(f"空 open3d mesh: V={len(V)}, F={len(F)}")
            return trimesh.Trimesh(vertices=V, faces=F, process=False, maintain_order=True)
    except ImportError:
        pass
    # (V, F) tuple
    if isinstance(mesh_input, (tuple, list)) and len(mesh_input) == 2:
        V = np.asarray(mesh_input[0], dtype=np.float64)
        F = np.asarray(mesh_input[1], dtype=np.int32)
        return trimesh.Trimesh(vertices=V, faces=F, process=False, maintain_order=True)
    raise TypeError(f"不支持的 mesh 输入类型: {type(mesh_input)}")

# =============================================================================
#                              参数构造
# =============================================================================
def build_args_from_overrides(preset: Optional[str] = None, **overrides) -> argparse.Namespace:
    """从 preset + 覆盖参数构造 argparse.Namespace（复用 mrs.build_parser 默认值）"""
    parser = mrs.build_parser()
    cli = []
    if preset is not None:
        cli += ["--preset", preset]
    args = parser.parse_args(cli)
    args = apply_preset_to_args(args)
    unknown = []
    for k, v in overrides.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            unknown.append(k)
    if unknown:
        log(f"⚠️  unknown args ignored: {unknown}")
    return args

# =============================================================================
#                              核心 Pipeline (内存版)
# =============================================================================
def run_pipeline_core(mesh: trimesh.Trimesh,
                      args: argparse.Namespace,
                      *,
                      save_intermediate: bool = False) -> PipelineResult:
    """
    核心 pipeline 逻辑（纯内存，不读不写文件）。

    与 mrs.process_mesh 的区别：
    - 输入是已加载的 trimesh，不读文件
    - 不调 save_outputs，由调用方决定是否落盘
    - 返回结构化 PipelineResult
    """
    total_t0 = time.perf_counter()

    sensor_origin = mrs.parse_vec3(args.sensor_origin)
    num_workers = args.num_workers if (args.num_workers and args.num_workers > 0) \
                  else max(1, cpu_count() - 1)

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
    intermediate = {} if save_intermediate else None
    before = mrs.mesh_report(mesh)

    log("========== PIPELINE START (API) ==========")
    log(f"num_workers: {num_workers}")
    log(f"input mesh: V={before['vertices']} F={before['faces']}")

    # [1/7]
    log("[1/7] 基础清理...")
    t0 = time.perf_counter()
    mesh = mrs.clean_mesh_light(mesh, fix_normals=False)
    timings["clean_mesh_1"] = round(time.perf_counter() - t0, 4)
    if intermediate is not None:
        intermediate["after_clean_1"] = mesh.copy()

    # [2/7]
    log("[2/7] 早期空洞修补 / 裂缝桥接...")
    t0 = time.perf_counter()
    mesh, hole_stats = mrs.run_early_hole_repair(
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
    if intermediate is not None:
        intermediate["after_hole_fill"] = mesh.copy()

    # [3/7]
    log("[3/7] 连通分量过滤...")
    t0 = time.perf_counter()
    mesh = mrs.filter_components_fast(
        mesh,
        args.min_component_faces, args.min_component_area,
        args.min_component_max_extent, args.keep_top_k_faces,
    )
    timings["filter_components"] = round(time.perf_counter() - t0, 4)
    if intermediate is not None:
        intermediate["after_filter"] = mesh.copy()

    # [4/7]
    t0 = time.perf_counter()
    planes, shared_cache = mrs.detect_planes(mesh, detect_cfg, args.verbose)
    timings["detect_planes"] = round(time.perf_counter() - t0, 4)

    # [5/7]
    t0 = time.perf_counter()
    mesh, reports1 = mrs.refine_planes_parallel(
        mesh, planes, refine_cfg, shared_cache=shared_cache,
        verbose=args.verbose, tag="[5/7]", num_workers=num_workers,
    )
    timings["refine_planes"] = round(time.perf_counter() - t0, 4)
    if intermediate is not None:
        intermediate["after_refine"] = mesh.copy()

    # [6/7]
    log("[6/7] 去除游离噪点碎块...")
    t0 = time.perf_counter()
    mesh = mrs.remove_floating_noise_fast(
        mesh,
        min_faces=args.noise_min_faces, min_area=args.noise_min_area,
        min_extent=args.noise_min_extent, keep_top_k=args.noise_keep_top_k,
    )
    timings["remove_floating_noise"] = round(time.perf_counter() - t0, 4)

    # [7/7]
    log("[7/7] 最终轻量清理...")
    t0 = time.perf_counter()
    mesh = mrs.clean_mesh_light(mesh, fix_normals=not args.skip_final_fix_normals)
    timings["clean_mesh_final"] = round(time.perf_counter() - t0, 4)

    after = mrs.mesh_report(mesh)
    timings["total"] = round(time.perf_counter() - total_t0, 4)

    log("")
    log("========== PIPELINE DONE ==========")
    log(f"  V: {before['vertices']} → {after['vertices']}")
    log(f"  F: {before['faces']} → {after['faces']}")
    log(f"  planes detected: {len(planes)}")
    log("Timings (sec):")
    for k, v in timings.items():
        log(f"  {k}: {v:.3f}")
    log("===================================")

    config_summary = {
        "preset": getattr(args, "preset", None),
        "sensor_origin": args.sensor_origin,
        "use_distance_strategy": args.use_distance_strategy,
        "num_workers": int(num_workers),
        "expand_rings": args.expand_rings,
        "boundary_protect": not args.no_boundary_protect,
    }

    return PipelineResult(
        mesh=mesh,
        before=before, after=after,
        timings=timings,
        planes=planes,
        refine_reports=reports1,
        hole_stats=hole_stats,
        intermediate_meshes=intermediate,
        config=config_summary,
    )

# =============================================================================
#                          MeshRefinePipeline (主类)
# =============================================================================
class MeshRefinePipeline:
    """
    Mesh refine pipeline 高层 API。

    Examples:
        # 最简使用
        pipe = MeshRefinePipeline()
        refined = pipe.process("input.ply")

        # 详细模式
        result = pipe.process("input.ply", return_details=True)
        print(f"detected {result.num_planes} planes, took {result.total_time:.2f}s")

        # 内存输入
        result = pipe.process(o3d_mesh, return_details=True)

        # 临时覆盖参数（不污染 self）
        refined = pipe.process("x.ply", verbose=False, sensor_origin="0,0,0")

        # 永久更新配置
        pipe.update_config(num_workers=4)

        # 处理 + 保存
        pipe.process_and_save("x.ply", "out_dir/", export_json=True)
    """

    def __init__(self, preset: str = "balanced", verbose: bool = True, **overrides):
        """
        Args:
            preset: 'fast' / 'balanced' / 'detail' (取决于 PRESET_CONFIGS)
            verbose: 默认是否打印日志
            **overrides: 任何 mrs.build_parser() 中的参数
        """
        self.preset = preset
        self.base_overrides = dict(overrides)
        self.base_overrides.setdefault("verbose", verbose)
        self._args = build_args_from_overrides(preset, **self.base_overrides)

    def update_config(self, **overrides):
        """永久更新配置"""
        for k, v in overrides.items():
            self.base_overrides[k] = v
            if hasattr(self._args, k):
                setattr(self._args, k, v)
        return self

    def get_config(self) -> Dict:
        """获取当前配置"""
        return vars(self._args).copy()

    def process(self,
                mesh_input,
                *,
                return_details: bool = False,
                save_intermediate: bool = False,
                verbose: Optional[bool] = None,
                **overrides):
        """
        处理 mesh。

        Args:
            mesh_input: 文件路径 / trimesh.Trimesh / open3d.TriangleMesh / (V, F)
            return_details: True 返回 PipelineResult，False 仅返回 trimesh.Trimesh
            save_intermediate: 是否保存各阶段中间 mesh
            verbose: 临时覆盖日志开关
            **overrides: 临时覆盖任何参数（不污染 self._args）

        Returns:
            return_details=False : trimesh.Trimesh
            return_details=True  : PipelineResult
        """
        # 临时构造 args（如果有覆盖）
        if overrides or verbose is not None:
            tmp = dict(self.base_overrides)
            tmp.update(overrides)
            if verbose is not None:
                tmp["verbose"] = verbose
            args = build_args_from_overrides(self.preset, **tmp)
        else:
            args = self._args

        # 设置全局日志开关
        enable_log(bool(args.verbose))

        # 输入归一化
        mesh = to_trimesh(mesh_input)

        # 太小直接跳过
        if len(mesh.vertices) < 50:
            log(f"⚠️  mesh 太小（V={len(mesh.vertices)}），跳过优化")
            empty = PipelineResult(
                mesh=mesh,
                before=mrs.mesh_report(mesh),
                after=mrs.mesh_report(mesh),
                timings={"total": 0.0},
                config={"skipped": True, "reason": "too_few_vertices"},
            )
            return empty if return_details else mesh

        # 跑核心
        result = run_pipeline_core(mesh, args, save_intermediate=save_intermediate)
        return result if return_details else result.mesh

    def process_and_save(self,
                         mesh_input,
                         output_dir: str,
                         *,
                         name_prefix: str = "refined",
                         export_json: bool = False,
                         return_details: bool = False,
                         **kwargs):
        """处理 + 落盘到 output_dir"""
        result = self.process(mesh_input, return_details=True, **kwargs)
        os.makedirs(output_dir, exist_ok=True)
        result.save_mesh(os.path.join(output_dir, f"{name_prefix}.ply"))
        if export_json:
            with open(os.path.join(output_dir, f"{name_prefix}.json"), "w", encoding="utf-8") as f:
                json.dump(result.to_json_dict(), f, indent=2, ensure_ascii=False, default=str)
        if result.intermediate_meshes:
            for stage, m in result.intermediate_meshes.items():
                p = os.path.join(output_dir, f"{name_prefix}_stage_{stage}.ply")
                try:
                    m.export(p, encoding="binary")
                except TypeError:
                    m.export(p)
        return result if return_details else result.mesh

# =============================================================================
#                              便利函数（懒人入口）
# =============================================================================
def refine_mesh(mesh_input,
                preset: str = "balanced",
                return_details: bool = False,
                **kwargs):
    """
    一行式入口（每次新建 pipeline，不缓存配置）：
        refined = refine_mesh("input.ply")
        result  = refine_mesh(o3d_mesh, return_details=True)
    """
    return MeshRefinePipeline(preset=preset).process(
        mesh_input, return_details=return_details, **kwargs
    )