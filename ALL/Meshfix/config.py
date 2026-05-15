# config.py
import os
import sys

# =========================================
# Environment / Thread control
# =========================================
# 放在主程序 import numpy/scipy 之前调用更稳妥
ENV_THREADS = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
}

def apply_env_threads():
    for k, v in ENV_THREADS.items():
        os.environ.setdefault(k, str(v))

# =========================================
# Default run config
# =========================================
DEFAULT_INPUT = r".\input\imu_fusion_model_final_20260413_193242.ply"
DEFAULT_OUTPUT_DIR = r".\outputs_default"
DEFAULT_SENSOR_ORIGIN = "0,0,0"

DEFAULT_NUM_WORKERS = 8
DEFAULT_PRESET = "balanced"

EXPORT_OBJ = False
EXPORT_PLY = True
PROFILE_FAST_MODE = False
SKIP_FINAL_FIX_NORMALS = True

# =========================================
# Presets
# =========================================
PRESET_CONFIGS = {
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

# =========================================
# Component filtering
# =========================================
COMPONENT_FILTER_CONFIG = {
    "min_component_faces": 80,
    "min_component_area": 0.0015,
    "min_component_max_extent": 0.04,
    "keep_top_k_faces": 20,
}

# =========================================
# Early hole fill / gap stitch / chain stitch
# =========================================
HOLE_FILL_CONFIG = {
    "disable_early_hole_fill": False,
    "disable_trimesh_fill_holes": True,

    "max_hole_edges": 160,
    "max_hole_diameter": 1.5,
    "max_hole_area": 1.2,
    "max_hole_plane_residual": 0.10,
    "max_hole_candidate_loops": 8000,

    "disable_gap_stitch": False,
    "max_bridge_dist": 0.30,
    "bridge_normal_dot_min": 0.15,
    "max_bridge_pairs": 20000,

    "disable_chain_stitch": True,
    "max_chain_endpoint_dist": 0.55,
    "max_chain_avg_gap": 0.40,
    "max_chain_plane_residual": 0.12,
    "chain_tangent_dot_min": 0.15,
    "chain_normal_dot_min": 0.20,
    "max_chain_pairs": 256,
    "max_chain_neighbor_candidates": 64,
}

# =========================================
# Plane detection
# =========================================
PLANE_DETECT_CONFIG = {
    "patch_normal_angle_deg": 24.0,
    "min_patch_faces": 25,
    "min_plane_faces": 80,
    "min_plane_area": 0.0008,
    "base_face_plane_dist": 0.014,
    "plane_normal_angle_deg": 18.0,
    "sigma_dist_mult": 3.6,
    "max_split_depth": 3,
}

# =========================================
# Plane refine
# =========================================
PLANE_REFINE_CONFIG = {
    "vertex_inlier_dist": 0.008,
    "vertex_outlier_dist": 0.085,
    "base_snap_strength": 1.0,
    "base_residual_shrink": 0.006,
    "base_normal_smooth_iterations": 22,
    "noise_adapt_gain": 1.8,
    "no_boundary_protect": False,
    "boundary_scale": 0.10,
    "expand_rings": 1,
}

# =========================================
# Distance strategy
# =========================================
DISTANCE_STRATEGY_CONFIG = {
    "sensor_origin": DEFAULT_SENSOR_ORIGIN,
    "distance_gain": 0.45,
    "use_distance_strategy": True,
    "near_ratio": 0.25,
    "far_ratio": 0.50,
}

# =========================================
# Far second pass
# =========================================
FAR_SECOND_PASS_CONFIG = {
    "enable_far_second_pass": False,
    "far_second_pass_outlier_mul": 1.60,
    "far_second_pass_shrink_mul": 0.45,
    "far_second_pass_smooth_mul": 1.80,
    "far_second_pass_snap_mul": 1.20,
    "far_second_pass_noise_mul": 1.30,
    "far_second_pass_expand_rings": 1,
    "far_second_pass_min_sigma": 0.003,
    "far_second_pass_min_faces": 60,
}

# =========================================
# Global smooth
# =========================================
GLOBAL_SMOOTH_CONFIG = {
    "global_smooth_iter": 0,
    "global_smooth_method": "taubin",  # taubin / laplacian / simple
}

# =========================================
# Floating noise filter
# =========================================
NOISE_FILTER_CONFIG = {
    "noise_min_faces": 18,
    "noise_min_area": 0.00035,
    "noise_min_extent": 0.018,
    "noise_keep_top_k": 10,
}

# =========================================
# Runtime / CLI-like defaults
# =========================================
RUNTIME_CONFIG = {
    "skip_final_fix_normals": True,
    "num_workers": DEFAULT_NUM_WORKERS,
    "profile_fast_mode": False,

    # 导出控制
    "export_obj": False,
    "disable_export_ply": False,

    # 默认关闭 JSON 导出
    "export_planes_json": False,
    "export_quality_report_json": False,

    # 控制台信息默认保留
    "verbose": True,
}

# =========================================
# Logging
# =========================================
_LOG_ENABLED = False
_DEBUG_ENABLED = False
_LOG_PREFIX = True

def enable_log(enabled=True):
    global _LOG_ENABLED
    _LOG_ENABLED = bool(enabled)

def enable_debug(enabled=True):
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(enabled)
    if enabled:
        enable_log(True)

def set_log_prefix(enabled=True):
    global _LOG_PREFIX
    _LOG_PREFIX = bool(enabled)

def is_log_enabled():
    return _LOG_ENABLED

def is_debug_enabled():
    return _DEBUG_ENABLED

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            text = " ".join(str(x) for x in args)
            sys.stdout.write(text + "\n")
        except Exception:
            pass

def log(*args, **kwargs):
    if not _LOG_ENABLED:
        return
    if _LOG_PREFIX:
        _safe_print("[INFO]", *args, **kwargs)
    else:
        _safe_print(*args, **kwargs)

def debug(*args, **kwargs):
    if not (_LOG_ENABLED and _DEBUG_ENABLED):
        return
    if _LOG_PREFIX:
        _safe_print("[DEBUG]", *args, **kwargs)
    else:
        _safe_print(*args, **kwargs)

def warn(*args, **kwargs):
    if not _LOG_ENABLED:
        return
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    if _LOG_PREFIX:
        _safe_print("[WARN]", *args, **kwargs)
    else:
        _safe_print(*args, **kwargs)

def banner(title, char="="):
    if not _LOG_ENABLED:
        return
    line = char * 10
    _safe_print(f"{line} {title} {line}")

def dump_dict(d, prefix=""):
    if not _LOG_ENABLED:
        return
    if d is None:
        log(f"{prefix}None")
        return
    for k, v in d.items():
        log(f"{prefix}{k}: {v}")

# =========================================
# Helpers
# =========================================
def get_preset_config(preset_name=None):
    name = preset_name or DEFAULT_PRESET
    return dict(PRESET_CONFIGS.get(name, PRESET_CONFIGS[DEFAULT_PRESET]))

def apply_preset_to_args(args):
    """
    将 preset 配置写回 argparse args 对象
    """
    preset_name = getattr(args, "preset", DEFAULT_PRESET)
    cfg = get_preset_config(preset_name)

    args.patch_normal_angle_deg = cfg["patch_normal_angle_deg"]
    args.min_patch_faces = cfg["min_patch_faces"]
    args.min_plane_faces = cfg["min_plane_faces"]
    args.min_plane_area = cfg["min_plane_area"]
    args.base_face_plane_dist = cfg["base_face_plane_dist"]
    args.plane_normal_angle_deg = cfg["plane_normal_angle_deg"]
    args.sigma_dist_mult = cfg["sigma_dist_mult"]
    args.max_split_depth = cfg["max_split_depth"]
    return args

def build_default_runtime_dict():
    """
    返回一个可用于主程序的完整默认参数字典
    便于后续你从 argparse 慢慢迁移出来
    """
    cfg = {}
    cfg.update(COMPONENT_FILTER_CONFIG)
    cfg.update(HOLE_FILL_CONFIG)
    cfg.update(PLANE_DETECT_CONFIG)
    cfg.update(PLANE_REFINE_CONFIG)
    cfg.update(DISTANCE_STRATEGY_CONFIG)
    cfg.update(FAR_SECOND_PASS_CONFIG)
    cfg.update(GLOBAL_SMOOTH_CONFIG)
    cfg.update(NOISE_FILTER_CONFIG)
    cfg.update(RUNTIME_CONFIG)

    cfg["input"] = DEFAULT_INPUT
    cfg["output_dir"] = DEFAULT_OUTPUT_DIR
    cfg["preset"] = DEFAULT_PRESET
    return cfg