# config_slim.py
import os

# =========================================
# Environment / Thread control
# =========================================
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
# Early hole fill / gap stitch
# (chain stitch / trimesh fill_holes 已删除)
# =========================================
HOLE_FILL_CONFIG = {
    "disable_early_hole_fill": False,

    "max_hole_edges": 160,
    "max_hole_diameter": 1.5,
    "max_hole_area": 1.2,
    "max_hole_plane_residual": 0.10,
    "max_hole_candidate_loops": 8000,

    "disable_gap_stitch": False,
    "max_bridge_dist": 0.30,
    "bridge_normal_dot_min": 0.15,
    "max_bridge_pairs": 20000,
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
# Floating noise filter
# =========================================
NOISE_FILTER_CONFIG = {
    "noise_min_faces": 18,
    "noise_min_area": 0.00035,
    "noise_min_extent": 0.018,
    "noise_keep_top_k": 10,
}

# =========================================
# Runtime / Output control
# =========================================
RUNTIME_CONFIG = {
    "skip_final_fix_normals": True,
    "num_workers": DEFAULT_NUM_WORKERS,

    # 导出控制
    "export_obj": False,
    "disable_export_ply": False,
    "export_planes_json": False,
    "export_quality_report_json": False,

    # 控制台输出默认开启
    "verbose": True,
}

# =========================================
# Logging
# =========================================
_LOG_ENABLED = True

def enable_log(enabled=True):
    global _LOG_ENABLED
    _LOG_ENABLED = bool(enabled)

def log(*args, **kwargs):
    if not _LOG_ENABLED:
        return
    print(*args, **kwargs)

# =========================================
# Helpers
# =========================================
def get_preset_config(preset_name=None):
    name = preset_name or DEFAULT_PRESET
    return dict(PRESET_CONFIGS.get(name, PRESET_CONFIGS[DEFAULT_PRESET]))

def apply_preset_to_args(args):
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