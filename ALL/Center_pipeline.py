from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


BASE_DIR = Path(__file__).resolve().parent
MESHFIX_DIR = BASE_DIR / "Meshfix"
MATERIAL_DIR = BASE_DIR / "Material"


def resolve_child_dir(parent: Path, name: str) -> Path:
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return parent / name


POINTCLOUD_DIR = resolve_child_dir(BASE_DIR, "pointcloud2mesh")
POINTCLOUD_PACKAGE = POINTCLOUD_DIR.name

for path in (BASE_DIR, MESHFIX_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

PipelineConfig = importlib.import_module(f"{POINTCLOUD_PACKAGE}.config").PipelineConfig
_pipeline_api_mod = importlib.import_module(f"{POINTCLOUD_PACKAGE}.pipeline_api")
RealtimeMappingPipeline = _pipeline_api_mod.RealtimeMappingPipeline
build_imu_world_initializer = _pipeline_api_mod.build_imu_world_initializer
_imu_world_mod = importlib.import_module(f"{POINTCLOUD_PACKAGE}.imu.world_initializer")
from Meshfix.mesh_refine_adapter_slim import o3d_to_trimesh, refine_mesh_in_memory
from Material.material_engine_adapter import MaterialEngineAdapter


POINTCLOUD_MODEL_DIR = POINTCLOUD_DIR / "model"
MESHFIX_OUTPUT_DIR = MESHFIX_DIR / "outputs_realtime"
MATERIAL_OUTPUT_DIR = MATERIAL_DIR / "outputs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _center_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return int(default)


def _center_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return float(default)


# Realm 扫描 mesh 底面默认高度（米），与前端 eyeHeightM 一致
REALM_DEFAULT_MESH_BASE_HEIGHT_M = _center_env_float("CENTER_MESH_BASE_HEIGHT_M", 1.65)


def recommended_view_distance_from_mesh(mesh: o3d.geometry.TriangleMesh | None) -> float | None:
    """
    Rough camera distance hint from reconstructed mesh vertices (meters).
    Uses AABB diagonal and max radial extent from center — not single-frame depth.
    """
    if mesh is None:
        return None
    try:
        v = np.asarray(mesh.vertices)
        if v.size == 0:
            return None
        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        center = (vmin + vmax) * 0.5
        diag = float(np.linalg.norm(vmax - vmin))
        radius = float(np.max(np.linalg.norm(v - center, axis=1)))
        span = max(radius, diag * 0.5)
        if span <= 1e-6:
            return None
        return float(span * 2.2 + 0.35)
    except Exception:
        return None


def _mesh_floor_y_percentile(vertices: np.ndarray, pct: float = 8.0) -> float:
    if vertices.size == 0:
        return 0.0
    ys = np.sort(vertices[:, 1])
    return float(np.percentile(ys, pct))


def compute_metric_mesh_placement(
    vertices: np.ndarray,
    *,
    floor_y_target: float | None = None,
    room_center_xz: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any] | None:
    """
    从 reconstruction_world 顶点（米）估计锚点放置：scale=1，底面对齐 floor_y_target，XZ 居中到 room_center。
    """
    if floor_y_target is None:
        floor_y_target = REALM_DEFAULT_MESH_BASE_HEIGHT_M
    if vertices.size == 0:
        return None
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        v = v.reshape(-1, 3)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    floor_y = _mesh_floor_y_percentile(v, 8.0)
    center_x = float((vmin[0] + vmax[0]) * 0.5)
    center_z = float((vmin[2] + vmax[2]) * 0.5)
    size_x = float(max(vmax[0] - vmin[0], 0.0))
    size_y = float(max(vmax[1] - floor_y, 0.0))
    size_z = float(max(vmax[2] - vmin[2], 0.0))
    return {
        "scale": 1.0,
        "position": [
            float(room_center_xz[0] - center_x),
            float(floor_y_target - floor_y),
            float(room_center_xz[1] - center_z),
        ],
        "extent_m": {"x": size_x, "y": size_y, "z": size_z},
        "floor_y_reconstruction": floor_y,
        "center_reconstruction": [center_x, float((vmin[1] + vmax[1]) * 0.5), center_z],
    }


def rel_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return rel_path(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def media_type_for(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".glb":
        return "model/gltf-binary"
    if suffix == ".obj":
        return "model/obj"
    if suffix == ".ply":
        return "application/octet-stream"
    return "application/octet-stream"


def safe_segment_filename(segment_key: str) -> str:
    return segment_key.replace("/", "_").replace("\\", "_") + ".glb"


class ModelProcessingWorker:
    def __init__(
        self,
        service: "CenterPipelineService",
        min_interval_sec: float = 5.0,
        material_change_threshold: float = 0.10,
        max_mesh_triangles: int = 150_000,
    ):
        self.service = service
        self.min_interval_sec = float(min_interval_sec)
        self.material_change_threshold = float(material_change_threshold)
        self.max_mesh_triangles = int(max_mesh_triangles)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.is_processing = False
        self.last_processed_vertices = 0
        self.last_processed_triangles = 0

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._loop,
            name="ModelProcessingWorker",
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            start = time.perf_counter()
            try:
                if self.service.is_running() and not self.is_processing:
                    self.is_processing = True
                    self.process_once()
            except Exception as exc:
                self.service.set_worker_error(exc)
            finally:
                self.is_processing = False

            elapsed = time.perf_counter() - start
            wait_time = max(0.2, self.min_interval_sec - elapsed)
            self.stop_event.wait(wait_time)

    def _should_process_material(self, vertices: int, triangles: int) -> bool:
        if self.last_processed_vertices <= 0 or self.last_processed_triangles <= 0:
            return True
        v_delta = abs(vertices - self.last_processed_vertices) / max(1, self.last_processed_vertices)
        f_delta = abs(triangles - self.last_processed_triangles) / max(1, self.last_processed_triangles)
        return max(v_delta, f_delta) >= self.material_change_threshold

    def _simplify_if_needed(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        triangles = len(mesh.triangles)
        if triangles <= self.max_mesh_triangles:
            return mesh
        simplified = mesh.simplify_quadric_decimation(self.max_mesh_triangles)
        simplified.compute_vertex_normals()
        return simplified

    def process_once(self) -> None:
        mesh = self.service.get_reconstruction_mesh_snapshot()
        if mesh is None or len(mesh.vertices) < 50 or len(mesh.triangles) < 20:
            return

        mesh.compute_vertex_normals()
        vertices = len(mesh.vertices)
        triangles = len(mesh.triangles)
        white_meta = self.service.export_white_mesh(mesh)

        if not self._should_process_material(vertices, triangles):
            return

        processing_mesh = self._simplify_if_needed(mesh)
        refined_mesh, meshfix_meta = refine_mesh_in_memory(
            processing_mesh,
            output_dir=str(MESHFIX_OUTPUT_DIR),
            prefix=f"refined_v{white_meta['version']:04d}",
            save_files=True,
            save_raw=False,
            verbose=False,
            preset="fast",
            num_workers=2,
            disable_early_hole_fill=True,
            disable_gap_stitch=True,
        )
        meshfix_version_path = MESHFIX_OUTPUT_DIR / f"refined_v{white_meta['version']:04d}.obj"
        meshfix_latest_path = MESHFIX_OUTPUT_DIR / "refined_latest.obj"
        o3d.io.write_triangle_mesh(str(meshfix_version_path), refined_mesh, write_vertex_normals=True)
        shutil.copyfile(meshfix_version_path, meshfix_latest_path)
        meshfix_meta["refined_version_path"] = str(meshfix_version_path)
        meshfix_meta["refined_latest_path"] = str(meshfix_latest_path)

        meshfix_meta_path = MESHFIX_OUTPUT_DIR / f"meta_v{white_meta['version']:04d}.json"
        with open(meshfix_meta_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(meshfix_meta), f, ensure_ascii=False, indent=2)
        meshfix_meta["meta_path"] = str(meshfix_meta_path)

        material_mesh = refined_mesh if refined_mesh is not None else processing_mesh
        material_adapter = MaterialEngineAdapter()
        material_adapter.register_pbr_styles_from_disk(MATERIAL_DIR)
        if not material_adapter.load_from_mesh(o3d_to_trimesh(material_mesh)):
            raise RuntimeError("Material segmentation failed")

        self.service.publish_material_result(material_adapter, meshfix_meta)
        self.last_processed_vertices = vertices
        self.last_processed_triangles = triangles


class CenterPipelineService:
    def __init__(self):
        self.lock = threading.RLock()
        self.pipeline: Any | None = None
        self.worker: ModelProcessingWorker | None = None
        self.material_engine: MaterialEngineAdapter | None = None

        self.white_version = 0
        self.material_version = 0
        self.latest_white: dict[str, Any] | None = None
        self.latest_material_scene: dict[str, Any] | None = None
        self.latest_material_targets: dict[str, Any] | None = None
        self.white_files: dict[int, Path] = {}
        self.material_versions: dict[int, dict[str, Any]] = {}
        self.worker_error: str | None = None
        self.last_stop_result: dict[str, Any] | None = None
        self._semantic_horizontal_up: list[float] | None = None
        self._event_queues: list[queue.Queue] = []

        self._center_recovery_state: str = "normal"
        self._consecutive_lost_frames: int = 0
        self._consecutive_success_frames: int = 0
        self._recovery_hint: str | None = None
        self._clf_pause_threshold: int = _center_env_int("CENTER_PAUSE_CLF_THRESHOLD", 30)
        self._clf_fail_threshold: int = _center_env_int("CENTER_FAIL_AFTER_RESUME_CLF", 200)
        self._recover_success_frames_threshold: int = _center_env_int("CENTER_RECOVER_SUCCESS_FRAMES", 5)
        self._recommended_view_distance_cache: tuple[float | None, float | None] = (None, None)

        # HTTP /api/center/latest-mesh/：内存网格版本（几何变化时递增）
        self._api_mesh_lock = threading.Lock()
        self._api_mesh_version = 0
        self._api_mesh_signature: str | None = None

        # 前几帧深度 + mesh 校准：真实尺寸（米）与位面锚点
        self._placement_lock = threading.Lock()
        self._placement_depth_samples: list[dict[str, Any]] = []
        self._placement_mesh_samples: list[dict[str, Any]] = []
        self._placement_locked: dict[str, Any] | None = None
        self._placement_reference_camera: list[list[float]] | None = None
        self._placement_floor_y_target = REALM_DEFAULT_MESH_BASE_HEIGHT_M

        # AR/VR 漫游：仅串口 IMU，不启深度相机 / TSDF
        self._imu_only_mode = False
        self._imu_only_stop = threading.Event()
        self._imu_only_thread: threading.Thread | None = None
        self._imu_initializer: Any | None = None
        self._imu_recon_world: np.ndarray | None = None
        self._imu_frame_id = 0
        self._imu_pose_lock = threading.Lock()
        self._latest_imu_pose: dict[str, Any] | None = None
        self._imu_last_error: str | None = None

        for directory in (POINTCLOUD_MODEL_DIR, MESHFIX_OUTPUT_DIR, MATERIAL_OUTPUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    def _reset_recovery_to_normal(self) -> None:
        self._center_recovery_state = "normal"
        self._consecutive_lost_frames = 0
        self._consecutive_success_frames = 0
        self._recovery_hint = None

    def _reset_placement_calibration(self) -> None:
        with self._placement_lock:
            self._placement_depth_samples = []
            self._placement_mesh_samples = []
            self._placement_locked = None
            self._placement_reference_camera = None

    def _capture_depth_placement_sample(self, tracking: Any) -> None:
        """跟踪成功时从前几帧深度图估计相机系/世界系包围盒（米）。"""
        max_depth_samples = 5
        with self._placement_lock:
            if self._placement_locked is not None:
                return
            if len(self._placement_depth_samples) >= max_depth_samples:
                return

        with self.lock:
            pipeline = self.pipeline
        if pipeline is None or not bool(getattr(tracking, "success", False)):
            return

        slot = getattr(pipeline, "latest_frame_slot", None)
        if slot is None:
            return
        frame = slot.get_latest()
        if frame is None or getattr(frame, "depth", None) is None:
            return

        mapper = getattr(pipeline, "mapper", None)
        if mapper is None:
            return

        depth_np = np.asarray(frame.depth)
        if depth_np.size == 0:
            return

        depth_scale = 1.0
        guess_fn = getattr(mapper, "_guess_depth_scale_for_local_map", None)
        if callable(guess_fn):
            try:
                depth_scale = float(guess_fn(depth_np))
            except Exception:
                depth_scale = 1.0

        fx = float(getattr(mapper, "fx", 525.0))
        fy = float(getattr(mapper, "fy", 525.0))
        cx = float(getattr(mapper, "cx", 319.5))
        cy = float(getattr(mapper, "cy", 239.5))
        depth_trunc = float(getattr(mapper, "depth_trunc", 3.0))
        stride = max(1, int(getattr(mapper, "local_map_frame_pcd_stride", 4)))

        d = depth_np.astype(np.float32, copy=False)
        z_m = d * depth_scale
        valid = np.isfinite(z_m) & (z_m > 0.05) & (z_m < depth_trunc)
        if not np.any(valid):
            return

        v_idx, u_idx = np.where(valid)
        v_idx = v_idx[::stride]
        u_idx = u_idx[::stride]
        z_m = z_m[v_idx, u_idx]
        x_m = (u_idx.astype(np.float64) - cx) * z_m / fx
        y_m = (v_idx.astype(np.float64) - cy) * z_m / fy

        cam_pts = np.stack([x_m, y_m, z_m], axis=1)
        cam_min = cam_pts.min(axis=0)
        cam_max = cam_pts.max(axis=0)
        cam_center = (cam_min + cam_max) * 0.5
        cam_size = cam_max - cam_min

        T_cw = pipeline.get_current_camera_pose_world()
        if T_cw is None:
            return
        T = np.asarray(T_cw, dtype=np.float64).reshape(4, 4)
        R = T[:3, :3]
        t = T[:3, 3]
        world_center = R @ cam_center + t

        sample = {
            "frame_id": int(getattr(tracking, "frame_id", -1)),
            "depth_scale": depth_scale,
            "extent_camera_m": {
                "x": float(cam_size[0]),
                "y": float(cam_size[1]),
                "z": float(cam_size[2]),
            },
            "center_camera_m": [float(cam_center[0]), float(cam_center[1]), float(cam_center[2])],
            "center_world_m": [float(world_center[0]), float(world_center[1]), float(world_center[2])],
        }

        with self._placement_lock:
            if self._placement_locked is not None:
                return
            if self._placement_reference_camera is None:
                self._placement_reference_camera = T.tolist()
            self._placement_depth_samples.append(sample)

    def _record_mesh_placement_sample(self, vertices: np.ndarray) -> None:
        placement = compute_metric_mesh_placement(
            vertices,
            floor_y_target=self._placement_floor_y_target,
        )
        if placement is None:
            return
        with self._placement_lock:
            if self._placement_locked is not None:
                return
            self._placement_mesh_samples.append(placement)
            mesh_n = len(self._placement_mesh_samples)
            depth_n = len(self._placement_depth_samples)
            ref_cam = self._placement_reference_camera

        if mesh_n >= 3:
            self._lock_placement_from_samples()

    def _lock_placement_from_samples(self) -> None:
        with self._placement_lock:
            if self._placement_locked is not None:
                return
            if not self._placement_mesh_samples:
                return
            scales = [float(s["scale"]) for s in self._placement_mesh_samples]
            px = py = pz = 0.0
            ext_x = ext_y = ext_z = 0.0
            for s in self._placement_mesh_samples:
                pos = s["position"]
                px += float(pos[0])
                py += float(pos[1])
                pz += float(pos[2])
                e = s.get("extent_m") or {}
                ext_x += float(e.get("x", 0.0))
                ext_y += float(e.get("y", 0.0))
                ext_z += float(e.get("z", 0.0))
            n = len(self._placement_mesh_samples)
            depth_samples = list(self._placement_depth_samples)
            ref_cam = self._placement_reference_camera

        depth_extent = None
        if depth_samples:
            wx = wy = wz = 0.0
            for ds in depth_samples:
                ec = ds.get("extent_camera_m") or {}
                wx += float(ec.get("x", 0.0))
                wy += float(ec.get("y", 0.0))
                wz += float(ec.get("z", 0.0))
            dn = len(depth_samples)
            depth_extent = {
                "x": wx / dn,
                "y": wy / dn,
                "z": wz / dn,
                "samples": dn,
            }

        locked = {
            "locked": True,
            "method": "metric_depth_mesh",
            "scale": float(sum(scales) / n),
            "position": [px / n, py / n, pz / n],
            "extent_m": {"x": ext_x / n, "y": ext_y / n, "z": ext_z / n},
            "depth_extent_m": depth_extent,
            "reference_camera_to_world": ref_cam,
            "mesh_samples": n,
            "floor_y_target": self._placement_floor_y_target,
            "coordinate_space": "reconstruction_world",
        }

        with self._placement_lock:
            if self._placement_locked is not None:
                return
            self._placement_locked = locked

    def _placement_api_payload(self) -> dict[str, Any] | None:
        with self._placement_lock:
            locked = copy.deepcopy(self._placement_locked)
            if locked is not None:
                return locked
            depth_n = len(self._placement_depth_samples)
            mesh_n = len(self._placement_mesh_samples)
            ref_cam = copy.deepcopy(self._placement_reference_camera)
        if depth_n == 0 and mesh_n == 0:
            return None
        return {
            "locked": False,
            "calibrating": True,
            "depth_samples": depth_n,
            "mesh_samples": mesh_n,
            "reference_camera_to_world": ref_cam,
        }

    def _attach_recovery_observer(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.set_tracking_result_observer(self._on_tracking_result)
        self.pipeline.set_center_recovery_gate(True, True)

    def _sync_recovery_pipeline_gates(self) -> None:
        """Caller must hold self.lock and ensure self.pipeline is not None."""
        enabled = self._center_recovery_state not in ("hard_paused_lost", "recovery_failed")
        self.pipeline.set_center_recovery_gate(enabled, enabled)

    def _on_tracking_result(self, tracking: Any) -> None:
        try:
            self._capture_depth_placement_sample(tracking)
        except Exception:
            pass

        with self.lock:
            if self.pipeline is None:
                return
            state = self._center_recovery_state
            success = bool(getattr(tracking, "success", False))

            pause_th = self._clf_pause_threshold
            fail_th = self._clf_fail_threshold
            rec_succ = self._recover_success_frames_threshold

            if state == "normal":
                if success:
                    self._consecutive_lost_frames = 0
                else:
                    self._consecutive_lost_frames += 1
                    if self._consecutive_lost_frames >= pause_th:
                        self._center_recovery_state = "hard_paused_lost"
                        self._recovery_hint = None
                        self._sync_recovery_pipeline_gates()

            elif state == "recovering":
                if success:
                    self._consecutive_lost_frames = 0
                    self._consecutive_success_frames += 1
                    if self._consecutive_success_frames >= rec_succ:
                        self._center_recovery_state = "normal"
                        self._consecutive_success_frames = 0
                        self._recovery_hint = None
                        self._sync_recovery_pipeline_gates()
                else:
                    self._consecutive_lost_frames += 1
                    self._consecutive_success_frames = 0
                    if self._consecutive_lost_frames >= fail_th:
                        self._center_recovery_state = "recovery_failed"
                        self._recovery_hint = "恢复失败，重新建模"
                        self._sync_recovery_pipeline_gates()

    def _recommended_view_distance_cached(self) -> float | None:
        now = time.perf_counter()
        cached_val, cached_ts = self._recommended_view_distance_cache
        if cached_ts is not None and (now - cached_ts) < 1.0:
            return cached_val
        mesh = self.get_reconstruction_mesh_snapshot()
        dist = recommended_view_distance_from_mesh(mesh)
        self._recommended_view_distance_cache = (dist, now)
        return dist

    def _recovery_status_slice(self, pipeline: Any | None) -> dict[str, Any]:
        capture_enabled = False
        tracking_enabled = False
        mapping_enabled = False
        if pipeline is not None:
            try:
                tracking_enabled = bool(pipeline.get_center_tracking_enabled())
                mapping_enabled = bool(pipeline.get_center_mapping_enabled())
                capture_enabled = bool(
                    pipeline.is_running()
                    and getattr(pipeline, "_background_thread_started", False)
                    and getattr(pipeline, "_input_mode", "") == "scanner"
                )
            except Exception:
                pass
        hint = self._recovery_hint
        message = hint if hint else None
        recommended_dist: float | None = None
        if pipeline is not None and pipeline.is_running():
            try:
                recommended_dist = self._recommended_view_distance_cached()
            except Exception:
                recommended_dist = None

        return {
            "center_recovery_state": self._center_recovery_state,
            "consecutive_lost_frames": self._consecutive_lost_frames,
            "consecutive_success_frames": self._consecutive_success_frames,
            "clf_pause_threshold": self._clf_pause_threshold,
            "clf_fail_threshold": self._clf_fail_threshold,
            "recover_success_frames_threshold": self._recover_success_frames_threshold,
            "tracking_enabled": tracking_enabled,
            "mapping_enabled": mapping_enabled,
            "capture_enabled": capture_enabled,
            "hint": hint,
            "message": message,
            "recommended_view_distance": recommended_dist,
        }

    def resume_center_recovery(self) -> dict[str, Any]:
        with self.lock:
            if self.pipeline is None or not self.pipeline.is_running():
                return {"ok": False, "running": False, "error": "pipeline_not_running"}
            if self._center_recovery_state != "hard_paused_lost":
                return {
                    "ok": False,
                    "running": True,
                    "error": "resume_only_from_hard_paused_lost",
                    "center_recovery_state": self._center_recovery_state,
                }
            self._center_recovery_state = "recovering"
            self._consecutive_lost_frames = 0
            self._consecutive_success_frames = 0
            self._recovery_hint = None
            self.pipeline.set_center_recovery_gate(True, True)
            tracker = self.pipeline.tracker
            if tracker is not None:
                fn = getattr(tracker, "reset_lost_counters_for_resume", None)
                if callable(fn):
                    fn()
        out = self.status()
        out["ok"] = True
        return out

    def reset_center_reconstruction(
        self,
        *,
        reset_scanner_temporal_state: bool = True,
        restart_workers: bool = True,
        join_timeout: float = 1.0,
        rebuild_renderer: bool = True,
    ) -> dict[str, Any]:
        with self.lock:
            pipeline = self.pipeline
            if pipeline is None or not pipeline.is_running():
                running = bool(pipeline is not None and pipeline.is_running())
                return {"ok": False, "running": running, "error": "pipeline_not_running"}

        try:
            reset_result = pipeline.reset_reconstruction(
                reset_scanner_temporal_state=reset_scanner_temporal_state,
                restart_workers=restart_workers,
                join_timeout=join_timeout,
                rebuild_renderer=rebuild_renderer,
            )
        except Exception as exc:
            return {"ok": False, "running": False, "error": repr(exc)}

        if not reset_result.get("success"):
            st = self.status()
            return {"ok": False, "reset": json_safe(reset_result), **st}

        with self.lock:
            self._reset_recovery_to_normal()
            self._recommended_view_distance_cache = (None, None)
            with self._api_mesh_lock:
                self._api_mesh_version = 0
                self._api_mesh_signature = None
            self._reset_placement_calibration()
            pl = self.pipeline
            if pl is not None:
                pl.set_tracking_result_observer(self._on_tracking_result)
                pl.set_center_recovery_gate(True, True)
                tracker = pl.tracker
                if tracker is not None:
                    fn = getattr(tracker, "reset_lost_counters_for_resume", None)
                    if callable(fn):
                        fn()

        with self.lock:
            pl_check = self.pipeline
        if pl_check is not None and not pl_check.is_running():
            try:
                restarted = bool(pl_check.start_background())
            except Exception as exc:
                st = self.status()
                return {"ok": False, "error": repr(exc), "reset": json_safe(reset_result), **st}
            if not restarted:
                st = self.status()
                return {
                    "ok": False,
                    "error": "start_background_failed_after_reset",
                    "reset": json_safe(reset_result),
                    **st,
                }

        st = self.status()
        return {"ok": True, "reset": json_safe(reset_result), **st}

    def _build_pipeline(self) -> Any:
        cfg = PipelineConfig()
        cfg.render.enabled = False
        cfg.mapping.model_dir = str(POINTCLOUD_MODEL_DIR)
        # Django /realm 实时预览：默认 cpu_rgbd，避免无 CUDA 或 ICP 质量未过门控时 pushed_to_mapping 恒为 0、TSDF 永远空。
        # 需要 GPU ICP 时设置环境变量 CENTER_TRACKING_BACKEND=gpu_icp
        backend = os.environ.get("CENTER_TRACKING_BACKEND", "cpu_rgbd").strip().lower()
        if hasattr(cfg.tracking, "tracking_backend"):
            cfg.tracking.tracking_backend = backend
        return RealtimeMappingPipeline(cfg=cfg, base_dir=POINTCLOUD_DIR)

    def is_running(self) -> bool:
        with self.lock:
            if self._imu_only_mode:
                return True
            return bool(self.pipeline is not None and self.pipeline.is_running())

    def _stop_imu_only_unlocked(self) -> None:
        """Caller holds self.lock."""
        self._imu_only_stop.set()
        thread = self._imu_only_thread
        initializer = self._imu_initializer
        self._imu_only_mode = False
        self._imu_only_thread = None
        self._imu_initializer = None
        self._imu_recon_world = None
        with self._imu_pose_lock:
            self._latest_imu_pose = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if initializer is not None:
            try:
                initializer.close()
            except Exception:
                pass
        self._imu_only_stop = threading.Event()

    def _imu_pose_from_quaternion(self, initializer: Any, q_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R_from_quat = _imu_world_mod.quaternion_wxyz_to_rotation(q_wxyz)
        if initializer.quaternion_convention == "imu_to_world":
            R_world_imu = R_from_quat
        else:
            R_world_imu = R_from_quat.T

        R_imu_cam = initializer.R_cam_imu.T
        R_world_cam = _imu_world_mod._normalize_rotation(R_world_imu @ R_imu_cam)

        R_recon = self._imu_recon_world
        if R_recon is None:
            mode = initializer.initial_alignment_mode
            if mode == "imu_absolute_world":
                R_recon = initializer.R_reconstruction_imu_world.copy()
            elif mode == "align_initial_camera_to_forward":
                R_target_cam = np.eye(3, dtype=np.float64)
                R_recon = _imu_world_mod._normalize_rotation(R_target_cam @ R_world_cam.T)
            else:
                R_recon = np.eye(3, dtype=np.float64)
            self._imu_recon_world = R_recon

        R_reconstruction_imu = _imu_world_mod._normalize_rotation(R_recon @ R_world_imu)
        R_reconstruction_camera = _imu_world_mod._normalize_rotation(R_recon @ R_world_cam)
        T_world_to_imu = _imu_world_mod.make_transform(R_reconstruction_imu)
        T_camera_to_world = _imu_world_mod.make_transform(R_reconstruction_camera)
        return T_world_to_imu, T_camera_to_world

    def _imu_only_loop(self) -> None:
        initializer = self._imu_initializer
        if initializer is None or not initializer.enabled:
            with self.lock:
                self._imu_last_error = "imu_disabled_in_config"
            return

        while not self._imu_only_stop.is_set():
            try:
                sample = initializer.provider.read_quaternion_sample(timeout_sec=0.12)
            except Exception as exc:
                with self.lock:
                    self._imu_last_error = repr(exc)
                time.sleep(0.05)
                continue

            if sample is None:
                continue

            try:
                T_imu, T_cam = self._imu_pose_from_quaternion(initializer, sample.quaternion_wxyz)
            except Exception as exc:
                with self.lock:
                    self._imu_last_error = repr(exc)
                continue

            with self.lock:
                self._imu_frame_id += 1
                frame_id = self._imu_frame_id
                self._imu_last_error = None

            payload = {
                "status": "ok",
                "tracking_success": True,
                "tracking_mode": "imu_only",
                "frame_id": frame_id,
                "timestamp": sample.timestamp,
                "imu_to_world": T_imu.tolist(),
                "camera_to_world": T_cam.tolist(),
                "coordinate_space": "reconstruction_world",
                "imu_only": True,
            }
            with self._imu_pose_lock:
                self._latest_imu_pose = payload

    def start_imu_only(self) -> dict[str, Any]:
        """仅启动串口 IMU 位姿流，供 AR/VR 漫游；不打开深度相机。"""
        with self.lock:
            if self.pipeline is not None and self.pipeline.is_running():
                st = self.status()
                return {**st, "ok": True, "imu_only": False, "message": "scanner_pipeline_already_running"}

            if self._imu_only_mode:
                st = self.status()
                return {**st, "ok": True, "imu_only": True}

            self._stop_imu_only_unlocked()
            self._imu_frame_id = 0
            self._imu_recon_world = None
            self._imu_last_error = None

            cfg = PipelineConfig()
            initializer = build_imu_world_initializer(cfg, logger=None)
            if not initializer.enabled:
                return {
                    "ok": False,
                    "running": False,
                    "imu_only": False,
                    "error": "IMU world init disabled in PipelineConfig.imu.enable_world_init",
                }
            if initializer.provider is None:
                return {
                    "ok": False,
                    "running": False,
                    "imu_only": False,
                    "error": "IMU serial provider not configured",
                }

            self._imu_initializer = initializer
            self._imu_only_stop.clear()
            self._imu_only_mode = True
            self._imu_only_thread = threading.Thread(
                target=self._imu_only_loop,
                name="CenterImuOnlyLoop",
                daemon=True,
            )
            self._imu_only_thread.start()

        time.sleep(0.08)
        self.emit_event({"type": "imu_only_started", "updated_at": utc_now_iso()})
        st = self.status()
        return {**st, "ok": True, "imu_only": True}

    def stop_imu_only(self) -> dict[str, Any]:
        with self.lock:
            self._stop_imu_only_unlocked()
        self.emit_event({"type": "imu_only_stopped", "updated_at": utc_now_iso()})
        st = self.status()
        return {**st, "ok": True}

    def start(self, input_mode: str = "scanner") -> dict[str, Any]:
        input_mode = str(input_mode).lower().strip()
        if input_mode != "scanner":
            raise ValueError("Only scanner input_mode is supported by this service entry")

        with self.lock:
            self._stop_imu_only_unlocked()

            if self.pipeline is not None and self.pipeline.is_running():
                return self.status()

            self._reset_recovery_to_normal()
            self._recommended_view_distance_cache = (None, None)
            with self._api_mesh_lock:
                self._api_mesh_version = 0
                self._api_mesh_signature = None
            self._reset_placement_calibration()

            self.pipeline = self._build_pipeline()
            ok = self.pipeline.start_background()
            if not ok:
                self.pipeline = None
                raise RuntimeError("RealtimeMappingPipeline failed to start")

            self._attach_recovery_observer()

            self.worker = ModelProcessingWorker(self)
            self.worker.start()
            self.worker_error = None

        # 后台 _background_loop 首帧 run_once 若抛错会立刻 set stop_event；不稍等则 status() 常误报「仍在运行」。
        time.sleep(0.12)

        self.emit_event({"type": "pipeline_started", "updated_at": utc_now_iso()})
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self._stop_imu_only_unlocked()
            worker = self.worker
            pipeline = self.pipeline

        if worker is not None:
            worker.stop()

        stop_result = None
        if pipeline is not None:
            stop_result = pipeline.stop(save_mesh=True, final_mesh_prefix="white_final")

        with self.lock:
            self.worker = None
            self.pipeline = None
            self.last_stop_result = json_safe(stop_result)
            self._reset_recovery_to_normal()
            with self._api_mesh_lock:
                self._api_mesh_version = 0
                self._api_mesh_signature = None
            self._reset_placement_calibration()

        self.emit_event({"type": "pipeline_stopped", "updated_at": utc_now_iso()})
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            pipeline = self.pipeline
            worker = self.worker
            imu_only = bool(self._imu_only_mode)
            scanner_running = bool(pipeline is not None and pipeline.is_running())
            status = {
                "running": scanner_running or imu_only,
                "imu_only": imu_only,
                "depth_capture_enabled": scanner_running,
                "imu_last_error": self._imu_last_error,
                "render_enabled": bool(pipeline.cfg.render.enabled) if pipeline is not None else False,
                "white_model": copy.deepcopy(self.latest_white),
                "material_scene": copy.deepcopy(self.latest_material_scene),
                "material_targets_version": self.material_version if self.latest_material_targets else None,
                "worker": {
                    "running": bool(worker is not None and worker.thread is not None and worker.thread.is_alive()),
                    "processing": bool(worker.is_processing) if worker is not None else False,
                    "last_error": self.worker_error,
                },
                "last_stop_result": copy.deepcopy(self.last_stop_result),
                "output_dirs": {
                    "pointcloud2mesh": rel_path(POINTCLOUD_MODEL_DIR),
                    "meshfix": rel_path(MESHFIX_OUTPUT_DIR),
                    "material": rel_path(MATERIAL_OUTPUT_DIR),
                },
            }
            status.update(self._recovery_status_slice(pipeline))

        if pipeline is not None:
            try:
                status["pipeline"] = json_safe(pipeline.get_status())
            except Exception as exc:
                status["pipeline"] = {"error": repr(exc)}
        else:
            status["pipeline"] = None

        return status

    def set_worker_error(self, exc: Exception) -> None:
        with self.lock:
            self.worker_error = repr(exc)
        self.emit_event({"type": "worker_error", "error": repr(exc), "updated_at": utc_now_iso()})

    def _extract_tsdf_triangle_mesh_once(self) -> tuple[o3d.geometry.TriangleMesh | None, int, int]:
        """
        单次 TSDF mesh 抽取（持 mapper 锁）。HTTP 轮询必须避免对同一请求做两次 extract。
        """
        with self.lock:
            pipeline = self.pipeline
        if pipeline is None or pipeline.mapper is None:
            return None, 0, 0
        if not getattr(pipeline, "_setup_done", False) or not pipeline.is_running():
            return None, 0, 0
        mapper = pipeline.mapper
        try:
            with mapper._lock:
                # 尚未写入 TSDF 时 extract_triangle_mesh 在部分 Open3D/驱动组合下与首帧 integrate 并发易触发原生崩溃；
                # 轮询 latest-mesh 仅在有融合帧后再抽取。
                if int(getattr(mapper, "integrated_frames", 0) or 0) <= 0:
                    return None, 0, 0
                raw = mapper.volume.extract_triangle_mesh()
                nv = int(len(raw.vertices))
                nt = int(len(raw.triangles))
                if nv == 0 or nt == 0:
                    return None, 0, 0
                # 与 Mapper.extract_output_mesh / 落盘导出一致：Y-up 输出轴修正 + 可选 yaw/平移
                out = mapper._apply_output_axis_transform(raw)
            nv = int(len(out.vertices))
            nt = int(len(out.triangles))
            return out, nv, nt
        except Exception:
            return None, 0, 0

    def get_reconstruction_mesh_snapshot(self) -> o3d.geometry.TriangleMesh | None:
        mesh, nv, nt = self._extract_tsdf_triangle_mesh_once()
        if mesh is None or nv == 0 or nt == 0:
            return None
        mesh.compute_vertex_normals()
        return mesh

    def _collect_no_mesh_diagnostics(
        self,
        *,
        tsdf_vertices: int,
        tsdf_triangles: int,
    ) -> dict[str, Any]:
        """
        mesh 尚为空时，把「相机→tracking→TSDF」各环节快照塞进 API，便于在浏览器里直接看原因。
        TSDF 顶点/面数由调用方在一次 extract 后传入，避免本函数再次 extract_triangle_mesh。
        """
        with self.lock:
            pipeline = self.pipeline
        if pipeline is None:
            return {"reason": "no_pipeline_object"}

        out: dict[str, Any] = {
            "tsdf_extract_vertices": int(tsdf_vertices),
            "tsdf_extract_triangles": int(tsdf_triangles),
        }

        try:
            st = pipeline.get_status()
            out["pipeline_last_error"] = st.get("last_error")
            tr = st.get("tracking")
            if isinstance(tr, dict):
                out["tracking_success"] = tr.get("success")
                out["tracking_mode"] = tr.get("mode")
                out["tracking_frame_id"] = tr.get("frame_id")
                ex = tr.get("extras")
                if isinstance(ex, dict):
                    tb = ex.get("tracker_backend")
                    if tb is not None:
                        out["tracker_backend"] = tb
            out["mapping_queue_size"] = st.get("mapping_queue_size")
            out["mapping_stats"] = st.get("mapping_stats")
            out["tracking_stats"] = st.get("tracking_stats")
        except Exception as exc:
            out["get_status_error"] = repr(exc)

        return out

    def get_latest_mesh_api_payload(self, max_vertices: int = 60_000) -> dict[str, Any]:
        """
        供 Django GET /api/center/latest-mesh/ 使用：从 TSDF 内存抽取 mesh，转 JSON 列表，不写 obj/glb。
        顶点数超过 max_vertices 时返回 too_large，避免浏览器卡死。
        """
        max_vertices = int(max_vertices)
        with self.lock:
            running = bool(self.pipeline is not None and self.pipeline.is_running())

        raw_mesh, nv0, nt0 = self._extract_tsdf_triangle_mesh_once()
        if raw_mesh is None or nv0 == 0 or nt0 == 0:
            with self._api_mesh_lock:
                ver = self._api_mesh_version
            msg = "NO_MESH_YET" if running else "PIPELINE_NOT_RUNNING"
            diag = self._collect_no_mesh_diagnostics(
                tsdf_vertices=nv0,
                tsdf_triangles=nt0,
            )
            payload: dict[str, Any] = {
                "ok": True,
                "running": running,
                "version": ver,
                "mesh": None,
                "message": msg,
                "coordinate_space": "reconstruction_world",
                "summary": {"vertices": 0, "faces": 0},
                "diagnostics": json_safe(diag),
            }
            with self.lock:
                sup0 = copy.deepcopy(self._semantic_horizontal_up)
            if sup0 is not None:
                payload["semantic_horizontal_up"] = sup0
            if running and isinstance(diag, dict):
                ms = diag.get("mapping_stats") if isinstance(diag.get("mapping_stats"), dict) else {}
                ts = diag.get("tracking_stats") if isinstance(diag.get("tracking_stats"), dict) else {}
                payload["progress"] = {
                    "integrated_frames": ms.get("integrated_frames"),
                    "pushed_to_mapping": ts.get("pushed_to_mapping"),
                    "processed_frames": ts.get("processed_frames"),
                    "failed_frames": ts.get("failed_frames"),
                    "tracking_success": diag.get("tracking_success"),
                    "tracking_mode": diag.get("tracking_mode"),
                    "tracker_backend": diag.get("tracker_backend"),
                }
                pm = payload["progress"].get("pushed_to_mapping")
                if diag.get("tsdf_extract_vertices") == 0 and pm == 0 and diag.get("tracking_success") is True:
                    payload["hint"] = (
                        "跟踪已成功但 pushed_to_mapping=0：若使用 gpu_icp，多为 fitness/rmse 未过建图门控。"
                        "Center 默认已用 cpu_rgbd；仍异常请看点云有效像素与缓慢移动相机。"
                    )
                elif diag.get("tsdf_extract_vertices") == 0 and diag.get("tracking_success") is False:
                    payload["hint"] = (
                        "TSDF 仍为 0 且 tracking_success=false：多为跟踪丢失/未初始化。"
                        "请缓慢平移相机、对准有纹理区域；确认深度图有足够有效像素。"
                    )
                elif diag.get("tsdf_extract_vertices") == 0 and diag.get("pipeline_last_error"):
                    payload["hint"] = "见 diagnostics.pipeline_last_error（后台采集线程可能已报错）。"
                elif diag.get("tsdf_extract_vertices") == 0:
                    payload["hint"] = (
                        "TSDF 体素里还没有有效表面：继续扫描几秒，或检查深度图是否有有效像素、"
                        "mapping 是否因「运动门控」未写入（需缓慢移动相机）。"
                    )
            return payload

        raw_mesh.compute_vertex_normals()
        work = copy.deepcopy(raw_mesh)
        nv = int(len(work.vertices))
        nf = int(len(work.triangles))
        if nv > max_vertices:
            with self._api_mesh_lock:
                ver = self._api_mesh_version
            out_large: dict[str, Any] = {
                "ok": True,
                "too_large": True,
                "running": running,
                "version": ver,
                "mesh": None,
                "message": "MESH_TOO_LARGE",
                "coordinate_space": "reconstruction_world",
                "summary": {"vertices": nv, "faces": nf, "max_vertices": max_vertices},
            }
            with self.lock:
                sup1 = copy.deepcopy(self._semantic_horizontal_up)
            if sup1 is not None:
                out_large["semantic_horizontal_up"] = sup1
            return out_large

        v_np = np.asarray(work.vertices, dtype=np.float32)
        t_np = np.asarray(work.triangles, dtype=np.int32)
        tail_n = min(20, nv)
        coord_sum = float(v_np[-tail_n:].sum()) if nv > 0 else 0.0
        sig = f"{nv}:{nf}:{coord_sum:.6f}"

        normals_list: list[float] | None = None
        if work.has_vertex_normals():
            nn = np.asarray(work.vertex_normals, dtype=np.float32)
            if len(nn) == nv:
                normals_list = nn.reshape(-1).tolist()

        colors_list: list[float] | None = None
        if work.has_vertex_colors():
            cc = np.asarray(work.vertex_colors, dtype=np.float32)
            if len(cc) == nv:
                colors_list = cc.reshape(-1).tolist()

        mesh_payload = {
            "vertices": v_np.reshape(-1).tolist(),
            "indices": t_np.reshape(-1).tolist(),
            "normals": normals_list,
            "colors": colors_list,
        }

        with self._api_mesh_lock:
            if sig != self._api_mesh_signature:
                self._api_mesh_signature = sig
                self._api_mesh_version += 1
            ver = self._api_mesh_version

        if nv >= 120:
            try:
                self._record_mesh_placement_sample(v_np)
            except Exception:
                pass

        placement = self._placement_api_payload()

        out_mesh: dict[str, Any] = {
            "ok": True,
            "running": running,
            "version": ver,
            "mesh": mesh_payload,
            "coordinate_space": "reconstruction_world",
            "summary": {"vertices": nv, "faces": nf},
        }
        if placement is not None:
            out_mesh["placement"] = placement
        with self.lock:
            sup2 = copy.deepcopy(self._semantic_horizontal_up)
        if sup2 is not None:
            out_mesh["semantic_horizontal_up"] = sup2
        return out_mesh

    def export_white_mesh(self, mesh: o3d.geometry.TriangleMesh) -> dict[str, Any]:
        with self.lock:
            self.white_version += 1
            version = self.white_version

        version_path = POINTCLOUD_MODEL_DIR / f"white_v{version:04d}.obj"
        latest_path = POINTCLOUD_MODEL_DIR / "white_latest.obj"
        o3d.io.write_triangle_mesh(str(version_path), mesh, write_vertex_normals=True)
        shutil.copyfile(version_path, latest_path)

        meta = {
            "status": "ready",
            "version": version,
            "format": "obj",
            "path": rel_path(version_path),
            "latest_path": rel_path(latest_path),
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.triangles),
            "coordinate_space": "reconstruction_world",
            "updated_at": utc_now_iso(),
            "source_module": "pointcloud2mesh",
        }

        with self.lock:
            self.latest_white = copy.deepcopy(meta)
            self.white_files[version] = version_path

        self.emit_event({"type": "white_model_updated", "version": version, "updated_at": meta["updated_at"]})
        return meta

    def _export_material_files(
        self,
        adapter: MaterialEngineAdapter,
        version: int,
    ) -> dict[str, Any]:
        version_dir = MATERIAL_OUTPUT_DIR / f"v{version:04d}"
        latest_dir = MATERIAL_OUTPUT_DIR / "latest"
        segments_dir = version_dir / "segments"
        latest_segments_dir = latest_dir / "segments"

        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        segments_dir.mkdir(parents=True, exist_ok=True)
        latest_segments_dir.mkdir(parents=True, exist_ok=True)

        scene_path = version_dir / "scene.glb"
        latest_scene_path = latest_dir / "scene.glb"
        adapter.export_scene_glb(scene_path)

        segment_paths: dict[str, Path] = {}
        for segment_type, segment_id, _mesh in adapter.iter_segments():
            segment_key = adapter.make_segment_key(segment_type, segment_id)
            segment_path = segments_dir / safe_segment_filename(segment_key)
            adapter.export_segment(segment_key, segment_path)
            segment_paths[segment_key] = segment_path

        shutil.copyfile(scene_path, latest_scene_path)
        for segment_key, segment_path in segment_paths.items():
            shutil.copyfile(segment_path, latest_segments_dir / segment_path.name)

        targets = adapter.get_material_targets(version)
        for target in targets:
            segment_key = target["segment_key"]
            target["model_url"] = None
            target["model_path"] = rel_path(segment_paths.get(segment_key))
        targets_payload = {"version": version, "targets": targets}

        targets_path = version_dir / "targets.json"
        latest_targets_path = latest_dir / "targets.json"
        with open(targets_path, "w", encoding="utf-8") as f:
            json.dump(targets_payload, f, ensure_ascii=False, indent=2)
        with open(latest_targets_path, "w", encoding="utf-8") as f:
            json.dump(targets_payload, f, ensure_ascii=False, indent=2)

        return {
            "scene_path": scene_path,
            "latest_scene_path": latest_scene_path,
            "segment_paths": segment_paths,
            "targets": targets_payload,
        }

    def publish_material_result(
        self,
        adapter: MaterialEngineAdapter,
        meshfix_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self.material_version += 1
            version = self.material_version

        exported = self._export_material_files(adapter, version)
        semantic_up = None
        try:
            semantic_up = adapter.get_semantic_horizontal_up_hint()
        except Exception:
            semantic_up = None
        scene_meta = {
            "status": "ready",
            "version": version,
            "path": rel_path(exported["scene_path"]),
            "latest_path": rel_path(exported["latest_scene_path"]),
            "targets_count": len(exported["targets"]["targets"]),
            "source_module": "Material",
            "meshfix": json_safe(meshfix_meta or {}),
            "updated_at": utc_now_iso(),
        }

        with self.lock:
            self.material_engine = adapter
            self.latest_material_scene = copy.deepcopy(scene_meta)
            self.latest_material_targets = copy.deepcopy(exported["targets"])
            self.material_versions[version] = {
                "scene_path": exported["scene_path"],
                "segment_paths": exported["segment_paths"],
                "targets": exported["targets"],
            }
            self._semantic_horizontal_up = semantic_up

        self.emit_event({"type": "material_updated", "version": version, "updated_at": scene_meta["updated_at"]})
        return scene_meta

    def change_material(self, segment_key: str, material_id: str, tiling: float = 1.0) -> dict[str, Any]:
        with self.lock:
            adapter = self.material_engine
        if adapter is None:
            raise RuntimeError("Material scene is not ready")

        if not adapter.change_material_by_key(segment_key, material_id, tiling=tiling):
            raise ValueError("Material change failed")

        scene_meta = self.publish_material_result(adapter, meshfix_meta={"skipped": True, "reason": "material_change"})
        return {
            "status": "success",
            "segment_key": segment_key,
            "material_id": material_id,
            "version": scene_meta["version"],
            "updated_segment_path": rel_path(
                self.material_versions[scene_meta["version"]]["segment_paths"].get(segment_key)
            ),
            "scene_path": scene_meta["path"],
        }

    def latest_pose(self) -> dict[str, Any]:
        with self.lock:
            imu_only = self._imu_only_mode
        if imu_only:
            with self._imu_pose_lock:
                snap = copy.deepcopy(self._latest_imu_pose) if self._latest_imu_pose else None
            if snap is None:
                err = None
                with self.lock:
                    err = self._imu_last_error
                out = {
                    "status": "not_ready",
                    "tracking_success": False,
                    "coordinate_space": "reconstruction_world",
                    "imu_only": True,
                }
                if err:
                    out["error"] = err
                return out
            return snap

        with self.lock:
            pipeline = self.pipeline
        if pipeline is None:
            return {
                "status": "not_running",
                "tracking_success": False,
                "coordinate_space": "reconstruction_world",
            }

        tracking = pipeline.get_latest_tracking()
        world_to_camera = pipeline.get_current_extrinsic_world_to_camera()
        camera_to_world = pipeline.get_current_camera_pose_world()
        imu_to_world = pipeline.get_current_imu_pose_world()
        if tracking is None or world_to_camera is None or camera_to_world is None:
            return {
                "status": "not_ready",
                "tracking_success": False,
                "coordinate_space": "reconstruction_world",
            }

        out = {
            "frame_id": tracking.frame_id,
            "timestamp": tracking.timestamp,
            "tracking_success": bool(tracking.success),
            "tracking_mode": tracking.mode,
            "world_to_camera": np.asarray(world_to_camera).tolist(),
            "camera_to_world": np.asarray(camera_to_world).tolist(),
            "coordinate_space": "reconstruction_world",
        }
        if imu_to_world is not None:
            out["imu_to_world"] = np.asarray(imu_to_world).tolist()
        return out

    def get_white_file(self, version: int | None) -> Path:
        with self.lock:
            if version is None:
                meta = self.latest_white
                if not meta:
                    raise KeyError("No white model has been exported")
                version = int(meta["version"])
            path = self.white_files.get(version)
        if path is None or not path.exists():
            raise KeyError(f"White model version not found: {version}")
        return path

    def get_material_scene_file(self, version: int | None) -> Path:
        with self.lock:
            if version is None:
                version = self.material_version
            entry = self.material_versions.get(version)
        if not entry:
            raise KeyError(f"Material scene version not found: {version}")
        path = Path(entry["scene_path"])
        if not path.exists():
            raise KeyError(f"Material scene file missing: {version}")
        return path

    def get_material_segment_file(self, segment_key: str, version: int | None) -> Path:
        with self.lock:
            if version is None:
                version = self.material_version
            entry = self.material_versions.get(version)
        if not entry:
            raise KeyError(f"Material version not found: {version}")
        path = entry["segment_paths"].get(segment_key)
        if path is None or not Path(path).exists():
            raise KeyError(f"Material segment not found: {segment_key}")
        return Path(path)

    def get_material_targets(self) -> dict[str, Any]:
        with self.lock:
            if self.latest_material_targets is None:
                return {"version": None, "targets": []}
            return copy.deepcopy(self.latest_material_targets)

    def get_material_status(self) -> dict[str, Any]:
        with self.lock:
            adapter = self.material_engine
            version = self.material_version
        if adapter is None:
            return {"version": None, "status": {}}
        return {"version": version, "status": adapter.get_flat_material_status()}

    def get_material_library(self) -> dict[str, Any]:
        with self.lock:
            adapter = self.material_engine or MaterialEngineAdapter()
        return {"materials": adapter.get_material_library()}

    def emit_event(self, event: dict[str, Any]) -> None:
        queues = list(self._event_queues)
        for event_queue in queues:
            try:
                event_queue.put_nowait(event)
            except Exception:
                pass

    async def subscribe(self):
        event_queue: queue.Queue = queue.Queue(maxsize=100)
        self._event_queues.append(event_queue)
        try:
            while True:
                yield await asyncio.to_thread(event_queue.get)
        finally:
            if event_queue in self._event_queues:
                self._event_queues.remove(event_queue)


service = CenterPipelineService()


def get_service() -> CenterPipelineService:
    """Return the process-wide service instance for Django or scripts."""
    return service


def start_pipeline(input_mode: str = "scanner") -> dict[str, Any]:
    return service.start(input_mode=input_mode)


def start_imu_pipeline() -> dict[str, Any]:
    return service.start_imu_only()


def stop_imu_pipeline() -> dict[str, Any]:
    return service.stop_imu_only()


def stop_pipeline() -> dict[str, Any]:
    return service.stop()


def get_pipeline_status() -> dict[str, Any]:
    return service.status()


def resume_center_recovery() -> dict[str, Any]:
    return service.resume_center_recovery()


def reset_center_reconstruction(**kwargs: Any) -> dict[str, Any]:
    return service.reset_center_reconstruction(**kwargs)


def get_white_model_latest() -> dict[str, Any]:
    with service.lock:
        meta = copy.deepcopy(service.latest_white)
    if meta is None:
        return {
            "status": "not_ready",
            "source_module": "pointcloud2mesh",
            "coordinate_space": "reconstruction_world",
        }
    return meta


def get_white_model_file(version: int | None = None) -> Path:
    return service.get_white_file(version)


def get_pose_latest() -> dict[str, Any]:
    return service.latest_pose()


def get_latest_mesh_api_payload(max_vertices: int = 60_000) -> dict[str, Any]:
    """JSON 可序列化的内存 mesh 快照（无文件 IO）。"""
    return service.get_latest_mesh_api_payload(max_vertices=max_vertices)


def get_material_targets() -> dict[str, Any]:
    return service.get_material_targets()


def get_material_status() -> dict[str, Any]:
    return service.get_material_status()


def get_material_library() -> dict[str, Any]:
    return service.get_material_library()


def change_material(segment_key: str, material_id: str, tiling: float = 1.0) -> dict[str, Any]:
    return service.change_material(segment_key, material_id, tiling)


def get_material_segment_file(segment_key: str, version: int | None = None) -> Path:
    return service.get_material_segment_file(segment_key, version)


def get_material_scene_latest() -> dict[str, Any]:
    with service.lock:
        meta = copy.deepcopy(service.latest_material_scene)
    if meta is None:
        return {"status": "not_ready", "source_module": "Material"}
    return meta


def get_material_scene_file(version: int | None = None) -> Path:
    return service.get_material_scene_file(version)


def get_event_subscriber():
    """Return an async generator yielding service events."""
    return service.subscribe()


def main():
    print("Center_pipeline.py is now a Python backend interface module.")
    print("Import it from Django, for example:")
    print("  from Center_pipeline import start_pipeline, get_pipeline_status")
    print("  start_pipeline()")
    print("  status = get_pipeline_status()")


if __name__ == "__main__":
    main()