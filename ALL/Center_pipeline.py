from __future__ import annotations

import asyncio
import copy
import importlib
import json
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
RealtimeMappingPipeline = importlib.import_module(
    f"{POINTCLOUD_PACKAGE}.pipeline_api"
).RealtimeMappingPipeline
from mesh_refine_adapter_slim import o3d_to_trimesh, refine_mesh_in_memory
from Material.material_engine_adapter import MaterialEngineAdapter


POINTCLOUD_MODEL_DIR = POINTCLOUD_DIR / "model"
MESHFIX_OUTPUT_DIR = MESHFIX_DIR / "outputs_realtime"
MATERIAL_OUTPUT_DIR = MATERIAL_DIR / "outputs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        if not material_adapter.load_from_mesh(o3d_to_trimesh(material_mesh)):
            raise RuntimeError("Material segmentation failed")

        self.service.publish_material_result(material_adapter, meshfix_meta)
        self.last_processed_vertices = vertices
        self.last_processed_triangles = triangles


class CenterPipelineService:
    def __init__(self):
        self.lock = threading.RLock()
        self.pipeline: RealtimeMappingPipeline | None = None
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
        self._event_queues: list[queue.Queue] = []

        for directory in (POINTCLOUD_MODEL_DIR, MESHFIX_OUTPUT_DIR, MATERIAL_OUTPUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    def _build_pipeline(self) -> RealtimeMappingPipeline:
        cfg = PipelineConfig()
        cfg.render.enabled = False
        cfg.mapping.model_dir = str(POINTCLOUD_MODEL_DIR)
        return RealtimeMappingPipeline(cfg=cfg, base_dir=POINTCLOUD_DIR)

    def is_running(self) -> bool:
        with self.lock:
            return bool(self.pipeline is not None and self.pipeline.is_running())

    def start(self, input_mode: str = "scanner") -> dict[str, Any]:
        input_mode = str(input_mode).lower().strip()
        if input_mode != "scanner":
            raise ValueError("Only scanner input_mode is supported by this service entry")

        with self.lock:
            if self.pipeline is not None and self.pipeline.is_running():
                return self.status()

            self.pipeline = self._build_pipeline()
            ok = self.pipeline.start_background()
            if not ok:
                self.pipeline = None
                raise RuntimeError("RealtimeMappingPipeline failed to start")

            self.worker = ModelProcessingWorker(self)
            self.worker.start()
            self.worker_error = None

        self.emit_event({"type": "pipeline_started", "updated_at": utc_now_iso()})
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self.lock:
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

        self.emit_event({"type": "pipeline_stopped", "updated_at": utc_now_iso()})
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            pipeline = self.pipeline
            worker = self.worker
            status = {
                "running": bool(pipeline is not None and pipeline.is_running()),
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

    def get_reconstruction_mesh_snapshot(self) -> o3d.geometry.TriangleMesh | None:
        with self.lock:
            pipeline = self.pipeline
        if pipeline is None or pipeline.mapper is None:
            return None

        mapper = pipeline.mapper
        with mapper._lock:
            mesh = mapper.volume.extract_triangle_mesh()

        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return None
        mesh.compute_vertex_normals()
        return mesh

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
        if tracking is None or world_to_camera is None or camera_to_world is None:
            return {
                "status": "not_ready",
                "tracking_success": False,
                "coordinate_space": "reconstruction_world",
            }

        return {
            "frame_id": tracking.frame_id,
            "timestamp": tracking.timestamp,
            "tracking_success": bool(tracking.success),
            "tracking_mode": tracking.mode,
            "world_to_camera": np.asarray(world_to_camera).tolist(),
            "camera_to_world": np.asarray(camera_to_world).tolist(),
            "coordinate_space": "reconstruction_world",
        }

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


def stop_pipeline() -> dict[str, Any]:
    return service.stop()


def get_pipeline_status() -> dict[str, Any]:
    return service.status()


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