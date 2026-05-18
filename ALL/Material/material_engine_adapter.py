from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import trimesh


MODULE_DIR = Path(__file__).resolve().parent

# 可选：专用贴图根目录（默认使用 Material 下的数字子文件夹 1/, 2/, …）
PBR_MATERIAL_ROOT_ENV = "CENTER_PBR_MATERIAL_ROOT"
LEGACY_ENGINE_CANDIDATES = (
    MODULE_DIR / "material_pipeline.py",
    MODULE_DIR / "material_pipeline_接口版.py",
)


def _load_legacy_engine_class():
    engine_path = next((path for path in LEGACY_ENGINE_CANDIDATES if path.exists()), None)
    if engine_path is None:
        names = ", ".join(path.name for path in LEGACY_ENGINE_CANDIDATES)
        raise ImportError(f"Cannot find Material engine file. Tried: {names}")

    source = engine_path.read_text(encoding="utf-8")
    marker = "# Web HTTP 接口层"
    if marker in source:
        source = source.split(marker, 1)[0]

    namespace = {
        "__file__": str(engine_path),
        "__name__": "material_pipeline_engine_only",
    }
    exec(compile(source, str(engine_path), "exec"), namespace)
    return namespace["MRMaterialEngine"]


MRMaterialEngine = _load_legacy_engine_class()


class MaterialEngineAdapter:
    """Frontend-friendly wrapper around the existing MRMaterialEngine."""

    DEFAULT_MATERIALS = {
        "default_wall": {
            "name": "默认墙面",
            "segment_types": {"walls"},
            "color": [200, 200, 200, 255],
        },
        "default_floor": {
            "name": "默认地面",
            "segment_types": {"floor"},
            "color": [40, 40, 45, 255],
        },
        "default_ceiling": {
            "name": "默认天花板",
            "segment_types": {"ceiling"},
            "color": [220, 220, 225, 255],
        },
        "default_door": {
            "name": "默认门",
            "segment_types": {"doors"},
            "color": [20, 20, 25, 255],
        },
        "default_others": {
            "name": "默认其他",
            "segment_types": {"others"},
            "color": [60, 60, 65, 255],
        },
        "style_1": {
            "name": "风格 1",
            "segment_types": {"floor", "ceiling", "walls", "doors", "others"},
            "color": [179, 159, 126, 255],
        },
        "style_2": {
            "name": "风格 2",
            "segment_types": {"floor", "ceiling", "walls", "doors", "others"},
            "color": [120, 140, 160, 255],
        },
        "style_3": {
            "name": "风格 3",
            "segment_types": {"floor", "ceiling", "walls", "doors", "others"},
            "color": [170, 130, 140, 255],
        },
    }

    def __init__(self):
        self.engine = MRMaterialEngine()
        self.material_metadata: dict[str, dict] = {}
        self.register_default_materials()

    @staticmethod
    def make_segment_key(segment_type: str, segment_id: str | None = None) -> str:
        if segment_type in {"floor", "ceiling", "others"}:
            return segment_type
        if segment_type in {"walls", "doors"} and segment_id:
            return f"{segment_type}/{segment_id}"
        raise ValueError(f"Invalid segment: type={segment_type!r}, id={segment_id!r}")

    @staticmethod
    def parse_segment_key(segment_key: str) -> tuple[str, str | None]:
        key = str(segment_key).strip().strip("/")
        if key in {"floor", "ceiling", "others"}:
            return key, None

        parts = key.split("/")
        if len(parts) == 2 and parts[0] in {"walls", "doors"} and parts[1]:
            return parts[0], parts[1]

        raise ValueError(f"Invalid segment_key: {segment_key!r}")

    def register_default_materials(self) -> None:
        for material_id, meta in self.DEFAULT_MATERIALS.items():
            color = meta["color"]
            metallic = 0.9 if material_id == "default_door" else 0.1
            self.engine.register_color_material(material_id, color, metallic=metallic)
            self.material_metadata[material_id] = {
                "material_id": material_id,
                "name": meta["name"],
                "type": "color",
                "preview_color": color,
                "preview_url": None,
                "segment_types": sorted(meta["segment_types"]),
            }

    def register_pbr_styles_from_disk(self, library_root: Path | str | None = None) -> int:
        """
        从磁盘加载 style_<n> PBR 套装（与 FastAPI 示例一致：<n>/<n>_albedo.jpg …）。
        覆盖同名 style_* 在 engine.material_library 中的条目，并更新 material_metadata。
        """
        import os

        env_root = os.environ.get(PBR_MATERIAL_ROOT_ENV, "").strip()
        if env_root:
            root = Path(env_root).expanduser().resolve()
        elif library_root is not None:
            root = Path(library_root).resolve()
        else:
            root = MODULE_DIR.resolve()

        registered = 0
        if not root.is_dir():
            return 0

        digit_dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        for mat_dir in digit_dirs:
            i = int(mat_dir.name)
            material_id = f"style_{i}"
            albedo = mat_dir / f"{i}_albedo.jpg"
            normal = mat_dir / f"{i}_normal.jpg"
            ao = mat_dir / f"{i}_ao.jpg"
            rough = mat_dir / f"{i}_roughness.jpg"
            metal = mat_dir / f"{i}_metallic.jpg"
            if not albedo.is_file():
                continue
            ok = self.engine.register_pbr_material(
                material_id=material_id,
                albedo_path=str(albedo),
                normal_path=str(normal),
                ao_path=str(ao),
                rough_path=str(rough),
                metal_path=str(metal),
            )
            if not ok:
                continue
            registered += 1
            meta_base = self.DEFAULT_MATERIALS.get(material_id, {})
            fallback_color = meta_base.get("color", [180, 180, 180, 255])
            self.material_metadata[material_id] = {
                "material_id": material_id,
                "name": meta_base.get("name", f"风格 {i}"),
                "type": "pbr",
                "preview_color": fallback_color,
                "preview_url": None,
                "segment_types": sorted(
                    meta_base.get(
                        "segment_types",
                        {"floor", "ceiling", "walls", "doors", "others"},
                    )
                ),
            }
        return registered

    def load_from_mesh(self, mesh: trimesh.Trimesh) -> bool:
        ok = self.engine.load_from_mesh(mesh.copy())
        if ok:
            self.engine.restore_default_materials()
        return bool(ok)

    def load_from_file(self, path: str | Path) -> bool:
        ok = self.engine.load_from_file(str(path))
        if ok:
            self.engine.restore_default_materials()
        return bool(ok)

    def iter_segments(self):
        for segment_type in ("floor", "ceiling", "others"):
            mesh = self.engine.segments.get(segment_type)
            if mesh is not None and not mesh.is_empty:
                yield segment_type, None, mesh

        for segment_type in ("walls", "doors"):
            for segment_id in sorted(self.engine.segments.get(segment_type, {}).keys()):
                mesh = self.engine.segments[segment_type][segment_id]
                if mesh is not None and not mesh.is_empty:
                    yield segment_type, segment_id, mesh

    def get_material_library(self) -> list[dict]:
        return [
            {
                key: value
                for key, value in meta.items()
                if key != "segment_types"
            }
            for meta in self.material_metadata.values()
        ]

    def get_available_materials(self, segment_type: str) -> list[str]:
        return [
            material_id
            for material_id, meta in self.material_metadata.items()
            if segment_type in set(meta.get("segment_types", []))
        ]

    def get_flat_material_status(self) -> dict[str, str | None]:
        state = self.engine.get_material_status()
        status: dict[str, str | None] = {}

        for segment_type, segment_id, _mesh in self.iter_segments():
            key = self.make_segment_key(segment_type, segment_id)
            if segment_type in {"floor", "ceiling", "others"}:
                status[key] = state.get(segment_type)
            else:
                status[key] = state.get(segment_type, {}).get(segment_id)

        return status

    def get_current_material(self, segment_type: str, segment_id: str | None) -> str | None:
        key = self.make_segment_key(segment_type, segment_id)
        return self.get_flat_material_status().get(key)

    @staticmethod
    def _display_name(segment_type: str, segment_id: str | None) -> str:
        if segment_type == "floor":
            return "地面"
        if segment_type == "ceiling":
            return "天花板"
        if segment_type == "others":
            return "其他"
        if segment_type == "walls":
            index = int(segment_id.split("_")[-1]) + 1 if segment_id else 1
            return f"墙面 {index}"
        if segment_type == "doors":
            index = int(segment_id.split("_")[-1]) + 1 if segment_id else 1
            return f"门 {index}"
        return segment_id or segment_type

    @staticmethod
    def _mesh_bounds(mesh: trimesh.Trimesh) -> list[list[float]] | None:
        if mesh is None or mesh.is_empty or mesh.vertices is None or len(mesh.vertices) == 0:
            return None
        return np.asarray(mesh.bounds, dtype=float).tolist()

    def _weighted_mean_face_normal(self, mesh: trimesh.Trimesh) -> np.ndarray | None:
        if mesh is None or mesh.is_empty:
            return None
        try:
            mesh.fix_normals()
        except Exception:
            pass
        fn = np.asarray(mesh.face_normals, dtype=np.float64)
        if fn.size == 0 or fn.ndim != 2 or fn.shape[1] != 3:
            return None
        n_face = fn.shape[0]
        ta = getattr(mesh, 'triangles_area', None)
        if ta is None:
            areas = np.ones(n_face, dtype=np.float64)
        else:
            areas = np.asarray(ta, dtype=np.float64).reshape(-1)
        if areas.size != n_face:
            areas = np.ones(n_face, dtype=np.float64)
        total = float(np.sum(areas))
        if total < 1e-18:
            return None
        w = areas / total
        v = np.sum(fn * w[:, np.newaxis], axis=0)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            return None
        return v / norm

    def get_semantic_horizontal_up_hint(self) -> list[float] | None:
        """
        用语义 floor / ceiling 三角面的面积加权法线估计竖直朝上方向（重建坐标），供前端锚点回正。
        """
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        hints: list[np.ndarray] = []

        floor_m = self.engine.segments.get("floor")
        nf = self._weighted_mean_face_normal(floor_m) if floor_m is not None else None
        if nf is not None:
            if float(np.dot(nf, ey)) < 0.0:
                nf = -nf
            hints.append(nf)

        ceil_m = self.engine.segments.get("ceiling")
        nc = self._weighted_mean_face_normal(ceil_m) if ceil_m is not None else None
        if nc is not None:
            room_up = -nc
            if float(np.dot(room_up, ey)) < 0.0:
                room_up = -room_up
            hints.append(room_up)

        if not hints:
            return None

        combined = np.sum(hints, axis=0)
        norm = float(np.linalg.norm(combined))
        if norm < 1e-12:
            return None
        up = combined / norm
        if float(np.dot(up, ey)) < 0.0:
            up = -up
        return [float(up[0]), float(up[1]), float(up[2])]

    def get_material_targets(
        self,
        version: int,
        model_url_builder: Callable[[str, int], str] | None = None,
    ) -> list[dict]:
        targets = []
        for segment_type, segment_id, mesh in self.iter_segments():
            segment_key = self.make_segment_key(segment_type, segment_id)
            targets.append(
                {
                    "segment_key": segment_key,
                    "segment_type": segment_type,
                    "segment_id": segment_id,
                    "display_name": self._display_name(segment_type, segment_id),
                    "current_material": self.get_current_material(segment_type, segment_id),
                    "available_materials": self.get_available_materials(segment_type),
                    "model_url": model_url_builder(segment_key, version)
                    if model_url_builder
                    else None,
                    "bbox": self._mesh_bounds(mesh),
                    "area": float(mesh.area) if mesh is not None else 0.0,
                    "source_module": "Material",
                }
            )
        return targets

    def get_segment_mesh(self, segment_key: str) -> trimesh.Trimesh:
        segment_type, segment_id = self.parse_segment_key(segment_key)
        if segment_type in {"floor", "ceiling", "others"}:
            mesh = self.engine.segments.get(segment_type)
        else:
            mesh = self.engine.segments.get(segment_type, {}).get(segment_id)

        if mesh is None or mesh.is_empty:
            raise KeyError(f"Segment not found: {segment_key}")

        return mesh.copy()

    def export_segment(self, segment_key: str, output_path: str | Path) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_segment_mesh(segment_key).export(output_path)
        return str(output_path)

    def change_material_by_key(self, segment_key: str, material_id: str, tiling: float = 1.0) -> bool:
        segment_type, segment_id = self.parse_segment_key(segment_key)
        return bool(
            self.engine.change_segment_material(
                segment_type,
                segment_id,
                material_id,
                tiling=tiling,
            )
        )

    def export_scene_glb(self, output_path: str | Path) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine.export_glb(str(output_path))
        return str(output_path)
