from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


class MRMaterialEngine:
    """
    MR 空间材质重构引擎
    """

    def __init__(self):
        self.segments = {
            "walls": {}, "doors": {},
            "floor": None, "ceiling": None, "others": None
        }
        self.material_library = {}
        self.current_state = {
            "walls": {}, "doors": {},
            "floor": None, "ceiling": None, "others": None
        }
        self.original_mesh = None
        self.current_scene = None

    # ==========================================
    # 接口 1：数据输入与几何拆分
    # ==========================================
    def load_from_file(self, obj_path):
        print(f"📥 [输入] 从文件加载模型: {obj_path}")
        try:
            mesh = trimesh.load(obj_path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0: return False
                mesh = list(mesh.geometry.values())[0]
            return self.load_from_mesh(mesh)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def load_from_mesh(self, trimesh_obj):
        print("📥 [输入] 接收 Mesh 对象，启动几何解析引擎...")
        valid_faces = trimesh_obj.nondegenerate_faces()
        trimesh_obj.update_faces(valid_faces)
        trimesh_obj.remove_unreferenced_vertices()

        self.original_mesh = trimesh_obj
        self.original_mesh.visual = trimesh.visual.ColorVisuals(mesh=self.original_mesh)
        return self._process_geometry(self.original_mesh)

    def _merge_nearby_mesh_parts(self, parts: list, max_centroid_dist: float) -> list:
        """按质心距离合并邻近连通域，减少碎片门洞被算作多扇门。"""
        if len(parts) <= 1:
            return parts
        centers = [np.mean(p.vertices, axis=0) for p in parts]
        n = len(parts)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(centers[i] - centers[j]) <= max_centroid_dist:
                    union(i, j)

        clusters: dict[int, list] = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(parts[i])

        merged: list = []
        for group in clusters.values():
            merged.append(trimesh.util.concatenate(group))
        return merged

    def _extract_doors_and_clean_walls(self, wall_mesh):
        if wall_mesh.is_empty:
            return wall_mesh, [], []
        wall_mesh.fix_normals()

        dominant_normal = np.median(wall_mesh.face_normals, axis=0)
        norm = np.linalg.norm(dominant_normal)
        if norm > 1e-6:
            dominant_normal /= norm
        else:
            dominant_normal = np.array([0.0, 1.0, 0.0])

        centroid = np.median(wall_mesh.vertices, axis=0)
        depths = np.dot(wall_mesh.triangles.mean(axis=1) - centroid, dominant_normal)

        # 略收紧深度带，减少窗套/墙饰被当成门洞
        door_face_mask = (np.abs(depths) > 0.13) & (np.abs(depths) < 0.22)
        door_candidate = trimesh.Trimesh(vertices=wall_mesh.vertices, faces=wall_mesh.faces[door_face_mask])
        door_candidate.remove_unreferenced_vertices()

        main_wall = trimesh.Trimesh(vertices=wall_mesh.vertices, faces=wall_mesh.faces[~door_face_mask])
        main_wall.remove_unreferenced_vertices()

        true_doors: list = []
        false_doors: list = []

        if door_candidate.is_empty:
            return main_wall, true_doors, false_doors

        raw_parts: list = []
        for part in door_candidate.split(only_watertight=False):
            ext = part.bounds[1] - part.bounds[0]
            height = float(ext[1])
            if part.area < 0.08 or height < 0.4:
                false_doors.append(part)
                continue
            min_y, max_y = float(part.bounds[0][1]), float(part.bounds[1][1])
            if min_y < 1.45 and (min_y < 0.65 or max_y > 1.15):
                raw_parts.append(part)
            else:
                false_doors.append(part)

        merged = self._merge_nearby_mesh_parts(raw_parts, max_centroid_dist=0.55)
        merged.sort(key=lambda m: float(m.area), reverse=True)

        MIN_DOOR_AREA = 0.28
        MIN_DOOR_HEIGHT = 0.92
        MIN_ASPECT_H_OVER_W = 1.05
        max_doors_keep = 5

        for part in merged:
            ext = part.bounds[1] - part.bounds[0]
            height = float(ext[1])
            horiz = float(max(ext[0], ext[2]))
            min_y = float(part.bounds[0][1])
            max_y = float(part.bounds[1][1])

            if part.area < MIN_DOOR_AREA or height < MIN_DOOR_HEIGHT:
                false_doors.append(part)
                continue
            if height / max(horiz, 0.05) < MIN_ASPECT_H_OVER_W:
                false_doors.append(part)
                continue
            if min_y > 1.35 and max_y < 2.2:
                false_doors.append(part)
                continue
            true_doors.append(part)

        if len(true_doors) > max_doors_keep:
            false_doors.extend(true_doors[max_doors_keep:])
            true_doors = true_doors[:max_doors_keep]

        return main_wall, true_doors, false_doors

    def _process_geometry(self, mesh):
        mesh.fix_normals()
        y_normals = mesh.face_normals[:, 1]
        face_centers = mesh.triangles.mean(axis=1)

        global_min_y = mesh.bounds[0][1]
        global_max_y = mesh.bounds[1][1]

        wall_mask = np.abs(y_normals) < 0.3
        floor_mask = (y_normals < -0.7) & (face_centers[:, 1] > global_max_y - 0.3)
        ceiling_mask = (y_normals > 0.7) & (face_centers[:, 1] < global_min_y + 0.3)

        others_mask = ~(wall_mask | floor_mask | ceiling_mask)

        floor_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[floor_mask])
        floor_mesh.remove_unreferenced_vertices()
        if not floor_mesh.is_empty: self.segments["floor"] = floor_mesh

        ceiling_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[ceiling_mask])
        ceiling_mesh.remove_unreferenced_vertices()
        if not ceiling_mesh.is_empty: self.segments["ceiling"] = ceiling_mesh

        others_pieces = []
        initial_others = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[others_mask])
        initial_others.remove_unreferenced_vertices()
        if not initial_others.is_empty: others_pieces.append(initial_others)

        all_vertical_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[wall_mask])
        all_vertical_mesh.remove_unreferenced_vertices()

        room_bounds = mesh.bounds
        min_bound, max_bound = room_bounds[0], room_bounds[1]

        MIN_WALL_AREA = 1.0

        wall_candidates = []
        for comp in all_vertical_mesh.split(only_watertight=False):
            c_min, c_max = comp.bounds[0], comp.bounds[1]
            touches_outer_shell = (
                    abs(c_min[0] - min_bound[0]) < 0.5 or abs(c_max[0] - max_bound[0]) < 0.5 or
                    abs(c_min[2] - min_bound[2]) < 0.5 or abs(c_max[2] - max_bound[2]) < 0.5
            )
            if comp.area >= MIN_WALL_AREA and touches_outer_shell:
                wall_candidates.append(comp)
            else:
                others_pieces.append(comp)

        initial_wall_mesh = trimesh.util.concatenate(wall_candidates) if wall_candidates else trimesh.Trimesh()
        main_wall, true_doors, false_doors = self._extract_doors_and_clean_walls(initial_wall_mesh)

        for i, part in enumerate(true_doors):
            self.segments["doors"][f"door_{i}"] = part

        others_pieces.extend(false_doors)

        logical_walls = []
        if not main_wall.is_empty:
            for part in main_wall.split(only_watertight=False):
                if part.area < MIN_WALL_AREA:
                    others_pieces.append(part)
                    continue

                part.fix_normals()
                normal = np.median(part.face_normals, axis=0)
                normal[1] = 0
                n_len = np.linalg.norm(normal)
                if n_len > 1e-6:
                    normal /= n_len
                else:
                    normal = np.array([1.0, 0.0, 0.0])

                centroid = np.median(part.vertices, axis=0)
                d = np.dot(normal, centroid)

                matched = False
                for w_data in logical_walls:
                    ref_n = w_data['normal']
                    ref_d = w_data['d']

                    dot_n = np.dot(ref_n, normal)
                    if abs(dot_n) > 0.85:
                        d_adj = d if dot_n > 0 else -d
                        if abs(ref_d - d_adj) < 0.5:
                            w_data['parts'].append(part)
                            matched = True
                            break

                if not matched:
                    logical_walls.append({'normal': normal, 'd': d, 'parts': [part]})

            for i, w_data in enumerate(logical_walls):
                merged_logical_wall = trimesh.util.concatenate(w_data['parts'])
                self.segments["walls"][f"wall_{i}"] = merged_logical_wall

        valid_others = [m for m in others_pieces if not m.is_empty]
        if valid_others:
            v_list, f_list = [], []
            v_offset = 0
            for m in valid_others:
                v_list.append(m.vertices)
                f_list.append(m.faces + v_offset)
                v_offset += len(m.vertices)

            final_others_mesh = trimesh.Trimesh(
                vertices=np.vstack(v_list),
                faces=np.vstack(f_list)
            )
            self.segments["others"] = final_others_mesh

        return True

    # ==========================================
    # 接口 2：材质库管理系统
    # ==========================================
    def register_pbr_material(self, material_id, albedo_path, normal_path, ao_path, rough_path, metal_path):
        try:
            img_ao = Image.open(ao_path).convert('L')
            img_rough = Image.open(rough_path).convert('L').resize(img_ao.size)
            img_metal = Image.open(metal_path).convert('L').resize(img_ao.size)
            orm_image = Image.merge('RGB', (img_ao, img_rough, img_metal))

            mat = trimesh.visual.material.PBRMaterial(
                name=material_id,
                baseColorTexture=Image.open(albedo_path).convert('RGB'),
                normalTexture=Image.open(normal_path),
                metallicRoughnessTexture=orm_image, occlusionTexture=orm_image
            )
            self.material_library[material_id] = mat
            return True
        except Exception as e:
            print(f"❌ 注册材质 {material_id} 失败: {e}")
            return False

    def register_color_material(self, material_id, rgba_color, metallic=0.1, roughness=0.9):
        mat = trimesh.visual.material.PBRMaterial(
            name=material_id, baseColorFactor=rgba_color,
            metallicFactor=metallic, roughnessFactor=roughness
        )
        self.material_library[material_id] = mat

    # ==========================================
    # 接口 3：智能 UV 映射系统
    # ==========================================
    def _generate_smart_uv(self, mesh, segment_type, tiling=1.0):
        uvs = np.zeros((len(mesh.vertices), 2))

        if segment_type in ["floor", "ceiling", "others"]:
            uvs[:, 0] = mesh.vertices[:, 0] * tiling
            uvs[:, 1] = mesh.vertices[:, 2] * tiling
        else:
            mesh.fix_normals()
            dominant_normal = np.abs(np.median(mesh.face_normals, axis=0))
            if dominant_normal[0] > dominant_normal[2]:
                uvs[:, 0] = mesh.vertices[:, 2] * tiling
                uvs[:, 1] = mesh.vertices[:, 1] * tiling
            else:
                uvs[:, 0] = mesh.vertices[:, 0] * tiling
                uvs[:, 1] = mesh.vertices[:, 1] * tiling
        return uvs

    # ==========================================
    # 接口 4：材质换装与状态管理
    # ==========================================
    def change_segment_material(self, segment_type, segment_id, material_id, tiling=1.0):
        if material_id not in self.material_library:
            print(f"⚠️ 材质 {material_id} 不存在！")
            return False

        mesh = None
        if segment_type in ["floor", "ceiling", "others"]:
            mesh = self.segments.get(segment_type)
        elif segment_type in ["walls", "doors"]:
            mesh = self.segments[segment_type].get(segment_id)

        if not mesh: return False

        uvs = self._generate_smart_uv(mesh, segment_type, tiling)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs, material=self.material_library[material_id])

        if segment_type in ["floor", "ceiling", "others"]:
            self.current_state[segment_type] = material_id
        else:
            self.current_state[segment_type][segment_id] = material_id
        return True

    def change_type_material(self, segment_type, material_id, tiling=1.0):
        if segment_type in ["floor", "ceiling", "others"]:
            self.change_segment_material(segment_type, None, material_id, tiling)
        else:
            for seg_id in self.segments[segment_type].keys():
                self.change_segment_material(segment_type, seg_id, material_id, tiling)

    def restore_default_materials(self):
        self.change_type_material("walls", "default_wall", tiling=1.0)
        self.change_type_material("floor", "default_floor", tiling=2.0)
        self.change_type_material("ceiling", "default_ceiling", tiling=1.0)
        self.change_type_material("doors", "default_door")
        self.change_type_material("others", "default_others")

    def get_material_status(self):
        return self.current_state

    def get_segment_counts(self):
        return {
            "walls": len(self.segments["walls"]),
            "doors": len(self.segments["doors"]),
            "floor": 1 if self.segments["floor"] else 0,
            "ceiling": 1 if self.segments["ceiling"] else 0,
            "others": 1 if self.segments["others"] else 0
        }

    # ==========================================
    # 接口 5：数据输出层
    # ==========================================
    def build_scene(self):
        elements = []
        if self.segments["floor"]: elements.append(self.segments["floor"])
        if self.segments["ceiling"]: elements.append(self.segments["ceiling"])
        if self.segments["others"]: elements.append(self.segments["others"])
        elements.extend(list(self.segments["walls"].values()))
        elements.extend(list(self.segments["doors"].values()))

        valid_elements = [e for e in elements if not e.is_empty]
        self.current_scene = trimesh.Scene(valid_elements)
        return self.current_scene

    def export_glb(self, output_path):
        self.build_scene()
        self.current_scene.export(output_path)
        print(f"GLB 已重新生成并保存至: {output_path}")


# ==========================================
# Web HTTP 接口层 (FastAPI)
# ==========================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 实例化 Web 框架
app = FastAPI(title="MR 空间材质引擎 API (对接前端)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端地址跨域访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许 POST, GET 等所有请求
    allow_headers=["*"],
)

global_engine = MRMaterialEngine()

_MATERIAL_API_ROOT = Path(__file__).resolve().parent


def _register_pbr_library_for_engine(engine: MRMaterialEngine, root: Path) -> int:
    """与 MaterialEngineAdapter 约定一致：<root>/<n>/<n>_albedo.jpg …"""
    if not root.is_dir():
        return 0
    n_ok = 0
    digit_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    for mat_dir in digit_dirs:
        i = int(mat_dir.name)
        material_id = f"style_{i}"
        albedo = mat_dir / f"{i}_albedo.jpg"
        if not albedo.is_file():
            continue
        ok = engine.register_pbr_material(
            material_id=material_id,
            albedo_path=str(albedo),
            normal_path=str(mat_dir / f"{i}_normal.jpg"),
            ao_path=str(mat_dir / f"{i}_ao.jpg"),
            rough_path=str(mat_dir / f"{i}_roughness.jpg"),
            metal_path=str(mat_dir / f"{i}_metallic.jpg"),
        )
        if ok:
            n_ok += 1
    return n_ok


# 定义前后端通信的 JSON 格式
class InitSceneRequest(BaseModel):
    obj_path: str  # 接收处理好的 obj 路径！


class MaterialChangeRequest(BaseModel):
    segment_type: str  # 例如 "walls" 或 "floor"
    segment_id: str  # 例如 "wall_0"
    material_id: str  # 例如 "style_2"


@app.post("/api/init_scene")
def api_init_scene(req: InitSceneRequest):
    """
    步骤 1：前端/总控调用，加载优化后的模型，切分空间，并返回可用房间结构。
    """
    if global_engine.load_from_file(req.obj_path):
        # 注册纯色防崩底色
        global_engine.register_color_material("default_ceiling", [220, 220, 225, 255])
        global_engine.register_color_material("default_wall", [200, 200, 200, 255])
        global_engine.register_color_material("default_floor", [40, 40, 45, 255])
        global_engine.register_color_material("default_door", [20, 20, 25, 255], metallic=0.9)
        global_engine.register_color_material("default_others", [60, 60, 65, 255])

        _register_pbr_library_for_engine(global_engine, _MATERIAL_API_ROOT)

        global_engine.restore_default_materials()

        # 将房间有几面墙、有几个门返回给前端，前端拿到后可以动态生成 UI 按钮
        return {
            "status": "success",
            "counts": global_engine.get_segment_counts(),
            "available_materials": list(global_engine.material_library.keys())
        }
    raise HTTPException(status_code=500, detail="模型加载失败，请检查金输出的 obj 路径是否正确")


@app.get("/api/status")
def api_get_status():
    """步骤 2：前端调用，获取当前每个面的材质状态"""
    return global_engine.get_material_status()


@app.post("/api/change_material")
def api_change_material(req: MaterialChangeRequest):
    """
    步骤 3：前端触发材质更换。用户在网页点击某面墙换风格时，前端发送请求到这里。
    """
    target_id = req.segment_id if req.segment_id else None
    tiling_val = 2.0 if req.segment_type == "floor" else 1.0

    success = global_engine.change_segment_material(req.segment_type, target_id, req.material_id, tiling=tiling_val)
    if success:
        return {"status": "success", "message": f"已将 {req.segment_id or req.segment_type} 替换为 {req.material_id}"}
    raise HTTPException(status_code=400, detail="材质替换失败，请检查参数")


@app.get("/api/download_glb")
def api_download_glb():
    """步骤 4：前端调用，获取拼接、换装完成后的最终 GLB 模型文件用于 Three.js 渲染"""
    out_dir = _MATERIAL_API_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "api_download_room.glb"
    global_engine.export_glb(str(out_path))
    return FileResponse(out_path, media_type="model/gltf-binary", filename="room.glb")


if __name__ == "__main__":
    print("正在启动后台材质渲染与通信引擎 (FastAPI)...")
    print("=========================================================")
    print("👉 把优化好的 obj 路径，通过 POST 传给 http://127.0.0.1:8000/api/init_scene")
    print("👉 接口文档地址在 http://127.0.0.1:8000/docs")
    print("=========================================================")
    # 启动服务器监听
    uvicorn.run(app, host="127.0.0.1", port=8000)