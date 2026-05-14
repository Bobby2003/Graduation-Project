import trimesh
import numpy as np
import os
from PIL import Image


class MRMaterialEngine:

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

    def _extract_doors_and_clean_walls(self, wall_mesh):
        if wall_mesh.is_empty: return wall_mesh, [], []
        wall_mesh.fix_normals()

        dominant_normal = np.median(wall_mesh.face_normals, axis=0)
        norm = np.linalg.norm(dominant_normal)
        if norm > 1e-6:
            dominant_normal /= norm
        else:
            dominant_normal = np.array([0.0, 1.0, 0.0])

        centroid = np.median(wall_mesh.vertices, axis=0)
        depths = np.dot(wall_mesh.triangles.mean(axis=1) - centroid, dominant_normal)

        door_face_mask = (np.abs(depths) > 0.1) & (np.abs(depths) < 0.25)
        door_candidate = trimesh.Trimesh(vertices=wall_mesh.vertices, faces=wall_mesh.faces[door_face_mask])
        door_candidate.remove_unreferenced_vertices()

        main_wall = trimesh.Trimesh(vertices=wall_mesh.vertices, faces=wall_mesh.faces[~door_face_mask])
        main_wall.remove_unreferenced_vertices()

        true_doors = []
        false_doors = []

        if not door_candidate.is_empty:
            for part in door_candidate.split(only_watertight=False):
                min_y, max_y = part.bounds[0][1], part.bounds[1][1]
                if part.area > 0.1 and min_y < 1.5 and (min_y < 0.6 or max_y > 1.2):
                    true_doors.append(part)
                else:
                    false_doors.append(part)

        return main_wall, true_doors, false_doors

    def _process_geometry(self, mesh):
        """核心处理管线：物理级防割裂重组 + 平面共面聚类 (Co-planar Clustering)"""
        mesh.fix_normals()
        y_normals = mesh.face_normals[:, 1]
        face_centers = mesh.triangles.mean(axis=1)

        global_min_y = mesh.bounds[0][1]
        global_max_y = mesh.bounds[1][1]

        wall_mask = np.abs(y_normals) < 0.3

        # 天花板：位于 Y 值最大处附近 (global_max_y)，法线指向上方 (Y 的负方向 <-0.7)
        ceiling_mask = (y_normals < -0.7) & (face_centers[:, 1] > global_max_y - 0.3)

        # 地板：位于 Y 值最小处附近 (global_min_y)，法线指向下方 (Y 的正方向 >0.7)
        floor_mask = (y_normals > 0.7) & (face_centers[:, 1] < global_min_y + 0.3)

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

        # ==========================================
        # 共面几何聚类算法 (Co-planar Clustering)
        # ==========================================
        logical_walls = []  # 记录真实的物理墙面集合
        if not main_wall.is_empty:
            for part in main_wall.split(only_watertight=False):
                if part.area < MIN_WALL_AREA:
                    others_pieces.append(part)
                    continue

                # 1. 提取这个碎片的法向量 (主朝向)
                part.fix_normals()
                normal = np.median(part.face_normals, axis=0)
                normal[1] = 0
                n_len = np.linalg.norm(normal)
                if n_len > 1e-6:
                    normal /= n_len
                else:
                    normal = np.array([1.0, 0.0, 0.0])

                # 2. 提取这个碎片的绝对空间位置 (平面深度距离 d)
                centroid = np.median(part.vertices, axis=0)
                d = np.dot(normal, centroid)

                # 3. 与已有的物理墙面进行对比，寻找归属
                matched = False
                for w_data in logical_walls:
                    ref_n = w_data['normal']
                    ref_d = w_data['d']

                    dot_n = np.dot(ref_n, normal)
                    # 如果法线夹角小于 31度 (dot > 0.85)，或者完全反过来 (扫描仪反转误差)
                    if abs(dot_n) > 0.85:
                        d_adj = d if dot_n > 0 else -d
                        if abs(ref_d - d_adj) < 0.5:
                            w_data['parts'].append(part)
                            matched = True
                            break

                # 如果没有匹配的，说明是一面全新的物理墙
                if not matched:
                    logical_walls.append({'normal': normal, 'd': d, 'parts': [part]})

            # 4. 把聚类好的碎片全部物理合并
            for i, w_data in enumerate(logical_walls):
                merged_logical_wall = trimesh.util.concatenate(w_data['parts'])
                self.segments["walls"][f"wall_{i}"] = merged_logical_wall
                print(f"  - 🧱 聚类生成实体墙: wall_{i} (由 {len(w_data['parts'])} 个碎片缝合而成)")

        # 处理 others
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

        print("空间几何解析完成")
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
        print("正在恢复房间默认防崩底色...")
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
# 交互演示
# ==========================================
if __name__ == "__main__":
    engine = MRMaterialEngine()

    if engine.load_from_file(r"C:\毕设\1.obj"):
        # 1. 注册兜底纯色材质
        engine.register_color_material("default_ceiling", [220, 220, 225, 255])
        engine.register_color_material("default_wall", [200, 200, 200, 255])
        engine.register_color_material("default_floor", [40, 40, 45, 255])
        engine.register_color_material("default_door", [20, 20, 25, 255], metallic=0.9)
        engine.register_color_material("default_others", [60, 60, 65, 255])

        # 2. 批量注册 PBR 材质
        for i in [1, 2, 3]:
            mat_dir = rf"C:\毕设\材质\{i}"
            if os.path.exists(mat_dir):
                engine.register_pbr_material(
                    material_id=f"style_{i}",
                    albedo_path=os.path.join(mat_dir, f"{i}_albedo.jpg"),
                    normal_path=os.path.join(mat_dir, f"{i}_normal.jpg"),
                    ao_path=os.path.join(mat_dir, f"{i}_ao.jpg"),
                    rough_path=os.path.join(mat_dir, f"{i}_roughness.jpg"),
                    metal_path=os.path.join(mat_dir, f"{i}_metallic.jpg")
                )

        # 3. 初始化防崩底色
        engine.restore_default_materials()

        # ==========================================
        # 历替换逻辑
        # ==========================================
        print(f"\n✅ 几何解析完毕！")

        available_materials = list(engine.material_library.keys())
        print("\n可选材质列表:")
        for idx, mat_name in enumerate(available_materials):
            print(f"  [{idx}] {mat_name}")

        # 收集所有需要替换的部位
        targets = []
        if engine.segments.get("floor"): targets.append(("floor", None, "🟩 地板"))
        if engine.segments.get("ceiling"): targets.append(("ceiling", None, "⬜ 天花板"))
        for w_id in engine.segments["walls"].keys():
            targets.append(("walls", w_id, f"墙面 {w_id}"))

        print("\n[开始分配材质] (输入材质编号即可，直接回车保持默认，输入 q 提前结束)")

        for seg_type, seg_id, display_name in targets:
            user_input = input(f"为【{display_name}】选择材质编号: ").strip()

            if user_input.lower() == 'q':
                print("结束分配，剩余结构保持默认。")
                break

            if user_input == '':
                continue  # 直接回车跳过

            try:
                mat_idx = int(user_input)
                if 0 <= mat_idx < len(available_materials):
                    target_mat = available_materials[mat_idx]

                    if seg_id:
                        engine.change_segment_material(seg_type, seg_id, target_mat, tiling=1.0)
                    else:
                        tiling_val = 2.0 if seg_type == "floor" else 1.0
                        engine.change_type_material(seg_type, target_mat, tiling=tiling_val)

                    print(f"   ✨ 已应用 {target_mat}")
                else:
                    print("   ⚠️ 编号超限，保持默认。")
            except ValueError:
                print("   ⚠️ 只能输入数字，保持默认。")

        # ==========================================
        # 导出结果
        # ==========================================
        print("\n正在生成最终混合模型...")
        engine.export_glb(r"C:\毕设\2.glb")
        print("导出成功！")