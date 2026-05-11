import trimesh
import numpy as np
import os
import math
from PIL import Image


# ==========================================
# 🚪 纯几何门框提取器子模块
# ==========================================
def extract_door_geometric(wall_mesh):
    """
    纯几何门框提取器 (高容错/参数放宽版)
    """
    print("🚪 启动纯几何门框提取算法 (高容错模式)...")

    if wall_mesh.is_empty:
        return wall_mesh, trimesh.Trimesh()

    wall_mesh.fix_normals()
    dominant_normal = np.median(wall_mesh.face_normals, axis=0)
    dominant_normal /= np.linalg.norm(dominant_normal)
    centroid = np.median(wall_mesh.vertices, axis=0)

    face_centers = wall_mesh.triangles.mean(axis=1)
    depths = np.dot(face_centers - centroid, dominant_normal)

    # ==========================================
    # 🔧 调整点 1：极限放宽深度容差
    # ==========================================
    # 即使门只凹陷/凸起了 1cm，或者深深凹进去了 40cm，都算数！
    MIN_DOOR_DEPTH = 0.015  # 原为 0.02
    MAX_DOOR_DEPTH = 0.30  # 原为 0.25

    door_face_mask = (np.abs(depths) > MIN_DOOR_DEPTH) & (np.abs(depths) < MAX_DOOR_DEPTH)

    door_candidate = trimesh.Trimesh(vertices=wall_mesh.vertices, faces=wall_mesh.faces[door_face_mask])
    door_candidate.remove_unreferenced_vertices()

    main_wall = trimesh.Trimesh(vertices=wall_mesh.vertices, faces=wall_mesh.faces[~door_face_mask])
    main_wall.remove_unreferenced_vertices()

    if door_candidate.is_empty:
        print(f"❌ 深度判定失败：墙面上没有任何 {MIN_DOOR_DEPTH * 100}cm 到 {MAX_DOOR_DEPTH * 100}cm 之间的起伏。")
        return wall_mesh, trimesh.Trimesh()

    print("📏 发现起伏区域，正在进行高度与碎片过滤...")
    door_parts = door_candidate.split(only_watertight=False)
    true_doors = []

    # ==========================================
    # 🔧 调整点 2：碎片化宽容检测
    # ==========================================
    MIN_PART_AREA = 0.1  # 面积大于 0.1 平方米才算数 (过滤墙面微小噪点)

    for part in door_parts:
        min_y = part.bounds[0][1]
        max_y = part.bounds[1][1]

        # 逻辑放宽：
        # 1. 面积不能太小 (排除纯噪点)
        # 2. 不能悬在天花板上 (最低点必须低于 1.5 米)
        # 3. 满足以下其一即可：要么接地(最低点<0.6米)，要么够高(最高点>1.2米)

        is_large_enough = part.area > MIN_PART_AREA
        is_not_ceiling_noise = min_y < 1.5
        is_door_like_height = (min_y < 0.6) or (max_y > 1.2)

        if is_large_enough and is_not_ceiling_noise and is_door_like_height:
            true_doors.append(part)

    if len(true_doors) > 0:
        final_door = trimesh.util.concatenate(true_doors)
        print(f"🎯 纯几何命中！成功提取出深度差异区，面积共计: {final_door.area:.2f} 平方米。")

        false_doors = [p for p in door_parts if p not in true_doors]
        if false_doors:
            main_wall = trimesh.util.concatenate([main_wall] + false_doors)

        return main_wall, final_door
    else:
        print("❌ 提取失败：起伏区域全都是面积太小的噪点，或者位置悬在半空中（可能是相框或窗户）。")
        return wall_mesh, trimesh.Trimesh()

# ==========================================
# 🚀 核心主处理管线
# ==========================================
def process_scanned_room(obj_path, output_path):
    print(f"🚀 正在导入模型: {obj_path}")

    # 1. 加载模型
    mesh = trimesh.load(obj_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return
        mesh = list(mesh.geometry.values())[0]

    # 清理遗留材质
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    # ==========================================
    # 阶段一：法向量初筛 (分离垂直与水平面)
    # ==========================================
    print("🔪 阶段一：启动法向量初筛算法...")
    mesh.fix_normals()
    vertical_components = np.abs(mesh.face_normals[:, 1])

    # 提取所有垂直面
    WALL_THRESHOLD = 0.3
    wall_mask = vertical_components < WALL_THRESHOLD

    all_vertical_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[wall_mask])
    all_vertical_mesh.remove_unreferenced_vertices()

    floor_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[~wall_mask])
    floor_mesh.remove_unreferenced_vertices()

    # ==========================================
    # 阶段二：剔除衣柜与垂直家具 (面积 + 绝对位置)
    # ==========================================
    print("🔍 阶段二：启动连通域分析，分离真墙壁与家具...")
    room_bounds = mesh.bounds
    min_bound, max_bound = room_bounds[0], room_bounds[1]

    vertical_parts = all_vertical_mesh.split(only_watertight=False)

    true_walls = []
    furniture_verticals = []

    MIN_WALL_AREA = 0.5
    BOUNDARY_TOLERANCE = 1

    for comp in vertical_parts:
        comp_bounds = comp.bounds
        comp_min, comp_max = comp_bounds[0], comp_bounds[1]

        is_large_enough = comp.area >= MIN_WALL_AREA
        touches_outer_shell = (
                abs(comp_min[0] - min_bound[0]) < BOUNDARY_TOLERANCE or
                abs(comp_max[0] - max_bound[0]) < BOUNDARY_TOLERANCE or
                abs(comp_min[2] - min_bound[2]) < BOUNDARY_TOLERANCE or
                abs(comp_max[2] - max_bound[2]) < BOUNDARY_TOLERANCE
        )

        if is_large_enough and touches_outer_shell:
            true_walls.append(comp)
        else:
            furniture_verticals.append(comp)

    initial_wall_mesh = trimesh.util.concatenate(true_walls) if true_walls else trimesh.Trimesh()

    if furniture_verticals:
        floor_and_furniture_mesh = trimesh.util.concatenate([floor_mesh] + furniture_verticals)
    else:
        floor_and_furniture_mesh = floor_mesh

    # ==========================================
    # 阶段三：从真墙壁中提取门框
    # ==========================================
    print("🚪 阶段三：进行几何结构解析...")
    door_mesh, final_wall_mesh = extract_door_geometric(initial_wall_mesh)

    # ==========================================
    # 阶段四：PBR 材质生成与合并
    # ==========================================
    print("🎨 阶段四：正在组装 PBR 材质...")
    base_dir = r"C:\毕设\材质\Space Blanket Folds_1K"
    path_albedo = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_albedo.jpg")
    path_ao = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_ao.jpg")
    path_metallic = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_metallic.jpg")
    path_normal = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_normal.jpg")
    path_roughness = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_roughness.jpg")

    # ORM 合并通道
    img_ao = Image.open(path_ao).convert('L')
    img_rough = Image.open(path_roughness).convert('L')
    img_metal = Image.open(path_metallic).convert('L')
    target_size = img_ao.size
    if img_rough.size != target_size: img_rough = img_rough.resize(target_size)
    if img_metal.size != target_size: img_metal = img_metal.resize(target_size)
    orm_image = Image.merge('RGB', (img_ao, img_rough, img_metal))

    # 墙面：科幻装甲
    scifi_material = trimesh.visual.material.PBRMaterial(
        name="True_Wall",
        baseColorTexture=Image.open(path_albedo).convert('RGB'),
        normalTexture=Image.open(path_normal),
        metallicRoughnessTexture=orm_image,
        occlusionTexture=orm_image
    )

    # 地板及家具：深色基础磨砂
    floor_material = trimesh.visual.material.PBRMaterial(
        name="Floor_And_Furniture", baseColorFactor=[40, 40, 45, 255], metallicFactor=0.2, roughnessFactor=0.8
    )

    # 舱门：深邃金属反射
    door_material = trimesh.visual.material.PBRMaterial(
        name="SciFi_Door", baseColorFactor=[20, 20, 25, 255], metallicFactor=0.9, roughnessFactor=0.2
    )

    # ==========================================
    # 阶段五：物理 UV 计算与装配
    # ==========================================
    print("📏 阶段五：计算绝对物理 UV...")
    tiling = 1.0

    if not final_wall_mesh.is_empty:
        wall_uvs = np.zeros((len(final_wall_mesh.vertices), 2))
        wall_uvs[:, 0] = (final_wall_mesh.vertices[:, 0] + final_wall_mesh.vertices[:, 2]) * tiling
        wall_uvs[:, 1] = final_wall_mesh.vertices[:, 1] * tiling
        final_wall_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=wall_uvs, material=scifi_material)

    if not floor_and_furniture_mesh.is_empty:
        floor_and_furniture_mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.zeros((len(floor_and_furniture_mesh.vertices), 2)), material=floor_material
        )

    if not door_mesh.is_empty:
        door_mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.zeros((len(door_mesh.vertices), 2)), material=door_material
        )

    # ==========================================
    # 📦 重新打包并导出
    # ==========================================
    print("📦 正在重新拼装并导出场景...")
    scene_elements = [m for m in [final_wall_mesh, door_mesh, floor_and_furniture_mesh] if not m.is_empty]
    final_scene = trimesh.Scene(scene_elements)
    final_scene.export(output_path)
    print(f"✅ MR 空间重组大功告成! 已保存至: {output_path}")


# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    teammate_obj = r"C:\毕设\imu_fusion_model_final_20260413_193242_refined.ply"
    final_output = r"C:\毕设\Final_MR_Room.glb"

    if os.path.exists(teammate_obj):
        process_scanned_room(teammate_obj, final_output)
    else:
        print(f"❌ 找不到输入模型: {teammate_obj}")