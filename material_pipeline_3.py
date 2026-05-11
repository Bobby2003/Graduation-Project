import trimesh
import numpy as np
import os
from PIL import Image


def process_scanned_room(obj_path, output_path):
    print(f"🚀 正在导入的模型: {obj_path}")

    # 1. 加载模型
    mesh = trimesh.load(obj_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return
        mesh = list(mesh.geometry.values())[0]

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    print("🔪 正在启动法向量初筛算法 (分离垂直与水平面)...")
    mesh.fix_normals()
    face_normals = mesh.face_normals

    vertical_components = np.abs(face_normals[:, 1])

    # 阈值判定: 提取所有的垂直面 (包含墙壁、门、衣柜侧面)
    WALL_THRESHOLD = 0.3
    wall_mask = vertical_components < WALL_THRESHOLD
    floor_desk_mask = ~wall_mask

    # 物理切割网格
    all_vertical_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[wall_mask])
    all_vertical_mesh.remove_unreferenced_vertices()

    floor_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[floor_desk_mask])
    floor_mesh.remove_unreferenced_vertices()

    # ==========================================
    # 💡 核心升级：面积 + 绝对位置 双重狙击！
    # ==========================================
    print("🔍 正在启动连通域分析，踢出家具与衣柜...")

    # 获取整个房间的绝对物理边界 (Min X/Y/Z, Max X/Y/Z)
    room_bounds = mesh.bounds
    min_bound, max_bound = room_bounds[0], room_bounds[1]

    # 打碎所有垂直面为独立的物理区块
    vertical_parts = all_vertical_mesh.split(only_watertight=False)

    true_walls = []
    furniture_verticals = []

    # ⚠️ 参数调整区
    MIN_WALL_AREA = 0.5  # 面积阈值：小于 1.5 的垂直面被认为是家具 (可根据真实比例微调)
    BOUNDARY_TOLERANCE = 1  # 边界容忍度：距离房间最外壳 1 米以内的才算墙壁

    for comp in vertical_parts:
        comp_bounds = comp.bounds
        comp_min, comp_max = comp_bounds[0], comp_bounds[1]

        # 1. 面积检测
        is_large_enough = comp.area >= MIN_WALL_AREA

        # 2. 绝对位置检测 (判断该面的边缘是否贴近房间的最外侧 X 或 Z 轴边界)
        # 假设房间的长宽沿 X 和 Z 轴分布
        touches_outer_shell = (
                abs(comp_min[0] - min_bound[0]) < BOUNDARY_TOLERANCE or
                abs(comp_max[0] - max_bound[0]) < BOUNDARY_TOLERANCE or
                abs(comp_min[2] - min_bound[2]) < BOUNDARY_TOLERANCE or
                abs(comp_max[2] - max_bound[2]) < BOUNDARY_TOLERANCE
        )

        # 必须同时满足“面积够大”且“位于房间最外围”
        if is_large_enough and touches_outer_shell:
            true_walls.append(comp)
        else:
            furniture_verticals.append(comp)

    print(f"✅ 过滤完毕！发现 {len(true_walls)} 面真墙壁，剔除了 {len(furniture_verticals)} 个衣柜/家具垂直面。")

    # 重新拼装真正的墙壁
    if len(true_walls) > 0:
        final_wall_mesh = trimesh.util.concatenate(true_walls)
    else:
        print("⚠️ 警告：未检测到符合条件的真墙壁！")
        final_wall_mesh = trimesh.Trimesh()

    # 将踢出来的衣柜侧面，和地板桌面合并在一起，准备统一涂成暗色
    if len(furniture_verticals) > 0:
        floor_and_furniture_mesh = trimesh.util.concatenate([floor_mesh] + furniture_verticals)
    else:
        floor_and_furniture_mesh = floor_mesh

    # ==========================================
    # 🎨 材质组装
    # ==========================================
    base_dir = r"C:\毕设\材质\Space Blanket Folds_1K"
    path_albedo = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_albedo.jpg")
    path_ao = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_ao.jpg")
    path_metallic = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_metallic.jpg")
    path_normal = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_normal.jpg")
    path_roughness = os.path.join(base_dir, "TCom_Plastic_SpaceBlanketFolds_1K_roughness.jpg")

    for p in [path_albedo, path_ao, path_metallic, path_normal, path_roughness]:
        if not os.path.exists(p):
            print(f"❌ 找不到贴图文件: {p}")
            return

    print("🔧 正在合并 ORM 贴图...")
    img_ao = Image.open(path_ao).convert('L')
    img_rough = Image.open(path_roughness).convert('L')
    img_metal = Image.open(path_metallic).convert('L')
    target_size = img_ao.size
    if img_rough.size != target_size: img_rough = img_rough.resize(target_size)
    if img_metal.size != target_size: img_metal = img_metal.resize(target_size)
    orm_image = Image.merge('RGB', (img_ao, img_rough, img_metal))

    scifi_material = trimesh.visual.material.PBRMaterial(
        name="True_Wall",
        baseColorTexture=Image.open(path_albedo).convert('RGB'),
        normalTexture=Image.open(path_normal),
        metallicRoughnessTexture=orm_image,
        occlusionTexture=orm_image
    )

    floor_material = trimesh.visual.material.PBRMaterial(
        name="Floor_And_Furniture",
        baseColorFactor=[40, 40, 45, 255],  # 偏暗蓝的深灰色
        metallicFactor=0.2,
        roughnessFactor=0.8
    )

    # ==========================================
    # 📏 为墙壁计算物理 UV
    # ==========================================
    print("📏 正在为墙面生成无缝 UV...")
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

    # ==========================================
    # 📦 重新打包并导出
    # ==========================================
    print("📦 正在重新拼装并导出...")
    final_scene = trimesh.Scene([final_wall_mesh, floor_and_furniture_mesh])
    final_scene.export(output_path)
    print(f"✅ 空间重组完成! 已保存至: {output_path}")


# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    teammate_obj = r"C:\毕设\imu_fusion_model_final_20260413_193242_refined.ply"
    final_output = r"C:\毕设\1.glb"

    if os.path.exists(teammate_obj):
        process_scanned_room(teammate_obj, final_output)
    else:
        print(f"❌ 找不到输入模型: {teammate_obj}")
