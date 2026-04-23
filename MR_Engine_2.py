import trimesh
import numpy as np
import os
import math
from PIL import Image


def process_scanned_room(obj_path, output_path):
    print(f"🚀 正在加载: {obj_path}")

    # 1. 加载模型
    mesh = trimesh.load(obj_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return
        mesh = list(mesh.geometry.values())[0]

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    print("正在启动法向量切割算法...")
    mesh.fix_normals()
    face_normals = mesh.face_normals

    vertical_components = np.abs(face_normals[:, 1])

    # 阈值判定
    WALL_THRESHOLD = 0.3
    wall_mask = vertical_components < WALL_THRESHOLD
    floor_desk_mask = ~wall_mask

    # 物理切割网格
    wall_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[wall_mask])
    wall_mesh.remove_unreferenced_vertices()

    floor_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[floor_desk_mask])
    floor_mesh.remove_unreferenced_vertices()

    print(f"✅ 分割完毕！墙壁面数: {len(wall_mesh.faces)}, 地板及桌椅面数: {len(floor_mesh.faces)}")

    # ==========================================
    # 材质组装
    # ==========================================
    base_dir = r"C:\毕设\材质\TCom_Scifi_Panel_1K"
    path_albedo = os.path.join(base_dir, "TCom_Scifi_Panel_1K_albedo.jpg")
    path_ao = os.path.join(base_dir, "TCom_Scifi_Panel_1K_ao.jpg")
    path_metallic = os.path.join(base_dir, "TCom_Scifi_Panel_1K_metallic.jpg")
    path_normal = os.path.join(base_dir, "TCom_Scifi_Panel_1K_normal.jpg")
    path_roughness = os.path.join(base_dir, "TCom_Scifi_Panel_1K_roughness.jpg")

    for p in [path_albedo, path_ao, path_metallic, path_normal, path_roughness]:
        if not os.path.exists(p):
            print(f"❌ 找不到贴图文件: {p}")
            return

    # 合并 ORM 贴图
    print("正在合并 ORM 贴图...")
    img_ao = Image.open(path_ao).convert('L')
    img_rough = Image.open(path_roughness).convert('L')
    img_metal = Image.open(path_metallic).convert('L')
    target_size = img_ao.size
    if img_rough.size != target_size: img_rough = img_rough.resize(target_size)
    if img_metal.size != target_size: img_metal = img_metal.resize(target_size)
    orm_image = Image.merge('RGB', (img_ao, img_rough, img_metal))

    scifi_material = trimesh.visual.material.PBRMaterial(
        name="SciFi_Wall",
        baseColorTexture=Image.open(path_albedo).convert('RGB'),
        normalTexture=Image.open(path_normal),
        metallicRoughnessTexture=orm_image,
        occlusionTexture=orm_image
    )

    floor_material = trimesh.visual.material.PBRMaterial(
        name="Dark_Floor",
        baseColorFactor=[40, 40, 45, 255],  # 偏暗蓝的深灰色
        metallicFactor=0.2,
        roughnessFactor=0.8
    )

    # ==========================================
    #  为墙壁计算物理 UV
    # ==========================================
    print("正在为墙面生成物理级无缝 UV...")
    tiling = 1.0

    wall_uvs = np.zeros((len(wall_mesh.vertices), 2))
    wall_uvs[:, 0] = (wall_mesh.vertices[:, 0] + wall_mesh.vertices[:, 2]) * tiling
    wall_uvs[:, 1] = wall_mesh.vertices[:, 1] * tiling

    wall_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=wall_uvs, material=scifi_material)

    floor_mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(floor_mesh.vertices), 2)), material=floor_material
    )

    # ==========================================
    #  重新打包并导出
    # ==========================================
    print("正在将墙壁与地板/桌椅重新拼装并导出...")
    final_scene = trimesh.Scene([wall_mesh, floor_mesh])
    final_scene.export(output_path)
    print(f"✅ MR 空间重组完成！已保存至: {output_path}")


# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    teammate_obj = r"C:\毕设\imu_fusion_model_final_20260413_193242.obj"

    # 输出的最终结果文件
    final_output = r"C:\毕设\Final_MR_Room.glb"

    if os.path.exists(teammate_obj):
        process_scanned_room(teammate_obj, final_output)
    else:
        print(f"❌ 找不到输入模型: {teammate_obj}")