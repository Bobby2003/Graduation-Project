import trimesh
import numpy as np
import os
from PIL import Image


def apply_scifi_pbr_generate_uv(obj_path, output_path):
    print(f"🚀 正在加载队友的扫描模型: {obj_path}")

    # 1. 加载模型 (force='mesh' 更适合处理纯粹的 .obj 文件)
    mesh = trimesh.load(obj_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            print("❌ 错误：空模型！")
            return
        mesh = list(mesh.geometry.values())[0]

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    # 2. 精确的 JPG 材质贴图路径
    base_dir = r"C:\毕设\材质\TCom_Scifi_Panel_1K"
    path_albedo = os.path.join(base_dir, "TCom_Scifi_Panel_1K_albedo.jpg")
    path_ao = os.path.join(base_dir, "TCom_Scifi_Panel_1K_ao.jpg")
    path_metallic = os.path.join(base_dir, "TCom_Scifi_Panel_1K_metallic.jpg")
    path_normal = os.path.join(base_dir, "TCom_Scifi_Panel_1K_normal.jpg")
    path_roughness = os.path.join(base_dir, "TCom_Scifi_Panel_1K_roughness.jpg")

    for p in [path_albedo, path_ao, path_metallic, path_normal, path_roughness]:
        if not os.path.exists(p):
            print(f"❌ 致命错误：找不到贴图文件 -> {p}")
            return

    # 3. 强行合并 ORM 贴图
    print("正在合并 AO、粗糙度、金属度为标准 ORM 贴图...")
    img_ao = Image.open(path_ao).convert('L')
    img_rough = Image.open(path_roughness).convert('L')
    img_metal = Image.open(path_metallic).convert('L')

    target_size = img_ao.size
    if img_rough.size != target_size: img_rough = img_rough.resize(target_size)
    if img_metal.size != target_size: img_metal = img_metal.resize(target_size)
    orm_image = Image.merge('RGB', (img_ao, img_rough, img_metal))

    print("正在压入 PBR 材质...")


    img_albedo_opaque = Image.open(path_albedo).convert('RGB')

    material = trimesh.visual.material.PBRMaterial(
        name="SciFi_Room",
        baseColorTexture=img_albedo_opaque,
        normalTexture=Image.open(path_normal),
        metallicRoughnessTexture=orm_image,
        occlusionTexture=orm_image
    )

    # ==========================================
    # 通过绝对物理坐标算 UV
    # ==========================================
    print("正在通过绝对物理坐标生成无缝 UV 映射...")
    vertices = mesh.vertices

    # 关键参数调整 (Tiling)：
    tiling = 1.0

    uvs = np.zeros((len(vertices), 2))

    # 将现实世界的 X 轴和 Z 轴的坐标相加作为水平贴图 (U)
    # 将现实世界的 Y 轴作为垂直贴图 (V)
    uvs[:, 0] = (vertices[:, 0] + vertices[:, 2]) * tiling
    uvs[:, 1] = vertices[:, 1] * tiling

    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uvs,
        material=material
    )

    # 4. 导出为现代引擎最友好的 GLB 格式
    print(f"正在打包并导出 GLB...")
    mesh.export(output_path)
    print(f"✅ 材质替换大功告成！已保存至: {output_path}")


# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    teammate_obj = r"C:\毕设\imu_fusion_model_final_20260413_193242.obj"

    # 输出的最终结果文件
    final_output = r"C:\毕设\Final_Scifi_Room.glb"

    if os.path.exists(teammate_obj):
        apply_scifi_pbr_generate_uv(teammate_obj, final_output)
    else:
        print(f"❌ 找不到输入模型: {teammate_obj}")