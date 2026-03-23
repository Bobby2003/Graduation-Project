import open3d as o3d
import numpy as np
import os
import sys
import time

# ─────────────────────────────
# 路径基准（兼容打包）
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(
    sys.executable if getattr(sys, 'frozen', False) else __file__
))

POINTCLOUD_DIR = os.path.join(BASE_DIR, "pointcloud")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ─────────────────────────────
# 选择点云文件
# ─────────────────────────────
def choose_ply_file(folder):
    if not os.path.exists(folder):
        print(f"[错误] 文件夹不存在: {folder}")
        return None

    files = [f for f in os.listdir(folder) if f.endswith(".ply")]

    if not files:
        print("[错误] pointcloud 文件夹里没有 .ply 文件")
        return None

    print("\n可用点云文件：")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    while True:
        try:
            idx = int(input("请输入要建模的文件编号: "))
            if 0 <= idx < len(files):
                return os.path.join(folder, files[idx])
            else:
                print("编号超出范围")
        except ValueError:
            print("请输入数字")

# ─────────────────────────────
# 点云 → 模型
# ─────────────────────────────
def build_mesh(ply_path):
    print(f"\n[INFO] 正在读取: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    if len(pcd.points) == 0:
        print("[错误] 点云为空")
        return

    print("[INFO] 原始点数:", len(pcd.points))

    # 1. 去噪
    print("[INFO] 去噪...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    # 2. 降采样
    print("[INFO] 降采样...")
    pcd = pcd.voxel_down_sample(0.005)
    print("[INFO] 采样后点数:", len(pcd.points))

    # 3. 法向量（更稳定）
    print("[INFO] 计算法向量...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03,
            max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(50)

    # 4. Poisson（更高分辨率）
    print("[INFO] 表面重建（Poisson）...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=10
    )

    # ✅ 5. 先去低密度区域（关键修复）
    print("[INFO] 清理低密度区域...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # ✅ 6. 再裁剪外壳（顺序必须在后）
    print("[INFO] 裁剪模型...")
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 7. 平滑（核心优化）
    print("[INFO] 平滑模型...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

    # 8. 简化（可选但推荐）
    print("[INFO] 简化模型...")
    mesh = mesh.simplify_quadric_decimation(100000)

    # 9. 保存
    print("[INFO] 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())
    base_name = os.path.basename(ply_path).replace(".ply", "")
    out_name = f"{base_name}_mesh_{timestamp}.obj"
    out_path = os.path.join(MODEL_DIR, out_name)

    o3d.io.write_triangle_mesh(out_path, mesh)

    print(f"[完成] 模型已保存: {out_path}")

    # 10. 可视化（建议保留）
    print("[INFO] 显示模型...")
    o3d.visualization.draw_geometries([mesh])


# ─────────────────────────────
# 主程序
# ─────────────────────────────
if __name__ == "__main__":
    print("=== 点云建模工具 ===")

    ply_file = choose_ply_file(POINTCLOUD_DIR)

    if ply_file:
        build_mesh(ply_file)
    else:
        print("未选择文件，程序退出")