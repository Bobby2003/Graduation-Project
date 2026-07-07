"""
点云转 Mesh 算法实时性与效果对比工具
依赖: open3d, numpy, scipy, (可选) pymeshlab

功能:
  - 采集点云并保存
  - 逐算法重建 mesh 并保存为 .ply
  - 输出性能对比表格
  - 支持一键可视化对比所有已保存模型
"""
import os
import time
import json
import numpy as np
import open3d as o3d
from datetime import datetime
from scipy.spatial import Delaunay

from end import UnifiedDepthScanner

# ─────────────────────────────────────────────
# 输出目录
# ─────────────────────────────────────────────
OUTPUT_DIR = "mesh_results"

def ensure_output_dir(session_dir):
    os.makedirs(session_dir, exist_ok=True)

def make_session_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(OUTPUT_DIR, ts)
    ensure_output_dir(d)
    return d

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def build_o3d_pcd(points, colors=None, estimate_normal=True, knn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if estimate_normal:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )
        pcd.orient_normals_consistent_tangent_plane(k=knn)
    return pcd

def mesh_stats(mesh):
    if mesh is None:
        return {"vertices": 0, "triangles": 0}
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
    }

def save_mesh(mesh, path):
    if mesh is None or len(mesh.triangles) == 0:
        return False
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)
    return True

def save_pointcloud(pcd, path):
    o3d.io.write_point_cloud(path, pcd)

def show_mesh(mesh, name):
    if mesh is None or len(mesh.triangles) == 0:
        print(f"  [{name}] 空网格，跳过显示")
        return
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], window_name=name, width=1024, height=768
    )

# ─────────────────────────────────────────────
# 重建算法
# ─────────────────────────────────────────────
def recon_poisson(pcd, depth=9):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
    return mesh

def recon_ball_pivoting(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_d = np.mean(distances)
    radii = [avg_d * r for r in (1.0, 1.5, 2.0, 3.0)]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh

def recon_alpha_shape(pcd, alpha=0.03):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

def recon_delaunay_2d(pcd):
    pts = np.asarray(pcd.points)
    tri = Delaunay(pts[:, :2])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    return mesh

def recon_convex_hull(pcd):
    mesh, _ = pcd.compute_convex_hull()
    return mesh

def recon_marching_cubes_approx(pcd, voxel_size=0.01):
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    mesh = o3d.geometry.TriangleMesh()
    for v in vg.get_voxels():
        c = vg.get_voxel_center_coordinate(v.grid_index)
        cube = o3d.geometry.TriangleMesh.create_box(
            voxel_size, voxel_size, voxel_size
        ).translate(c - voxel_size / 2)
        mesh += cube
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    return mesh

def recon_greedy_projection_pymeshlab(pcd):
    try:
        import pymeshlab
    except ImportError:
        print("  [跳过] 未安装 pymeshlab, pip install pymeshlab")
        return None
    ms = pymeshlab.MeshSet()
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    m = pymeshlab.Mesh(vertex_matrix=pts, v_normals_matrix=nrm)
    ms.add_mesh(m, "pc")
    ms.generate_surface_reconstruction_ball_pivoting()
    out = ms.current_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(out.vertex_matrix())
    mesh.triangles = o3d.utility.Vector3iVector(out.face_matrix())
    return mesh

def recon_screened_poisson_pymeshlab(pcd):
    try:
        import pymeshlab
    except ImportError:
        print("  [跳过] 未安装 pymeshlab")
        return None
    ms = pymeshlab.MeshSet()
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    m = pymeshlab.Mesh(vertex_matrix=pts, v_normals_matrix=nrm)
    ms.add_mesh(m, "pc")
    ms.generate_surface_reconstruction_screened_poisson(depth=9)
    out = ms.current_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(out.vertex_matrix())
    mesh.triangles = o3d.utility.Vector3iVector(out.face_matrix())
    return mesh

# ─────────────────────────────────────────────
# 算法注册表
# ─────────────────────────────────────────────
ALGORITHMS = {
    "1":  ("Poisson_depth9",              lambda p: recon_poisson(p, depth=9)),
    "2":  ("Poisson_depth7_fast",         lambda p: recon_poisson(p, depth=7)),
    "3":  ("BallPivoting",                recon_ball_pivoting),
    "4":  ("AlphaShape_0.03",             lambda p: recon_alpha_shape(p, 0.03)),
    "5":  ("AlphaShape_0.01_fine",        lambda p: recon_alpha_shape(p, 0.01)),
    "6":  ("Delaunay2D",                  recon_delaunay_2d),
    "7":  ("ConvexHull",                  recon_convex_hull),
    "8":  ("MarchingCubes_approx",        lambda p: recon_marching_cubes_approx(p, 0.01)),
    "9":  ("ScreenedPoisson_pymeshlab",   recon_screened_poisson_pymeshlab),
    "10": ("GreedyProjection_pymeshlab",  recon_greedy_projection_pymeshlab),
}

SPECIAL_KEYS = {"A", "V", "R", "Q"}

DL_NOTE = """
[提示] 深度学习算法 (PointNet/DeepSDF/Point2Mesh/NKSR 等)
       需要预训练模型 + GPU, 不适合实时菜单对比。
       建议用本工具保存的点云 .ply 离线跑推理后放入同目录对比。
"""

# ─────────────────────────────────────────────
# 运行 & 保存
# ─────────────────────────────────────────────
def run_one(name, fn, pcd, session_dir):
    print(f"\n>>> {name}")
    t0 = time.time()
    try:
        mesh = fn(pcd)
    except Exception as e:
        print(f"  [错误] {e}")
        return None, None

    dt = time.time() - t0
    stats = mesh_stats(mesh)
    stats["time_ms"] = round(dt * 1000, 1)

    print(f"  耗时 {stats['time_ms']} ms | "
          f"顶点 {stats['vertices']} | 三角面 {stats['triangles']}")

    # 保存 mesh
    if mesh is not None and stats["triangles"] > 0:
        path = os.path.join(session_dir, f"{name}.ply")
        save_mesh(mesh, path)
        stats["file"] = path
        print(f"  已保存 → {path}")
    else:
        stats["file"] = None

    return mesh, stats

def run_all(pcd, session_dir):
    all_results = []
    for k in sorted(ALGORITHMS.keys(), key=lambda x: int(x)):
        name, fn = ALGORITHMS[k]
        _, stats = run_one(name, fn, pcd, session_dir)
        if stats:
            all_results.append((name, stats))

    # 保存汇总 JSON
    summary_path = os.path.join(session_dir, "summary.json")
    summary = []
    for name, s in all_results:
        summary.append({
            "algorithm": name,
            "time_ms": s["time_ms"],
            "vertices": s["vertices"],
            "triangles": s["triangles"],
            "file": s.get("file"),
        })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印表格
    print("\n" + "=" * 78)
    print(f"{'算法':<32} {'耗时(ms)':>10} {'顶点':>10} {'三角面':>10} {'文件':>12}")
    print("-" * 78)
    for name, s in all_results:
        saved = "✓" if s.get("file") else "✗"
        print(f"{name:<32} {s['time_ms']:>10.1f} "
              f"{s['vertices']:>10} {s['triangles']:>10} {saved:>12}")
    print("=" * 78)
    print(f"汇总已保存 → {summary_path}")

    return all_results

def view_all_saved(session_dir):
    """加载当前 session 目录下所有 .ply mesh 做并排可视化"""
    files = sorted([
        f for f in os.listdir(session_dir)
        if f.endswith(".ply") and f != "pointcloud.ply"
    ])
    if not files:
        print("当前 session 无已保存的 mesh")
        return

    print(f"\n找到 {len(files)} 个模型, 逐个显示 (关闭窗口看下一个):")
    for fname in files:
        path = os.path.join(session_dir, fname)
        print(f"  → {fname}")
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name=fname.replace(".ply", ""),
                width=1024, height=768,
            )

    # 最后尝试全部叠加显示
    print("\n全部叠加显示...")
    geoms = []
    offset_x = 0.0
    for fname in files:
        path = os.path.join(session_dir, fname)
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0:
            continue
        mesh.compute_vertex_normals()
        # 沿 X 轴平移排列
        bb = mesh.get_axis_aligned_bounding_box()
        span = bb.get_extent()[0]
        mesh.translate([offset_x, 0, 0])
        offset_x += span * 1.3
        geoms.append(mesh)

    if geoms:
        o3d.visualization.draw_geometries(
            geoms, window_name="All Models Side-by-Side",
            width=1600, height=900,
        )

# ─────────────────────────────────────────────
# 采集
# ─────────────────────────────────────────────
def capture_pointcloud(scanner, warmup=30, min_points=5000):
    print("正在采集点云 (预热中)...")
    frame = None
    for i in range(warmup):
        frame = scanner.get_pointcloud()
        time.sleep(0.03)

    if frame is None or frame["points"] is None or len(frame["points"]) < min_points:
        print("采集失败, 点数不足")
        return None, None

    pts, col = frame["points"], frame["colors"]
    print(f"  采集完成, 点数: {len(pts)}")
    return pts, col

# ─────────────────────────────────────────────
# 菜单
# ─────────────────────────────────────────────
def print_menu():
    print("\n" + "=" * 60)
    print("  点云 → Mesh 算法对比 (模型自动保存)")
    print("=" * 60)
    for k in sorted(ALGORITHMS.keys(), key=lambda x: int(x)):
        name, _ = ALGORITHMS[k]
        print(f"  [{k:>2}]  {name}")
    print(f"  [ A]  全部跑一遍 + 输出表格")
    print(f"  [ V]  查看已保存的所有模型")
    print(f"  [ R]  重新采集一帧点云")
    print(f"  [ Q]  退出")
    print("=" * 60)

# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    print(DL_NOTE)

    scanner = UnifiedDepthScanner()
    if not scanner.init():
        print("设备初始化失败")
        return

    session_dir = make_session_dir()
    print(f"本次结果保存目录: {session_dir}")

    try:
        pts, col = capture_pointcloud(scanner)
        if pts is None:
            return

        pcd = build_o3d_pcd(pts, col, estimate_normal=True)

        # 保存原始点云
        pc_path = os.path.join(session_dir, "pointcloud.ply")
        save_pointcloud(pcd, pc_path)
        print(f"原始点云已保存 → {pc_path}")

        while True:
            print_menu()
            choice = input("选择: ").strip().upper()

            if choice == "Q":
                break

            if choice == "R":
                pts, col = capture_pointcloud(scanner)
                if pts is not None:
                    pcd = build_o3d_pcd(pts, col, estimate_normal=True)
                    session_dir = make_session_dir()
                    pc_path = os.path.join(session_dir, "pointcloud.ply")
                    save_pointcloud(pcd, pc_path)
                    print(f"新点云已保存 → {pc_path}")
                continue

            if choice == "A":
                run_all(pcd, session_dir)
                continue

            if choice == "V":
                view_all_saved(session_dir)
                continue

            if choice not in ALGORITHMS:
                print("无效选择")
                continue

            name, fn = ALGORITHMS[choice]
            mesh, _ = run_one(name, fn, pcd, session_dir)
            if mesh is not None:
                show_mesh(mesh, name)

    finally:
        scanner.close()
        print(f"\n所有结果在: {session_dir}/")
        print("已退出")

if __name__ == "__main__":
    main()