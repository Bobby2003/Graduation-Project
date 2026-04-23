import numpy as np
import os
import random
import trimesh
import trimesh.creation as tc

# ─── 输出目录 ───────────────────────────────────────────────
STL_DIR = "dataset/stl"
PCD_DIR = "dataset/pointcloud"
os.makedirs(STL_DIR, exist_ok=True)
os.makedirs(PCD_DIR, exist_ok=True)

NUM_SAMPLES = 1000
NUM_PCD_POINTS = 4096  # 输入残缺点云点数（模拟扫描）
NUM_GT_POINTS = 409600  # GT 完整表面采样（用于后续训练 Mesh 重建也够用）


# ─── 基础形状生成函数 ────────────────────────────────────────

def make_box(scale=None):
    if scale is None:
        scale = np.random.uniform(0.3, 2.0, 3)
    return tc.box(extents=scale)


def make_sphere(radius=None):
    if radius is None:
        radius = np.random.uniform(0.3, 1.5)
    return tc.icosphere(subdivisions=3, radius=radius)


def make_cylinder(radius=None, height=None):
    if radius is None:
        radius = np.random.uniform(0.2, 1.0)
    if height is None:
        height = np.random.uniform(0.4, 2.0)
    return tc.cylinder(radius=radius, height=height, sections=40)


def make_cone(radius=None, height=None):
    if radius is None:
        radius = np.random.uniform(0.2, 1.0)
    if height is None:
        height = np.random.uniform(0.5, 2.0)
    return tc.cone(radius=radius, height=height, sections=40)


def make_capsule():
    r = np.random.uniform(0.2, 0.8)
    h = np.random.uniform(0.5, 2.0)
    cyl = tc.cylinder(radius=r, height=h, sections=40)
    sp1 = tc.icosphere(subdivisions=3, radius=r)
    sp2 = tc.icosphere(subdivisions=3, radius=r)
    sp1.apply_translation([0, 0, h / 2])
    sp2.apply_translation([0, 0, -h / 2])
    return trimesh.util.concatenate([cyl, sp1, sp2])


def make_torus():
    major = np.random.uniform(0.5, 1.5)
    minor = np.random.uniform(0.1, major * 0.4)
    return tc.torus(major_radius=major, minor_radius=minor)


def make_ellipsoid():
    r = np.random.uniform(0.3, 1.5, 3)
    m = tc.icosphere(subdivisions=3, radius=1.0)
    m.apply_scale(r)
    return m


# ─── 随机变换 ────────────────────────────────────────────────

def random_transform(mesh):
    angle = np.random.uniform(0, 2 * np.pi, 3)
    Rx = trimesh.transformations.rotation_matrix(angle[0], [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(angle[1], [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(angle[2], [0, 0, 1])
    mesh.apply_transform(Rx @ Ry @ Rz)
    return mesh


# ─── 组合形状 ────────────────────────────────────────────────

PRIMITIVES = [make_box, make_sphere, make_cylinder, make_cone,
              make_capsule, make_ellipsoid, make_torus]


def make_combined():
    n = random.randint(2, 4)
    parts = []
    offsets = np.random.uniform(-1.2, 1.2, (n, 3))
    for i in range(n):
        fn = random.choice(PRIMITIVES)
        mesh = fn()
        mesh = random_transform(mesh)
        mesh.apply_translation(offsets[i])
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


# ─── 形状类型路由 ────────────────────────────────────────────

SHAPE_TYPES = [
    "box", "sphere", "cylinder", "cone",
    "capsule", "ellipsoid", "torus",
    "combined"
]

WEIGHTS = [0.08, 0.08, 0.08, 0.08,
           0.08, 0.08, 0.07,
           0.45]


def make_shape(shape_type):
    if shape_type == "box":
        return make_box()
    elif shape_type == "sphere":
        return make_sphere()
    elif shape_type == "cylinder":
        return make_cylinder()
    elif shape_type == "cone":
        return make_cone()
    elif shape_type == "capsule":
        return make_capsule()
    elif shape_type == "ellipsoid":
        return make_ellipsoid()
    elif shape_type == "torus":
        return make_torus()
    else:
        return make_combined()


# ─── 归一化 ─────────────────────────────────────────────────

def normalize_mesh(mesh):
    center = mesh.vertices.mean(axis=0)
    mesh.apply_translation(-center)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh


# ─── ✅ 模拟深度相机扫描（核心改动）──────────────────────────

def simulate_depth_scan(mesh, n_points, num_views=None):
    """
    模拟深度相机扫描：
    - 随机 1~3 个视角
    - 每个视角只保留法线朝向相机的点（可见面）
    - 添加深度相机特有噪声

    返回模拟的残缺点云
    """
    if num_views is None:
        num_views = random.randint(1, 3)

    all_visible_pts = []

    for _ in range(num_views):
        # 随机一个相机观察方向（从外部看物体）
        view_dir = np.random.randn(3).astype(np.float32)
        view_dir /= np.linalg.norm(view_dir)

        # 过采样
        oversample = max(n_points * 4 // num_views, 5000)
        pts, face_idx = trimesh.sample.sample_surface(mesh, oversample)

        # 获取每个点对应面的法线
        normals = mesh.face_normals[face_idx]

        # 只保留朝向相机的点（法线与观察方向点积 > 0）
        dots = np.dot(normals, view_dir)
        visible_mask = dots > 0

        # 模拟边缘衰减：越接近掠射角(dot接近0)的点越容易丢失
        edge_prob = np.clip(dots[visible_mask] / 0.3, 0, 1)
        keep_mask = np.random.random(len(edge_prob)) < edge_prob

        pts_visible = pts[visible_mask][keep_mask]

        if len(pts_visible) > 0:
            all_visible_pts.append(pts_visible)

    if len(all_visible_pts) == 0:
        # 极端情况回退：全表面采样
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        return pts.astype(np.float32)

    all_visible_pts = np.concatenate(all_visible_pts, axis=0)

    # 采样到目标点数
    if len(all_visible_pts) >= n_points:
        idx = np.random.choice(len(all_visible_pts), n_points, replace=False)
    else:
        idx = np.random.choice(len(all_visible_pts), n_points, replace=True)

    result = all_visible_pts[idx].astype(np.float32)

    # 添加深度相机噪声（比均匀高斯更真实）
    # 1. 全局高斯噪声
    noise_std = np.random.uniform(0.002, 0.008)
    result += np.random.normal(0, noise_std, result.shape).astype(np.float32)

    # 2. 偶尔的离群飞点（模拟深度相机边缘飞点）
    n_outliers = int(n_points * np.random.uniform(0.0, 0.02))
    if n_outliers > 0:
        outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
        result[outlier_idx] += np.random.normal(0, 0.05, (n_outliers, 3)).astype(np.float32)

    return result


def save_pointcloud_ply(pts, filepath):
    pc = trimesh.PointCloud(pts)
    pc.export(filepath)


# ─── 主循环 ──────────────────────────────────────────────────

def main():
    print(f"开始生成 {NUM_SAMPLES} 个样本（模拟深度相机扫描）...")
    print(f"  输入点云: {NUM_PCD_POINTS} 点（单面/多面扫描）")
    print(f"  GT 表面:  {NUM_GT_POINTS} 点（完整表面）")

    shape_choices = random.choices(SHAPE_TYPES, weights=WEIGHTS, k=NUM_SAMPLES)

    success = 0
    fail = 0
    view_stats = {1: 0, 2: 0, 3: 0}

    for i, shape_type in enumerate(shape_choices):
        try:
            mesh = make_shape(shape_type)
            mesh = normalize_mesh(mesh)

            if len(mesh.vertices) < 4 or len(mesh.faces) < 1:
                raise ValueError(f"mesh 不合法：{len(mesh.vertices)} 个顶点，{len(mesh.faces)} 个面")

            name = f"{i:04d}_{shape_type}"
            stl_path = os.path.join(STL_DIR, f"{name}.stl")
            pcd_path = os.path.join(PCD_DIR, f"{name}.ply")

            # 保存完整 STL
            mesh.export(stl_path)

            # ✅ 生成模拟深度扫描的残缺点云（替换原来的均匀采样）
            num_views = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            pts = simulate_depth_scan(mesh, n_points=NUM_PCD_POINTS, num_views=num_views)
            save_pointcloud_ply(pts, pcd_path)

            view_stats[num_views] += 1
            success += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{NUM_SAMPLES}] 已完成 {success} 个，失败 {fail} 个")

        except Exception as e:
            fail += 1
            print(f"  [警告] 第 {i} 个 ({shape_type}) 生成失败: {e}")

    print(f"\n✅ 完成！成功 {success} 个，失败 {fail} 个")
    print(f"   视角分布: 1视角={view_stats[1]}, 2视角={view_stats[2]}, 3视角={view_stats[3]}")
    print(f"   STL  → {os.path.abspath(STL_DIR)}")
    print(f"   点云 → {os.path.abspath(PCD_DIR)}")


if __name__ == "__main__":
    main()