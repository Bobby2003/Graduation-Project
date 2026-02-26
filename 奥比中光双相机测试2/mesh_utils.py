import numpy as np
import open3d as o3d
import tempfile
import os

def estimate_normals_for_pcd(pcd, radius=None, max_nn=30):
    """
    估计并统一朝向点云法线
    """
    if radius is None:
        # 基于点云尺度自动选择 radius（粗略）
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
        radius = max(diag * 0.01, 0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # 令法线朝向一致（k 可调）
    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except Exception:
        pass
    return pcd

def pcd_to_mesh(pcd,
                method='poisson',           # 'poisson' | 'ball_pivot' | 'alpha'
                poisson_depth=9,
                poisson_scale=1.1,
                poisson_linear_fit=False,
                bp_radii=None,             # list/tuple of radii for ball pivot, e.g. [0.005, 0.01, 0.02]
                alpha=0.01,                # alpha value for alpha shape
                remove_low_density=True,
                density_quantile=0.01):
    """
    将点云转为三角网格，返回 mesh（open3d.geometry.TriangleMesh）
    - 对 pcd 进行法线估计与方向一致化
    - poisson 会返回一个可能很大且带外部零散片段的网格，默认用密度过滤去掉低置信部分
    """
    assert isinstance(pcd, o3d.geometry.PointCloud)
    if len(pcd.points) < 10:
        raise ValueError("点云太少，无法生成网格")

    pcd = estimate_normals_for_pcd(pcd)

    if method == 'poisson':
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=poisson_depth,
            width=0,
            scale=poisson_scale,
            linear_fit=poisson_linear_fit
        )
        mesh.compute_vertex_normals()
        if remove_low_density:
            densities = np.asarray(densities)
            # 过滤掉密度低的顶点与其相关三角形
            thresh = np.quantile(densities, density_quantile)
            verts_to_keep = densities > thresh
            # build vertex mask -> remove vertices and triangles referencing removed verts
            verts_idx = np.where(verts_to_keep)[0]
            if len(verts_idx) == 0:
                # 如果全部被过滤，退回未过滤的 mesh
                return mesh
            mesh = mesh.select_by_index(verts_idx)
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.compute_vertex_normals()
        return mesh

    elif method == 'ball_pivot':
        # 需要选择合理的 radii，根据点云尺度
        if bp_radii is None:
            # 自动设置 radii：基于点云分布，取若干倍数
            bbox = pcd.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
            base = max(diag * 0.002, 0.001)
            bp_radii = [base, base * 2, base * 4]
        radii_o3d = o3d.utility.DoubleVector(bp_radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_o3d)
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh

    elif method == 'alpha':
        # alpha shape，alpha 值需要根据点云尺度调整
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh

    else:
        raise ValueError("Unknown mesh method: %s" % method)

def smooth_mesh(mesh, method='taubin', iterations=10, lamb=0.5, mu=-0.53):
    """
    对 mesh 做平滑处理
    - method: 'taubin' 或 'simple'
    - iterations: 迭代次数
    - taubin 参数 lamb, mu
    返回平滑后的 mesh（副本）
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise ValueError("输入必须是 TriangleMesh")
    mesh_out = mesh.clone()
    if method == 'taubin':
        # Taubin smoothing（保形较好）
        mesh_out = mesh_out.filter_smooth_taubin(number_of_iterations=iterations, lambda_param=lamb, mu=mu)
    elif method == 'simple':
        mesh_out = mesh_out.filter_smooth_simple(number_of_iterations=iterations)
    else:
        raise ValueError("Unknown smoothing method: %s" % method)
    mesh_out.compute_vertex_normals()
    return mesh_out

def export_mesh_to_stl(mesh, filename, write_ascii=False):
    """
    导出 STL。Open3D 支持写入 stl（若需要更复杂修补可用 trimesh）
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise ValueError("输入必须是 TriangleMesh")
    # Ensure folder exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    # Ensure normals computed
    mesh.compute_vertex_normals()
    # Open3D supports writing stl
    o3d.io.write_triangle_mesh(filename, mesh, write_ascii=write_ascii)
    return filename

def visualize_mesh_and_pcd(mesh, pcd=None, mesh_color=(0.8, 0.6, 0.2)):
    """
    使用 Open3D 可视化网格与（可选）点云
    """
    vis_list = []
    if pcd is not None:
        # 点云着色为深灰
        pcd_for_vis = pcd.clone()
        try:
            pcd_for_vis.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.5,0.5,0.5]]), (len(pcd_for_vis.points),1)))
        except Exception:
            pass
        vis_list.append(pcd_for_vis)
    mesh_for_vis = mesh.clone()
    mesh_for_vis.paint_uniform_color(mesh_color)
    mesh_for_vis.compute_vertex_normals()
    vis_list.append(mesh_for_vis)
    o3d.visualization.draw_geometries(vis_list)

# 一个一键式流程函数，方便在你的主流程里直接调用
def generate_smoothed_stl_from_pcd(pcd,
                                   out_stl_path,
                                   mesh_method='poisson',
                                   poisson_depth=9,
                                   bp_radii=None,
                                   alpha=0.01,
                                   smoothing_method='taubin',
                                   smoothing_iters=10,
                                   smoothing_lamb=0.5,
                                   smoothing_mu=-0.53,
                                   write_ascii=False,
                                   visualize=True):
    """
    从点云生成网格、平滑、导出 stl 并可视化
    返回 (mesh_raw, mesh_smoothed, out_stl_path)
    """
    # 1) mesh
    mesh_raw = pcd_to_mesh(pcd,
                           method=mesh_method,
                           poisson_depth=poisson_depth,
                           bp_radii=bp_radii,
                           alpha=alpha)
    # 2) smoothing
    mesh_smoothed = smooth_mesh(mesh_raw, method=smoothing_method, iterations=smoothing_iters, lamb=smoothing_lamb, mu=smoothing_mu)
    # 3) export
    export_mesh_to_stl(mesh_smoothed, out_stl_path, write_ascii=write_ascii)
    # 4) visualize
    if visualize:
        try:
            visualize_mesh_and_pcd(mesh_smoothed, pcd)
        except Exception as e:
            print("可视化失败：", e)
    return mesh_raw, mesh_smoothed, out_stl_path