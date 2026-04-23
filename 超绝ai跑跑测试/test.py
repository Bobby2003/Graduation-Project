"""
DMTet 测试脚本
==============
功能：
  1. 单文件推理：输入 .ply 点云 → 输出 .stl Mesh
  2. 批量推理：遍历文件夹，全部转换
  3. 与 GT 对比评估（Chamfer / Hausdorff）
  4. 可视化：输入点云 vs 输出 Mesh vs GT Mesh
  5. 支持 --model 参数选择模型文件
  6. 支持 --tet_res 参数指定四面体分辨率

用法：
  python test.py                                                # 交互式选择模型，自动测试
  python test.py --model checkpoints_mesh/best_mesh_model.pth single input.ply
  python test.py --model checkpoints_mesh/mesh_epoch_200.pth batch dataset/pointcloud
  python test.py --list_models                                  # 列出所有可用模型
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import trimesh
import time

from model import DMTetModel, Config

cfg = Config()


# ═══════════════════════════════════════════════════════════════
# 模型管理
# ═══════════════════════════════════════════════════════════════

def list_available_models(search_dirs=None):
    """列出所有可用的模型文件"""
    if search_dirs is None:
        search_dirs = [cfg.SAVE_DIR, ".", "checkpoints", "checkpoints_mesh"]

    models = []
    seen = set()

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.pth"))):
            full = os.path.abspath(f)
            if full not in seen:
                seen.add(full)
                size_mb = os.path.getsize(f) / (1024 * 1024)
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))
                models.append({
                    "path": f,
                    "name": os.path.basename(f),
                    "size_mb": size_mb,
                    "mtime": mtime,
                })
    return models


def interactive_select_model():
    """交互式选择模型"""
    models = list_available_models()

    if not models:
        print("❌ 没找到任何 .pth 模型文件！")
        print(f"   搜索目录: {cfg.SAVE_DIR}, ., checkpoints, checkpoints_mesh")
        sys.exit(1)

    if len(models) == 1:
        print(f"找到 1 个模型，自动选择: {models[0]['path']}")
        return models[0]['path']

    print(f"\n{'='*60}")
    print(f"找到 {len(models)} 个模型文件:")
    print(f"{'='*60}")
    print(f"{'序号':<6} {'文件名':<35} {'大小':<10} {'修改时间':<20}")
    print(f"{'-'*71}")

    for i, m in enumerate(models):
        marker = " ⭐" if "best" in m["name"].lower() else ""
        print(f"  {i+1:<4} {m['name']:<35} {m['size_mb']:.1f}MB    {m['mtime']}{marker}")

    print(f"\n  ⭐ = 推荐（best model）")

    while True:
        try:
            choice = input(f"\n请选择模型 [1-{len(models)}] (回车=1): ").strip()
            if choice == "":
                idx = 0
            else:
                idx = int(choice) - 1
            if 0 <= idx < len(models):
                print(f"✅ 选择: {models[idx]['path']}")
                return models[idx]['path']
            print(f"请输入 1~{len(models)} 之间的数字")
        except ValueError:
            print("请输入数字")
        except KeyboardInterrupt:
            print("\n已取消")
            sys.exit(0)


def load_model(model_path, tet_res=None, device=None):
    """加载训练好的模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tet_res is None:
        tet_res = cfg.TET_GRID_RES

    print(f"\n加载模型: {model_path}")
    print(f"设备: {device}")
    print(f"四面体分辨率: {tet_res}")

    model = DMTetModel(tet_res=tet_res, latent_dim=512).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        err_str = str(e)
        if "size mismatch" in err_str:
            print(f"\n⚠️  模型参数尺寸不匹配！")
            print(f"   可能是 --tet_res 与训练时不一致。")
            print(f"   当前 tet_res={tet_res}，请尝试其他值（如 32, 48, 64）")
            print(f"\n错误详情: {err_str[:200]}")
            sys.exit(1)
        raise

    model.eval()
    print("✅ 模型加载成功")
    return model, device


# ═══════════════════════════════════════════════════════════════
# 预处理
# ═══════════════════════════════════════════════════════════════

def preprocess_pointcloud(ply_path, n_points=4096):
    """加载并预处理点云"""
    pc = trimesh.load(ply_path)
    pts = np.array(pc.vertices, dtype=np.float32)

    if len(pts) == 0:
        raise ValueError(f"点云为空: {ply_path}")

    if len(pts) >= n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
    else:
        idx = np.random.choice(len(pts), n_points, replace=True)

    pts = pts[idx]
    return pts


# ═══════════════════════════════════════════════════════════════
# 推理
# ═══════════════════════════════════════════════════════════════

def inference_single(model, device, input_ply, output_stl=None):
    """
    单个文件推理
    返回: trimesh.Trimesh 对象 或 None
    """
    if output_stl is None:
        name = os.path.splitext(os.path.basename(input_ply))[0]
        output_stl = f"output_{name}.stl"

    print(f"\n{'='*60}")
    print(f"输入: {input_ply}")

    # 预处理
    pts = preprocess_pointcloud(input_ply, cfg.NUM_INPUT_PTS)
    print(f"  点云: {len(pts)} 个点")
    print(f"  范围: X[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] "
          f"Y[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] "
          f"Z[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

    pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(device)

    # 推理
    t0 = time.time()
    with torch.no_grad():
        results, sdf = model(pts_tensor)
        verts, faces = results[0]
    t1 = time.time()

    v_np = verts.cpu().numpy()
    f_np = faces.cpu().numpy()

    # SDF 分析
    print(f"  推理耗时: {t1-t0:.3f}s")
    print(f"  SDF 范围: [{sdf[0].min().item():.4f}, {sdf[0].max().item():.4f}]")

    neg_ratio = (sdf[0] < 0).float().mean().item()
    pos_ratio = (sdf[0] > 0).float().mean().item()
    near_zero = (sdf[0].abs() < 0.01).float().mean().item()
    print(f"  SDF 分布: 负{neg_ratio:.1%} | 正{pos_ratio:.1%} | 近零(<0.01){near_zero:.1%}")

    if len(v_np) == 0 or len(f_np) == 0:
        print("  ❌ 未能提取 Mesh！")
        print(f"  SDF 均值: {sdf[0].mean().item():.4f}")
        return None

    # 后处理
    mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
    mesh.process(validate=True)
    mesh.fix_normals()

    print(f"  输出 Mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面")
    if mesh.is_watertight:
        print(f"  水密性: ✅ 水密")
        print(f"  体积: {mesh.volume:.6f}")
    else:
        print(f"  水密性: ❌ 非水密")
    print(f"  表面积: {mesh.area:.6f}")
    print(f"  包围盒: {mesh.bounds}")

    mesh.export(output_stl)
    print(f"  ✅ 保存到 {output_stl}")

    return mesh


def inference_batch(model, device, input_dir, output_dir="output_meshes"):
    """批量推理：遍历文件夹中所有 .ply 文件"""
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    print(f"\n批量推理: 共 {len(ply_files)} 个文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    results_summary = []
    total_time = 0

    for i, ply_path in enumerate(ply_files):
        name = os.path.splitext(os.path.basename(ply_path))[0]
        out_path = os.path.join(output_dir, f"{name}.stl")

        t0 = time.time()
        mesh = inference_single(model, device, ply_path, out_path)
        t1 = time.time()
        total_time += (t1 - t0)

        status = "✅" if mesh is not None else "❌"
        n_verts = len(mesh.vertices) if mesh else 0
        n_faces = len(mesh.faces) if mesh else 0
        results_summary.append({
            "name": name,
            "status": status,
            "vertices": n_verts,
            "faces": n_faces,
            "time": t1 - t0,
        })

    # 汇总
    print(f"\n{'='*60}")
    print(f"批量推理完毕")
    print(f"{'='*60}")
    print(f"{'文件名':<30} {'状态':<4} {'顶点':<10} {'面':<10} {'耗时':<8}")
    print(f"{'-'*62}")

    success = 0
    for r in results_summary:
        print(f"{r['name']:<30} {r['status']:<4} {r['vertices']:<10} {r['faces']:<10} {r['time']:.2f}s")
        if r['status'] == '✅':
            success += 1

    print(f"\n成功率: {success}/{len(ply_files)} ({success/max(len(ply_files),1)*100:.1f}%)")
    print(f"总耗时: {total_time:.1f}s, 平均: {total_time/max(len(ply_files),1):.2f}s/个")


# ═══════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════

def evaluate_with_gt(model, device, input_ply, gt_stl, output_stl=None):
    """
    推理并与 GT Mesh 对比评估
    计算 Chamfer Distance, Hausdorff Distance 等指标
    """
    pred_mesh = inference_single(model, device, input_ply, output_stl)
    if pred_mesh is None:
        print("  跳过评估（无 Mesh 输出）")
        return None

    # 加载 GT
    gt_mesh = trimesh.load(gt_stl)
    print(f"\n  GT Mesh: {len(gt_mesh.vertices)} 顶点, {len(gt_mesh.faces)} 面")

    # 从两个 Mesh 采样点云
    n_eval = 10000
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, n_eval)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, n_eval)

    pred_pts = pred_pts.astype(np.float32)
    gt_pts = gt_pts.astype(np.float32)

    # Chamfer Distance
    pred_t = torch.from_numpy(pred_pts).to(device)
    gt_t = torch.from_numpy(gt_pts).to(device)

    dist_matrix = torch.cdist(pred_t.unsqueeze(0), gt_t.unsqueeze(0)).squeeze(0)
    cd_pred2gt = dist_matrix.min(dim=1)[0]
    cd_gt2pred = dist_matrix.min(dim=0)[0]

    chamfer = (cd_pred2gt.mean() + cd_gt2pred.mean()).item()
    hausdorff = max(cd_pred2gt.max().item(), cd_gt2pred.max().item())

    print(f"\n  📊 评估指标:")
    print(f"     Chamfer Distance:   {chamfer:.6f}")
    print(f"     Hausdorff Distance: {hausdorff:.6f}")
    print(f"     Pred→GT 平均距离:   {cd_pred2gt.mean().item():.6f}")
    print(f"     GT→Pred 平均距离:   {cd_gt2pred.mean().item():.6f}")
    print(f"     Pred→GT 最大距离:   {cd_pred2gt.max().item():.6f}")
    print(f"     GT→Pred 最大距离:   {cd_gt2pred.max().item():.6f}")

    return {
        "chamfer": chamfer,
        "hausdorff": hausdorff,
        "pred_vertices": len(pred_mesh.vertices),
        "pred_faces": len(pred_mesh.faces),
    }


def evaluate_batch(model, device, pcd_dir, stl_dir, output_dir="eval_output"):
    """批量评估：对数据集中所有样本推理并与 GT 对比"""
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))
    print(f"\n批量评估: {len(ply_files)} 个样本")

    all_metrics = []

    for ply_path in ply_files:
        name = os.path.splitext(os.path.basename(ply_path))[0]
        stl_path = os.path.join(stl_dir, f"{name}.stl")

        if not os.path.exists(stl_path):
            print(f"  ⚠ 找不到 GT: {stl_path}, 跳过")
            continue

        out_stl = os.path.join(output_dir, f"{name}_pred.stl")
        metrics = evaluate_with_gt(model, device, ply_path, stl_path, out_stl)

        if metrics:
            metrics["name"] = name
            all_metrics.append(metrics)

    if all_metrics:
        print(f"\n{'='*60}")
        print(f"批量评估汇总")
        print(f"{'='*60}")

        chamfers = [m["chamfer"] for m in all_metrics]
        hausdorffs = [m["hausdorff"] for m in all_metrics]

        print(f"  样本数: {len(all_metrics)}")
        print(f"  Chamfer Distance:")
        print(f"    平均: {np.mean(chamfers):.6f}")
        print(f"    中位: {np.median(chamfers):.6f}")
        print(f"    最优: {np.min(chamfers):.6f} ({all_metrics[np.argmin(chamfers)]['name']})")
        print(f"    最差: {np.max(chamfers):.6f} ({all_metrics[np.argmax(chamfers)]['name']})")
        print(f"  Hausdorff Distance:")
        print(f"    平均: {np.mean(hausdorffs):.6f}")
        print(f"    中位: {np.median(hausdorffs):.6f}")

        # 保存结果
        import json
        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  结果已保存到 {output_dir}/eval_results.json")


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

def visualize_result(input_ply, output_stl, gt_stl=None):
    """
    用 trimesh 可视化：输入点云 + 输出 Mesh (+ GT Mesh)
    需要桌面环境
    """
    try:
        scene = trimesh.Scene()

        # 输入点云（蓝色）
        pc = trimesh.load(input_ply)
        pts = np.array(pc.vertices)
        colors_blue = np.tile([0, 100, 255, 200], (len(pts), 1))
        pc_vis = trimesh.PointCloud(pts, colors=colors_blue)
        scene.add_geometry(pc_vis, node_name="Input PointCloud")

        # 输出 Mesh（绿色半透明）
        if os.path.exists(output_stl):
            pred = trimesh.load(output_stl)
            pred.visual.face_colors = [0, 200, 100, 150]
            pred.apply_translation([2.5, 0, 0])
            scene.add_geometry(pred, node_name="Predicted Mesh")

        # GT Mesh（红色半透明）
        if gt_stl and os.path.exists(gt_stl):
            gt = trimesh.load(gt_stl)
            gt.visual.face_colors = [255, 80, 80, 150]
            gt.apply_translation([5.0, 0, 0])
            scene.add_geometry(gt, node_name="GT Mesh")

        scene.show()
    except Exception as e:
        print(f"可视化失败: {e}")
        print("提示: 可视化需要桌面环境和 pyglet/pyrender")


# ═══════════════════════════════════════════════════════════════
# 命令行参数解析
# ═══════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="DMTet 点云→Mesh 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式选择模型，自动测试
  python test.py

  # 指定模型，单文件推理
  python test.py --model checkpoints_mesh/best_mesh_model.pth single input.ply output.stl

  # 指定模型，批量推理
  python test.py --model checkpoints_mesh/mesh_epoch_200.pth batch dataset/pointcloud output_meshes

  # 指定模型和分辨率（必须与训练时一致）
  python test.py --model best.pth --tet_res 32 single input.ply

  # 列出所有可用模型
  python test.py --list_models

  # 批量评估（与 GT 对比）
  python test.py --model best.pth eval_batch dataset/pointcloud dataset/stl eval_output
        """
    )

    parser.add_argument("--model", "-m", type=str, default=None,
                        help="模型文件路径 (.pth)。不指定则交互式选择")
    parser.add_argument("--tet_res", "-r", type=int, default=None,
                        help=f"四面体网格分辨率（必须与训练时一致，默认={cfg.TET_GRID_RES}）")
    parser.add_argument("--list_models", "-l", action="store_true",
                        help="列出所有可用的模型文件")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="设备: cuda 或 cpu")

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # single
    p_single = subparsers.add_parser("single", help="单文件推理")
    p_single.add_argument("input_ply", type=str, help="输入点云 .ply 文件")
    p_single.add_argument("output_stl", type=str, nargs="?", default=None,
                          help="输出 .stl 文件（可选，默认 output_<名称>.stl）")

    # batch
    p_batch = subparsers.add_parser("batch", help="批量推理")
    p_batch.add_argument("input_dir", type=str, nargs="?", default=cfg.PCD_DIR,
                         help=f"输入目录（默认={cfg.PCD_DIR}）")
    p_batch.add_argument("output_dir", type=str, nargs="?", default="output_meshes",
                         help="输出目录（默认=output_meshes）")

    # eval
    p_eval = subparsers.add_parser("eval", help="单文件推理 + GT 对比评估")
    p_eval.add_argument("input_ply", type=str, help="输入点云 .ply")
    p_eval.add_argument("gt_stl", type=str, help="GT Mesh .stl")
    p_eval.add_argument("output_stl", type=str, nargs="?", default=None,
                        help="输出 .stl（可选）")

    # eval_batch
    p_evalb = subparsers.add_parser("eval_batch", help="批量评估（与 GT 对比）")
    p_evalb.add_argument("pcd_dir", type=str, nargs="?", default=cfg.PCD_DIR,
                         help=f"点云目录（默认={cfg.PCD_DIR}）")
    p_evalb.add_argument("stl_dir", type=str, nargs="?", default=cfg.STL_DIR,
                         help=f"GT STL 目录（默认={cfg.STL_DIR}）")
    p_evalb.add_argument("output_dir", type=str, nargs="?", default="eval_output",
                         help="输出目录（默认=eval_output）")

    # viz
    p_viz = subparsers.add_parser("viz", help="可视化对比")
    p_viz.add_argument("input_ply", type=str, help="输入点云 .ply")
    p_viz.add_argument("output_stl", type=str, help="预测 Mesh .stl")
    p_viz.add_argument("gt_stl", type=str, nargs="?", default=None,
                       help="GT Mesh .stl（可选）")

    return parser


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # ── 列出模型 ──
    if args.list_models:
        models = list_available_models()
        if not models:
            print("没找到任何 .pth 模型文件")
        else:
            print(f"\n找到 {len(models)} 个模型:")
            print(f"{'序号':<6} {'路径':<45} {'大小':<10} {'修改时间':<20}")
            print(f"{'-'*81}")
            for i, m in enumerate(models):
                marker = " ⭐" if "best" in m["name"].lower() else ""
                print(f"  {i+1:<4} {m['path']:<45} {m['size_mb']:.1f}MB    {m['mtime']}{marker}")
        sys.exit(0)

    # ── 选择模型 ──
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            sys.exit(1)
    else:
        model_path = interactive_select_model()

    # ── 加载模型 ──
    tet_res = args.tet_res if args.tet_res else cfg.TET_GRID_RES
    model, device = load_model(model_path, tet_res=tet_res, device=args.device)

    # ── 无子命令 → 交互式自动测试 ──
    if args.mode is None:
        print("\n" + "=" * 60)
        print("用法:")
        print("  python test.py [--model MODEL] single <input.ply> [output.stl]")
        print("  python test.py [--model MODEL] batch  <input_dir> [output_dir]")
        print("  python test.py [--model MODEL] eval   <input.ply> <gt.stl>")
        print("  python test.py [--model MODEL] eval_batch [pcd_dir] [stl_dir]")
        print("  python test.py [--model MODEL] viz    <input.ply> <output.stl> [gt.stl]")
        print("  python test.py --list_models")
        print("=" * 60)

        # 自动找测试文件
        test_files = sorted(glob.glob(os.path.join(cfg.PCD_DIR, "*.ply")))
        if not test_files:
            print("\n没找到测试文件，请指定输入路径")
            sys.exit(0)

        print(f"\n快速测试 — 找到 {len(test_files)} 个点云文件")
        show_count = min(len(test_files), 20)
        for i in range(show_count):
            print(f"  {i+1}. {os.path.basename(test_files[i])}")
        if len(test_files) > show_count:
            print(f"  ... 还有 {len(test_files) - show_count} 个")

        try:
            choice = input(f"\n选择测试文件 [1-{show_count}] (回车=1): ").strip()
            if choice == "":
                test_idx = 0
            else:
                test_idx = int(choice) - 1
            test_idx = max(0, min(test_idx, show_count - 1))
        except (ValueError, KeyboardInterrupt):
            test_idx = 0

        test_file = test_files[test_idx]
        print(f"\n🚀 自动测试: {test_file}")
        mesh = inference_single(model, device, test_file)

        # 如果有 GT，自动评估
        name = os.path.splitext(os.path.basename(test_file))[0]
        gt_path = os.path.join(cfg.STL_DIR, f"{name}.stl")
        if os.path.exists(gt_path) and mesh is not None:
            evaluate_with_gt(model, device, test_file, gt_path)

        sys.exit(0)

    # ── 执行对应模式 ──
    if args.mode == "single":
        inference_single(model, device, args.input_ply, args.output_stl)

    elif args.mode == "batch":
        inference_batch(model, device, args.input_dir, args.output_dir)

    elif args.mode == "eval":
        evaluate_with_gt(model, device, args.input_ply, args.gt_stl, args.output_stl)

    elif args.mode == "eval_batch":
        evaluate_batch(model, device, args.pcd_dir, args.stl_dir, args.output_dir)

    elif args.mode == "viz":
        visualize_result(args.input_ply, args.output_stl, args.gt_stl)

    else:
        print(f"未知模式: {args.mode}")
        print("可用模式: single, batch, eval, eval_batch, viz")
        sys.exit(1)