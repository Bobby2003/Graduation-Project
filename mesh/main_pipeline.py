from pathlib import Path
from pipeline_api import RealtimeMappingPipeline
from mesh_refine_adapter import refine_mesh_in_memory

BASE_DIR = Path(__file__).resolve().parent

def main():
    # ============================================================
    # 阶段 1+2：实时采集 + 建图（原本的逻辑，完全不动）
    # ============================================================
    pipeline = RealtimeMappingPipeline()

    # save_mesh=True 保留 → 仍然会输出 model/fusion_model_final.obj 作为对比
    result = pipeline.run_forever(
        save_mesh=True,
        final_mesh_prefix="fusion_model_final",
    )

    # ============================================================
    # 阶段 3：从 mapper 内存里抽 mesh，直接优化（新增）
    # ============================================================
    print("\n🔄 进入在线优化阶段...")

    # ⚠️ 关键：run_forever 返回后 _setup_done 已经被置 False，
    # 但 self.mapper 引用还在（stop() 没主动 None 它），可以直接拿
    if pipeline.mapper is None:
        print("❌ mapper 不存在，跳过优化")
        return

    try:
        # 从 TSDF volume 直接抽最完整的 mesh
        final_mesh = pipeline.mapper.volume.extract_triangle_mesh()
        final_mesh.compute_vertex_normals()
    except Exception as e:
        print(f"❌ 无法从 mapper.volume 抽 mesh: {e}")
        return

    if len(final_mesh.vertices) == 0:
        print("⚠️ TSDF volume 为空，无可优化模型")
        return

    refined_mesh, meta = refine_mesh_in_memory(
        final_mesh,
        output_dir=str(BASE_DIR / "outputs_online"),
        prefix="fusion_refined",
        save_files=True,
        save_raw=True,
        verbose=True,
        # ---- 推荐参数 ----
        global_smooth_iter=1,
        enable_far_second_pass=True,
        sensor_origin="0,0,0",
        use_distance_strategy=True,
    )

    if not meta.get("skipped"):
        print(f"\n🎉 全流程完成！")
        print(f"   原始 mesh:  {meta['before']['vertices']} 顶点")
        print(f"   优化 mesh:  {meta['after']['vertices']} 顶点")
        print(f"   优化耗时:   {meta['total_elapsed_sec']}s")
        print(f"   保存位置:   {meta.get('refined_latest_path')}")

if __name__ == "__main__":
    main()