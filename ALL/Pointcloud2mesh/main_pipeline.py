try:
    from .Pointcloud2mesh.pipeline_api import RealtimeMappingPipeline
except ImportError:
    # 允许在 Pointcloud2mesh 目录下直接 python main_pipeline.py（非包方式）
    from Pointcloud2mesh.pipeline_api import RealtimeMappingPipeline

def main():
    pipeline = RealtimeMappingPipeline()
    pipeline.run_forever(
        save_mesh=True,
        final_mesh_prefix="fusion_model_final",
    )

if __name__ == "__main__":
    main()