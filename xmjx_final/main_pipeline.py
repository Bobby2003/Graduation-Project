from pipeline_api import RealtimeMappingPipeline

def main():
    pipeline = RealtimeMappingPipeline()
    pipeline.run_forever(
        save_mesh=True,
        final_mesh_prefix="fusion_model_final",
    )

if __name__ == "__main__":
    main()