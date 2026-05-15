import trimesh

src = r".\input\imu_fusion_model_final_20260413_193242.obj"
dst = r".\input\imu_fusion_model_final_20260413_193242.ply"

mesh = trimesh.load(src, force="mesh", process=False)
mesh.export(dst)
print("done")