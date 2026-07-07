import cv2
import numpy as np
import open3d as o3d
import time
from orbbec_sdk import OrbbecCameraSDK


def save_refined_model(pcd_l, pcd_r, distance_cm):
    """应用精细间距并保存对比点云"""
    dist_m = distance_cm / 100.0

    # 假设左相机为原点，右相机在 X 轴正方向偏移
    transform = np.eye(4)
    transform[0, 3] = dist_m

    pcd_r_moved = o3d.geometry.PointCloud(pcd_r)
    pcd_r_moved.transform(transform)

    # 合并点云
    combined = pcd_l + pcd_r_moved

    # 为了观察更精细，稍微减小下采样粒度（2mm）
    combined = combined.voxel_down_sample(0.002)

    filename = f"refine_{distance_cm}cm.ply"
    o3d.io.write_point_cloud(filename, combined)
    print(f"📍 精细模型已生成: {filename}")


def run_refine_test():
    cam_l = OrbbecCameraSDK()
    cam_r = OrbbecCameraSDK()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 319.5, 239.5)

    try:
        cam_l.initialize()
        devs = cam_l.get_devices()
        cam_l.open(devs[0]['uri'])
        cam_r.open(devs[1]['uri'])
        cap_l = cv2.VideoCapture(0)
        cap_r = cv2.VideoCapture(1)

        print("📸 正在最后捕捉，请保持相机完全静止...")
        time.sleep(3)

        d_l = cam_l.get_depth()
        d_r = cam_r.get_depth()
        _, c_l = cap_l.read()
        _, c_r = cap_r.read()

        def to_pcd(d, c):
            color_rgb = cv2.cvtColor(cv2.resize(c, (640, 480)), cv2.COLOR_BGR2RGB)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(d.astype(np.uint16)),
                1000.0, 2.5, False
            )
            return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        pcd_l = to_pcd(d_l, c_l)
        pcd_r = to_pcd(d_r, c_r)

        # 执行 27, 28, 29 测试
        target_list = [27, 28, 29]
        for dist in target_list:
            save_refined_model(pcd_l, pcd_r, dist)

        print("\n🎯 探测完成！请在 MeshLab 中重点观察 27-29cm 的细微差别。")
        print("💡 技巧：旋转模型到侧面，看哪一个数值下‘重影’最少。")

    except Exception as e:
        print(f"错误: {e}")
    finally:
        cam_l.close();
        cam_r.close()


if __name__ == "__main__":
    run_refine_test()