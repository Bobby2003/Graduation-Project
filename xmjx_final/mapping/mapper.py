import os
import shutil
from datetime import datetime

import numpy as np
import open3d as o3d

from common.types import MapPacket, MapSnapshot
from input.rgbd_preprocessor import RGBDPreprocessor

class Mapper:
    """
    TSDF 建图模块。
    使用 TrackingResult 中的 T_wc(这里实际沿用原型语义：world -> current_camera)
    直接作为 Open3D integrate 的 extrinsic。
    """

    def __init__(
        self,
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,

        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        input_color_is_bgr=False,
        depth_scale=None,
        depth_trunc=2.0,

        integrate_only_when_motion=True,
        min_integrate_rot_deg=0.30,
        max_integrate_rot_deg=8.0,
        min_integrate_trans_m=0.0008,

        mesh_update_interval=10,
        mesh_min_vertices_to_show=10,

        model_dir="model",
    ):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=color_type,
        )

        self.preprocessor = RGBDPreprocessor(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            input_color_is_bgr=input_color_is_bgr,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )

        self.integrate_only_when_motion = integrate_only_when_motion
        self.min_integrate_rot_deg = min_integrate_rot_deg
        self.max_integrate_rot_deg = max_integrate_rot_deg
        self.min_integrate_trans_m = min_integrate_trans_m

        self.mesh_update_interval = mesh_update_interval
        self.mesh_min_vertices_to_show = mesh_min_vertices_to_show

        self.total_packets = 0
        self.integrated_frames = 0
        self.last_mesh = None
        self.last_mesh_vertex_count = 0

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    @staticmethod
    def normalize_rotation(Rm):
        U, _, Vt = np.linalg.svd(Rm)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    @staticmethod
    def rot_deg(Rm):
        trace = np.trace(Rm)
        trace = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))

    def _should_integrate(self, packet: MapPacket) -> bool:
        if not self.integrate_only_when_motion:
            return True

        tracking = packet.tracking
        extras = tracking.extras if tracking.extras is not None else {}

        rot_used = float(extras.get("vo_rot_deg", 0.0))
        trans_used = float(extras.get("translation_norm", 0.0))

        moving = (
            (trans_used >= self.min_integrate_trans_m)
            or (rot_used >= self.min_integrate_rot_deg)
        )
        rot_too_fast = (rot_used > self.max_integrate_rot_deg)

        return moving and (not rot_too_fast)

    def update(self, packet: MapPacket) -> MapSnapshot:
        self.total_packets += 1

        frame = packet.frame
        tracking = packet.tracking

        current_rgbd, rgbd_info = self.preprocessor.preprocess(frame.color, frame.depth)
        if current_rgbd is None:
            return MapSnapshot(
                frame_id=frame.frame_id,
                timestamp=frame.device_timestamp,
                map_data={
                    "integrated": False,
                    "reason": "rgbd_preprocess_failed",
                    "rgbd_info": rgbd_info,
                    "integrated_frames": self.integrated_frames,
                },
            )

        do_integrate = self._should_integrate(packet)

        if do_integrate:
            extrinsic = tracking.T_wc.copy()
            self.volume.integrate(current_rgbd, self.preprocessor.get_intrinsic(), extrinsic)
            self.integrated_frames += 1

        map_data = {
            "integrated": do_integrate,
            "integrated_frames": self.integrated_frames,
            "total_packets": self.total_packets,
            "last_frame_id": frame.frame_id,
            "mesh_vertex_count": self.last_mesh_vertex_count,
        }

        # 周期性抽 mesh
        if self.total_packets % self.mesh_update_interval == 0:
            new_mesh = self.volume.extract_triangle_mesh()
            vnum = len(new_mesh.vertices)

            if vnum > self.mesh_min_vertices_to_show:
                new_mesh.compute_vertex_normals()
                self.last_mesh = new_mesh
                self.last_mesh_vertex_count = vnum

            map_data["mesh_vertex_count"] = self.last_mesh_vertex_count
            map_data["mesh_updated"] = True
        else:
            map_data["mesh_updated"] = False

        return MapSnapshot(
            frame_id=frame.frame_id,
            timestamp=frame.device_timestamp,
            map_data=map_data,
            extras={
                "mesh": self.last_mesh,
                "rgbd_info": rgbd_info,
            }
        )

    def get_latest_mesh(self):
        return self.last_mesh

    def save_final_mesh(self, prefix="imu_fusion_model_final"):
        final_mesh = self.volume.extract_triangle_mesh()
        if len(final_mesh.vertices) == 0:
            print("⚠️ 没有有效模型数据可保存")
            return False, None, None

        final_mesh.compute_vertex_normals()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_path = os.path.join(self.model_dir, f"{prefix}_{ts}.obj")
        latest_path = os.path.join(self.model_dir, f"{prefix}.obj")

        ok_hist = o3d.io.write_triangle_mesh(hist_path, final_mesh, write_vertex_normals=True)

        if ok_hist:
            shutil.copyfile(hist_path, latest_path)
            print(f"✅ 模型历史版: {hist_path}")
            print(f"✅ 模型最新版: {latest_path}")
            return True, hist_path, latest_path

        print("⚠️ 历史版模型保存失败")
        return False, None, None