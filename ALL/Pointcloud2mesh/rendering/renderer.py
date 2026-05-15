import threading
import numpy as np
import open3d as o3d

class Renderer:
    """
    Open3D 可视化模块。
    只负责：
    - 创建窗口
    - 更新 mesh
    - 刷新渲染
    不参与 tracking / mapping 计算。
    """

    def __init__(
        self,
        window_name="TSDF Fusion",
        width=1280,
        height=720,
        background_color=(0.2, 0.2, 0.2),
        mesh_show_back_face=True,
        axis_size=0.3,
        enable_log=True,
        logger=None,
    ):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.background_color = np.asarray(background_color, dtype=np.float64)
        self.mesh_show_back_face = mesh_show_back_face
        self.axis_size = axis_size
        self.enable_log = enable_log
        self.logger = logger

        self._vis = None
        self._mesh = None
        self._mesh_added = False
        self._axes = None
        self._initialized = False
        self._view_initialized = False
        self._lock = threading.RLock()

        self._last_rendered_frame_id = None
        self._last_mesh_vertex_count = 0

    def _log(self, msg: str, level: str = "status", force: bool = False):
        if not self.enable_log:
            return

        if self.logger is None:
            return

        if level == "warning":
            self.logger.warning(msg, force=force)
        elif level == "debug":
            self.logger.debug(msg, force=force)
        elif level == "profile":
            if hasattr(self.logger, "profile"):
                self.logger.profile(msg, force=force)
            else:
                self.logger.status(msg, force=force)
        else:
            self.logger.status(msg, force=force)

    def initialize(self):
        with self._lock:
            if self._initialized:
                return True

            self._vis = o3d.visualization.Visualizer()
            ok = self._vis.create_window(
                window_name=self.window_name,
                width=self.width,
                height=self.height,
            )
            if not ok:
                self._log("failed to create window", level="warning", force=True)
                return False

            opt = self._vis.get_render_option()
            opt.background_color = self.background_color
            opt.mesh_show_back_face = self.mesh_show_back_face

            self._axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.axis_size,
                origin=[0, 0, 0],
            )
            self._vis.add_geometry(self._axes)

            self._initialized = True
            self._log("initialized", force=True)
            return True

    def _update_mesh_geometry(self, new_mesh):
        """
        将 new_mesh 内容拷贝到内部持有的 mesh 对象，
        避免反复 add/remove geometry。
        """
        if new_mesh is None:
            return

        vnum = len(new_mesh.vertices)
        if vnum == 0:
            return

        tnum = len(new_mesh.triangles)
        if tnum == 0:
            return

        if not new_mesh.has_vertex_normals():
            new_mesh.compute_vertex_normals()

        if self._mesh is None:
            self._mesh = o3d.geometry.TriangleMesh()

        self._mesh.vertices = new_mesh.vertices
        self._mesh.triangles = new_mesh.triangles
        self._mesh.vertex_colors = new_mesh.vertex_colors
        self._mesh.vertex_normals = new_mesh.vertex_normals

        if not self._mesh_added:
            self._vis.add_geometry(self._mesh, reset_bounding_box=True)
            self._mesh_added = True
        else:
            self._vis.update_geometry(self._mesh)

        self._last_mesh_vertex_count = vnum

        if not self._view_initialized:
            self._vis.reset_view_point(True)
            self._view_initialized = True
            self._log("auto focus initialized", force=True)

    def render(self, tracking, map_snapshot):
        """
        每次调用只渲染“最新状态”
        """
        with self._lock:
            if not self._initialized:
                ok = self.initialize()
                if not ok:
                    return False

            mesh = None
            mesh_updated = False
            if map_snapshot is not None:
                extras = getattr(map_snapshot, "extras", None)
                if extras is not None:
                    mesh = extras.get("mesh", None)

                map_data = getattr(map_snapshot, "map_data", None)
                if isinstance(map_data, dict):
                    mesh_updated = bool(map_data.get("mesh_updated", False))

            if mesh is not None and mesh_updated:
                self._update_mesh_geometry(mesh)

            alive = self._vis.poll_events()
            self._vis.update_renderer()

            if tracking is not None:
                self._last_rendered_frame_id = tracking.frame_id

            return alive

    def print_status(self, tracking, map_snapshot):
        tracking_info = None
        if tracking is not None:
            tracking_info = {
                "frame_id": tracking.frame_id,
                "mode": tracking.mode,
                "success": tracking.success,
                "score": tracking.score,
            }

        map_info = None
        if map_snapshot is not None:
            map_info = map_snapshot.map_data

        self._log(f"tracking={tracking_info}, map={map_info}")

    def close(self):
        with self._lock:
            if self._vis is not None:
                self._vis.destroy_window()
                self._vis = None
            self._mesh = None
            self._mesh_added = False
            self._initialized = False
            self._view_initialized = False
            self._log("closed", force=True)