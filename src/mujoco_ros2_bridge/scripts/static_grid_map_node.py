#!/usr/bin/env python3
import math
import os
from typing import Iterable

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class StaticGridMapNode(Node):
    """Publish a 2D occupancy grid from static MuJoCo collision geometry."""

    def __init__(self):
        super().__init__("static_grid_map_node")

        default_model_path = self._default_model_path()
        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("origin_x", -10.0)
        self.declare_parameter("origin_y", -10.0)
        self.declare_parameter("width", 400)
        self.declare_parameter("height", 400)
        self.declare_parameter("occupied_height_min", 0.05)
        self.declare_parameter("occupied_height_max", 2.0)
        self.declare_parameter("collision_groups", [3])
        self.declare_parameter("inflation_radius", 0.0)
        self.declare_parameter("publish_period", 1.0)
        self.declare_parameter("save_map", False)
        self.declare_parameter("save_dir", "/tmp")
        self.declare_parameter("save_name", "static_map")

        self.model_path = str(self.get_parameter("model_path").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.origin_x = float(self.get_parameter("origin_x").value)
        self.origin_y = float(self.get_parameter("origin_y").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.occupied_height_min = float(self.get_parameter("occupied_height_min").value)
        self.occupied_height_max = float(self.get_parameter("occupied_height_max").value)
        self.collision_groups = self._as_int_set(self.get_parameter("collision_groups").value)
        self.inflation_radius = float(self.get_parameter("inflation_radius").value)
        publish_period = float(self.get_parameter("publish_period").value)

        self.publisher = self.create_publisher(OccupancyGrid, self.map_topic, 1)
        self.grid = self._build_grid()
        self.map_msg = self._make_map_msg(self.grid)

        if bool(self.get_parameter("save_map").value):
            self._save_map_files(self.grid)

        self.timer = self.create_timer(publish_period, self._publish_map)
        self._publish_map()

    def _default_model_path(self) -> str:
        try:
            share_dir = get_package_share_directory("mujoco_ros2_bridge")
            return os.path.join(share_dir, "model", "robot.xml")
        except Exception:
            return os.path.join(
                os.getcwd(),
                "src",
                "mujoco_ros2_bridge",
                "model",
                "robot.xml",
            )

    def _as_int_set(self, value) -> set[int]:
        if isinstance(value, (list, tuple)):
            return {int(v) for v in value}
        return {int(value)}

    def _build_grid(self) -> np.ndarray:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

        model = mujoco.MjModel.from_xml_path(self.model_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        grid = np.zeros((self.height, self.width), dtype=np.int8)
        occupied_geoms = 0

        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) not in self.collision_groups:
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue

            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue

            triangles = self._mesh_triangles_world(model, data, geom_id, mesh_id)
            marked = self._rasterize_triangles(grid, triangles)
            if marked:
                occupied_geoms += 1

        if self.inflation_radius > 0.0:
            grid = self._inflate_grid(grid, self.inflation_radius)

        occupied_cells = int(np.count_nonzero(grid == 100))
        self.get_logger().info(
            f"Published static grid map from {occupied_geoms} geoms: "
            f"{self.width}x{self.height}, resolution={self.resolution:.3f} m, "
            f"occupied_cells={occupied_cells}, groups={sorted(self.collision_groups)}"
        )
        return grid

    def _mesh_triangles_world(self, model, data, geom_id: int, mesh_id: int) -> np.ndarray:
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])

        verts_local = np.asarray(model.mesh_vert)[vert_adr : vert_adr + vert_num]
        faces = np.asarray(model.mesh_face)[face_adr : face_adr + face_num]

        rot = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        pos = np.asarray(data.geom_xpos[geom_id])
        verts_world = verts_local @ rot.T + pos
        return verts_world[faces]

    def _rasterize_triangles(self, grid: np.ndarray, triangles: np.ndarray) -> bool:
        marked_any = False
        for tri in triangles:
            min_z = float(np.min(tri[:, 2]))
            max_z = float(np.max(tri[:, 2]))
            if max_z < self.occupied_height_min or min_z > self.occupied_height_max:
                continue

            xs = tri[:, 0]
            ys = tri[:, 1]
            min_col = self._world_x_to_col(float(np.min(xs)))
            max_col = self._world_x_to_col(float(np.max(xs)))
            min_row = self._world_y_to_row(float(np.min(ys)))
            max_row = self._world_y_to_row(float(np.max(ys)))

            min_col = max(0, min_col)
            max_col = min(self.width - 1, max_col)
            min_row = max(0, min_row)
            max_row = min(self.height - 1, max_row)
            if min_col > max_col or min_row > max_row:
                continue

            cols = np.arange(min_col, max_col + 1)
            rows = np.arange(min_row, max_row + 1)
            cell_x = self.origin_x + (cols + 0.5) * self.resolution
            cell_y = self.origin_y + (rows + 0.5) * self.resolution
            xx, yy = np.meshgrid(cell_x, cell_y)

            mask = self._points_inside_triangle(xx, yy, tri[:, :2])
            if np.any(mask):
                grid[min_row : max_row + 1, min_col : max_col + 1][mask] = 100
                marked_any = True
        return marked_any

    def _points_inside_triangle(self, xx: np.ndarray, yy: np.ndarray, tri_xy: np.ndarray) -> np.ndarray:
        x1, y1 = tri_xy[0]
        x2, y2 = tri_xy[1]
        x3, y3 = tri_xy[2]

        area = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if abs(float(area)) < 1e-12:
            return np.zeros_like(xx, dtype=bool)

        d1 = (xx - x2) * (y1 - y2) - (x1 - x2) * (yy - y2)
        d2 = (xx - x3) * (y2 - y3) - (x2 - x3) * (yy - y3)
        d3 = (xx - x1) * (y3 - y1) - (x3 - x1) * (yy - y1)
        has_neg = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
        has_pos = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
        return ~(has_neg & has_pos)

    def _world_x_to_col(self, x: float) -> int:
        return int(math.floor((x - self.origin_x) / self.resolution))

    def _world_y_to_row(self, y: float) -> int:
        return int(math.floor((y - self.origin_y) / self.resolution))

    def _inflate_grid(self, grid: np.ndarray, radius: float) -> np.ndarray:
        radius_cells = int(math.ceil(radius / self.resolution))
        if radius_cells <= 0:
            return grid

        occupied_rows, occupied_cols = np.nonzero(grid == 100)
        inflated = grid.copy()
        offsets = self._disk_offsets(radius_cells)
        for row, col in zip(occupied_rows, occupied_cols):
            rr = row + offsets[:, 0]
            cc = col + offsets[:, 1]
            valid = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
            inflated[rr[valid], cc[valid]] = 100
        return inflated

    def _disk_offsets(self, radius_cells: int) -> np.ndarray:
        offsets = []
        limit_sq = radius_cells * radius_cells
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= limit_sq:
                    offsets.append((dr, dc))
        return np.asarray(offsets, dtype=np.int32)

    def _make_map_msg(self, grid: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).astype(np.int8).tolist()
        return msg

    def _publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.map_msg)

    def _save_map_files(self, grid: np.ndarray):
        save_dir = str(self.get_parameter("save_dir").value)
        save_name = str(self.get_parameter("save_name").value)
        os.makedirs(save_dir, exist_ok=True)

        pgm_path = os.path.join(save_dir, f"{save_name}.pgm")
        yaml_path = os.path.join(save_dir, f"{save_name}.yaml")

        image = np.full_like(grid, 254, dtype=np.uint8)
        image[grid == 100] = 0
        with open(pgm_path, "wb") as pgm_file:
            pgm_file.write(f"P5\n{self.width} {self.height}\n255\n".encode("ascii"))
            pgm_file.write(np.flipud(image).tobytes())

        with open(yaml_path, "w", encoding="ascii") as yaml_file:
            yaml_file.write(f"image: {os.path.basename(pgm_path)}\n")
            yaml_file.write("mode: trinary\n")
            yaml_file.write(f"resolution: {self.resolution}\n")
            yaml_file.write(f"origin: [{self.origin_x}, {self.origin_y}, 0.0]\n")
            yaml_file.write("negate: 0\n")
            yaml_file.write("occupied_thresh: 0.65\n")
            yaml_file.write("free_thresh: 0.196\n")

        self.get_logger().info(f"Saved map files: {yaml_path}, {pgm_path}")


def main(args: Iterable[str] | None = None):
    rclpy.init(args=args)
    node = StaticGridMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
