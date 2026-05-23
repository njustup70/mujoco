#!/usr/bin/env python3
import json
import os
from typing import Iterable

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster


STATIC_MAP_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

def _default_config_path() -> str:
    try:
        package_share_dir = get_package_share_directory("mujoco_ros2_bridge")
        return os.path.join(
            package_share_dir, "config", "static_map_obstacles.json"
        )
    except Exception:
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_dir, "config", "static_map_obstacles.json")


class StaticGridMapNode(Node):
    """Publish the fixed field occupancy grid without simulator dependencies."""

    def __init__(self):
        super().__init__("static_grid_map_node")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("obstacle_config", _default_config_path())
        self.declare_parameter("save_map", False)
        self.declare_parameter("save_dir", "/tmp")
        self.declare_parameter("save_name", "static_map")

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.config_path = str(self.get_parameter("obstacle_config").value)
        self.map_definition = self._load_map_definition(self.config_path)

        map_config = self.map_definition["map"]
        self.obstacles = self.map_definition["obstacles"]
        self.resolution = float(map_config["resolution"])
        self.origin_x = float(map_config["origin_x"])
        self.origin_y = float(map_config["origin_y"])
        self.width = int(map_config["width"])
        self.height = int(map_config["height"])

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.publisher = self.create_publisher(
            OccupancyGrid, self.map_topic, STATIC_MAP_QOS
        )
        self.grid = self._build_grid()
        self.map_msg = self._make_map_msg(self.grid)

        if bool(self.get_parameter("save_map").value):
            self._save_map_files(self.grid)

        self._publish_map_to_odom_tf()
        self._publish_map()

    def _load_map_definition(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="ascii") as config_file:
            config = json.load(config_file)

        if "map" not in config or "obstacles" not in config:
            raise ValueError(
                f"Static map config is missing required keys: {config_path}"
            )

        if not isinstance(config["obstacles"], list) or not config["obstacles"]:
            raise ValueError(f"Static map config has no obstacles: {config_path}")

        return config

    def _build_grid(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        for obstacle in self.obstacles:
            name = str(obstacle["name"])
            for box in obstacle["boxes"]:
                row_start, row_end, col_start, col_end = self._box_to_grid_indices(
                    name, box
                )
                grid[row_start : row_end + 1, col_start : col_end + 1] = 100

        occupied_cells = int(np.count_nonzero(grid == 100))
        self.get_logger().info(
            f"Published fixed static grid map: "
            f"{self.width}x{self.height}, resolution={self.resolution:.3f} m, "
            f"occupied_cells={occupied_cells}, obstacles={len(self.obstacles)}, "
            f"config={self.config_path}"
        )
        return grid

    def _box_to_grid_indices(
        self, obstacle_name: str, box: dict
    ) -> tuple[int, int, int, int]:
        x_min = float(box["x_min"])
        x_max = float(box["x_max"])
        y_min = float(box["y_min"])
        y_max = float(box["y_max"])

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Invalid box for obstacle '{obstacle_name}': {box}"
            )

        col_start = self._aligned_index(
            (x_min - self.origin_x) / self.resolution,
            obstacle_name,
            "x_min",
        )
        col_end_exclusive = self._aligned_index(
            (x_max - self.origin_x) / self.resolution,
            obstacle_name,
            "x_max",
        )
        row_start = self._aligned_index(
            (y_min - self.origin_y) / self.resolution,
            obstacle_name,
            "y_min",
        )
        row_end_exclusive = self._aligned_index(
            (y_max - self.origin_y) / self.resolution,
            obstacle_name,
            "y_max",
        )

        if not (
            0 <= col_start < col_end_exclusive <= self.width
            and 0 <= row_start < row_end_exclusive <= self.height
        ):
            raise ValueError(
                f"Obstacle '{obstacle_name}' is outside the map bounds: {box}"
            )

        return (
            row_start,
            row_end_exclusive - 1,
            col_start,
            col_end_exclusive - 1,
        )

    def _aligned_index(
        self, value: float, obstacle_name: str, field_name: str
    ) -> int:
        rounded = round(value)
        if abs(value - rounded) > 1e-6:
            raise ValueError(
                f"Obstacle '{obstacle_name}' field '{field_name}' is not aligned "
                f"to the {self.resolution:.3f} m grid: {value}"
            )
        return int(rounded)

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
        self.publisher.publish(self.map_msg)

    def _publish_map_to_odom_tf(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.frame_id
        transform.child_frame_id = self.odom_frame_id
        transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(transform)

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
