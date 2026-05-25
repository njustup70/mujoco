#!/usr/bin/env python3
from typing import Iterable

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.ndimage import distance_transform_edt


MAP_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class CostMapNode(Node):
    """Build and publish an inflated cost map from a static occupancy grid."""

    def __init__(self):
        super().__init__("cost_map_node")

        self.declare_parameter("robot_radius", 0.35)
        self.declare_parameter("soft_inflation_radius", 0.15)
        self.declare_parameter("cost_scaling_factor", 8.0)
        self.declare_parameter("cost_map_topic", "/map_cost")
        self.declare_parameter("publish_cost_map", True)

        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.soft_inflation_radius = float(
            self.get_parameter("soft_inflation_radius").value
        )
        self.cost_scaling_factor = float(
            self.get_parameter("cost_scaling_factor").value
        )
        self.cost_map_topic = str(self.get_parameter("cost_map_topic").value)
        self.publish_cost_map = bool(self.get_parameter("publish_cost_map").value)
        self.total_inflation_radius = (
            self.robot_radius + self.soft_inflation_radius
        )

        self.cost_map_pub = None
        if self.publish_cost_map:
            self.cost_map_pub = self.create_publisher(
                OccupancyGrid, self.cost_map_topic, MAP_QOS
            )

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_callback, MAP_QOS
        )

        self.get_logger().info(
            "Cost map node ready: "
            f"robot_radius={self.robot_radius:.2f}, "
            f"soft_inflation_radius={self.soft_inflation_radius:.2f}, "
            f"cost_scaling_factor={self.cost_scaling_factor:.2f}, "
            f"publish_cost_map={self.publish_cost_map}, "
            f"topic={self.cost_map_topic}"
        )

    def _map_callback(self, map_msg: OccupancyGrid):
        grid = np.asarray(map_msg.data, dtype=np.int16).reshape(
            map_msg.info.height, map_msg.info.width
        )
        cost_grid = self._build_cost_grid(grid, map_msg.info.resolution)
        cost_map_msg = self._make_cost_map_msg(map_msg, cost_grid)

        if self.cost_map_pub is not None:
            self.cost_map_pub.publish(cost_map_msg)

        occupied_cells = int(np.count_nonzero(grid == 100))
        lethal_cells = int(np.count_nonzero(cost_grid == 100))
        soft_cells = int(np.count_nonzero((cost_grid > 0) & (cost_grid < 100)))
        self.get_logger().info(
            "Published cost map: "
            f"occupied={occupied_cells}, lethal={lethal_cells}, soft={soft_cells}"
        )

    def _build_cost_grid(
        self, occupancy_grid: np.ndarray, resolution: float
    ) -> np.ndarray:
        occupied_mask = occupancy_grid >= 100
        unknown_mask = occupancy_grid < 0
        distances = distance_transform_edt(~occupied_mask) * resolution
        epsilon = max(1e-9, resolution * 1e-6)

        cost_grid = np.zeros_like(occupancy_grid, dtype=np.int8)
        cost_grid[unknown_mask] = -1

        lethal_mask = occupied_mask | (distances <= self.robot_radius + epsilon)
        cost_grid[lethal_mask] = 100

        soft_mask = (
            (distances > self.robot_radius + epsilon)
            & (distances <= self.total_inflation_radius + epsilon)
            & (~unknown_mask)
        )
        if np.any(soft_mask):
            soft_distance = distances[soft_mask] - self.robot_radius
            soft_cost = np.exp(-self.cost_scaling_factor * soft_distance) * 100.0
            soft_cost = np.clip(np.rint(soft_cost), 1, 99).astype(np.int8)
            cost_grid[soft_mask] = soft_cost

        return cost_grid

    def _make_cost_map_msg(
        self, source_map_msg: OccupancyGrid, cost_grid: np.ndarray
    ) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = source_map_msg.header
        msg.info = source_map_msg.info
        msg.data = cost_grid.reshape(-1).astype(np.int8).tolist()
        return msg


def main(args: Iterable[str] | None = None):
    rclpy.init(args=args)
    node = CostMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
