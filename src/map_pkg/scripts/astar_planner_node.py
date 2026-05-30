#!/usr/bin/env python3
from math import atan2, cos, sin
from typing import Iterable

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from astar import AStarPlanner, AstarConfig, grid_to_world, world_to_grid


MAP_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

PATH_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class AStarPlannerNode(Node):
    """Plan a path on /map_cost and publish world-frame path points."""

    def __init__(self):
        super().__init__("astar_planner_node")

        self.declare_parameter("cost_map_topic", "/map_cost")
        self.declare_parameter("path_topic", "/astar/path")
        self.declare_parameter("waypoints_topic", "/astar/waypoints")
        self.declare_parameter("start_x", 0.5)
        self.declare_parameter("start_y", 0.5)
        self.declare_parameter("goal_x", 11.0)
        self.declare_parameter("goal_y", 5.0)
        self.declare_parameter("cost_weight", 5.0)
        self.declare_parameter("lethal_cost", 100.0)
        self.declare_parameter("allow_diagonal", True)
        self.declare_parameter("unknown_is_obstacle", True)
        self.declare_parameter("replan_on_map_update", True)

        self.cost_map_topic = str(self.get_parameter("cost_map_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.waypoints_topic = str(self.get_parameter("waypoints_topic").value)
        self.replan_on_map_update = bool(
            self.get_parameter("replan_on_map_update").value
        )
        self.has_planned = False

        planner_config = AstarConfig(
            cost_weight=float(self.get_parameter("cost_weight").value),
            lethal_cost=float(self.get_parameter("lethal_cost").value),
            allow_diagonal=bool(self.get_parameter("allow_diagonal").value),
            unknown_is_obstacle=bool(
                self.get_parameter("unknown_is_obstacle").value
            ),
        )
        self.planner = AStarPlanner(planner_config)

        self.path_pub = self.create_publisher(Path, self.path_topic, PATH_QOS)
        self.waypoints_pub = self.create_publisher(
            PoseArray, self.waypoints_topic, PATH_QOS
        )
        self.cost_map_sub = self.create_subscription(
            OccupancyGrid, self.cost_map_topic, self._cost_map_callback, MAP_QOS
        )

        self.get_logger().info(
            "A* planner node ready: "
            f"cost_map_topic={self.cost_map_topic}, "
            f"path_topic={self.path_topic}, "
            f"waypoints_topic={self.waypoints_topic}"
        )

    def _cost_map_callback(self, map_msg: OccupancyGrid):
        if self.has_planned and not self.replan_on_map_update:
            return

        try:
            cost_grid = self._cost_grid_from_msg(map_msg)
            start = self._world_param_to_grid("start", map_msg)
            goal = self._world_param_to_grid("goal", map_msg)
            grid_path = self.planner.plan(start, goal, cost_grid)
        except ValueError as error:
            self.get_logger().error(f"A* planning input error: {error}")
            return

        if grid_path is None:
            self.get_logger().warn(
                "A* failed to find a path: "
                f"start={start}, goal={goal}, topic={self.cost_map_topic}"
            )
            return

        path_msg = self._make_path_msg(map_msg, grid_path)
        waypoints_msg = self._make_waypoints_msg(path_msg)
        self.path_pub.publish(path_msg)
        self.waypoints_pub.publish(waypoints_msg)
        self.has_planned = True

        self.get_logger().info(
            "Published A* path: "
            f"grid_points={len(grid_path)}, "
            f"frame_id={path_msg.header.frame_id}, "
            f"start={start}, goal={goal}"
        )

    def _cost_grid_from_msg(self, map_msg: OccupancyGrid) -> np.ndarray:
        width = int(map_msg.info.width)
        height = int(map_msg.info.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid map size: {width}x{height}")
        if len(map_msg.data) != width * height:
            raise ValueError(
                "map data length does not match map size: "
                f"{len(map_msg.data)} != {width * height}"
            )

        return np.asarray(map_msg.data, dtype=np.int16).reshape(height, width)

    def _world_param_to_grid(
        self, prefix: str, map_msg: OccupancyGrid
    ) -> tuple[int, int]:
        resolution = float(map_msg.info.resolution)
        if resolution <= 0.0:
            raise ValueError(f"invalid map resolution: {resolution}")

        x = float(self.get_parameter(f"{prefix}_x").value)
        y = float(self.get_parameter(f"{prefix}_y").value)
        origin = map_msg.info.origin.position
        return world_to_grid(x, y, origin.x, origin.y, resolution)

    def _make_path_msg(
        self, map_msg: OccupancyGrid, grid_path: list[tuple[int, int]]
    ) -> Path:
        path_msg = Path()
        path_msg.header = map_msg.header
        path_msg.header.stamp = self.get_clock().now().to_msg()

        origin = map_msg.info.origin.position
        resolution = float(map_msg.info.resolution)
        points = [
            grid_to_world(cell, origin.x, origin.y, resolution)
            for cell in grid_path
        ]

        for index, (x, y) in enumerate(points):
            yaw = self._path_yaw(points, index)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation = self._yaw_to_quaternion(yaw)
            path_msg.poses.append(pose)

        return path_msg

    def _make_waypoints_msg(self, path_msg: Path) -> PoseArray:
        waypoints_msg = PoseArray()
        waypoints_msg.header = path_msg.header
        waypoints_msg.poses = [pose_stamped.pose for pose_stamped in path_msg.poses]
        return waypoints_msg

    def _path_yaw(self, points: list[tuple[float, float]], index: int) -> float:
        if len(points) < 2:
            return 0.0

        if index < len(points) - 1:
            current = points[index]
            target = points[index + 1]
        else:
            current = points[index - 1]
            target = points[index]

        return atan2(target[1] - current[1], target[0] - current[0])

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        quaternion = Quaternion()
        quaternion.z = sin(yaw * 0.5)
        quaternion.w = cos(yaw * 0.5)
        return quaternion


def main(args: Iterable[str] | None = None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
