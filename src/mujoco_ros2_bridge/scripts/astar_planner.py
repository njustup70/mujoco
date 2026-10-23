#!/usr/bin/env python3
"""全局规划节点：/goal_pose + 代价地图 -> /plan

订阅
    /global_costmap/costmap   nav_msgs/OccupancyGrid
    /goal_pose                geometry_msgs/PoseStamped
    /odom                     nav_msgs/Odometry
发布
    /plan                     nav_msgs/Path   (frame_id = map)
"""
import heapq
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from grid_utils import (grid_to_numpy, grid_to_world, make_path_msg,
                        world_to_grid, yaw_from_quaternion)

SQRT2 = math.sqrt(2.0)
NEIGHBORS = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2))


def astar(cost, passable, start, goal, cost_weight):
    """8 邻域 A*，返回 [(i, j), ...]（栅格索引）或 None。"""
    h, w = cost.shape
    si, sj = start
    gi, gj = goal
    if not (0 <= si < w and 0 <= sj < h and 0 <= gi < w and 0 <= gj < h):
        return None
    if not (passable[sj, si] and passable[gj, gi]):
        return None

    n = h * w
    g = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=bool)
    flat_cost = cost.ravel()
    flat_pass = passable.ravel()

    s = sj * w + si
    t = gj * w + gi
    if s == t:
        return [(si, sj)]

    g[s] = 0.0
    heap = [(math.hypot(si - gi, sj - gj), s)]
    push, pop = heapq.heappush, heapq.heappop

    while heap:
        _, cur = pop(heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == t:
            break
        ci, cj = cur % w, cur // w
        gc = g[cur]
        for di, dj, dc in NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= w or nj < 0 or nj >= h:
                continue
            nxt = nj * w + ni
            if closed[nxt] or not flat_pass[nxt]:
                continue
            ng = gc + dc * (1.0 + cost_weight * flat_cost[nxt])
            if ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = cur
                push(heap, (ng + math.hypot(ni - gi, nj - gj), nxt))

    if not closed[t]:
        return None

    path = []
    cur = t
    while cur != -1:
        path.append((cur % w, cur // w))
        if cur == s:
            break
        cur = parent[cur]
    path.reverse()
    return path


def smooth(path_idx, passable, iterations, alpha):
    """栅格坐标下的拉普拉斯平滑，每步校验可通行。"""
    if len(path_idx) < 3:
        return path_idx
    pts = np.asarray(path_idx, dtype=np.float64)
    h, w = passable.shape
    for _ in range(iterations):
        for k in range(1, len(pts) - 1):
            cand = pts[k] + alpha * (pts[k - 1] + pts[k + 1] - 2.0 * pts[k])
            i, j = int(round(cand[0])), int(round(cand[1]))
            if 0 <= i < w and 0 <= j < h and passable[j, i]:
                pts[k] = cand
    return [(int(round(p[0])), int(round(p[1]))) for p in pts]


def nearest_free(passable, idx, radius=6):
    """起点/终点落在障碍上时，就近找一个可通行格。"""
    h, w = passable.shape
    i0, j0 = idx
    if 0 <= i0 < w and 0 <= j0 < h and passable[j0, i0]:
        return idx
    for r in range(1, radius + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                i, j = i0 + di, j0 + dj
                if 0 <= i < w and 0 <= j < h and passable[j, i]:
                    return i, j
    return None


class AStarPlanner(Node):
    def __init__(self):
        super().__init__("astar_planner")

        self.declare_parameter("lethal_threshold", 98)   # >= 该值视为不可通行
        self.declare_parameter("cost_weight", 0.02)      # 靠近障碍的额外代价
        self.declare_parameter("smooth_iterations", 80)
        self.declare_parameter("smooth_alpha", 0.5)
        self.declare_parameter("replan_period", 3.0)
        self.declare_parameter("goal_tolerance", 0.3)

        self.lethal = int(self.get_parameter("lethal_threshold").value)
        self.cost_w = float(self.get_parameter("cost_weight").value)
        self.smooth_it = int(self.get_parameter("smooth_iterations").value)
        self.smooth_a = float(self.get_parameter("smooth_alpha").value)
        self.goal_tol = float(self.get_parameter("goal_tolerance").value)
        period = float(self.get_parameter("replan_period").value)

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 self.on_map, qos)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.pub = self.create_publisher(Path, "/plan", 10)

        self.info = None
        self.cost = None
        self.passable = None
        self.goal = None
        self.pose = None

        self.create_timer(period, self.try_plan)
        self.get_logger().info("A* 全局规划节点已启动，等待 /goal_pose")

    # ---------- 回调 ----------
    def on_map(self, msg: OccupancyGrid):
        self.info = msg.info
        self.cost = grid_to_numpy(msg).astype(np.float32)
        self.passable = self.cost < self.lethal

    def on_goal(self, msg: PoseStamped):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"收到目标 ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        self.try_plan()

    def on_odom(self, msg: Odometry):
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    # ---------- 规划 ----------
    def try_plan(self):
        if self.info is None or self.goal is None or self.pose is None:
            return

        start = world_to_grid(self.pose[0], self.pose[1], self.info)
        goal = world_to_grid(self.goal[0], self.goal[1], self.info)
        start = nearest_free(self.passable, start) if start != (-1, -1) else None
        goal = nearest_free(self.passable, goal) if goal != (-1, -1) else None
        if start is None or goal is None:
            self.get_logger().warn("起点或终点超出地图/被占据，跳过本次规划")
            return

        path = astar(self.cost, self.passable, start, goal, self.cost_w)
        if path is None:
            self.get_logger().warn("A* 无可行路径")
            return

        path = smooth(path, self.passable, self.smooth_it, self.smooth_a)
        pts = [grid_to_world(i, j, self.info) for i, j in path]

        msg = make_path_msg(pts, "map", self.get_clock().now().to_msg())
        self.pub.publish(msg)

        length = sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
                     for k in range(len(pts) - 1))
        self.get_logger().info(
            f"规划完成: {len(pts)} 点, 长度 {length:.2f} m")


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
