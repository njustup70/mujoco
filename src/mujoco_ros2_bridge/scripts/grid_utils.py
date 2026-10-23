#!/usr/bin/env python3
"""栅格地图与坐标转换的公共工具，被 A* 与 DWA 共用。"""
import numpy as np
from nav_msgs.msg import OccupancyGrid


def grid_to_numpy(msg: OccupancyGrid) -> np.ndarray:
    """OccupancyGrid -> (H, W) int8，arr[j, i] 对应世界坐标 (x_i, y_j)。"""
    return np.asarray(msg.data, dtype=np.int8).reshape(
        msg.info.height, msg.info.width)


def world_to_grid(x: float, y: float, info) -> tuple[int, int]:
    """世界坐标 -> 栅格索引 (i, j)。越界返回 (-1, -1)。"""
    i = int((x - info.origin.position.x) / info.resolution)
    j = int((y - info.origin.position.y) / info.resolution)
    if 0 <= i < info.width and 0 <= j < info.height:
        return i, j
    return -1, -1


def grid_to_world(i: int, j: int, info) -> tuple[float, float]:
    """栅格索引 -> 该格中心的世界坐标。"""
    x = info.origin.position.x + (i + 0.5) * info.resolution
    y = info.origin.position.y + (j + 0.5) * info.resolution
    return x, y


def make_path_msg(points, frame_id: str, stamp) -> "Path":
    """[(x, y), ...] -> nav_msgs/Path"""
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    for x, y in points:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def yaw_from_quaternion(q) -> float:
    """四元数 -> yaw"""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))
