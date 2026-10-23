#!/usr/bin/env python3
"""代价地图节点：把 MuJoCo 场景投影成栅格，再做分层膨胀。

流程
    scene.xml --(逐格垂直 raycast)--> 占据栅格 --(欧氏距离场)--> 分层代价
    cost 分级: lethal(占用) / inscribed(内切圆内,不可通行) / 膨胀带(按 decay 衰减)

发布
    /map                     OccupancyGrid  0=free 100=occupied -1=unknown
    /global_costmap/costmap  OccupancyGrid  0..100 的代价
    map -> odom              静态 TF

参数见 config/costmap.yaml。
"""
import os

import mujoco
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy import ndimage


class CostmapNode(Node):
    def __init__(self):
        super().__init__("costmap_node")

        # ---------------- 参数 ----------------
        defaults = {
            "scene": "scene_demo.xml",   # MuJoCo 场景（唯一地图来源）
            "resolution": 0.05,          # 栅格边长 [m]
            "width_m": 20.0,             # 场地边长 [m]
            "origin_x": 0.0,             # 栅格原点（左下角）世界坐标 [m]
            "origin_y": 0.0,
            "scan_height": 3.0,          # 投影射线起始高度 [m]
            "geometry_group": 0,         # 只投影该 geom group；机器人是 2
            "inscribed_radius": 0.383,   # 内切圆半径：小于它=致命
            "inflation_radius": 0.90,    # 膨胀外边界：超过它=代价 0
            "decay_power": 2.0,          # 膨胀带衰减指数，越大衰减越快
            "lethal_cost": 254,
            "inscribed_cost": 253,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        gp = self.get_parameter
        self.scene = str(gp("scene").value)
        self.res = float(gp("resolution").value)
        self.width = float(gp("width_m").value)
        self.ox = float(gp("origin_x").value)
        self.oy = float(gp("origin_y").value)
        self.scan_h = float(gp("scan_height").value)
        self.group = int(gp("geometry_group").value)
        self.inscribed = float(gp("inscribed_radius").value)
        self.inflation = float(gp("inflation_radius").value)
        self.decay = float(gp("decay_power").value)
        self.lethal = int(gp("lethal_cost").value)
        self.insc_cost = int(gp("inscribed_cost").value)

        self._validate()
        self.n = int(round(self.width / self.res))

        # ---------------- 从 MuJoCo 场景投影 ----------------
        path = self._resolve(self.scene)
        if not os.path.exists(path):
            raise FileNotFoundError(f"场景文件不存在: {path}")
        self.get_logger().info(f"场景: {path}")
        model = mujoco.MjModel.from_xml_path(path)

        occ = self._project(model)
        cost, dist = self._inflate(occ)

        # ---------------- 发布 ----------------
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", qos)
        self.pub_cost = self.create_publisher(OccupancyGrid, "/global_costmap/costmap", qos)
        self.occ, self.cost = occ, cost
        self._publish()
        self.create_timer(1.0, self._publish)

        tf = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.rotation.w = 1.0
        tf.sendTransform(t)

        free_pct = 100.0 * (~occ).sum() / occ.size
        self.get_logger().info(
            f"栅格 {self.n}x{self.n} @ {self.res} m | 原点 ({self.ox}, {self.oy}) | "
            f"障碍 {int(occ.sum())} 格 ({100 - free_pct:.1f}%)")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if bid >= 0:
            sx = float(model.body_pos[bid][0])
            sy = float(model.body_pos[bid][1])
            ci = min(max(int((sx - self.ox) / self.res), 0), self.n - 1)
            cj = min(max(int((sy - self.oy) / self.res), 0), self.n - 1)
            self.get_logger().info(
                f"膨胀 inscribed={self.inscribed} m, inflation={self.inflation} m, "
                f"decay={self.decay} | 起点 ({sx:.2f},{sy:.2f}) 处代价 {cost[cj, ci]:.0f}")

    def _validate(self):
        if self.res <= 0:
            raise ValueError("resolution 必须 > 0")
        if self.width <= 0:
            raise ValueError("width_m 必须 > 0")
        if not (0 < self.inscribed <= self.inflation):
            raise ValueError(
                f"需满足 0 < inscribed_radius <= inflation_radius，"
                f"当前 {self.inscribed} / {self.inflation}")
        if self.decay <= 0:
            raise ValueError("decay_power 必须 > 0")

    @staticmethod
    def _resolve(scene):
        """绝对路径 / 相对路径 / 包内 model 下的文件名"""
        if os.path.isabs(scene) or os.path.exists(scene):
            return scene
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(get_package_share_directory("mujoco_ros2_bridge"),
                                "model", scene)
        except Exception:
            return scene

    def _project(self, model):
        """MuJoCo 场景 -> 二维占据栅格（逐格垂直射线投影）"""
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        xs = self.ox + self.res * (np.arange(self.n) + 0.5)
        ys = self.oy + self.res * (np.arange(self.n) + 0.5)

        geomgroup = np.zeros(6, dtype=np.uint8)
        geomgroup[self.group] = 1     # 只打场景层，避开机器人
        pnt = np.zeros(3)
        vec = np.array([0.0, 0.0, -1.0])
        gid = np.array([-1], dtype=np.int32)

        occ = np.zeros((self.n, self.n), dtype=bool)
        for i, x in enumerate(xs):
            pnt[0] = x
            for j, y in enumerate(ys):
                pnt[1], pnt[2] = y, self.scan_h
                dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, -1, gid, None)
                if dist > 0.0 and gid[0] != floor_id:
                    occ[i, j] = True
        return occ

    def _inflate(self, occ):
        """占据栅格 -> 距离场 -> 分层代价"""
        dist = ndimage.distance_transform_edt(~occ) * self.res
        cost = np.zeros(occ.shape, dtype=np.float32)

        free = ~occ
        blocked = free & (dist < self.inscribed)          # 内切圆内，物理上过不去
        band = free & (dist >= self.inscribed) & (dist < self.inflation)

        cost[occ] = self.lethal
        cost[blocked] = self.insc_cost
        # 膨胀带：从 inscribed 到 inflation，按 (1-t)^decay 衰减到 0
        span = max(self.inflation - self.inscribed, 1e-9)
        t = (dist[band] - self.inscribed) / span
        cost[band] = self.insc_cost * np.power(1.0 - t, self.decay)
        return cost, dist

    def _grid_msg(self, values, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.info.resolution = self.res
        msg.info.width = self.n
        msg.info.height = self.n
        msg.info.origin.position.x = self.ox
        msg.info.origin.position.y = self.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = values.T.astype(np.int8).flatten().tolist()
        return msg

    def _publish(self):
        stamp = self.get_clock().now().to_msg()

        grid = np.zeros_like(self.occ, dtype=np.int8)
        grid[self.occ] = 100
        self.pub_map.publish(self._grid_msg(grid, stamp))

        scaled = np.clip(np.round(self.cost * 100.0 / self.lethal), 0, 100).astype(np.int8)
        self.pub_cost.publish(self._grid_msg(scaled, stamp))


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
