#!/usr/bin/env python3
"""局部规划节点：DWA 采样 + 打分 -> /local_plan

订阅
    /odom                     nav_msgs/Odometry
    /plan                     nav_msgs/Path        全局路径（取前视点）
    /global_costmap/costmap   nav_msgs/OccupancyGrid
发布
    /local_plan               nav_msgs/Path        未来 horizon 秒的轨迹

说明：本节点只输出轨迹，由 MPC 负责跟踪。二者不抢 /cmd_vel。
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from grid_utils import grid_to_numpy, make_path_msg, yaw_from_quaternion


class DWAPlanner(Node):
    def __init__(self):
        super().__init__("dwa_planner")

        # ---- 视界与频率 ----
        self.declare_parameter("horizon", 1.5)        # 前向模拟时长 [s]
        self.declare_parameter("sim_dt", 0.05)        # 积分步长 [s]
        self.declare_parameter("control_rate", 10.0)  # 重规划频率 [Hz]
        # ---- 速度与加速度约束（要与 MPC 模型对齐）----
        self.declare_parameter("v_max", 1.2)
        self.declare_parameter("v_min", 0.0)
        self.declare_parameter("w_max", 1.5)
        self.declare_parameter("a_max", 1.0)
        self.declare_parameter("alpha_max", 2.0)
        # ---- 采样数 ----
        self.declare_parameter("n_vx", 5)
        self.declare_parameter("n_vy", 5)
        self.declare_parameter("n_wz", 7)
        # ---- 打分权重 ----
        self.declare_parameter("w_goal", 1.0)
        self.declare_parameter("w_clear", 2.0)
        self.declare_parameter("w_vel", 0.3)
        self.declare_parameter("w_path", 0.5)
        # ---- 其他 ----
        self.declare_parameter("lookahead", 1.5)       # 沿全局路径前视距离 [m]
        self.declare_parameter("lethal_threshold", 98)

        gp = self.get_parameter
        self.horizon = float(gp("horizon").value)
        self.sim_dt = float(gp("sim_dt").value)
        self.rate = float(gp("control_rate").value)
        self.v_max = float(gp("v_max").value)
        self.v_min = float(gp("v_min").value)
        self.w_max = float(gp("w_max").value)
        self.a_max = float(gp("a_max").value)
        self.alpha_max = float(gp("alpha_max").value)
        self.n_vx = int(gp("n_vx").value)
        self.n_vy = int(gp("n_vy").value)
        self.n_wz = int(gp("n_wz").value)
        self.w_goal = float(gp("w_goal").value)
        self.w_clear = float(gp("w_clear").value)
        self.w_vel = float(gp("w_vel").value)
        self.w_path = float(gp("w_path").value)
        self.lookahead = float(gp("lookahead").value)
        self.lethal = int(gp("lethal_threshold").value)

        self.steps = max(1, int(round(self.horizon / self.sim_dt)))

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Path, "/plan", self.on_plan, 10)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 self.on_map, qos)
        self.pub = self.create_publisher(Path, "/local_plan", 10)

        self.pose = None       # (x, y, yaw)
        self.vel = None        # (vx, vy, wz) 车体系
        self.path = None       # np.ndarray (N, 2)
        self.cost = None
        self.info = None

        self.create_timer(1.0 / self.rate, self.step)
        self.get_logger().info(
            f"DWA 局部规划已启动: horizon={self.horizon}s, "
            f"采样 {self.n_vx}x{self.n_vy}x{self.n_wz}")

    # ---------- 回调 ----------
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        t = msg.twist.twist
        self.pose = (p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation))
        self.vel = (t.linear.x, t.linear.y, t.angular.z)

    def on_plan(self, msg: Path):
        if not msg.poses:
            self.path = None
            return
        self.path = np.array([[q.pose.position.x, q.pose.position.y]
                              for q in msg.poses], dtype=np.float64)

    def on_map(self, msg: OccupancyGrid):
        self.info = msg.info
        self.cost = grid_to_numpy(msg).astype(np.float32)

    # ---------- 前视点 ----------
    def lookahead_point(self):
        """沿全局路径从最近点往前取 lookahead 距离处的点"""
        if self.path is None or len(self.path) < 2:
            return None
        x, y = self.pose[0], self.pose[1]
        d2 = (self.path[:, 0] - x) ** 2 + (self.path[:, 1] - y) ** 2
        k0 = int(np.argmin(d2))
        acc = 0.0
        for k in range(k0, len(self.path) - 1):
            acc += float(np.hypot(*(self.path[k + 1] - self.path[k])))
            if acc >= self.lookahead:
                return self.path[k + 1]
        return self.path[-1]

    # ---------- 主循环 ----------
    def step(self):
        if self.pose is None or self.vel is None or self.cost is None:
            return
        goal = self.lookahead_point()
        if goal is None:
            return

        best = self._sample_and_score(goal)
        if best is None:
            self.get_logger().warn("DWA 无可行轨迹（全部碰撞）", throttle_duration_sec=1.0)
            return

        traj, _cmd = best
        pts = [(float(p[0]), float(p[1])) for p in traj]
        msg = make_path_msg(pts, "map", self.get_clock().now().to_msg())
        self.pub.publish(msg)

    def _sample_and_score(self, goal):
        x0, y0, th0 = self.pose
        vx0, vy0, wz0 = self.vel
        # 1. 采样窗口：按整个 horizon 可达的速度范围，而不是单帧 a*dt。
        # 若按单帧算，从静止起步每帧只能采到 0.1 m/s，轨迹只有 0.15 m，
        # MPC 拿不到足够长的参考，闭环会锁死在低速。
        span = self.a_max * self.horizon
        vx_lo = max(self.v_min, vx0 - span)
        vx_hi = min(self.v_max, vx0 + span)
        vy_lo = max(-self.v_max, vy0 - span)
        vy_hi = min(self.v_max, vy0 + span)
        wspan = self.alpha_max * self.horizon
        wz_lo = max(-self.w_max, wz0 - wspan)
        wz_hi = min(self.w_max, wz0 + wspan)

        vxs = np.linspace(vx_lo, vx_hi, self.n_vx)
        vys = np.linspace(vy_lo, vy_hi, self.n_vy)
        wzs = np.linspace(wz_lo, wz_hi, self.n_wz)
        gx, gy, gw = np.meshgrid(vxs, vys, wzs, indexing="ij")
        U = np.column_stack([gx.ravel(), gy.ravel(), gw.ravel()])   # (S,3)

        # 合成速度包络：vx/vy 的每个采样组合必须落在半径 v_max 的圆内。
        # 只按单轴限幅时，vx=vy=v_max 的合成速度会到 1.41*v_max，
        # 超出底盘/轮胎的稳定域（仿真里约 1.4 m/s 就开始明显侧倾）。
        speed_norm = np.hypot(U[:, 0], U[:, 1])
        keep = speed_norm <= self.v_max + 1e-9
        if not np.any(keep):
            keep[np.argmin(speed_norm)] = True
        U = U[keep]

        # 2. 前向模拟
        s = U.shape[0]
        traj = np.empty((s, self.steps + 1, 3), dtype=np.float64)
        x = np.tile([x0, y0, th0], (s, 1))
        traj[:, 0] = x
        for t in range(self.steps):
            th = x[:, 2]
            vx, vy = U[:, 0], U[:, 1]
            x = np.empty_like(x)
            x[:, 0] = traj[:, t, 0] + (vx * np.cos(th) - vy * np.sin(th)) * self.sim_dt
            x[:, 1] = traj[:, t, 1] + (vx * np.sin(th) + vy * np.cos(th)) * self.sim_dt
            x[:, 2] = traj[:, t, 2] + U[:, 2] * self.sim_dt
            traj[:, t + 1] = x

        # 3. 碰撞检查（查代价地图）
        info = self.info
        ii = ((traj[:, :, 0] - info.origin.position.x) / info.resolution).astype(np.int32)
        jj = ((traj[:, :, 1] - info.origin.position.y) / info.resolution).astype(np.int32)
        inside = ((ii >= 0) & (ii < info.width) & (jj >= 0) & (jj < info.height))
        ii = np.clip(ii, 0, info.width - 1)
        jj = np.clip(jj, 0, info.height - 1)
        vals = self.cost[jj, ii]
        vals[~inside] = 100.0
        collision = (vals >= self.lethal).any(axis=1)

        # 4. 打分
        d_goal = np.hypot(traj[:, -1, 0] - goal[0], traj[:, -1, 1] - goal[1])
        d_scale = max(float(d_goal.max()), 1e-3)
        clearance = (100.0 - vals.max(axis=1)) / 100.0
        speed = np.hypot(U[:, 0], U[:, 1]) / max(self.v_max, 1e-3)

        if self.path is not None and len(self.path) > 1:
            dx = traj[:, -1:, 0] - self.path[None, :, 0]
            dy = traj[:, -1:, 1] - self.path[None, :, 1]
            d_path = np.sqrt(dx ** 2 + dy ** 2).min(axis=1)
            path_score = 1.0 / (1.0 + d_path)
        else:
            path_score = np.ones(s)

        score = (self.w_goal * (1.0 - d_goal / d_scale)
                 + self.w_clear * clearance
                 + self.w_vel * speed
                 + self.w_path * path_score)
        score[collision] = -np.inf

        if not np.any(np.isfinite(score)):
            return None

        k = int(np.argmax(score))
        return traj[k].tolist(), U[k]


def main(args=None):
    rclpy.init(args=args)
    node = DWAPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
