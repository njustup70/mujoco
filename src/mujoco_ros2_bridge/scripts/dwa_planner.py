#!/usr/bin/env python3
"""局部规划节点：DWA 采样 + 打分 -> /local_plan

订阅
    /odom                     nav_msgs/Odometry
    /plan                     nav_msgs/Path        全局路径（取前视点）
    /global_costmap/costmap   nav_msgs/OccupancyGrid
    /dynamic_obstacle         geometry_msgs/PoseStamped  动态障碍位姿（可选）
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
from std_msgs.msg import Bool

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
        self.declare_parameter("n_vx", 7)
        self.declare_parameter("n_vy", 7)
        self.declare_parameter("n_wz", 5)
        # ---- 打分权重 ----
        self.declare_parameter("w_goal", 1.0)
        self.declare_parameter("w_clear", 2.0)
        self.declare_parameter("w_vel", 0.3)
        self.declare_parameter("w_approach", 1.0)
        self.declare_parameter("w_lateral", 0.8)
        self.declare_parameter("w_stop", 0.5)
        self.declare_parameter("w_wz", 0.15)
        self.declare_parameter("w_path_max", 1.5)
        self.declare_parameter("w_path_min", 0.8)
        # ---- 其他 ----
        self.declare_parameter("lookahead", 1.5)       # 沿全局路径前视距离 [m]
        self.declare_parameter("lethal_threshold", 98)
        # 动态障碍安全半径 = 障碍半径 0.4 + 机器人半宽 0.383 + 余量 0.1
        self.declare_parameter("dynamic_safe_radius", 0.88)
        # 预警半径（dyn_score 归一化范围，不再作为唯一触发条件）
        self.declare_parameter("dynamic_warn_radius", 1.8)
        # 按相对运动预测会进入安全圈时，TTC 小于该值才逐步启用 DWA
        self.declare_parameter("dynamic_ttc", 2.0)
        self.declare_parameter("w_dynamic", 1.0)
        # 软代价范围随相对接近速度线性变化：范围 = safe + gain * closing
        self.declare_parameter("dynamic_warn_gain", 3.0)
        self.declare_parameter("dynamic_warn_min", 1.8)
        self.declare_parameter("dynamic_warn_accel_weight", 0.5)
        self.declare_parameter("dynamic_curve_k", 9.0)
        # blend 的权重映射指数：0.5 = sqrt，把权重往远处扩
        self.declare_parameter("dynamic_weight_power", 0.5)
        # blend 一阶低通系数：越小越平滑，0.25 对应约 0.4 s 时间常数
        self.declare_parameter("blend_alpha", 0.25)
        # blend 下降更慢：障碍离开后 DWA 多保持一段，避免刚绕开就回 A*
        self.declare_parameter("blend_release_alpha", 0.08)
        # blend 超过该值才从 A* 直通切到 DWA（约 1.69 m）
        self.declare_parameter("dwa_switch_blend", 0.12)

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
        self.w_approach = float(gp("w_approach").value)
        self.w_lateral = float(gp("w_lateral").value)
        self.w_stop = float(gp("w_stop").value)
        self.w_wz = float(gp("w_wz").value)
        self.w_path_max = float(gp("w_path_max").value)
        self.w_path_min = float(gp("w_path_min").value)
        self.lookahead = float(gp("lookahead").value)
        self.lethal = int(gp("lethal_threshold").value)
        self.dynamic_safe_radius = float(gp("dynamic_safe_radius").value)
        self.dynamic_warn_radius = float(gp("dynamic_warn_radius").value)
        self.dynamic_ttc = float(gp("dynamic_ttc").value)
        self.w_dynamic = float(gp("w_dynamic").value)
        self.dynamic_warn_gain = float(gp("dynamic_warn_gain").value)
        self.dynamic_warn_min = float(gp("dynamic_warn_min").value)
        self.dynamic_warn_accel_weight = float(
            gp("dynamic_warn_accel_weight").value)
        self.dynamic_warn_eff = self.dynamic_warn_radius
        self.dynamic_curve_k = float(gp("dynamic_curve_k").value)
        self.dynamic_weight_power = float(gp("dynamic_weight_power").value)
        self.blend_alpha = float(gp("blend_alpha").value)
        self.blend_release_alpha = float(gp("blend_release_alpha").value)
        self.dwa_switch_blend = float(gp("dwa_switch_blend").value)
        self.blend_smooth = 0.0

        self.steps = max(1, int(round(self.horizon / self.sim_dt)))

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Path, "/plan", self.on_plan, 10)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 self.on_map, qos)
        self.create_subscription(PoseStamped, "/dynamic_obstacle",
                                 self.on_dynamic_obstacle, 10)
        self.pub = self.create_publisher(Path, "/local_plan", 10)
        self.dwa_active_pub = self.create_publisher(Bool, "/dwa_active", 10)
        self.last_dwa_active = None

        self.pose = None       # (x, y, yaw)
        self.vel = None        # (vx, vy, wz) 车体系
        self.path = None       # np.ndarray (N, 2)
        self.last_path_idx = 0  # 上一帧最近点索引，用于单调前视
        self.cost = None
        self.info = None
        self.dynamic_obstacle = None   # (x, y)，未收到则为 None
        # 动态障碍位置/速度：位置由 /dynamic_obstacle 给出，速度用时间戳差分估计
        self.dyn_obs_x = 0.0
        self.dyn_obs_y = 0.0
        self.dyn_obs_vx = 0.0
        self.dyn_obs_vy = 0.0
        self.dyn_obs_ax = 0.0
        self.dyn_obs_ay = 0.0
        self.dyn_obs_ahead = 0.0
        self.dyn_obs_closing = 0.0
        self.dyn_obs_ux = 0.0
        self.dyn_obs_uy = 0.0
        self.dyn_obs_vx_prev_raw = None
        self.dyn_obs_vy_prev_raw = None
        self.dyn_obs_t = None

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
        # 新路径从车辆当前位置开始，索引重新从 0 单调前进
        self.last_path_idx = 0

    def on_map(self, msg: OccupancyGrid):
        self.info = msg.info
        self.cost = grid_to_numpy(msg).astype(np.float32)

    def on_dynamic_obstacle(self, msg: PoseStamped):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        now = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        # 用时间戳差分估计障碍速度，供未来位置预测使用
        if self.dyn_obs_t is not None and now > self.dyn_obs_t:
            dt = now - self.dyn_obs_t
            if dt > 1e-4:
                vx_raw = (x - self.dyn_obs_x) / dt
                vy_raw = (y - self.dyn_obs_y) / dt
                vx_raw = max(-2.0, min(2.0, vx_raw))
                vy_raw = max(-2.0, min(2.0, vy_raw))
                ax_raw = (0.0 if self.dyn_obs_vx_prev_raw is None
                          else (vx_raw - self.dyn_obs_vx_prev_raw) / dt)
                ay_raw = (0.0 if self.dyn_obs_vy_prev_raw is None
                          else (vy_raw - self.dyn_obs_vy_prev_raw) / dt)
                ax_raw = max(-3.0, min(3.0, ax_raw))
                ay_raw = max(-3.0, min(3.0, ay_raw))
                self.dyn_obs_vx = 0.7 * self.dyn_obs_vx + 0.3 * vx_raw
                self.dyn_obs_vy = 0.7 * self.dyn_obs_vy + 0.3 * vy_raw
                self.dyn_obs_ax = 0.8 * self.dyn_obs_ax + 0.2 * ax_raw
                self.dyn_obs_ay = 0.8 * self.dyn_obs_ay + 0.2 * ay_raw
                self.dyn_obs_vx_prev_raw = vx_raw
                self.dyn_obs_vy_prev_raw = vy_raw
        self.dyn_obs_x, self.dyn_obs_y, self.dyn_obs_t = x, y, now
        self.dynamic_obstacle = (x, y)

    # ---------- 前视点 ----------
    def _dynamic_blend_raw(self) -> float:
        """CPA/TTC 为主，接近速度 + 距离兜底；已超车/远离不触发。"""
        if self.dynamic_obstacle is None:
            return 0.0

        dx = self.dynamic_obstacle[0] - self.pose[0]
        dy = self.dynamic_obstacle[1] - self.pose[1]
        d_now = math.hypot(dx, dy)
        if d_now <= self.dynamic_safe_radius:
            return 1.0

        # 车体速度转到世界系
        th = self.pose[2]
        cvx = self.vel[0] * math.cos(th) - self.vel[1] * math.sin(th)
        cvy = self.vel[0] * math.sin(th) + self.vel[1] * math.cos(th)
        rvx = self.dyn_obs_vx - cvx
        rvy = self.dyn_obs_vy - cvy
        vr2 = rvx * rvx + rvy * rvy

        # 距离变化率：>0 表示仍在接近。圆柱正弦换向时 CPA 可能误判，
        # 这里用接近速度 + 距离做一个兜底，保证 DWA 至少会被唤醒。
        closing = -(dx * rvx + dy * rvy) / max(d_now, 1e-3)

        # 预警范围 = safe + v_closing*T + 0.5*T²*a_closing
        # T 是避让时间，速度/加速度都用实测值，不预设速度范围。
        ux = dx / max(d_now, 1e-3)
        uy = dy / max(d_now, 1e-3)
        self.dyn_obs_ux = ux
        self.dyn_obs_uy = uy
        acc_closing = -(self.dyn_obs_ax * ux + self.dyn_obs_ay * uy)
        acc_closing = max(acc_closing, 0.0)
        warn_eff = min(
            self.dynamic_warn_radius,
            max(self.dynamic_warn_min,
                self.dynamic_safe_radius
                + self.dynamic_warn_gain * max(closing, 0.0)
                + 0.5 * self.dynamic_warn_accel_weight
                  * self.dynamic_warn_gain ** 2 * acc_closing))
        self.dynamic_warn_eff = warn_eff

        dist_blend = 0.0
        # 整个软代价范围内按距离缓慢加权；只在仍在接近时生效。
        # 远处权重很小，不会一进范围就停；近处再逐渐增大。
        # 允许相对速度接近 0（同向跟车）时也按最小范围生效；
        # 只有明确在远离（closing < -0.1）或障碍在正后方且不接近时才不触发。
        ahead = (dx * math.cos(self.pose[2])
                 + dy * math.sin(self.pose[2]))
        self.dyn_obs_ahead = ahead
        self.dyn_obs_closing = closing
        approach = closing > 0.1 or ahead > -0.2 * d_now
        # 远处由 TTC 主导；距离兜底只在近距离使用，避免固定距离提前接管
        if (approach
                and warn_eff > self.dynamic_safe_radius
                and d_now < self.dynamic_safe_radius + 0.8):
            span = max(warn_eff - self.dynamic_safe_radius, 1e-3)
            dist_blend = float(np.clip(
                (warn_eff - d_now) / span, 0.0, 1.0))

        if vr2 < 1e-6:
            return dist_blend

        # 最近接近点（CPA）：未来是否会进安全圈
        t_cpa = -(dx * rvx + dy * rvy) / vr2
        if t_cpa <= 0.0:
            return dist_blend  # 正在远离
        d_cpa = math.hypot(dx + rvx * t_cpa, dy + rvy * t_cpa)
        if d_cpa >= self.dynamic_safe_radius:
            return dist_blend  # 最近点也在安全圈外

        # TTC 越短权重越高；和距离兜底取较大值
        cpa_blend = float(np.clip(
            1.0 - t_cpa / max(self.dynamic_ttc, 1e-3), 0.0, 1.0))
        return max(cpa_blend, dist_blend)

    def lookahead_point(self):
        """沿全局路径从最近点往前取 lookahead 距离处的点"""
        if self.path is None or len(self.path) < 2:
            return None
        x, y = self.pose[0], self.pose[1]
        n = len(self.path)
        # 只在上一帧索引附近搜索，且索引只前进，避免接近终点时
        # 最近点在路径折角两侧跳变，导致前视点突然指向后方。
        lo = max(0, self.last_path_idx - 5)
        hi = min(n, self.last_path_idx + 60)
        d2 = ((self.path[lo:hi, 0] - x) ** 2
              + (self.path[lo:hi, 1] - y) ** 2)
        k0 = lo + int(np.argmin(d2))
        k0 = max(k0, self.last_path_idx)
        self.last_path_idx = k0

        acc = 0.0
        for k in range(k0, n - 1):
            acc += float(np.hypot(*(self.path[k + 1] - self.path[k])))
            if acc >= self.lookahead:
                return self.path[k + 1]
        return self.path[-1]

    def _path_segment_clear(self) -> bool:
        """A* 前视段在未来 horizon 内是否无碰撞（静态 + 预测动态）。"""
        if self.path is None or self.info is None or self.cost is None:
            return False
        k0 = min(self.last_path_idx, len(self.path) - 1)
        seg = self.path[k0:k0 + self.steps + 1]
        if len(seg) < 2:
            return False

        info = self.info
        ii = ((seg[:, 0] - info.origin.position.x)
              / info.resolution).astype(np.int32)
        jj = ((seg[:, 1] - info.origin.position.y)
              / info.resolution).astype(np.int32)
        inside = ((ii >= 0) & (ii < info.width)
                  & (jj >= 0) & (jj < info.height))
        ii = np.clip(ii, 0, info.width - 1)
        jj = np.clip(jj, 0, info.height - 1)
        vals = self.cost[jj, ii]
        vals[~inside] = 100.0
        if (vals >= self.lethal).any():
            return False

        if self.dynamic_obstacle is not None:
            t_pred = np.arange(len(seg), dtype=np.float64) * self.sim_dt
            ox = (self.dyn_obs_x + self.dyn_obs_vx * t_pred
                  + 0.5 * self.dyn_obs_ax * t_pred * t_pred)
            oy = (self.dyn_obs_y + self.dyn_obs_vy * t_pred
                  + 0.5 * self.dyn_obs_ay * t_pred * t_pred)
            d = np.hypot(seg[:, 0] - ox, seg[:, 1] - oy)
            if (d < self.dynamic_safe_radius).any():
                return False
        return True

    def _publish_path(self, pts):
        """把点列发布成 /local_plan，减少重复的 make_path_msg 代码。"""
        msg = make_path_msg(
            [(float(p[0]), float(p[1])) for p in pts],
            "map", self.get_clock().now().to_msg())
        self.pub.publish(msg)

    # ---------- 主循环 ----------
    def step(self):
        if self.pose is None or self.vel is None or self.cost is None:
            return
        goal = self.lookahead_point()
        if goal is None:
            return

        # 先更新平滑权重；接管用原始值，避免相向快速接近时被低通延迟
        raw_blend = self._dynamic_blend_raw()
        alpha = (self.blend_alpha if raw_blend > self.blend_smooth
                 else self.blend_release_alpha)
        self.blend_smooth += alpha * (raw_blend - self.blend_smooth)
        active = raw_blend >= self.dwa_switch_blend
        if active != self.last_dwa_active:
            self.dwa_active_pub.publish(Bool(data=active))
            self.last_dwa_active = active
        # 只有 A* 前视段真的会碰撞时才切 DWA；否则保持直线，
        # 避免提前画小弧浪费横向空间。
        if self._path_segment_clear():
            k0 = min(self.last_path_idx, len(self.path) - 1)
            seg = self.path[k0:k0 + self.steps + 1]
            if len(seg) >= 2:
                self._publish_path(seg)
                return

        best = self._sample_and_score(goal)
        if best is None:
            self.get_logger().warn("DWA 无可行轨迹（全部碰撞）", throttle_duration_sec=1.0)
            # 无解时发布当前姿态附近的短路径，让 MPC 停车；
            # 不要直接 return，否则 MPC 会继续跟旧轨迹（可能含 wz）导致原地自转。
            th = self.pose[2]
            pts = [(self.pose[0], self.pose[1]),
                   (self.pose[0] + 0.02 * math.cos(th),
                    self.pose[1] + 0.02 * math.sin(th))]
            self._publish_path(pts)
            return

        traj, cmd = best
        # 纯原地自转不执行：改发 2 cm 短路径让 MPC 刹车，
        # 等下一帧再找侧移/后退解，避免“先自转再侧移”。
        if (math.hypot(float(cmd[0]), float(cmd[1])) < 0.05
                and abs(float(cmd[2])) > 0.1):
            th = self.pose[2]
            pts = [(self.pose[0], self.pose[1]),
                   (self.pose[0] + 0.02 * math.cos(th),
                    self.pose[1] + 0.02 * math.sin(th))]
            self._publish_path(pts)
            return

        self._publish_path(traj)

    def _sample_and_score(self, goal):
        x0, y0, th0 = self.pose
        vx0, vy0, wz0 = self.vel
        # 1. 采样窗口：按整个 horizon 可达的速度范围，而不是单帧 a*dt。
        # 若按单帧算，从静止起步每帧只能采到 0.1 m/s，轨迹只有 0.15 m，
        # MPC 拿不到足够长的参考，闭环会锁死在低速。
        blend = self.blend_smooth
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
        static_collision = (vals >= self.lethal).any(axis=1)
        collision = static_collision.copy()

        # 动态障碍：按估计速度预测未来 horizon 内的位置，再做碰撞检查。
        # 只查当前位姿会让避障滞后到“快撞上”才反应。
        d_obs = None
        if self.dynamic_obstacle is not None:
            t_pred = np.arange(self.steps + 1, dtype=np.float64) * self.sim_dt
            ox = (self.dyn_obs_x + self.dyn_obs_vx * t_pred
                  + 0.5 * self.dyn_obs_ax * t_pred * t_pred)
            oy = (self.dyn_obs_y + self.dyn_obs_vy * t_pred
                  + 0.5 * self.dyn_obs_ay * t_pred * t_pred)
            d_obs = np.hypot(traj[:, :, 0] - ox[None, :],
                             traj[:, :, 1] - oy[None, :])
            collision |= (d_obs < self.dynamic_safe_radius).any(axis=1)

        # 动态障碍影响程度：0=无/远，1=靠近
        # sqrt 映射把权重往远处扩：blend=0.15 时有效权重约 0.39，
        # 远处就开始轻微避让，而不是等近处才突然加权。
        blend_w = max(blend, 0.0) ** self.dynamic_weight_power
        w_path_eff = (self.w_path_max
                      - (self.w_path_max - self.w_path_min) * blend_w)
        w_dyn_eff = self.w_dynamic * blend_w

        # 4. 打分
        d_goal = np.hypot(traj[:, -1, 0] - goal[0], traj[:, -1, 1] - goal[1])
        d_scale = max(float(d_goal.max()), 1e-3)
        # costmap 是 0~254，先裁到 0~1；否则膨胀带会变成负的 clearance，
        # 让 DWA 过度远离障碍，甚至在窄通道口选零速轨迹。
        clearance = np.clip(
            (100.0 - vals.max(axis=1)) / 100.0, 0.0, 1.0)
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
                 + w_path_eff * path_score)

        # 零速惩罚：避免 DWA 在空旷处选“停在原地”的轨迹
        score -= self.w_stop * (np.hypot(U[:, 0], U[:, 1]) < 0.1)

        # 温和的原地自转惩罚，避免用打转代替避障
        score -= self.w_wz * np.abs(U[:, 2]) / max(self.w_max, 1e-3)

        # 速度障碍软约束：紧急时惩罚朝圆柱方向的速度，奖励侧向速度，
        # 让 DWA 倾向“纯侧移”而不是“往前顺带一点侧移”。
        if (self.dynamic_obstacle is not None
                and self.dyn_obs_ahead > 0.0
                and blend > 0.2):
            th = self.pose[2]
            ux_b = (math.cos(th) * self.dyn_obs_ux
                    + math.sin(th) * self.dyn_obs_uy)
            uy_b = (-math.sin(th) * self.dyn_obs_ux
                    + math.cos(th) * self.dyn_obs_uy)
            v_along = U[:, 0] * ux_b + U[:, 1] * uy_b
            v_lat = -U[:, 0] * uy_b + U[:, 1] * ux_b
            score -= (self.w_approach * np.maximum(v_along, 0.0)
                      / max(self.v_max, 1e-3) * blend)
            score += (self.w_lateral * np.abs(v_lat)
                      / max(self.v_max, 1e-3) * blend)

        # 动态障碍预警：进入 warn_radius 后，越近得分越低，促使提前远离
        if d_obs is not None:
            d_min = d_obs.min(axis=1)
            span = max(self.dynamic_warn_eff - self.dynamic_safe_radius, 1e-3)
            t = np.clip(
                (d_min - self.dynamic_safe_radius) / span, 0.0, 1.0)
            # 对数曲线：远处已有小代价，近处增长放缓，避免贴边才猛打
            dyn_score = (np.log1p(self.dynamic_curve_k * t)
                         / math.log1p(self.dynamic_curve_k))
            score += w_dyn_eff * dyn_score

        score[collision] = -np.inf

        if not np.any(np.isfinite(score)):
            # 全部轨迹碰撞：如果只是动态圆柱导致（静态可通行），
            # 选一条“最不坏”的移动轨迹，不要停车等撞。
            if d_obs is not None:
                moving = ((np.hypot(U[:, 0], U[:, 1]) > 0.1)
                          & (~static_collision))
                if np.any(moving):
                    d_min = d_obs.min(axis=1)
                    metric = np.where(
                        moving, d_min + 0.2 * speed, -np.inf)
                    k = int(np.argmax(metric))
                    return traj[k].tolist(), U[k]
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
