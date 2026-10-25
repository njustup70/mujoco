import time

import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import PoseStamped
import linear
import mpc_acados as mpc
import foxgloveTools
from state_observer import PoseVelocityObserver,PoseVelocityESO

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        self.dt = 0.1
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            0)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.cmd_state_pub = self.create_publisher(Vector3Stamped, '/state/cmd_vel', 10)
        self.observer_state_pub = self.create_publisher(Vector3Stamped, '/state/observe_vel', 10)
        self.frame_id = 'odom'
        self.max_tracked_points = 2000
        self.path_visual = foxgloveTools.PathVisual(
            self,
            frame_id=self.frame_id,
            max_len=self.max_tracked_points,
        )
        self.ref_path_topic = '/mpc/reference_path'
        self.tracked_path_topic = '/mpc/tracked_path'
        # motor 控制器的速度包络：与 DWA 的 v_max（nav.yaml）保持一致
        self.declare_parameter('v_max', 1.2)
        self.v_max = float(self.get_parameter('v_max').value)
        # 参考速度：与 DWA 的 v_max 对齐，保证 MPC 视界（1 s）不超过局部路径长度
        self.declare_parameter('ref_speed', 1.2)
        self.ref_speed = float(self.get_parameter('ref_speed').value)
        if self.ref_speed > self.v_max:
            self.get_logger().warn(
                f'ref_speed={self.ref_speed} 大于 v_max={self.v_max}，'
                f'参考速度会被速度约束截断')

        # self.control.set_target_point(np.array([0.0, 10.0, 3.0]))  # 设置目标点
        self.path_follwer = mpc.AugmentedSwerveMPC(
            0.01, n_horizon=100, v_max=self.v_max)
        # self.path_follwer=AcadosMPC(0.05,model_type='swerve',n_horizon=100)
        self.cube=linear.SplinePlanner()

        # 参考路径由 /local_plan 提供，不再硬编码
        self.has_path = False
        # /odom 实际跑到 200 Hz，而 MPC 按 dt=0.01 设计，必须节流到 100 Hz
        self.solve_period = 0.01
        self.last_solve_t = 0.0
        self.local_plan_sub = self.create_subscription(
            Path, '/local_plan', self.on_local_plan, 10)
        # 终点判断：到达位置后先转到目标朝向，朝向对齐才停车
        self.declare_parameter('goal_tolerance', 0.25)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.declare_parameter('goal_yaw_tolerance', 0.15)
        self.goal_yaw_tolerance = float(
            self.get_parameter('goal_yaw_tolerance').value)
        self.goal_xy = None
        self.goal_yaw = 0.0
        self.reached_goal = False
        self.final_aligning = False
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        # 新目标第一条局部轨迹到达时，把 MPC 的 alpha 对齐到路径方向
        self.pending_goal_start = False
        # 新目标后短暂忽略旧 local_plan，避免 MPC 先跟旧路径掉头
        self.ignore_local_plan_until = 0.0
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.on_goal, 10)

        self.ref_path_timer = self.create_timer(0.5, self._publish_reference_path_once)
        self.initialized = False
        self.state_observer = PoseVelocityObserver(
            min_dt=1e-3,
            max_dt=0.2,
            q_linear_acc=20.0,
            q_yaw_acc=4.0,
            r_pos=1e-5,
            r_yaw=2.0e-4,
            reset_threshold_pos=0.5,
            reset_threshold_yaw=0.8,
        )
        # self.state_observer=PoseVelocityESO()
        self.observed_body_velocity = np.zeros(3, dtype=float)

        # --- 新增：底层控制输出平滑（模拟物理电机的响应过程与惯性） ---
        self.last_u = np.array([0.0, 0.0, 0.0])
        self.lpf_alpha = 0.2  # 低通滤波系数(0.0~1.0)，越小底盘响应越柔和，舵轮转向过程越明显

        import asyncio,threading
        self.loop=asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.thread=threading.Thread(target=self.loop.run_forever,daemon=True)
        self.thread.start()
        # asyncio.run_coroutine_threadsafe(test(), self.loop)
        # self.server=foxglove.start_server(port=8766)

    def _publish_reference_path_once(self):
        planner = self.path_follwer.path_planner
        if len(planner.x_path) == 0:
            return

        points = [np.array([x, y, 0.0], dtype=float) for x, y in zip(planner.x_path, planner.y_path)]
        yaws = [float(yaw) for yaw in planner.yaw_path]
        self.path_visual.publish_points(self.ref_path_topic, points, yaws=yaws)

    def _append_tracked_pose(self, measured_x: float, measured_y: float, measured_theta: float):
        self.path_visual.add_point(
            self.tracked_path_topic,
            np.array([measured_x, measured_y, 0.0], dtype=float),
            yaw=float(measured_theta),
        )
    def on_goal(self, msg: PoseStamped):
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.goal_yaw = float(np.arctan2(siny, cosy))
        self.reached_goal = False
        self.final_aligning = False
        # 等 DWA 针对新目标重新规划出局部轨迹后再恢复跟踪
        self.has_path = False
        self.pending_goal_start = True
        # 收到目标立刻把运动方向对准目标，避免沿用上一个目标的 alpha
        goal_dir = float(np.arctan2(
            self.goal_xy[1] - self.current_y,
            self.goal_xy[0] - self.current_x))
        alpha0 = float(np.arctan2(
            np.sin(goal_dir - self.current_yaw),
            np.cos(goal_dir - self.current_yaw)))
        self.path_follwer.reset_aug_state(alpha0)
        self.ignore_local_plan_until = (
            self.get_clock().now().nanoseconds * 1e-9 + 0.3)
        self.get_logger().info(
            f'MPC 收到新目标 ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}), '
            f'yaw={self.goal_yaw:.2f}')

    def on_local_plan(self, msg: Path):
        """接收 DWA 输出的局部轨迹，作为 MPC 的参考路径。"""
        if self.reached_goal or self.final_aligning:
            return
        # 新目标刚下发时忽略旧轨迹，等 DWA 用新 /plan 重规划
        if self.get_clock().now().nanoseconds * 1e-9 < self.ignore_local_plan_until:
            return
        if len(msg.poses) < 2:
            return
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        # 剔除过近的重复点：样条要求弧长严格递增
        keep = [0]
        for k in range(1, len(pts)):
            if np.hypot(pts[k, 0] - pts[keep[-1], 0],
                        pts[k, 1] - pts[keep[-1], 1]) > 1e-3:
                keep.append(k)
        if len(keep) < 2:
            return
        pts = pts[keep]
        if self.pending_goal_start:
            # 新目标起步：把运动方向对齐到路径第一段，避免 MPC 先转 alpha
            dx = pts[1, 0] - pts[0, 0]
            dy = pts[1, 1] - pts[0, 1]
            path_yaw = float(np.arctan2(dy, dx))
            alpha0 = float(np.arctan2(
                np.sin(path_yaw - self.current_yaw),
                np.cos(path_yaw - self.current_yaw)))
            self.path_follwer.reset_aug_state(alpha0)
            self.pending_goal_start = False
        self.path_follwer.set_path(pts, target_yaw=0.0, ref_speed=self.ref_speed)
        if not self.has_path:
            self.get_logger().info(f'收到局部轨迹 {len(pts)} 点，开始跟踪')
        self.has_path = True

    from decorder import time_print
    # @time_print(10)
    def odom_callback(self, msg: Odometry):
        # 从 Odometry 消息中提取测量值
        measured_x = msg.pose.pose.position.x
        measured_y = msg.pose.pose.position.y
        # 从四元数提取 Yaw 角
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        measured_theta = np.arctan2(siny_cosp, cosy_cosp)
        self.current_yaw = float(measured_theta)
        self.current_x = float(measured_x)
        self.current_y = float(measured_y)

        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.observed_body_velocity = self.state_observer.update(
            measured_x,
            measured_y,
            measured_theta,
            stamp_sec=stamp_sec,
        )
        obs_msg = Vector3Stamped()
        obs_msg.header.stamp = msg.header.stamp
        obs_msg.header.frame_id = 'base_link'
        obs_msg.vector.x = float(self.observed_body_velocity[0])
        obs_msg.vector.y = float(self.observed_body_velocity[1])
        obs_msg.vector.z = float(self.observed_body_velocity[2])
        self.observer_state_pub.publish(obs_msg)

        self._append_tracked_pose(measured_x, measured_y, measured_theta)

        # 到达位置后先原地转到目标朝向，朝向对齐才停车
        if self.goal_xy is not None and not self.reached_goal:
            gx, gy = self.goal_xy
            pos_err = np.hypot(measured_x - gx, measured_y - gy)
            if self.final_aligning or pos_err < self.goal_tolerance:
                self.final_aligning = True
                yaw_err = float(np.arctan2(
                    np.sin(self.goal_yaw - measured_theta),
                    np.cos(self.goal_yaw - measured_theta)))
                if abs(yaw_err) > self.goal_yaw_tolerance:
                    cmd = Twist()
                    # 转向时用很小的位置保持，避免原地转时漂走又回来
                    cmd.linear.x = float(np.clip(
                        0.4 * (gx - measured_x), -0.1, 0.1))
                    cmd.linear.y = float(np.clip(
                        0.4 * (gy - measured_y), -0.1, 0.1))
                    # 角速度 P + D，抑制超调和来回摆
                    cmd.angular.z = float(np.clip(
                        1.2 * yaw_err - 0.3 * self.observed_body_velocity[2],
                        -0.8, 0.8))
                    self.pub.publish(cmd)
                    return
                self.reached_goal = True
                self.final_aligning = False
                self.has_path = False
                self.pub.publish(Twist())
                self.get_logger().info(
                    f'到达目标 ({gx:.2f}, {gy:.2f}) 且朝向对齐，停车')
                return

        if not self.has_path:
            return

        # 节流：只按设计频率求解
        now = time.monotonic()
        if now - self.last_solve_t < self.solve_period:
            return
        self.last_solve_t = now

        x_mpc = np.array([[measured_x], [measured_y], [measured_theta]])

        # 新模型下 U 直接是速度 [vx, vy, vw]
        
        u=self.path_follwer.update(x_mpc)
        cmd_msg = Twist()
        cmd_msg.linear.x = u[0]
        cmd_msg.linear.y = u[1]
        cmd_msg.angular.z = u[2]
        # 发布控制命令
        # if(u[0]**2+u[1]**2<1e-2):
        #     cmd_msg.angular.z=0.0  # 当线速度非常小时，直接将角速度设为0，避免不必要的旋转
        self.pub.publish(cmd_msg)
def main():
    import rclpy
    rclpy.init()
    node = MPCControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()