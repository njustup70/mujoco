#!/usr/bin/env python3
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3Stamped
import mpc
import foxgloveTools
from state_observer import PoseVelocityObserver,PoseVelocityESO

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('astar_path_topic', '/astar/path')
        self.declare_parameter('target_yaw', 0.0)
        self.declare_parameter('ref_speed', 1.0)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('path_min_point_spacing', 0.10)
        self.declare_parameter('start_when_path_received', True)

        self.dt = 0.1
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.astar_path_topic = str(self.get_parameter('astar_path_topic').value)
        self.target_yaw = float(self.get_parameter('target_yaw').value)
        self.ref_speed = float(self.get_parameter('ref_speed').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.path_min_point_spacing = float(
            self.get_parameter('path_min_point_spacing').value
        )
        self.wait_for_path = bool(
            self.get_parameter('start_when_path_received').value
        )
        self.path_received = False
        self.goal_position: np.ndarray | None = None

        self.subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            0)
        self.path_subscription = self.create_subscription(
            Path,
            self.astar_path_topic,
            self.path_callback,
            10)
        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
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
        # self.control.set_target_point(np.array([0.0, 10.0, 3.0]))  # 设置目标点
        self.path_follwer= mpc.MPCPathFollower(0.01, type='swerve')
        # self.path_follwer=AcadosMPC(0.05,model_type='swerve',n_horizon=100)
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
        self.get_logger().info(
            f'MPC waiting for A* path on {self.astar_path_topic}; '
            f'odom={self.odom_topic}, cmd_vel={self.cmd_vel_topic}'
        )

    def path_callback(self, msg: Path):
        target_points = self._path_msg_to_points(msg)
        if target_points is None:
            return

        try:
            self.path_follwer.set_path(
                target_points,
                target_yaw=self.target_yaw,
                ref_speed=self.ref_speed,
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to set MPC path: {exc}')
            return

        self.path_received = True
        self.goal_position = target_points[-1].copy()
        self._publish_reference_path_once()
        self.get_logger().info(
            f'Received A* path for MPC: raw_points={len(msg.poses)}, '
            f'control_points={len(target_points)}, '
            f'goal=({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})'
        )

    def _path_msg_to_points(self, msg: Path) -> np.ndarray | None:
        if len(msg.poses) < 2:
            self.get_logger().warn('Ignoring A* path with fewer than 2 poses')
            return None

        points = []
        last_point = None
        min_spacing = max(0.0, self.path_min_point_spacing)
        for pose_stamped in msg.poses:
            point = np.array(
                [
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y,
                ],
                dtype=float,
            )
            if last_point is None or np.linalg.norm(point - last_point) >= min_spacing:
                points.append(point)
                last_point = point

        final_point = np.array(
            [
                msg.poses[-1].pose.position.x,
                msg.poses[-1].pose.position.y,
            ],
            dtype=float,
        )
        if not points or np.linalg.norm(final_point - points[-1]) > 1e-6:
            points.append(final_point)

        if len(points) < 2:
            self.get_logger().warn('Ignoring A* path after downsampling')
            return None

        return np.vstack(points)

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

    def _publish_zero_cmd(self):
        self.pub.publish(Twist())

    def _goal_reached(self, measured_x: float, measured_y: float) -> bool:
        if self.goal_position is None:
            return False
        current = np.array([measured_x, measured_y], dtype=float)
        return np.linalg.norm(current - self.goal_position) <= self.goal_tolerance

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

        x_mpc = np.array([[measured_x], [measured_y], [measured_theta]])

        if self.wait_for_path and not self.path_received:
            self._publish_zero_cmd()
            return

        if self._goal_reached(measured_x, measured_y):
            self._publish_zero_cmd()
            return

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
