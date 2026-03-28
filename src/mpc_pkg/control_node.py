import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import linear
from mpc import MPCModel,MPCPathFollower
import foxgloveTools
from state_observer import PoseVelocityObserver

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        self.dt = 0.1
        self.control = MPCModel(dt=self.dt)
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.mpc_speed_pub = self.create_publisher(Twist, '/mpc/control_input', 10)
        self.observer_speed_pub = self.create_publisher(Twist, '/mpc/observed_velocity', 10)
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
        self.path_follwer=MPCPathFollower(0.1,type='swerve')
        self.cube=linear.SplinePlanner()
        # 生成一条简单的路径
        target_points = np.array([[0, 0], [2, 4]])
        # self.cube.generate_path(x_pts, y_pts, step_cm=10.0)
        self.path_follwer.set_path(target_points, target_yaw=2.0, ref_speed=2.0)
        self._publish_reference_path_once()
        self.ref_path_timer = self.create_timer(0.5, self._publish_reference_path_once)
        self.initialized = False
        self.state_observer = PoseVelocityObserver(
            min_dt=1e-3,
            max_dt=0.2,
            q_linear_acc=10.0,
            q_yaw_acc=4.0,
            r_pos=5e-4,
            r_yaw=2.0e-4,
            reset_threshold_pos=0.5,
            reset_threshold_yaw=0.8,
        )
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
        obs_msg = Twist()
        obs_msg.linear.x = float(self.observed_body_velocity[0])
        obs_msg.linear.y = float(self.observed_body_velocity[1])
        obs_msg.angular.z = float(self.observed_body_velocity[2])
        self.observer_speed_pub.publish(obs_msg)

        self._append_tracked_pose(measured_x, measured_y, measured_theta)

        x_mpc = np.array([[measured_x], [measured_y], [measured_theta]])

        if not self.initialized:
            self.control.mpc.x0 = x_mpc
            self.control.mpc.set_initial_guess()
            self.path_follwer.set_state_init(x_mpc)
            self.initialized = True
            return

        # 新模型下 U 直接是速度 [vx, vy, vw]
        import asyncio
        u=asyncio.run_coroutine_threadsafe(self.path_follwer.async_update(x_mpc), self.loop).result()  # 等待结果
        cmd_msg = Twist()
        cmd_msg.linear.x = u[0]
        cmd_msg.linear.y = u[1]
        cmd_msg.angular.z = u[2]
        # 发布控制命令
        if(u[0]**2+u[1]**2<1e-2):
            cmd_msg.angular.z=0.0  # 当线速度非常小时，直接将角速度设为0，避免不必要的旋转
        self.pub.publish(cmd_msg)
        self.mpc_speed_pub.publish(cmd_msg)

def main():
    import rclpy
    rclpy.init()
    node = MPCControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()