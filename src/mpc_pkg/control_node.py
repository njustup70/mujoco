import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from mpc import ExtendedMPCModel

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        self.dt = 0.1
        self.control = ExtendedMPCModel(dt=self.dt)
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.control.set_target_point(np.array([0.0, 10.0, 5.0]))  # 设置目标点
        self.initialized = False
        # 缓存上一拍控制量（速度），避免 MPC 求解慢或失败时 cmd_vel 出现 0 或断流
        self.last_u = np.array([0.0, 0.0, 0.0])  # [V, alpha, omega]
        self.last_u_lock = None  # 可选：用 threading.Lock 若多线程写 last_u
        # 目标死区：进入后发 0 并视为到达，解决终点难以收敛/抖动
        self.declare_parameter('goal_dist_thresh', 0.02)
        self.declare_parameter('goal_yaw_thresh_rad', 0.02)
        self.goal_dist_thresh = self.get_parameter('goal_dist_thresh').value
        self.goal_yaw_thresh = self.get_parameter('goal_yaw_thresh_rad').value
        # 固定频率发布 cmd_vel（如 50Hz），用 last_u 填充，避免因 MPC 延迟导致话题“断流”被底盘当成 0
        self.cmd_publish_timer = self.create_timer(0.02, self.cmd_publish_callback)  # 50Hz
        import asyncio
        import threading
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def _goal_reached(self, x_mpc: np.ndarray) -> bool:
        """检查是否到达目标点（只依赖前 3 维：x, y, theta）"""
        ref = np.array(self.control.p_template['_p', 0, 'ref']).flatten()
        dist = np.hypot(x_mpc[0, 0] - ref[0], x_mpc[1, 0] - ref[1])
        yaw_err = abs(np.arctan2(np.sin(x_mpc[2, 0] - ref[2]), np.cos(x_mpc[2, 0] - ref[2])))
        return dist < self.goal_dist_thresh and yaw_err < self.goal_yaw_thresh

    def cmd_publish_callback(self):
        """固定频率发布 cmd_vel，MPC 直接输出速度"""
        # MPC 输出直接是速度 [V, alpha, omega]
        V = self.last_u[0]
        alpha = self.last_u[1]
        omega = self.last_u[2]
        
        # 转换为身体坐标系速度
        vx_body = V * np.cos(alpha)
        vy_body = V * np.sin(alpha)
        
        cmd_msg = Twist()
        cmd_msg.linear.x = float(vx_body)
        cmd_msg.linear.y = float(vy_body)
        cmd_msg.angular.z = float(omega)
        self.pub.publish(cmd_msg)

    def odom_callback(self, msg: Odometry):
        # 一旦到达目标，就不再处理任何 odom 更新（忽略后续的噪声）
        measured_x = msg.pose.pose.position.x
        measured_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        measured_theta = np.arctan2(siny_cosp, cosy_cosp)
        
        # 构建 3 维状态向量 [x, y, theta]
        x_mpc = np.array([[measured_x], [measured_y], [measured_theta]])

        if not self.initialized:
            self.control.set_state_init(x_mpc)
            self.initialized = True
            return

        # 进入目标死区则发 0，**并上锁**，后续不再启动 MPC
        if self._goal_reached(x_mpc):
            self.last_u = np.array([0.0, 0.0, 0.0])
            return

        import asyncio
        future = asyncio.run_coroutine_threadsafe(self.control.async_update(x_mpc), self.loop)
        try:
            u = future.result(timeout=self.dt * 2.0)  # 略大于一步周期，避免无限等
        except Exception:
            # 求解超时或异常时保留上一拍控制，不置 0
            return
        u = np.asarray(u).flatten()
        if u.size >= 3:
            self.last_u = u

def main():
    import rclpy
    rclpy.init()
    node = MPCControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()