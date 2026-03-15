import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from mpc import MPCModel
import foxglove
from foxgloveTools import  FoxgloveVisual
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
        self.control.set_target_point(np.array([0.0, 10.0, 3.0]))  # 设置目标点
        self.initialized = False
        import asyncio,threading
        self.loop=asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.thread=threading.Thread(target=self.loop.run_forever,daemon=True)
        self.thread.start()
        # asyncio.run_coroutine_threadsafe(test(), self.loop)
        # self.server=foxglove.start_server(port=8766)
        self.server=FoxgloveVisual(port=8766)
    def odom_callback(self, msg: Odometry):
        # 从 Odometry 消息中提取测量值
        measured_x = msg.pose.pose.position.x
        measured_y = msg.pose.pose.position.y
        # 从四元数提取 Yaw 角
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        measured_theta = np.arctan2(siny_cosp, cosy_cosp)

        x_mpc = np.array([[measured_x], [measured_y], [measured_theta]])

        if not self.initialized:
            self.control.mpc.x0 = x_mpc
            self.control.mpc.set_initial_guess()
            self.initialized = True
            return

        # 新模型下 U 直接是速度 [vx, vy, vw]
        # u = self.control.mpc.make_step(x_mpc)
        # vx = float(u[0][0])
        # vy = float(u[1][0])
        # vw = float(u[2][0])
        import asyncio
        u=asyncio.run_coroutine_threadsafe(self.control.async_update(x_mpc), self.loop).result()  # 等待结果
        # mpc_data=self.control.mpc.data['_p']
        # self.server.send(mpc_data, topic="/mpc_control")
        self.server.send(u,topic='/mpc_control')

        cmd_msg = Twist()
        cmd_msg.linear.x = u[0]
        cmd_msg.linear.y = u[1]
        cmd_msg.angular.z = u[2]
        # 发布控制命令
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