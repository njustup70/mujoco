import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from mpc import MPCModel,MPCStateObserver

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        self.control=MPCModel(dt=0.01)
        self.observer=MPCStateObserver(dt=0.01)
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.pub=self.create_publisher(Twist, 'cmd_vel', 10)
        self.control.p_template['_p', 0, 'ref'] = np.array([1.0, 0.0, 0.0])
        self.vel=[0.0,0.0,0.0]
    def odom_callback(self, msg:Odometry):
        # 从 Odometry 消息中提取测量值
        measured_x = msg.pose.pose.position.x
        measured_y = msg.pose.pose.position.y
        # 从四元数提取 Yaw 角
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        measured_theta = np.arctan2(siny_cosp, cosy_cosp)
        # 更新状态观测器
        self.observer.update(z=[measured_x, measured_y, measured_theta])
        # 将观测到的状态作为 MPC 的当前状态
        u=self.control.mpc.make_step(self.observer.ekf.x)
        # 从 MPC 获取控制输入
        # u = self.control.mpc.u0
        ax= u[0][0]
        ay= u[1][0]
        alpha= u[2][0]
        # 将控制输入转换为 Twist 消息
        self.vel[0]+=ax*0.01
        self.vel[1]+=ay*0.01
        self.vel[2]+=alpha*0.01
        cmd_msg = Twist()
        cmd_msg.linear.x = self.vel[0]
        cmd_msg.linear.y = self.vel[1]
        cmd_msg.angular.z = self.vel[2]
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