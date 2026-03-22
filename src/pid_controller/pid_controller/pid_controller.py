import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,Vector3Stamped
from nav_msgs.msg import Odometry
import math
from .pid_algorithm import PidAlgorithm

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def quaternion_to_euler(w, x, y, z):
    # 将 MuJoCo 的 [w, x, y, z] 四元数转换为 Z-axis 偏航角 (yaw)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class PidController(Node):
    def __init__(self):
        super().__init__("pid_controller")
        #pi参数
        self.declare_parameter('kp_x', 1)
        self.declare_parameter('kp_y', 1)
        self.declare_parameter('kp_yaw', 1.5)

        self.declare_parameter('ki_x', 0.02)
        self.declare_parameter('ki_y', 0.02)
        self.declare_parameter('ki_yaw', 0.02)
        
        self.declare_parameter('kd_x', 0.4)
        self.declare_parameter('kd_y', 0.4)
        self.declare_parameter('kd_yaw', 0.4)
        #限速
        self.declare_parameter('max_linear_x', 1.0)
        self.declare_parameter('max_linear_y', 1.0)
        self.declare_parameter('max_angular_z', 1.0)
        #积分限幅
        self.declare_parameter('max_integral_x', 0.5)
        self.declare_parameter('max_integral_y', 0.5)
        self.declare_parameter('max_integral_yaw', 0.5)
        #到位阈值
        self.declare_parameter('pos_tolerance', 0.05)
        self.declare_parameter('yaw_tolerance', 0.05)
        #误差容许
        self.declare_parameter('body_error_threshold', 0.005)
        self.declare_parameter('yaw_error_threshold', 0.01)
        
        self.kp_x = self.get_parameter('kp_x').value
        self.kp_y = self.get_parameter('kp_y').value
        self.kp_yaw = self.get_parameter('kp_yaw').value
        
        self.ki_x = self.get_parameter('ki_x').value
        self.ki_y = self.get_parameter('ki_y').value
        self.ki_yaw = self.get_parameter('ki_yaw').value
        
        self.kd_x = self.get_parameter('kd_x').value
        self.kd_y = self.get_parameter('kd_y').value
        self.kd_yaw = self.get_parameter('kd_yaw').value
        
        self.max_linear_x = self.get_parameter('max_linear_x').value
        self.max_linear_y = self.get_parameter('max_linear_y').value
        self.max_angular_z = self.get_parameter('max_angular_z').value
        
        self.max_integral_x = self.get_parameter('max_integral_x').value
        self.max_integral_y = self.get_parameter('max_integral_y').value
        self.max_integral_yaw = self.get_parameter('max_integral_yaw').value
        
        self.pos_tolerance = self.get_parameter('pos_tolerance').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        
        self.body_error_threshold = self.get_parameter('body_error_threshold').value
        self.yaw_error_threshold = self.get_parameter('yaw_error_threshold').value
        
        #状态
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw = 0.0
        self.target_received = False
        
        self.last_time = self.get_clock().now()
        
        self.pid_x = PidAlgorithm(self.kp_x, self.ki_x, self.kd_x, self.max_integral_x, self.max_linear_x, self.body_error_threshold)
        self.pid_y = PidAlgorithm(self.kp_y, self.ki_y, self.kd_y, self.max_integral_y, self.max_linear_y, self.body_error_threshold)
        self.pid_yaw = PidAlgorithm(self.kp_yaw, self.ki_yaw, self.kd_yaw, self.max_integral_yaw, self.max_angular_z, self.yaw_error_threshold)

        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.target_sub = self.create_subscription(Vector3Stamped, 'target_pose', self.target_callback, 10)
        self.control_timer = self.create_timer(0.01, self.control_loop)
        self.get_logger().info("PID Controller node has been started.")
        
    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        self.current_yaw = quaternion_to_euler(quat.w, quat.x, quat.y, quat.z)
        self.odom_received = True
        
    def target_callback(self, msg: Vector3Stamped):
        new_target_x = msg.vector.x
        new_target_y = msg.vector.y
        new_target_yaw = msg.vector.z

        target_changed = (
            abs(new_target_x - self.target_x) > 1e-6 or
            abs(new_target_y - self.target_y) > 1e-6 or
            abs(normalize_angle(new_target_yaw - self.target_yaw)) > 1e-6
        )

        if target_changed:
            self.reset_pid()

        self.target_x = new_target_x
        self.target_y = new_target_y
        self.target_yaw = new_target_yaw
        self.target_received = True
        
    def reset_pid(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_yaw.reset()
        
    def publish_cmd_zero(self):
        zero_msg = Twist()
        self.cmd_vel_pub.publish(zero_msg)
        
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        
        if not self.odom_received or not self.target_received:
            self.publish_cmd_zero()
            return
        if dt <= 0.0:
            dt = 0.01  
        #计算误差
        error_x = self.target_x - self.current_x
        error_y = self.target_y - self.current_y
        error_distance = math.sqrt(error_x**2 + error_y**2)

        cos_yaw = math.cos(self.current_yaw)
        sin_yaw = math.sin(self.current_yaw)
        # error_body_x = cos_yaw * error_x + sin_yaw * error_y
        # error_body_y = -sin_yaw * error_x + cos_yaw * error_y
        
        error_yaw = normalize_angle(self.target_yaw - self.current_yaw)

        # #到位判断
        # if error_distance < self.pos_tolerance and abs(error_yaw) < self.yaw_tolerance:
        #     self.publish_cmd_zero()
        #     self.reset_pid()
        #     self.target_received = False
        #     self.get_logger().info("Target reached. Waiting for new target...")
        #     return
        
        #PID计算
        vx_world = self.pid_x.compute(error_x, dt)
        vy_world = self.pid_y.compute(error_y, dt)
        vtheta_cmd = self.pid_yaw.compute(error_yaw, dt)
        #发指令并进行坐标变换
        vx_cmd = cos_yaw * vx_world + sin_yaw * vy_world
        vy_cmd = -sin_yaw * vx_world + cos_yaw * vy_world
        cmd=Twist()
        cmd.linear.x = vx_cmd
        cmd.linear.y = vy_cmd
        cmd.angular.z = vtheta_cmd
        self.cmd_vel_pub.publish(cmd)
        
        self.get_logger().debug(
            f"Error(world): x={error_x:.3f}, y={error_y:.3f}, yaw={error_yaw:.3f}"
        )
        self.get_logger().debug(
            f"Vel(world): vx={vx_world:.3f}, vy={vy_world:.3f}, wz={vtheta_cmd:.3f}"
        )
        self.get_logger().debug(
            f"Vel(body): vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, wz={vtheta_cmd:.3f}"
        )
        
        
def main(args=None):
    rclpy.init(args=args)
    pid_controller = PidController()
    rclpy.spin(pid_controller)
    pid_controller.destroy_node()
    rclpy.shutdown()