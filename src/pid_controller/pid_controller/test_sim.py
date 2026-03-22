import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CircleController(Node):
    def __init__(self):
        super().__init__('circle_controller')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_circle_cmd)
        self.get_logger().info('Circle controller started')

    def publish_circle_cmd(self):
        msg = Twist()
        msg.linear.x = 0.5
        msg.angular.z = 0.5
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        # 先停掉定时器，防止继续发转圈命令
        if self.timer is not None:
            self.timer.cancel()

        msg = Twist()   # 全 0
        # 连发几次，更稳一点
        for _ in range(5):
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05)

        self.get_logger().info('Published zero cmd_vel, robot should stop')


def main(args=None):
    rclpy.init(args=args)
    node = CircleController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            # 只在 context 还活着时停车
            if rclpy.ok():
                node.stop_robot()
        except Exception as e:
            print(f'Failed to stop robot cleanly: {e}')
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()