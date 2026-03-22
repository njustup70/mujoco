import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
class GoalSetter(Node):
    def __init__(self):
        super().__init__('target_setter')
        self.goal_pub = self.create_publisher(Vector3Stamped, 'target_pose', 10)
        self.timer = self.create_timer(1.0, self.publish_goal)
        self.get_logger().info('Target setter started')

    def publish_goal(self):
        goal_msg = Vector3Stamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.vector.x = 2.0
        goal_msg.vector.y = 2.0
        goal_msg.vector.z = 3.14
        self.goal_pub.publish(goal_msg)
        self.get_logger().info('Published new target pose: (2.0, 2.0)')
        
def main(args=None):
    rclpy.init(args=args)
    node = GoalSetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()