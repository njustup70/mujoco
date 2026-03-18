from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 启动 MuJoCo 桥接节点 (仿真器)
        Node(
            package='mujoco_ros2_bridge',
            executable='mujoco_node.py',
            name='mujoco_node',
            output='screen'
        ),
        
        # 2. odom_noise 逻辑已内聚到 mujoco_node.py，不再单独启动节点
        
        # # 3. 启动手柄转换节点
        # Node(
        #     package='mujoco_ros2_bridge',
        #     executable='teleop_joy_node.py',
        #     name='teleop_joy_node',
        #     output='screen',
        #     parameters=[{
        #         'scale_linear': 2.0,
        #         'scale_angular': 2.0
        #     }]
        # ),
        
        # # 4. 启动 ROS 2 标准手柄驱动
        # Node(
        #     package='joy',
        #     executable='joy_node',
        #     name='joy_node',
        #     output='screen',
        #     parameters=[{
        #         'device_id': 0,
        #         'device_filepath': '/dev/input/js0',
        #         'deadzone': 0.1,
        #         'autorepeat_rate': 20.0,
        #     }]
        # ),
        #启动foxglove节点
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
        )
    ])
