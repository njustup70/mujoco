from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('mujoco_ros2_bridge')
    
    return LaunchDescription([
        # 1. 启动 MuJoCo 桥接节点
        Node(
            package='mujoco_ros2_bridge',
            executable='mujoco_node',
            name='mujoco_node',
            output='screen'
        ),
        
        # 2. 启动手柄转换节点
        Node(
            package='mujoco_ros2_bridge',
            executable='teleop_joy_node',
            name='teleop_joy_node',
            output='screen',
            parameters=[{
                'scale_linear': 2.0,
                'scale_angular': 2.0
            }]
        ),
        
        # 3. 可选：启动 ROS 2 标准手柄驱动 (需要先安装 ros-humble-joy)
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        )
    ])
