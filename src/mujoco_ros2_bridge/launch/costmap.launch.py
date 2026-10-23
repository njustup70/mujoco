"""启动代价地图节点，参数从 config/costmap.yaml 读取。"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('mujoco_ros2_bridge'), 'config', 'costmap.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='代价地图参数文件，默认 config/costmap.yaml'
        ),
        Node(
            package='mujoco_ros2_bridge',
            executable='costmap_node.py',
            name='costmap_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
