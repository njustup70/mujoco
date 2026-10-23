"""启动 A* 全局规划与 DWA 局部规划，参数来自 config/nav.yaml。"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('mujoco_ros2_bridge'), 'config', 'nav.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='A*/DWA 参数文件，默认 config/nav.yaml'
        ),
        Node(
            package='mujoco_ros2_bridge',
            executable='astar_planner.py',
            name='astar_planner',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
        Node(
            package='mujoco_ros2_bridge',
            executable='dwa_planner.py',
            name='dwa_planner',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
