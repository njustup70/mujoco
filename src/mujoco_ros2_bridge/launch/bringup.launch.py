"""一键启动：MuJoCo + 代价地图 + A* + DWA + MPC（可选 Foxglove）。

用法：
    ros2 launch mujoco_ros2_bridge bringup.launch.py
    ros2 launch mujoco_ros2_bridge bringup.launch.py scene:=scene_demo.xml v_max:=1.2 ref_speed:=1.0
    ros2 launch mujoco_ros2_bridge bringup.launch.py mpc_dir:=/path/to/src/mpc_pkg use_foxglove:=false

然后在另一个终端发终点：
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "map"}, pose: {position: {x: 8.0, y: 5.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}'
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _default_mpc_dir() -> str:
    """control_node.py 不在 ROS 包里，按常见工作区布局找一个可用目录。"""
    candidates = [
        os.path.join(os.getcwd(), 'src', 'mpc_pkg'),
        os.path.expanduser('~/ros2_ws/src/mpc_pkg'),
        '/home/Elaina/ros2_ws/src/mpc_pkg',
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, 'control_node.py')):
            return path
    return candidates[0]


def generate_launch_description():
    share = get_package_share_directory('mujoco_ros2_bridge')
    costmap_params = os.path.join(share, 'config', 'costmap.yaml')
    nav_params = os.path.join(share, 'config', 'nav.yaml')

    scene = LaunchConfiguration('scene')
    v_max = LaunchConfiguration('v_max')
    ref_speed = LaunchConfiguration('ref_speed')
    mpc_dir = LaunchConfiguration('mpc_dir')
    use_foxglove = LaunchConfiguration('use_foxglove')

    return LaunchDescription([
        DeclareLaunchArgument(
            'scene', default_value='scene_demo.xml',
            description='MuJoCo 场景，mujoco_node 与 costmap_node 共用'),
        DeclareLaunchArgument(
            'v_max', default_value='1.2',
            description='合成速度上限 [m/s]，同时约束 DWA 与 MPC'),
        DeclareLaunchArgument(
            'ref_speed', default_value='1.0',
            description='MPC 参考速度 [m/s]，建议 <= v_max，跟踪滞后时先降它'),
        DeclareLaunchArgument(
            'mpc_dir', default_value=_default_mpc_dir(),
            description='mpc_pkg 目录（含 control_node.py）'),
        DeclareLaunchArgument(
            'use_foxglove', default_value='true',
            description='是否启动 foxglove_bridge；没装就传 false'),

        LogInfo(msg=['[bringup] mpc_dir = ', mpc_dir]),

        # 1. 仿真
        Node(
            package='mujoco_ros2_bridge',
            executable='mujoco_node.py',
            name='mujoco_node',
            output='screen',
            parameters=[{'scene': ParameterValue(scene, value_type=str)}],
        ),

        # 2. 代价地图
        Node(
            package='mujoco_ros2_bridge',
            executable='costmap_node.py',
            name='costmap_node',
            output='screen',
            parameters=[
                costmap_params,
                {'scene': ParameterValue(scene, value_type=str)},
            ],
        ),

        # 3. 全局规划
        Node(
            package='mujoco_ros2_bridge',
            executable='astar_planner.py',
            name='astar_planner',
            output='screen',
            parameters=[nav_params],
        ),

        # 4. 局部规划（v_max 与 MPC 保持一致）
        Node(
            package='mujoco_ros2_bridge',
            executable='dwa_planner.py',
            name='dwa_planner',
            output='screen',
            parameters=[
                nav_params,
                {'v_max': ParameterValue(v_max, value_type=float)},
            ],
        ),

        # 5. MPC：control_node.py 不是 ROS 包，用 ExecuteProcess 指定 cwd 运行
        ExecuteProcess(
            cmd=[
                'python3', 'control_node.py',
                '--ros-args',
                '-p', ['v_max:=', v_max],
                '-p', ['ref_speed:=', ref_speed],
            ],
            cwd=mpc_dir,
            name='mpc_control_node',
            output='screen',
        ),

        # 6. 可选可视化
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            condition=IfCondition(use_foxglove),
        ),
    ])
