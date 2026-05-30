import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_pkg_share_dir = get_package_share_directory('map_pkg')
    cost_map_params = os.path.join(
        map_pkg_share_dir, 'config', 'cost_map_params.yaml'
    )

    return LaunchDescription([
        # 1. 启动 MuJoCo 桥接节点 (仿真器)
        Node(
            package='mujoco_ros2_bridge',
            executable='mujoco_node.py',
            name='mujoco_node',
            output='screen'
        ),

        # 2. 发布固定真值栅格地图
        Node(
            package='map_pkg',
            executable='static_grid_map_node.py',
            name='static_grid_map_node',
            output='screen',
            parameters=[{
                'frame_id': 'map',
                'odom_frame_id': 'odom',
                'map_topic': '/map',
            }]
        ),

        # 3. 基于/map生成代价地图
        Node(
            package='map_pkg',
            executable='cost_map_node.py',
            name='cost_map_node',
            output='screen',
            parameters=[cost_map_params],
        ),

        # 4. 在/map_cost上规划全局路径，输出给MPC
        Node(
            package='map_pkg',
            executable='astar_planner_node.py',
            name='astar_planner_node',
            output='screen',
            parameters=[{
                'cost_map_topic': '/map_cost',
                'path_topic': '/astar/path',
                'waypoints_topic': '/astar/waypoints',
                'start_x': 0.4,
                'start_y': 0.4,
                'goal_x': 11.0,
                'goal_y': 5.0,
                'cost_weight': 5.0,
                'allow_diagonal': True,
                'replan_on_map_update': True,
            }],
        ),

        # 5. MPC订阅A*路径并发布cmd_vel控制MuJoCo小车
        Node(
            package='mujoco_ros2_bridge',
            executable='control_node.py',
            name='mpc_control_node',
            output='screen',
            parameters=[{
                'odom_topic': 'odom',
                'cmd_vel_topic': 'cmd_vel',
                'astar_path_topic': '/astar/path',
                'target_yaw': 0.0,
                'ref_speed': 1.0,
                'goal_tolerance': 0.15,
                'path_min_point_spacing': 0.10,
                'start_when_path_received': True,
            }],
        ),
        
        # odom_noise 逻辑已内聚到 mujoco_node.py，不再单独启动节点
        
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
        # 6. 启动foxglove节点
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
        )
    ])
