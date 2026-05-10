#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import tf2_ros
import mujoco
import mujoco.viewer
import os
import threading
import time
import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from ament_index_python.packages import get_package_share_directory
from odom_noise_node import OdomNoiseConfig, OdomNoiseGenerator
from swerve_solver import SwerveSolver

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # 0. 配置（只保留必要项，去除所有滞后和噪声参数）
        self.use_viewer = True
        self.noise_cfg = OdomNoiseConfig(
            std_pos_100hz=0.0002,
            std_ori_100hz=0.002,
            std_pos_10hz=0.0001,
            std_ori_10hz=0.001,
            std_vel=0.02,
        )
        self.noise_gen = OdomNoiseGenerator(self.noise_cfg)

        # 1. 加载模型
        try:
            package_share_dir = get_package_share_directory('mujoco_ros2_bridge')
            model_path = os.path.join(package_share_dir, 'model', 'robot.xml')
        except Exception:
            model_path = 'src/mujoco_ros2_bridge/model/robot.xml'

        self.get_logger().info(f'Loading MuJoCo model from: {model_path}')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.chassis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'chassis')

        # 2. 通讯组件
        self.cmd_vel_sub = self.create_subscription(Twist, '/control/cmd_vel', self.cmd_vel_callback, 10)
        self.cmd_swerve_sub = self.create_subscription(Float64MultiArray, '/control/cmd_swerve', self.cmd_swerve_callback, 10)

        # 发布 TF 和消息
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.truth_odom_pub = self.create_publisher(Odometry, '/sim/truth/odom', 10)
        self.noisy_odom_pub = self.create_publisher(Odometry, '/sim/odom', 10)
        self.real_vel_pub = self.create_publisher(Vector3Stamped, '/sim/real_vel', 10)
        self.steer_state_pub = self.create_publisher(Float64MultiArray, '/sim/steer_state', 10)

        # 3. 状态变量
        self.target_v_x = 0.0
        self.target_v_y = 0.0
        self.target_v_yaw = 0.0
        
        # 舵轮直接控制
        self.control_mode = 'chassis_vel'
        self.target_swerve_angles = [0.0, 0.0, 0.0, 0.0]
        self.target_swerve_speeds = [0.0, 0.0, 0.0, 0.0]
        self.swerve_cmd_timeout = 0.5
        self.last_swerve_cmd_time = 0.0
        
        self.wheels_pos = [(0.325, 0.325), (0.325, -0.325), (-0.325, 0.325), (-0.325, -0.325)]
        self.wheel_radius = 0.058
        
        # 舵关节顺序：FL, FR, RL, RR (按照 control_node 约定)
        self.steering_names = ['wheel0_steer', 'wheel1_steer', 'wheel2_steer', 'wheel3_steer']
        self.driving_names = ['wheel0_drive', 'wheel1_drive', 'wheel2_drive', 'wheel3_drive']
        
        self.steer_qposadr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.steering_names
        ]

        # 轮子位置也需要与舵关节顺序对应：FL, FR, RL, RR
        self.wheels_pos = [(0.325, 0.325), (0.325, -0.325), (-0.325, 0.325), (-0.325, -0.325)]

        # 舵轮解算器（只保留运动学解算，无滞后无噪声）
        self.swerve_solver = SwerveSolver(
            wheels_pos=self.wheels_pos,
            wheel_radius=self.wheel_radius,
        )

        # 4. 定时器：100Hz 发布
        self.timer = self.create_timer(0.01, self.publish_truth_callback)

        # 5. 启动仿真线程
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.simulation_loop)
        self.thread.start()

    def cmd_vel_callback(self, msg):
        self.target_v_x = msg.linear.x
        self.target_v_y = msg.linear.y
        self.target_v_yaw = msg.angular.z
        # 切换回底盘速度控制模式
        self.control_mode = 'chassis_vel'

    def cmd_swerve_callback(self, msg):
        """接收舵轮直接控制指令 [steer0, steer1, steer2, steer3, speed0, speed1, speed2, speed3]
        按照 control_node 顺序：FL, FR, RL, RR
        数值为归一化值，在 swerve_solver 中转换为物理量
        """
        if len(msg.data) >= 8:
            # 前四个为角度，后四个为速度
            self.target_swerve_angles = list(msg.data[0:4])
            self.target_swerve_speeds = list(msg.data[4:8])
            self.control_mode = 'swerve_direct'
            self.last_swerve_cmd_time = time.time()

    def simulation_loop(self):
        
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        while not self.stop_event.is_set():
            if viewer is not None and not viewer.is_running():
                break
                
            step_start = time.time()
            
            # 1. 读取当前实际舵角 (底盘反馈)
            current_steer_angles = [self.data.qpos[self.steer_qposadr[i]] for i in range(4)]
            
            # 2. 根据控制模式计算电机指令
            # 我们始终使用归一化角度和线速度 (m/s)
            # 检查舵轮指令是否超时
            current_time = time.time()
            if current_time - self.last_swerve_cmd_time > self.swerve_cmd_timeout:
                # 超时或未收到指令，设为 0
                norm_angles = [0.0] * 4
                norm_speeds = [0.0] * 4
            else:
                norm_angles = self.target_swerve_angles
                norm_speeds = self.target_swerve_speeds

            # 使用舵轮直接控制（归一化输入 -> 在 swerve_solver 内转换为物理量并做角度积分，无滞后无噪声）
            dt = float(self.model.opt.timestep)
            motor_commands = self.swerve_solver.get_direct_actuator_commands(
                norm_angles,
                norm_speeds,
                current_steer_angles,
                dt,
            )
            
            # 3. 下达电机指令给 MuJoCo 促动器 (顺序需与 steering_names 一一对应)
            for i, (steer_ctrl, drive_ctrl) in enumerate(motor_commands):
                # 利用 steering_names/driving_names 获取正确的 actuator 名称
                self.data.actuator(f'steer{i}').ctrl[0] = steer_ctrl
                self.data.actuator(f'drive{i}').ctrl[0] = drive_ctrl

            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
                
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        if viewer is not None:
            viewer.close()

    def publish_truth_callback(self):
        """发布基于 MuJoCo 仿真数据的 ROS2 消息"""
        stamp_msg = self.get_clock().now().to_msg()
        pos = self.data.body('chassis').xpos.copy()
        quat = self.data.body('chassis').xquat.copy()
        body_vel = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.chassis_body_id,
            body_vel,
            1,
        )

        # 发布舵机真实角度反馈 (用于 control_node 的流形速度分解)
        current_steer_angles = [self.data.qpos[self.steer_qposadr[i]] for i in range(4)]
        steer_msg = Float64MultiArray()
        steer_msg.data = [float(a) for a in current_steer_angles]
        self.steer_state_pub.publish(steer_msg)

        # 调用 odom_noise_node 构建所有 ROS2 消息类型
        truth_odom, truth_tf, noisy_tf, noisy_odom, real_vel = self.noise_gen.apply_to_truth(
            stamp_msg, pos, quat, body_vel
        )

        # 广播 TF 和发布消息
        self.tf_broadcaster.sendTransform(noisy_tf)
        self.truth_odom_pub.publish(truth_odom)
        self.noisy_odom_pub.publish(noisy_odom)
        self.real_vel_pub.publish(real_vel)



    def destroy_node(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
