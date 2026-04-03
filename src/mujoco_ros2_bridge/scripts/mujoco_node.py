#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
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

        # 0. 简洁默认配置（不依赖 ROS 参数读取）
        self.use_viewer = True
        self.wheel_steer_noise_std = 0.008
        self.wheel_drive_noise_std = 0.1
        self.wheel_steer_lag_alpha = 0.0
        self.wheel_drive_lag_alpha = 0.0
        self.noise_cfg = OdomNoiseConfig(
            std_pos_100hz=0.02,
            std_ori_100hz=0.02,
            std_pos_10hz=0.01,
            std_ori_10hz=0.01,
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
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        # 发布带噪声里程计与 TF
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.base_link_state_pub = self.create_publisher(Vector3Stamped, '/state/base_link', 10)
        self.real_vel_pub = self.create_publisher(Vector3Stamped, '/mujoco/real_vel', 10)

        # 3. 状态变量
        self.target_v_x = 0.0
        self.target_v_y = 0.0
        self.target_v_yaw = 0.0
        
        self.wheels_pos = [(0.25, 0.2), (0.25, -0.2), (-0.25, 0.2), (-0.25, -0.2)]
        self.wheel_radius = 0.08
        
        # 舵关节在 qpos 中的下标（用于读取当前实际舵角）
        self.steer_qposadr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}_steer')]
            for i in range(4)
        ]

        # 舵轮解算器：内部管理一阶滞后和噪声
        self.swerve_solver = SwerveSolver(
            wheels_pos=self.wheels_pos,
            wheel_radius=self.wheel_radius,
            steer_lag_alpha=self.wheel_steer_lag_alpha,
            drive_lag_alpha=self.wheel_drive_lag_alpha,
            steer_noise_std=self.wheel_steer_noise_std,
            drive_noise_std=self.wheel_drive_noise_std,
        )

        # 4. 定时器：100Hz 发布真值与带噪声 odom/TF
        self.timer = self.create_timer(0.01, self.publish_truth_callback)

        # 5. 启动仿真线程
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.simulation_loop)
        self.thread.start()

    @staticmethod
    def _sim_time_to_msg(sim_time_sec: float) -> TimeMsg:
        # 使用 MuJoCo 仿真时间作为消息时间戳，保证下游用 stamp 求 dt 时与物理积分一致。
        sec = int(sim_time_sec)
        nanosec = int(round((sim_time_sec - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        msg = TimeMsg()
        msg.sec = sec
        msg.nanosec = nanosec
        return msg

    def cmd_vel_callback(self, msg):
        self.target_v_x = msg.linear.x
        self.target_v_y = msg.linear.y
        self.target_v_yaw = msg.angular.z

    def simulation_loop(self):
        viewer = None
        
        # 仅在启用 viewer 且存在 DISPLAY 环境变量时尝试启动
        if self.use_viewer and os.environ.get('DISPLAY'):
            try:
                viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.get_logger().info("MuJoCo viewer launched successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to launch MuJoCo viewer: {e}. Running in headless mode.")
                viewer = None
        else:
            self.get_logger().info("Running in headless mode (no viewer).")

        while not self.stop_event.is_set():
            if viewer is not None and not viewer.is_running():
                break
                
            step_start = time.time()
            
            # 1. 舵轮运动解算：根据速度指令得到目标舵角和速度
            target_steer_list, target_drive_list = zip(*self.swerve_solver.solve(
                self.target_v_x,
                self.target_v_y,
                self.target_v_yaw,
            ))
            
            # 2. 读取当前实际舵角
            current_steer_angles = [self.data.qpos[self.steer_qposadr[i]] for i in range(4)]
            
            # 3. 应用电机动力学：一阶滞后 + 响应噪声
            motor_commands = self.swerve_solver.apply_motor_dynamics(
                list(target_steer_list),
                list(target_drive_list),
                current_steer_angles
            )
            
            # 4. 下达电机指令
            for i, (steer_ctrl, drive_ctrl) in enumerate(motor_commands):
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
        stamp_msg = self._sim_time_to_msg(float(self.data.time))
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
        
        # 1. 发布 Odometry 消息
        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.pose.pose.position.x = pos[0]
        odom.pose.pose.position.y = pos[1]
        odom.pose.pose.position.z = pos[2]
        odom.pose.pose.orientation.w = quat[0]
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]
        odom.twist.twist.linear.x = body_vel[3]
        odom.twist.twist.linear.y = body_vel[4]
        odom.twist.twist.linear.z = body_vel[5]
        odom.twist.twist.angular.x = body_vel[0]
        odom.twist.twist.angular.y = body_vel[1]
        odom.twist.twist.angular.z = body_vel[2]

        # 2. 由 odom_noise_node.py 模块计算并发布 noisy odom 与 odom->base_link TF
        noisy_pos, noisy_quat, noisy_v_lin, noisy_v_ang = self.noise_gen.apply_to_truth(odom)
        noisy_yaw = float(np.arctan2(
            2.0 * (noisy_quat[3] * noisy_quat[2] + noisy_quat[0] * noisy_quat[1]),
            1.0 - 2.0 * (noisy_quat[1] * noisy_quat[1] + noisy_quat[2] * noisy_quat[2]),
        ))

        base_link_state = Vector3Stamped()
        base_link_state.header.stamp = stamp_msg
        base_link_state.header.frame_id = 'odom'
        base_link_state.vector.x = float(noisy_pos[0])
        base_link_state.vector.y = float(noisy_pos[1])
        base_link_state.vector.z = noisy_yaw
        self.base_link_state_pub.publish(base_link_state)

        real_vel_msg = Vector3Stamped()
        real_vel_msg.header.stamp = stamp_msg
        real_vel_msg.header.frame_id = 'base_link'
        real_vel_msg.vector.x = float(body_vel[3])
        real_vel_msg.vector.y = float(body_vel[4])
        real_vel_msg.vector.z = float(body_vel[2])
        self.real_vel_pub.publish(real_vel_msg)

        noisy_odom = Odometry()
        noisy_odom.header.stamp = stamp_msg
        noisy_odom.header.frame_id = 'odom'
        noisy_odom.child_frame_id = 'base_link'
        noisy_odom.pose.pose.position.x = noisy_pos[0]
        noisy_odom.pose.pose.position.y = noisy_pos[1]
        noisy_odom.pose.pose.position.z = noisy_pos[2]
        noisy_odom.pose.pose.orientation.x = noisy_quat[0]
        noisy_odom.pose.pose.orientation.y = noisy_quat[1]
        noisy_odom.pose.pose.orientation.z = noisy_quat[2]
        noisy_odom.pose.pose.orientation.w = noisy_quat[3]
        noisy_odom.twist.twist.linear.x = noisy_v_lin[0]
        noisy_odom.twist.twist.linear.y = noisy_v_lin[1]
        noisy_odom.twist.twist.linear.z = noisy_v_lin[2]
        noisy_odom.twist.twist.angular.x = noisy_v_ang[0]
        noisy_odom.twist.twist.angular.y = noisy_v_ang[1]
        noisy_odom.twist.twist.angular.z = noisy_v_ang[2]
        self.odom_pub.publish(noisy_odom)

        t = TransformStamped()
        t.header.stamp = stamp_msg
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = noisy_pos[0]
        t.transform.translation.y = noisy_pos[1]
        t.transform.translation.z = noisy_pos[2]
        t.transform.rotation.x = noisy_quat[0]
        t.transform.rotation.y = noisy_quat[1]
        t.transform.rotation.z = noisy_quat[2]
        t.transform.rotation.w = noisy_quat[3]
        self.tf_broadcaster.sendTransform(t)

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
