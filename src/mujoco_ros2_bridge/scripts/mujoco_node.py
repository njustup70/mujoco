#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
import mujoco
import mujoco.viewer
import os
import threading
import time
import numpy as np
from ament_index_python.packages import get_package_share_directory
import math
from collections import deque
from odom_noise_node import OdomNoiseConfig, OdomNoiseGenerator

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # 0. 简洁默认配置（不依赖 ROS 参数读取）
        self.use_viewer = True
        self.wheel_steer_noise_std = 0.008
        self.wheel_drive_noise_std = 0.1
        self.wheel_steer_lag_alpha = 0.4
        self.wheel_drive_lag_alpha = 0.6
        self.noise_cfg = OdomNoiseConfig(
            std_pos_100hz=0.0001,
            std_ori_100hz=0.0001,
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

        # 2. 通讯组件
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        # 发布真值里程计 (Ground Truth)
        self.odom_truth_pub = self.create_publisher(Odometry, 'odom_truth', 10)
        # 发布带噪声里程计与 TF
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 3. 状态变量
        self.target_v_x = 0.0
        self.target_v_y = 0.0
        self.target_v_yaw = 0.0
        self.smooth_v_x = 0.0
        self.smooth_v_y = 0.0
        self.smooth_v_yaw = 0.0
        
        self.wheels_pos = [(0.25, 0.2), (0.25, -0.2), (-0.25, 0.2), (-0.25, -0.2)]
        self.last_target_angles = [0.0] * 4
        # 轮子响应状态（用于一阶滞后）
        self.last_steer_ctrl = [0.0] * 4   # rad
        self.last_drive_ctrl = [0.0] * 4  # rad/s
        # 舵关节在 qpos 中的下标（用于读取当前角并做 2π 归一化，避免 180° 突变抖动）
        self.steer_qposadr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}_steer')]
            for i in range(4)
        ]

        # 控制延迟缓冲
        self.reaction_delay_steps = 40 
        self.cmd_buffers = [deque([0.0]*self.reaction_delay_steps, maxlen=self.reaction_delay_steps) for _ in range(8)]

        # 4. 定时器：100Hz 发布真值与带噪声 odom/TF
        self.timer = self.create_timer(0.01, self.publish_truth_callback)

        # 5. 启动仿真线程
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.simulation_loop)
        self.thread.start()

    def cmd_vel_callback(self, msg):
        self.target_v_x = msg.linear.x
        self.target_v_y = msg.linear.y
        self.target_v_yaw = msg.angular.z

    def simulation_loop(self):
        wheel_radius = 0.08
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
            
            # 平滑处理和死区控制
            alpha = 0.05
            self.smooth_v_x = (1 - alpha) * self.smooth_v_x + alpha * self.target_v_x
            self.smooth_v_y = (1 - alpha) * self.smooth_v_y + alpha * self.target_v_y
            self.smooth_v_yaw = (1 - alpha) * self.smooth_v_yaw + alpha * self.target_v_yaw
            
            if abs(self.smooth_v_x) < 1e-3: self.smooth_v_x = 0.0
            if abs(self.smooth_v_y) < 1e-3: self.smooth_v_y = 0.0
            if abs(self.smooth_v_yaw) < 1e-3: self.smooth_v_yaw = 0.0

            for i in range(4):
                wx, wy = self.wheels_pos[i]
                v_ix = self.smooth_v_x - self.smooth_v_yaw * wy
                v_iy = self.smooth_v_y + self.smooth_v_yaw * wx
                
                speed = math.sqrt(v_ix**2 + v_iy**2)
                target_angle = self.last_target_angles[i]
                signed_speed = speed  # 默认正向
                if speed > 0.01 or abs(self.smooth_v_yaw) > 0.01:
                    desired_deg = math.degrees(math.atan2(v_iy, v_ix))
                    # 归一化到 [-180, 180]
                    while desired_deg > 180.0:
                        desired_deg -= 360.0
                    while desired_deg < -180.0:
                        desired_deg += 360.0
                    # 两种等价表示：(舵角, 正转) 或 (舵角+180°, 反转)，选与上一帧舵角更接近的，前后/左右都避免舵轮转 180°
                    steer_a = desired_deg
                    drive_a = speed
                    steer_b = desired_deg + 180.0
                    if steer_b > 180.0:
                        steer_b -= 360.0
                    drive_b = -speed
                    last_rad = math.radians(self.last_target_angles[i])
                    def norm_diff(a_deg, b_rad):
                        d = math.radians(a_deg) - b_rad
                        while d > math.pi:
                            d -= 2.0 * math.pi
                        while d < -math.pi:
                            d += 2.0 * math.pi
                        return abs(d)
                    if norm_diff(steer_a, last_rad) <= norm_diff(steer_b, last_rad):
                        target_angle = steer_a
                        signed_speed = drive_a
                    else:
                        target_angle = steer_b
                        signed_speed = drive_b
                    self.last_target_angles[i] = target_angle

                # 目标指令
                target_steer_rad = math.radians(target_angle)
                target_drive_rads = signed_speed / wheel_radius
                # 读取当前舵角（hinge 会连续累加，可能超过 ±π），避免目标与当前差 2π 导致位置环突变抖动
                current_steer = self.data.qpos[self.steer_qposadr[i]]
                def wrap_to_near(angle, center):
                    d = angle - center
                    while d > math.pi:
                        d -= 2.0 * math.pi
                    while d < -math.pi:
                        d += 2.0 * math.pi
                    return center + d
                target_steer_rad = wrap_to_near(target_steer_rad, current_steer)
                self.last_steer_ctrl[i] = wrap_to_near(self.last_steer_ctrl[i], current_steer)
                # 一阶滞后（模拟电机/舵机响应惯性）
                self.last_steer_ctrl[i] += self.wheel_steer_lag_alpha * (target_steer_rad - self.last_steer_ctrl[i])
                self.last_drive_ctrl[i] += self.wheel_drive_lag_alpha * (target_drive_rads - self.last_drive_ctrl[i])
                # 叠加响应噪声（模拟真实误差），指令仍归一化到当前角附近
                steer_ctrl = wrap_to_near(
                    self.last_steer_ctrl[i] + np.random.normal(0.0, self.wheel_steer_noise_std),
                    current_steer
                )
                drive_ctrl = self.last_drive_ctrl[i] + np.random.normal(0.0, self.wheel_drive_noise_std)
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
        current_time = self.get_clock().now()
        pos = self.data.body('chassis').xpos.copy()
        quat = self.data.body('chassis').xquat.copy()
        
        # 1. 发布 Odometry 消息
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link_truth'
        odom.pose.pose.position.x = pos[0]
        odom.pose.pose.position.y = pos[1]
        odom.pose.pose.position.z = pos[2]
        odom.pose.pose.orientation.w = quat[0]
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]
        self.odom_truth_pub.publish(odom)

        # 2. 由 odom_noise_node.py 模块计算并发布 noisy odom 与 odom->base_link TF
        noisy_pos, noisy_quat, noisy_v_lin, noisy_v_ang = self.noise_gen.apply_to_truth(odom)

        noisy_odom = Odometry()
        noisy_odom.header.stamp = current_time.to_msg()
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
        t.header.stamp = current_time.to_msg()
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
