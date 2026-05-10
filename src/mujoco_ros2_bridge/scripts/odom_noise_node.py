#!/usr/bin/env python3
from dataclasses import dataclass
import math
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Vector3Stamped

def euler_from_quaternion(x, y, z, w):
    """从四元数转换到欧拉角 (roll, pitch, yaw)"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def quaternion_from_euler(ai, aj, ak):
    """从欧拉角 (roll, pitch, yaw) 转换到四元数 [x, y, z, w]"""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss
    return q

@dataclass
class OdomNoiseConfig:
    # 100Hz 叠加噪声（每帧刷新）
    std_pos_100hz: float = 0.01
    std_ori_100hz: float = 0.01
    # 10Hz 叠加噪声（每 10 帧刷新一次）
    std_pos_10hz: float = 0.01
    std_ori_10hz: float = 0.01
    # 速度噪声
    std_vel: float = 0.02


class OdomNoiseGenerator:
    def __init__(self, cfg: OdomNoiseConfig | None = None):
        self.cfg = cfg or OdomNoiseConfig()
        self.tick_count = 0
        self.n_pos_10 = [0.0, 0.0, 0.0]
        self.n_yaw_10 = 0.0

    def generate_noise(self, std_pos: float, std_ori: float):
        """生成随机噪声"""
        nx = np.random.normal(0, std_pos)
        ny = np.random.normal(0, std_pos)
        nyaw = np.random.normal(0, std_ori)
        return [nx, ny, 0.0], nyaw

    def apply_to_truth(self, stamp_msg, pos, quat, body_vel) -> tuple:
        """根据 MuJoCo 原始仿真数据，构建所有相关的 ROS2 消息类型。
        返回: (truth_odom, truth_tf, noisy_tf, noisy_odom, real_vel)
        """
        # 1. 基础信息解析
        _, _, true_yaw = euler_from_quaternion(quat[1], quat[2], quat[3], quat[0])

        # 2. 噪声计算 (100Hz & 10Hz)
        n_pos_100, n_yaw_100 = self.generate_noise(self.cfg.std_pos_100hz, self.cfg.std_ori_100hz)
        if self.tick_count % 10 == 0:
            self.n_pos_10, self.n_yaw_10 = self.generate_noise(self.cfg.std_pos_10hz, self.cfg.std_ori_10hz)
        
        final_pos = pos + np.array(n_pos_100) + np.array(self.n_pos_10)
        final_yaw = true_yaw + n_yaw_100 + self.n_yaw_10
        final_quat = quaternion_from_euler(0, 0, final_yaw)
        self.tick_count += 1

        # 3. 构建消息 - truth_odom
        truth_odom = Odometry()
        truth_odom.header.stamp = stamp_msg
        truth_odom.header.frame_id = 'odom'
        truth_odom.child_frame_id = 'base_link'
        truth_odom.pose.pose.position.x = float(pos[0])
        truth_odom.pose.pose.position.y = float(pos[1])
        truth_odom.pose.pose.position.z = float(pos[2])
        truth_odom.pose.pose.orientation.w = float(quat[0])
        truth_odom.pose.pose.orientation.x = float(quat[1])
        truth_odom.pose.pose.orientation.y = float(quat[2])
        truth_odom.pose.pose.orientation.z = float(quat[3])
        truth_odom.twist.twist.linear.x = float(body_vel[3])
        truth_odom.twist.twist.linear.y = float(body_vel[4])
        truth_odom.twist.twist.linear.z = float(body_vel[5])
        truth_odom.twist.twist.angular.x = float(body_vel[0])
        truth_odom.twist.twist.angular.y = float(body_vel[1])
        truth_odom.twist.twist.angular.z = float(body_vel[2])

        # 4. 构建消息 - truth_tf
        truth_tf = TransformStamped()
        truth_tf.header = truth_odom.header
        truth_tf.child_frame_id = 'base_link'
        truth_tf.transform.translation.x = truth_odom.pose.pose.position.x
        truth_tf.transform.translation.y = truth_odom.pose.pose.position.y
        truth_tf.transform.translation.z = truth_odom.pose.pose.position.z
        truth_tf.transform.rotation = truth_odom.pose.pose.orientation

        # 5. 构建消息 - noisy_tf
        noisy_tf = TransformStamped()
        noisy_tf.header = truth_odom.header
        noisy_tf.child_frame_id = 'base_link'
        noisy_tf.transform.translation.x = float(final_pos[0])
        noisy_tf.transform.translation.y = float(final_pos[1])
        noisy_tf.transform.translation.z = float(final_pos[2])
        noisy_tf.transform.rotation.x = float(final_quat[0])
        noisy_tf.transform.rotation.y = float(final_quat[1])
        noisy_tf.transform.rotation.z = float(final_quat[2])
        noisy_tf.transform.rotation.w = float(final_quat[3])

        # 6. 构建消息 - noisy_odom
        noisy_odom = Odometry()
        noisy_odom.header = noisy_tf.header
        noisy_odom.child_frame_id = 'base_link'
        noisy_odom.pose.pose.position.x = noisy_tf.transform.translation.x
        noisy_odom.pose.pose.position.y = noisy_tf.transform.translation.y
        noisy_odom.pose.pose.position.z = noisy_tf.transform.translation.z
        noisy_odom.pose.pose.orientation = noisy_tf.transform.rotation

        # 7. 构建消息 - real_vel
        real_vel = Vector3Stamped()
        real_vel.header.stamp = stamp_msg
        real_vel.header.frame_id = 'base_link'
        real_vel.vector.x = float(body_vel[3])
        real_vel.vector.y = float(body_vel[4])
        real_vel.vector.z = float(body_vel[5])

        return truth_odom, truth_tf, noisy_tf, noisy_odom, real_vel
