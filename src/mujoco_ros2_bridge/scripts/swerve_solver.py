#!/usr/bin/env python3
import math
import numpy as np


def wrap_to_near(angle: float, center: float) -> float:
    """Wrap angle to be the nearest representation around center."""
    d = angle - center
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return center + d


def decompose_wheel_velocity(vx: float, vy: float, vyaw: float, wheel_xy: tuple[float, float]) -> tuple[float, float]:
    """速度分解：将底盘速度分解到单个舵轮接地点线速度。"""
    wx, wy = wheel_xy
    v_ix = vx - vyaw * wy
    v_iy = vy + vyaw * wx
    return v_ix, v_iy


def optimize_steer_arc(desired_steer: float, speed: float, last_steer: float) -> tuple[float, float]:
    """优劣弧优化：在(舵角+正转)与(舵角+pi+反转)中选择转角更小的一组。"""
    steer_a = desired_steer
    drive_a = speed
    steer_b = desired_steer + math.pi
    drive_b = -speed

    steer_a = wrap_to_near(steer_a, last_steer)
    steer_b = wrap_to_near(steer_b, last_steer)
    if abs(steer_a - last_steer) <= abs(steer_b - last_steer):
        return steer_a, drive_a
    return steer_b, drive_b


class SwerveSolver:
    """舵轮解算器：根据 vx/vy/yaw 计算四个舵轮目标，并处理电机一阶滞后和噪声。"""

    def __init__(self, 
                 wheels_pos: list[tuple[float, float]], 
                 wheel_radius: float,
                 steer_lag_alpha: float = 0.0,
                 drive_lag_alpha: float = 0.0,
                 steer_noise_std: float = 0.0,
                 drive_noise_std: float = 0.0):
        """
        初始化舵轮解算器。
        
        Args:
            wheels_pos: 四个轮子相对底盘中心的位置 [(x1,y1), (x2,y2), ...]
            wheel_radius: 轮子半径
            steer_lag_alpha: 舵机一阶滞后系数 (0 ~ 1)
            drive_lag_alpha: 驱动电机一阶滞后系数 (0 ~ 1)
            steer_noise_std: 舵控响应噪声标准差
            drive_noise_std: 驱动控响应噪声标准差
        """
        self.wheels_pos = wheels_pos
        self.wheel_radius = wheel_radius
        self.steer_lag_alpha = steer_lag_alpha
        self.drive_lag_alpha = drive_lag_alpha
        self.steer_noise_std = steer_noise_std
        self.drive_noise_std = drive_noise_std
        
        # 舵轮目标角度统计（用于优劣弧选择）
        self.last_target_angles = [0.0] * len(wheels_pos)
        
        # 电机响应状态（一阶滞后）
        self.last_steer_ctrl = [0.0] * len(wheels_pos)
        self.last_drive_ctrl = [0.0] * len(wheels_pos)

    def solve(self, vx: float, vy: float, vyaw: float) -> list[tuple[float, float]]:
        """
        纯舵轮运动解算：根据底盘速度计算每个轮子的目标舵角和目标速度。
        
        Returns:
            list[(target_steer_rad, target_drive_rad_s)] 每个轮子的目标
        """
        targets = []
        for i, wheel_xy in enumerate(self.wheels_pos):
            v_ix, v_iy = decompose_wheel_velocity(vx, vy, vyaw, wheel_xy)
            speed = math.hypot(v_ix, v_iy)

            # 默认保持上一步舵角，除非有明确的速度指令
            target_steer = self.last_target_angles[i]
            signed_speed = speed
            
            if speed > 0.01 or abs(vyaw) > 0.01:
                desired_steer = math.atan2(v_iy, v_ix)
                target_steer, signed_speed = optimize_steer_arc(
                    desired_steer,
                    speed,
                    self.last_target_angles[i],
                )

            self.last_target_angles[i] = target_steer
            target_drive_rads = signed_speed / self.wheel_radius
            targets.append((target_steer, target_drive_rads))

        return targets

    def apply_motor_dynamics(self, 
                             target_steer_list: list[float], 
                             target_drive_list: list[float],
                             current_steer_angles: list[float]) -> list[tuple[float, float]]:
        """
        应用电机一阶滞后和响应噪声。
        
        Args:
            target_steer_list: 目标舵角 (rad)
            target_drive_list: 目标驱动速度 (rad/s)
            current_steer_angles: 当前实际舵角 (rad) - 用于 2π 归一化
            
        Returns:
            list[(steer_ctrl, drive_ctrl)] 最终电机指令
        """
        controls = []
        for i in range(len(target_steer_list)):
            target_steer = target_steer_list[i]
            target_drive = target_drive_list[i]
            current_steer = current_steer_angles[i]
            
            # 将目标舵角归一化到当前舵角附近，避免 ±π 跳变
            target_steer = wrap_to_near(target_steer, current_steer)
            
            # 一阶滞后更新（或直接用目标值）
            if self.steer_lag_alpha > 0:
                self.last_steer_ctrl[i] = wrap_to_near(self.last_steer_ctrl[i], current_steer)
                self.last_steer_ctrl[i] += self.steer_lag_alpha * (target_steer - self.last_steer_ctrl[i])
                steer_output = self.last_steer_ctrl[i]
            else:
                steer_output = target_steer
            
            if self.drive_lag_alpha > 0:
                self.last_drive_ctrl[i] += self.drive_lag_alpha * (target_drive - self.last_drive_ctrl[i])
                drive_output = self.last_drive_ctrl[i]
            else:
                drive_output = target_drive
            
            # 加入响应高频噪声
            steer_ctrl = wrap_to_near(
                steer_output + np.random.normal(0.0, self.steer_noise_std),
                current_steer
            )
            drive_ctrl = drive_output + np.random.normal(0.0, self.drive_noise_std)
            
            controls.append((steer_ctrl, drive_ctrl))
            
        return controls
