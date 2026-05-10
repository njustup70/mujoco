#!/usr/bin/env python3
import math
import numpy as np


def wrap_to_near(angle: float, center: float) -> float:
    """Wrap angle to be the nearest representation around center."""
    return center + (angle - center + math.pi) % (2.0 * math.pi) - math.pi


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
                 wheel_radius: float):
        """
        初始化舵轮解算器。
        
        Args:
            wheels_pos: 四个轮子相对底盘中心的位置 [(x1,y1), (x2,y2), ...]
            wheel_radius: 轮子半径
        """
        self.wheels_pos = wheels_pos
        self.wheel_radius = wheel_radius
        
        # 记录累积的角度（用于累积积分）
        self.accumulated_steer_angles = [0.0] * len(wheels_pos)
        
        # 舵轮目标角度统计（用于优劣弧选择）
        self.last_target_angles = [0.0] * len(wheels_pos)
        
        # 直接控制时的参数（归一化 -> 物理量）
        self.max_wheel_linear_speed = 5.0  # m/s

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

    def get_direct_actuator_commands(self,
                                     norm_steer_list: list[float],
                                     norm_speed_list: list[float],
                                     current_steer_angles: list[float],
                                     dt: float) -> list[tuple[float, float]]:
        """针对接收到的指令直接转至 MuJoCo 促动器：
        - norm_steer_list: 映射后的目标角度 (rad)
        - norm_speed_list: 映射后的目标线速度 (m/s)
        - dt: 仿真步长，用于做舵角积分（限幅）

        返回值为 list[(steer_ctrl, drive_ctrl)]，其中 steer_ctrl 为角度(rad)，drive_ctrl 为轮子角速度(rad/s)
        """
        controls = []
        # 直接使用传入的物理量（已在 control_node 完成分解和映射）
        target_angles = norm_steer_list
        target_speeds_linear = norm_speed_list

        for i in range(len(target_angles)):
            # 将目标角度映射到当前实际角度附近，避免受外部控制器 +/- PI 突变影响
            current_steer = current_steer_angles[i]
            target_angle = wrap_to_near(target_angles[i], current_steer)
            
            # 直接透传映射后的连续角度
            new_angle = target_angle
            
            # 线速度 -> 轮子转速 (rad/s)
            wheel_linear = target_speeds_linear[i]
            wheel_rads = wheel_linear / self.wheel_radius if self.wheel_radius != 0 else 0.0

            controls.append((new_angle, wheel_rads))

        return controls

