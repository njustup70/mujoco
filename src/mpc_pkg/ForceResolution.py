import numpy as np
import casadi
from mpc import ChessicModel

class SwerveForceAllocator:
    def __init__(self, chessic_model=None):
        if chessic_model is None:
            chessic_model = ChessicModel()
        self.p = chessic_model
        self.num_wheels = 4
        
        # 轮子相对于质心的位置 (考虑到 dx_cg, dy_cg 偏移)
        # 1:FL, 2:FR, 3:RL, 4:RR
        self.lx = self.p.L / 2.0
        self.ly = self.p.W / 2.0
        
        # 考虑到质心偏移后的坐标
        self.wheel_coords = np.array([
            [ self.lx - self.p.dx_cg,  self.ly - self.p.dy_cg], # FL
            [ self.lx - self.p.dx_cg, -self.ly - self.p.dy_cg], # FR
            [-self.lx - self.p.dx_cg,  self.ly - self.p.dy_cg], # RL
            [-self.lx - self.p.dx_cg, -self.ly - self.p.dy_cg]  # RR
        ])

    def get_allocation(self, F_total, current_steer_angles):
        """
        F_total: [Fx, Fy, Mz] 来自 MPC
        current_steer_angles: 当前四个舵轮的角度 [theta1, theta2, theta3, theta4]
        """
        Fx, Fy, Mz = F_total
        mass, g, h = self.p.mass, self.p.g, self.p.h_cg
        L, W = self.p.L, self.p.W

        # --- 1. 计算动态负载 Fzi ---
        ax, ay = Fx / mass, Fy / mass
        delta_Fz_long = (mass * ax * h) / L
        # 简单的横向转移分配
        delta_Fz_lat_f = (mass * ay * h / W) * (0.5) 
        delta_Fz_lat_r = (mass * ay * h / W) * (0.5)

        # 静态负荷 (简化处理)
        Fz_static = (mass * g) / 4.0
        Fz = np.array([
            Fz_static - delta_Fz_long/2 - delta_Fz_lat_f, # FL
            Fz_static - delta_Fz_long/2 + delta_Fz_lat_f, # FR
            Fz_static + delta_Fz_long/2 - delta_Fz_lat_r, # RL
            Fz_static + delta_Fz_long/2 + delta_Fz_lat_r  # RR
        ])
        Fz = np.maximum(Fz, 1.0) # 防止压力为负或零

        # --- 2. 构建加权伪逆 A 矩阵 (3x8) ---
        # 状态变量 u = [Fx1, Fy1, Fx2, Fy2, Fx3, Fy3, Fx4, Fy4]
        A = np.zeros((3, 8))
        for i in range(4):
            xi, yi = self.wheel_coords[i]
            A[0, 2*i] = 1.0   # Fx 合力
            A[1, 2*i+1] = 1.0 # Fy 合力
            A[2, 2*i] = -yi   # Mz 贡献: -y * Fx
            A[2, 2*i+1] = xi  # Mz 贡献: x * Fy

        # --- 3. 权重矩阵 W (考虑抓地力) ---
        # 压力越大，权重越小，分配的力越多
        W_diag = []
        for i in range(4):
            weight = 1.0 / (Fz[i]**2)
            W_diag.extend([weight, weight])
        W_inv = np.diag(1.0 / np.array(W_diag))

        # --- 4. 求解最优轮端力 (最小化内力) ---
        # u = W_inv * A^T * (A * W_inv * A^T)^-1 * b
        b = np.array([Fx, Fy, Mz])
        temp = A @ W_inv @ A.T
        u_optimal = W_inv @ A.T @ np.linalg.inv(temp) @ b

        # --- 5. 映射到执行器指令 ---
        results = {
            'steer_torques': [],  # 舵向力矩
            'drive_angles': [],   # 轴向角度 (目标舵角)
            'drive_forces': []    # 驱动力 (可转力矩)
        }

        for i in range(4):
            fix, fiy = u_optimal[2*i], u_optimal[2*i+1]
            
            # 目标舵角 (轴向角度)
            target_angle = np.arctan2(fiy, fix)
            results['drive_angles'].append(target_angle)
            
            # 驱动力 (用于驱动电机力矩)
            f_drive = np.sqrt(fix**2 + fiy**2)
            results['drive_forces'].append(f_drive)

            # 舵向力矩 (考虑非瞬时性，使用 PD 反算)
            # 这里模拟：如果要消除这个角度偏差，需要施加多大的力矩
            angle_error = target_angle - current_steer_angles[i]
            # 角度回绕处理
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            
            kp_steer = 50.0 # 舵向刚度
            results['steer_torques'].append(kp_steer * angle_error)

        return results


class ServeForceAllocator(SwerveForceAllocator):
    def __init__(self, chessic_model=None):
        super().__init__(chessic_model=chessic_model)