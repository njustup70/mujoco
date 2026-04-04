import numpy as np
from decorder import time_print


import numpy as np

class PoseVelocityObserver:
    """使用恒定加速度模型 (CA) 的卡尔曼滤波器，估计位姿、速度及加速度。"""

    _STATE_SIZE = 9  # [x, y, yaw, vx, vy, wz, ax, ay, alpha]
    _MEAS_SIZE = 3   # [x, y, yaw]

    def __init__(
        self,
        min_dt: float = 1e-3,
        max_dt: float = 0.2,
        q_jerk: float = 10.0,       # 线加加速度过程噪声 (Jerk)
        q_angular_jerk: float = 5.0, # 角加加速度过程噪声
        r_pos: float = 6.4e-5,
        r_yaw: float = 2.0e-4,
        reset_threshold_pos: float = 0.5,
        reset_threshold_yaw: float = 0.8,
    ):
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        self.q_jerk = float(q_jerk)
        self.q_angular_jerk = float(q_angular_jerk)
        self.r_pos = float(r_pos)
        self.r_yaw = float(r_yaw)
        self.reset_threshold_pos = float(reset_threshold_pos)
        self.reset_threshold_yaw = float(reset_threshold_yaw)

        self._last_t = 0.0
        self._x_hat = np.zeros((self._STATE_SIZE, 1), dtype=float)
        self._P = np.eye(self._STATE_SIZE, dtype=float) * 1.0
        
        # H 矩阵：只观测前三个状态 [x, y, yaw]
        self._H = np.zeros((self._MEAS_SIZE, self._STATE_SIZE), dtype=float)
        self._H[:3, :3] = np.eye(3)
        
        self._R = np.diag([self.r_pos, self.r_pos, self.r_yaw])
        self._I = np.eye(self._STATE_SIZE, dtype=float)
        self._initialized = False

    def _build_A_Q(self, dt: float):
        """构造 CA 模型的 A 和 Q 矩阵。"""
        dt2 = 0.5 * dt * dt
        
        # 1. 构造状态转移矩阵 A (9x9)
        # 包含：pos = pos + v*dt + 0.5*a*dt^2; v = v + a*dt; a = a
        A = np.eye(self._STATE_SIZE)
        # 位置项受速度和加速度影响
        for i in range(3):
            A[i, i+3] = dt      # p += v*dt
            A[i, i+6] = dt2     # p += 0.5*a*dt^2
            A[i+3, i+6] = dt    # v += a*dt

        # 2. 构造过程噪声矩阵 Q (9x9)
        # 假设加加速度 (Jerk) 是白噪声，积分得到 Q
        Q = np.zeros((9, 9))
        
        def fill_block(q_val, start_idx):
            # 基于积分推导的标准 CA 模型过程噪声块
            # 对应的项为 [pos, vel, acc]
            m = np.array([
                [dt**5/20, dt**4/8,  dt**3/6],
                [dt**4/8,  dt**3/3,  dt**2/2],
                [dt**3/6,  dt**2/2,  dt]
            ]) * q_val
            indices = [start_idx, start_idx+3, start_idx+6]
            for r_i, r_idx in enumerate(indices):
                for c_i, c_idx in enumerate(indices):
                    Q[r_idx, c_idx] = m[r_i, c_i]

        fill_block(self.q_jerk, 0) # X 轴块
        fill_block(self.q_jerk, 1) # Y 轴块
        fill_block(self.q_angular_jerk, 2) # Yaw 轴块

        return A, Q

    def _state_to_body_velocity(self) -> np.ndarray:
        """从世界系 9 维状态中提取并旋转车体系速度。"""
        yaw_est = self._x_hat[2, 0]
        vx_w, vy_w, wz = self._x_hat[3, 0], self._x_hat[4, 0], self._x_hat[5, 0]
        
        cy, sy = np.cos(yaw_est), np.sin(yaw_est)
        # vx_body = vx_w*cos + vy_w*sin
        # vy_body = -vx_w*sin + vy_w*cos
        return np.array([
            cy * vx_w + sy * vy_w,
            -sy * vx_w + cy * vy_w,
            wz,
        ])

    def _reinitialize_with_measurement(self, z: np.ndarray, now: float):
        self._x_hat.fill(0.0)
        self._x_hat[:3, 0] = z[:3, 0]
        # 初始协方差：位置给小，速度和加速度给大，允许快速收敛
        self._P = np.diag([1e-2, 1e-2, 1e-3, 1.0, 1.0, 0.5, 2.0, 2.0, 1.0])
        self._last_t = now
        self._initialized = True
    def update(self, x: float, y: float, yaw: float, stamp_sec: float) -> np.ndarray:
        now = float(stamp_sec)
        z = np.array([[x], [y], [yaw]])

        if not self._initialized:
            self._reinitialize_with_measurement(z, now)
            return np.zeros(3)

        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0: return self._state_to_body_velocity()
        if dt < self.min_dt: return self._state_to_body_velocity()
        dt = min(dt, self.max_dt)

        A, Q = self._build_A_Q(dt)

        # --- 卡尔曼五步 ---
        # 1. 预测
        x_pred = A @ self._x_hat
        x_pred[2, 0] = np.arctan2(np.sin(x_pred[2, 0]), np.cos(x_pred[2, 0]))
        P_pred = A @ self._P @ A.T + Q

        # 2. 残差
        y_tilde = z - (self._H @ x_pred)
        y_tilde[2, 0] = np.arctan2(np.sin(y_tilde[2, 0]), np.cos(y_tilde[2, 0]))

        # 3. 异常重置
        if (abs(y_tilde[0,0]) > self.reset_threshold_pos or 
            abs(y_tilde[2,0]) > self.reset_threshold_yaw):
            self._reinitialize_with_measurement(z, now)
            return np.zeros(3)

        # 4. 增益 (使用 solve 提高稳定性)
        S = self._H @ P_pred @ self._H.T + self._R
        K = np.linalg.solve(S, (P_pred @ self._H.T).T).T

        # 5. 更新
        self._x_hat = x_pred + K @ y_tilde
        self._x_hat[2, 0] = np.arctan2(np.sin(self._x_hat[2, 0]), np.cos(self._x_hat[2, 0]))
        self._P = (self._I - K @ self._H) @ P_pred
        self._last_t = now

        return self._state_to_body_velocity()


class LowPassFilter:
    """简单的一阶低通滤波器，用于平滑衰减系数"""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.state = None

    def update(self, val: float) -> float:
        if self.state is None:
            self.state = val
            return val
        self.state = self.alpha * val + (1.0 - self.alpha) * self.state
        return self.state


class PoseVelocityESO:
    """
    带动态衰减系数的线性扩张状态观测器 (LESO)
    用于从 [x, y, yaw] 测量值中平滑地提取 [vx, vy, wz]
    """

    def __init__(
        self,
        omega_x: float = 12.0,    # x方向带宽
        omega_y: float = 12.0,    # y方向带宽
        omega_yaw: float = 15.0,  # 航向角带宽
        eta_base: float = 0.25,   # 基础衰减系数 (推荐 0.1 ~ 0.5)
        reset_threshold_pos: float = 0.5,
        reset_threshold_yaw: float = 0.8,
    ):
        # 观测器增益配置 (极点配置在 -omega 处)
        # beta1 = 3*w, beta2 = 3*w^2, beta3 = w^3
        self.params = {
            'x': [3 * omega_x, 3 * (omega_x ** 2), omega_x ** 3],
            'y': [3 * omega_y, 3 * (omega_y ** 2), omega_y ** 3],
            'yaw': [3 * omega_yaw, 3 * (omega_yaw ** 2), omega_yaw ** 3]
        }

        # 状态: [位置, 速度, 扰动项]
        self._z_x = np.zeros(3)
        self._z_y = np.zeros(3)
        self._z_yaw = np.zeros(3)

        # 衰减管理
        self.eta_base = eta_base
        self._eta_filt = LowPassFilter(alpha=0.05)  # 平滑 eta 的变化

        self.reset_threshold_pos = reset_threshold_pos
        self.reset_threshold_yaw = reset_threshold_yaw
        self._last_t = 0.0
        self._initialized = False

    def _wrap_angle(self, angle: float) -> float:
        """将角度限制在 [-pi, pi]"""
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _get_dynamic_eta(self, error_norm: float) -> float:
        """
        根据误差大小动态调整 eta。
        误差越大，eta 越接近 1.0 (加速跟踪)；
        误差越小，eta 越接近 eta_base (增加平滑度)。
        """
        # 这里的 2.0 是灵敏度系数，可根据实际噪声调整
        gate = 1.0 - np.exp(-abs(error_norm) * 2.0)
        target_eta = self.eta_base + (1.0 - self.eta_base) * gate
        return self._eta_filt.update(target_eta)

    def _update_eso_unit(self, z: np.ndarray, meas: float, dt: float, betas: list, is_angle: bool = False) -> np.ndarray:
        # 1. 计算原始误差
        err = meas - z[0]
        if is_angle:
            err = self._wrap_angle(err)

        # 2. 获取动态衰减后的有效误差
        # 注意：这里可以根据 err 的模长统一计算一次 eta，也可以每个通道独立计算
        current_eta = self._get_dynamic_eta(err)
        eff_err = err * current_eta

        # 3. 状态更新 (欧拉积分)
        # dz1 = z2 + beta1 * eff_err
        # dz2 = z3 + beta2 * eff_err
        # dz3 = beta3 * eff_err
        dz1 = z[1] + betas[0] * eff_err
        dz2 = z[2] + betas[1] * eff_err
        dz3 = betas[2] * eff_err

        new_z = z + np.array([dz1, dz2, dz3]) * dt

        if is_angle:
            new_z[0] = self._wrap_angle(new_z[0])

        return new_z

    def update(self, x: float, y: float, yaw: float, stamp_sec: float) -> np.ndarray:
        now = float(stamp_sec)

        if not self._initialized:
            self._z_x[0], self._z_y[0], self._z_yaw[0] = x, y, yaw
            self._last_t = now
            self._initialized = True
            return np.zeros(3)

        dt = now - self._last_t
        if dt <= 0.0 or dt > 0.5:  # 处理时间回退或间隔过大的情况
            self._last_t = now
            return self._get_body_velocity()

        # 异常跳变检测 (重置逻辑)
        if self._need_reinitialize(x, y, yaw):
            self._z_x[0], self._z_y[0], self._z_yaw[0] = x, y, yaw
            self._z_x[1:], self._z_y[1:], self._z_yaw[1:] = 0, 0, 0
            self._last_t = now
            return np.zeros(3)

        # 核心更新
        self._z_x = self._update_eso_unit(self._z_x, x, dt, self.params['x'])
        self._z_y = self._update_eso_unit(self._z_y, y, dt, self.params['y'])
        self._z_yaw = self._update_eso_unit(self._z_yaw, yaw, dt, self.params['yaw'], is_angle=True)

        self._last_t = now
        return self._get_body_velocity()

    def _need_reinitialize(self, x: float, y: float, yaw: float) -> bool:
        dx = abs(x - self._z_x[0])
        dy = abs(y - self._z_y[0])
        dyaw = abs(self._wrap_angle(yaw - self._z_yaw[0]))
        return dx > self.reset_threshold_pos or dy > self.reset_threshold_pos or dyaw > self.reset_threshold_yaw

    def _get_body_velocity(self) -> np.ndarray:
        """从世界坐标系 z1, z2 转换到车体系速度"""
        yaw_est = self._z_yaw[0]
        vx_w, vy_w, wz = self._z_x[1], self._z_y[1], self._z_yaw[1]

        c, s = np.cos(yaw_est), np.sin(yaw_est)
        # 坐标变换: R^T * V_world
        vx_b = c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w

        return np.array([vx_b, vy_b, wz])