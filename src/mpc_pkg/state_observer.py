import numpy as np
from decorder import time_print


class PoseVelocityObserver:
    """使用简化卡尔曼滤波器, 由位姿估计车体系速度 [vx_body, vy_body, yaw_rate]。"""

    _STATE_SIZE = 6
    _MEAS_SIZE = 3

    def __init__(
        self,
        min_dt: float = 1e-3,
        max_dt: float = 0.2,
        q_linear_acc: float = 1.0,
        q_yaw_acc: float = 0.8,
        r_pos: float = 6.4e-5,
        r_yaw: float = 2.0e-4,
        reset_threshold_pos: float = 0.5,
        reset_threshold_yaw: float = 0.8,
    ):
        if min_dt <= 0.0:
            raise ValueError("min_dt must be > 0.")
        if max_dt <= min_dt:
            raise ValueError("max_dt must be > min_dt.")
        if q_linear_acc <= 0.0 or q_yaw_acc <= 0.0:
            raise ValueError("process noise must be > 0.")
        if r_pos <= 0.0 or r_yaw <= 0.0:
            raise ValueError("measurement noise must be > 0.")
        if reset_threshold_pos <= 0.0 or reset_threshold_yaw <= 0.0:
            raise ValueError("reset thresholds must be > 0.")

        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        self.q_linear_acc = float(q_linear_acc)
        self.q_yaw_acc = float(q_yaw_acc)
        self.r_pos = float(r_pos)
        self.r_yaw = float(r_yaw)
        self.reset_threshold_pos = float(reset_threshold_pos)
        self.reset_threshold_yaw = float(reset_threshold_yaw)

        self._last_t = None
        # 状态向量定义:
        # x_hat = [x, y, yaw, vx_world, vy_world, yaw_rate]^T
        # 其中速度状态在世界系中估计, 最终输出前再旋转到车体系。
        self._x_hat = np.zeros((self._STATE_SIZE, 1), dtype=float)
        self._P = np.eye(self._STATE_SIZE, dtype=float) * 1e-2

        # 量测向量定义:
        # z = [x, y, yaw]^T
        # H 矩阵表示“只直接观测位置和航向, 不直接观测速度”。
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self._R = np.diag([self.r_pos, self.r_pos, self.r_yaw])
        self._I = np.eye(self._STATE_SIZE, dtype=float)
        self._initialized = False

    def reset(self):
        self._last_t = None
        self._x_hat = np.zeros((self._STATE_SIZE, 1), dtype=float)
        self._P = np.eye(self._STATE_SIZE, dtype=float) * 1e-2
        self._initialized = False

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        # 将角度包裹到 (-pi, pi], 防止航向残差在 +/-pi 附近跳变。
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _reinitialize_with_measurement(self, z: np.ndarray, now: float):
        # 观测器重置策略:
        # 1) 位置/航向直接对齐当前量测
        # 2) 速度状态清零
        # 3) 协方差恢复初值
        # 用于处理量测突变或模型暂时失配导致的发散。
        self._x_hat.fill(0.0)
        self._x_hat[0, 0] = z[0, 0]
        self._x_hat[1, 0] = z[1, 0]
        self._x_hat[2, 0] = z[2, 0]
        self._P = np.eye(self._STATE_SIZE, dtype=float) * 1e-2
        self._last_t = now
        self._initialized = True

    def _state_to_body_velocity(self) -> np.ndarray:
        """将状态中的世界系速度旋转到车体系, 并返回 [vx_body, vy_body, yaw_rate]。"""
        yaw_est = self._x_hat[2, 0]
        vx_world = self._x_hat[3, 0]
        vy_world = self._x_hat[4, 0]
        wz = self._x_hat[5, 0]
        cy = np.cos(yaw_est)
        sy = np.sin(yaw_est)
        return np.array(
            [
                cy * vx_world + sy * vy_world,
                -sy * vx_world + cy * vy_world,
                wz,
            ],
            dtype=float,
        )

    def _build_A_Q(self, dt: float):
        # 离散状态转移模型(常速度模型):
        # x(k+1) = x(k) + dt * vx(k)
        # y(k+1) = y(k) + dt * vy(k)
        # yaw(k+1) = yaw(k) + dt * wz(k)
        # vx, vy, wz 视为短时常值。
        A = np.array(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q_lin = self.q_linear_acc
        q_yaw = self.q_yaw_acc

        # 过程噪声 Q 的构造基于“连续白噪声加速度模型”离散化:
        # 对 x-vx 与 y-vy 子系统使用同样的线加速度谱强度 q_lin。
        # 对 yaw-wz 子系统使用角加速度谱强度 q_yaw。
        # 这样可以在保持模型简洁的前提下, 允许速度状态随时间缓慢漂移,
        # 从而吸收执行器动态误差、地面扰动与建模误差。
        Q = np.zeros((6, 6), dtype=float)
        # x-vx block
        Q[0, 0] = 0.25 * dt4 * q_lin
        Q[0, 3] = 0.5 * dt3 * q_lin
        Q[3, 0] = 0.5 * dt3 * q_lin
        Q[3, 3] = dt2 * q_lin
        # y-vy block
        Q[1, 1] = 0.25 * dt4 * q_lin
        Q[1, 4] = 0.5 * dt3 * q_lin
        Q[4, 1] = 0.5 * dt3 * q_lin
        Q[4, 4] = dt2 * q_lin
        # yaw-wz block
        Q[2, 2] = 0.25 * dt4 * q_yaw
        Q[2, 5] = 0.5 * dt3 * q_yaw
        Q[5, 2] = 0.5 * dt3 * q_yaw
        Q[5, 5] = dt2 * q_yaw

        return A, Q

    def _need_reinitialize(self, innovation: np.ndarray) -> bool:
        return (
            abs(innovation[0, 0]) > self.reset_threshold_pos
            or abs(innovation[1, 0]) > self.reset_threshold_pos
            or abs(innovation[2, 0]) > self.reset_threshold_yaw
        )

    @time_print(10)
    def update(self, x: float, y: float, yaw: float, stamp_sec: float | None = None) -> np.ndarray:
        """输入位姿与时间戳, 输出车体系速度 [vx_body, vy_body, yaw_rate]。"""
        if stamp_sec is None:
            raise ValueError("stamp_sec is required for deterministic observer timing.")

        now = float(stamp_sec)
        z = np.array([[float(x)], [float(y)], [float(yaw)]], dtype=float)

        if not self._initialized:
            self._reinitialize_with_measurement(z, now)
            return np.zeros(3, dtype=float)

        last_t = self._last_t
        assert last_t is not None

        dt = now - last_t
        # dt 过小时跳过滤波更新:
        # 这能避免数值病态(尤其是高频重复时间戳), 并复用上一次稳定估计值。
        if dt < self.min_dt:
            return self._state_to_body_velocity()

        dt = min(dt, self.max_dt)
        A, Q = self._build_A_Q(dt)

        # 1) 预测步骤
        # 使用上一时刻状态和常速度模型外推到当前时刻。
        x_pred = A @ self._x_hat
        x_pred[2, 0] = self._wrap_angle(x_pred[2, 0])
        P_pred = A @ self._P @ A.T + Q

        # 2) 创新(残差)计算
        # y_tilde = z - H x_pred
        # 航向残差需要做角度包裹, 否则在 +/-pi 附近会出现假大误差。
        y_tilde = z - (self._H @ x_pred)
        y_tilde[2, 0] = self._wrap_angle(y_tilde[2, 0])

        # 3) 异常值保护
        # 若任一残差超过阈值, 认为当前滤波状态与量测严重不一致,
        # 直接重置可避免错误状态持续污染后续估计。
        if self._need_reinitialize(y_tilde):
            self._reinitialize_with_measurement(z, now)
            return np.zeros(3, dtype=float)

        S = self._H @ P_pred @ self._H.T + self._R
        # 使用线性方程求解卡尔曼增益 K, 避免显式求逆带来的数值不稳定。
        K = np.linalg.solve(S, (P_pred @ self._H.T).T).T

        # 4) 校正步骤
        # 用量测残差修正预测状态与协方差。
        self._x_hat = x_pred + K @ y_tilde
        self._x_hat[2, 0] = self._wrap_angle(self._x_hat[2, 0])
        self._P = (self._I - K @ self._H) @ P_pred
        self._last_t = now

        return self._state_to_body_velocity()
