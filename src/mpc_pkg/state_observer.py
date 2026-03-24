import numpy as np


class PoseVelocityObserver:
    """Kalman observer: estimate body-frame velocity from pose (x, y, yaw)."""

    def __init__(
        self,
        min_dt: float = 1e-3,
        max_dt: float = 0.2,
        q_linear_acc: float = 4.0,
        q_yaw_acc: float = 2.0,
        r_pos: float = 0.01,
        r_yaw: float = 0.02,
        reset_threshold_pos: float = 1.0,
        reset_threshold_yaw: float = 1.0,
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
        # State: [x, y, yaw, vx_world, vy_world, yaw_rate]
        self._x_hat = np.zeros((6, 1), dtype=float)
        self._P = np.eye(6, dtype=float) * 1e-2
        # Measurement: [x, y, yaw]
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self._R = np.diag([self.r_pos, self.r_pos, self.r_yaw])
        self._I = np.eye(6, dtype=float)
        self._initialized = False

    def reset(self):
        self._last_t = None
        self._x_hat = np.zeros((6, 1), dtype=float)
        self._P = np.eye(6, dtype=float) * 1e-2
        self._initialized = False

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _reinitialize_with_measurement(self, z: np.ndarray, now: float):
        self._x_hat.fill(0.0)
        self._x_hat[0, 0] = z[0, 0]
        self._x_hat[1, 0] = z[1, 0]
        self._x_hat[2, 0] = z[2, 0]
        self._P = np.eye(6, dtype=float) * 1e-2
        self._last_t = now
        self._initialized = True

    def _build_A_Q(self, dt: float):
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

    def update(self, x: float, y: float, yaw: float, stamp_sec: float | None = None) -> np.ndarray:
        """Return [vx_body, vy_body, yaw_rate] in SI units."""
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
        if dt < self.min_dt:
            yaw_est = self._x_hat[2, 0]
            vx_world = self._x_hat[3, 0]
            vy_world = self._x_hat[4, 0]
            wz = self._x_hat[5, 0]
            cy = np.cos(yaw_est)
            sy = np.sin(yaw_est)
            return np.array([
                cy * vx_world + sy * vy_world,
                -sy * vx_world + cy * vy_world,
                wz,
            ], dtype=float)

        dt = min(dt, self.max_dt)
        A, Q = self._build_A_Q(dt)

        # Predict
        x_pred = A @ self._x_hat
        x_pred[2, 0] = self._wrap_angle(x_pred[2, 0])
        P_pred = A @ self._P @ A.T + Q

        # Update with wrapped yaw innovation
        y_tilde = z - (self._H @ x_pred)
        y_tilde[2, 0] = self._wrap_angle(y_tilde[2, 0])

        if (
            abs(y_tilde[0, 0]) > self.reset_threshold_pos
            or abs(y_tilde[1, 0]) > self.reset_threshold_pos
            or abs(y_tilde[2, 0]) > self.reset_threshold_yaw
        ):
            self._reinitialize_with_measurement(z, now)
            return np.zeros(3, dtype=float)

        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        self._x_hat = x_pred + K @ y_tilde
        self._x_hat[2, 0] = self._wrap_angle(self._x_hat[2, 0])
        self._P = (self._I - K @ self._H) @ P_pred
        self._last_t = now

        yaw_est = self._x_hat[2, 0]
        vx_world = self._x_hat[3, 0]
        vy_world = self._x_hat[4, 0]
        wz = self._x_hat[5, 0]
        cy = np.cos(yaw_est)
        sy = np.sin(yaw_est)

        vx_body = cy * vx_world + sy * vy_world
        vy_body = -sy * vx_world + cy * vy_world
        return np.array([vx_body, vy_body, wz], dtype=float)
