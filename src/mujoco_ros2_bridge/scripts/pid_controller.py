#!/usr/bin/env python3
import math


class PIDController:
    """Simple PID controller with integral/output clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        output_min: float = -math.inf,
        output_max: float = math.inf,
        integral_min: float = -math.inf,
        integral_max: float = math.inf,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.integral_min = float(integral_min)
        self.integral_max = float(integral_max)

        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def update(self, setpoint: float, measurement: float) -> float:
        error = float(setpoint) - float(measurement)

        # Integral term with anti-windup clamp.
        self._integral += error * self.dt
        self._integral = min(max(self._integral, self.integral_min), self.integral_max)

        if self._initialized:
            derivative = (error - self._prev_error) / self.dt
        else:
            derivative = 0.0
            self._initialized = True

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = min(max(output, self.output_min), self.output_max)

        self._prev_error = error
        return output
