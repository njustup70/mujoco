
class PidAlgorithm:
    def __init__(self, kp, ki, kd, max_integral,max_output, deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = abs(max_integral)
        self.max_output = abs(max_output)
        self.deadband = deadband
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True
    
    def _clamp_symmetric(self, value, limit):
        if limit is None:
            return value
        return max(-limit, min(value, limit))

    def compute(self, error, dt):
        if abs(error) < self.deadband:
            error = 0.0
        if dt <= 0.0:
            return 0.0
        # 积分项计算并限幅
        self.integral += error * dt
        self.integral = self._clamp_symmetric(self.integral, self.max_integral)
        i_term = self.ki * self.integral
     
        # 微分项计算
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
            derivative = (error - self.prev_error) / dt

        # PID输出计算
        output = self.kp * error + i_term + self.kd * derivative

        # 更新前一个误差
        self.prev_error = error
        return self._clamp_symmetric(output, self.max_output)
        