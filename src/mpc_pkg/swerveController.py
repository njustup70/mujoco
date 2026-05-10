import numpy as np
import math
class SwerveManifoldController:
    def __init__(self):
        """
        初始化舵轮几何参数
        :param wheel_coords: 列表或数组，形状为 (N, 2)，代表每个舵轮相对于中心的 [r_ix, r_iy]

        舵轮参数定义为前左、前右、后左、后右四个轮子相对于底盘中心的坐标 (单位: 米)
        """
        self.r = [
        [0.325, 0.325],   # 前左
        [0.325, -0.325],  # 前右
        [-0.325, 0.325],  # 后左
        [-0.325, -0.325]  # 后右
        ]
        self.v2rpm=60/(math.pi*0.058) # 线速度转轮速的比例，单位是 rpm/(m/s)，其中0.058是轮子半径
        self.num_wheels = len(self.r)
        self.I = np.eye(3)  # 底盘自由度为 3 (vx, vy, w)

    def compute_safe_velocity(self, theta_real, v_target):
        """
        执行流形速度分解，防止内力
        :param theta_real: 当前四个舵轮的真实角度反馈 (弧度)，形状为 (N,) 前左、前右、后左、后右
        :param v_target: 目标底盘速度 [vx, vy, w]
        :return: V_safe (过滤后的底盘速度), wheel_speeds (对应的各轮线速度指令)
        """
        v_des = np.array(v_target).reshape(3, 1)
        
        # 1. 构建约束雅可比矩阵 A (基于普法夫约束：侧向速度为0)
        # A 的每一行 a_i = [-sin(theta), cos(theta), r_ix*cos(theta) + r_iy*sin(theta)]
        A = np.zeros((self.num_wheels, 3))
        for i in range(self.num_wheels):
            s_i = np.sin(theta_real[i])
            c_i = np.cos(theta_real[i])
            r_ix, r_iy = self.r[i]
            
            A[i, 0] = -s_i
            A[i, 1] = c_i
            A[i, 2] = r_ix * c_i + r_iy * s_i

        # 2. 计算投影矩阵 P = I - A_pinv * A
        # A_pinv 是 A 的摩尔-彭若斯伪逆
        A_pinv = np.linalg.pinv(A)
        P = self.I - np.dot(A_pinv, A)

        # 3. 流形分解：将目标速度投影到零空间 (合法运动空间)
        # V_safe 是距离 V_des 最近且绝对不产生内力的解
        v_safe = np.dot(P, v_des).flatten()

        # 4. 逆运动学映射：将 V_safe 转化为各轮滚动线速度
        # v_i = vx*cos(theta) + vy*sin(theta) + w*(r_ix*sin(theta) - r_iy*cos(theta))
        wheel_speeds = np.zeros(self.num_wheels)
        vx, vy, w = v_safe
        for i in range(self.num_wheels):
            s_i = np.sin(theta_real[i])
            c_i = np.cos(theta_real[i])
            r_ix, r_iy = self.r[i]
            
            wheel_speeds[i] = vx * c_i + vy * s_i + w * (r_ix * s_i - r_iy * c_i)

        return v_safe, wheel_speeds
    def cmd_vel_compute(self,v_target,theta_real):
        '''
        根据目标底盘速度和当前舵轮角度，计算每个轮子的目标舵角和线速度指令。
        :param v_target: 目标底盘速度 [vx, vy, w]
        :param theta_real: 当前四个舵轮的真实角度反馈 (弧度)，形状为 (N,)
        :return: wheel_thetas (每个轮子的目标舵角), wheel_speeds (每个轮子的目标rpm)
        '''
        #进行原始的向量分解，得到每个轮子线速度，角度指令
        wheel_speeds = np.zeros(self.num_wheels)
        wheel_thetas = np.zeros(self.num_wheels)
        for i in range(self.num_wheels):
            #用r[i][0]和r[i][1]算出来的速度向量在轮子坐标系下的分量
            vxi=v_target[0] - v_target[2]*self.r[i][1]
            vyi=v_target[1] + v_target[2]*self.r[i][0]
            speed = math.hypot(vxi, vyi)
            theta = math.atan2(vyi, vxi)
            wheel_thetas[i], wheel_speeds[i] =self.optimize_steer_arc(theta,speed,theta_real[i])
        #进行流形分解
        v_safe, speeds = self.compute_safe_velocity(theta_real, v_target)
        speeds = speeds /0.058 # 将线速度转换为轮速指令（rpm）
        return wheel_thetas,speeds
    def optimize_steer_arc(self,desired_steer: float, speed: float, last_steer: float) -> tuple[float, float]:
        """优劣弧优化：在(舵角+正转)与(舵角+pi+反转)中选择转角更小的一组。"""
        steer_a = (desired_steer) % (2.0 * math.pi)
        drive_a = speed
        #归一化到2pi范围内，避免
        steer_b = ( desired_steer + math.pi) % (2.0 * math.pi)
        drive_b = -speed

        da=math.atan2(math.sin(steer_a), math.cos(steer_a))  # 将角度规范化到 [-pi, pi]
        db=math.atan2(math.sin(steer_b), math.cos(steer_b))
        if abs(da) <= abs(db):
            return steer_a, drive_a
        return steer_b, drive_b

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设底盘是 0.5m x 0.5m 的正方形，定义四个轮子的位置 [r_ix, r_iy]
    coords = [
        [0.325, 0.325],   # 前左
        [0.25, -0.325],  # 前右
        [-0.325, 0.325],  # 后左
        [-0.325, -0.325]  # 后右
    ]
    
    controller = SwerveManifoldController()

    # 情况：你想前进 [vx=1.0, vy=0, w=0]，但舵轮由于延迟，角度还没转到0，全在 45度(pi/4)
    current_thetas = [np.pi/4] * 4 
    target_v = [-0.5, 0.0, 0.0]

    v_safe, speeds = controller.compute_safe_velocity(current_thetas, target_v)
    
    print(f"原始目标速度: {target_v}")
    print(f"过滤后安全速度 (V_safe): {v_safe}")
    print(f"各轮执行线速度: {speeds}")
    thetas,speeds=controller.cmd_vel_compute(target_v,current_thetas)
    print(f"各轮目标舵角: {thetas}")
    print(f"各轮目标线速度: {speeds}")