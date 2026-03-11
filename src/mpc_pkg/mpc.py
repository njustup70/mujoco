import do_mpc
from casadi import vertcat,sin,cos
import casadi
class MPCModel:
    def __init__(self, dt):
        type='discrete'   # either 'discrete' or 'continuous
        self.model = do_mpc.model.Model(type)
        # state 包括 x y 
        self.xy_state=self.model.set_variable(var_type='_x', var_name='x_state', shape=(4,1))
        self.theta_state=self.model.set_variable(var_type='_x', var_name='theta_state', shape=(2,1))
        # 控制输入的角加速度和线加速度
        self.u_vec=self.model.set_variable(var_type='_u', var_name='u_input', shape=(3,1))
        #定义参数
        self.ref=self.model.set_variable(var_type='_p', var_name='ref', shape=(3,1))
        # 定义运动学方程
        self.A1=vertcat(
            self.xy_state[0]+dt*self.xy_state[2]*cos(self.theta_state[0])+0.5*dt**2*self.u_vec[0]*cos(self.theta_state[0]),
            self.xy_state[1]+dt*self.xy_state[3]*sin(self.theta_state[0])+0.5*dt**2*self.u_vec[1]*sin(self.theta_state[0]),
            self.xy_state[2]+dt*self.u_vec[0],
            self.xy_state[3]+dt*self.u_vec[1]
        )
        self.A2=vertcat(
            self.theta_state[0]+dt*self.theta_state[1]+0.5*dt**2*self.u_vec[2],
            self.theta_state[1]+dt*self.u_vec[2]
        )
        self.model.set_rhs('x_state', self.A1)
        self.model.set_rhs('theta_state', self.A2)
        self.model.setup()
        mterm=casadi.sum1((self.xy_state[0:2]-self.ref[0:2])**2+(self.theta_state-self.ref[2])**2)
        lterm=casadi.sum1((self.xy_state[0:2]-self.ref[0:2])**2+(self.theta_state-self.ref[2])**2)
        self.mpc=do_mpc.controller.MPC(self.model)
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_param(n_horizon=20, t_step=dt)
        self.p_template = self.mpc.get_p_template(1)
        
        # 2. 定义一个内部函数，它必须接收 t_now 参数
        def p_fun(t_now):
            # 这个函数在 MPC 运行时会被自动调用 N 遍（N = n_horizon）
            # 它会返回当前及未来预测步的目标值
            return self.p_template

        # 3. 关联函数
        self.mpc.set_p_fun(p_fun)
        self.mpc.setup()

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class MPCStateObserver:
    def __init__(self, dt):
        # 状态向量 x = [x, y, vx, vy, theta, omega] (共6维)
        # 观测向量 z = [x, y, theta] (共3维)
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.dt = dt
        
        # 1. 初始状态
        self.ekf.x = np.zeros(6)
        
        # 2. 过程噪声 Q：影响响应速度
        # 调参建议：如果觉得速度估计不敏感，调大最后四项
        self.ekf.Q = np.diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-3, 1e-2])
        
        # 3. 测量噪声 R：影响平滑度
        # 调参建议：根据传感器精度定，位置一般设 0.01~0.1
        self.ekf.R = np.diag([0.05, 0.05, 0.02])
        
        # 4. 初始协方差
        self.ekf.P *= 1.0

    def predict(self, u_vec):
        """
        u_vec: [ax, ay, alpha] 来自 MPC 的控制输入
        """
        def f_non_linear(x, u, dt):
            # 完全对应你的 do-mpc A1, A2 方程
            x_pos, y_pos, vx, vy, theta, omega = x
            ax, ay, alpha = u
            
            new_x = x_pos + dt*vx*np.cos(theta) + 0.5*dt**2 * ax*np.cos(theta)
            new_y = y_pos + dt*vy*np.sin(theta) + 0.5*dt**2 * ay*np.sin(theta)
            new_vx = vx + dt*ax
            new_vy = vy + dt*ay
            new_theta = theta + dt*omega + 0.5*dt**2 * alpha
            new_omega = omega + dt*alpha
            return np.array([new_x, new_y, new_vx, new_vy, new_theta, new_omega])

        # 线性化雅可比矩阵 (这里是 EKF 的核心)
        def get_F(x, u, dt):
            _, _, vx, vy, theta, _ = x
            ax, ay, _ = u
            F = np.eye(6)
            F[0, 2] = dt * np.cos(theta)
            F[0, 4] = -dt*vx*np.sin(theta) - 0.5*dt**2*ax*np.sin(theta)
            F[1, 3] = dt * np.sin(theta)
            F[1, 4] = dt*vy*np.cos(theta) + 0.5*dt**2*ay*np.cos(theta)
            F[2, 0] = 0 # vx 与位置无关
            F[3, 1] = 0
            F[4, 5] = dt
            return F

        F = get_F(self.ekf.x, u_vec, self.dt)
        self.ekf.predict(u=u_vec, f=f_non_linear, F=F, dt=self.dt)

    def update(self, z):
        """
        z: [measured_x, measured_y, measured_theta] 传感器实测值
        """
        def H_jacobian(x):
            # 观测矩阵：状态 [x,y,vx,vy,theta,omega] -> 观测 [x,y,theta]
            return np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0]])

        def H_x(x):
            # 观测方程
            return np.array([x[0], x[1], x[4]])

        self.ekf.update(z, HJacobian=H_jacobian, Hx=H_x)
        return self.ekf.x  # 返回估计出的全状态，喂给 MPC
def test():
    dt=0.1
    m=MPCModel(dt)
    print(m.model.x['x_state'])
    print(m.model.A_fun)
if __name__ == '__main__':
    test()

