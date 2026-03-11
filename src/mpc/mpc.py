import do_mpc
from casadi import vertcat,sin,cos
import casadi
class model:
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
def test():
    dt=0.1
    m=model(dt)
    print(m.model.x['x_state'])
    print(m.model.A_fun)
if __name__ == '__main__':
    test()

