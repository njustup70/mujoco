import warnings
# 屏蔽所有 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)
import do_mpc
import casadi
from casadi import vertcat, cos, sin
import numpy as np
from decorder import time_print
import asyncio
from linear import SplinePlanner
import foxgloveTools
'''
位置闭环mpc,不会追踪路径点。
'''
class MPCModel:
    def __init__(self, dt):
        model_type = 'discrete'
        self.model = do_mpc.model.Model(model_type)

        # State: [x, y, theta]
        #self.state= mpc.x['state'] 等价写法
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        # Input (body frame): [vx, vy, vw]
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        # 参数,即参考轨迹点 [x_ref, y_ref, theta_ref]
        ref = self.model.set_variable(var_type='_p', var_name='ref', shape=(3, 1))
        
        #定义系统运动学
        theta = self.state[2]
        vx_body = self.u_vec[0]
        vy_body = self.u_vec[1]
        vw = self.u_vec[2]
        # Convert body-frame velocity to world-frame using heading theta.
        vx_world = cos(theta) * vx_body - sin(theta) * vy_body
        vy_world = sin(theta) * vx_body + cos(theta) * vy_body

        next_state = vertcat(
            self.state[0] + dt * vx_world,
            self.state[1] + dt * vy_world,
            self.state[2] + dt * vw,
        )
        assert isinstance(next_state, casadi.SX)
        self.model.set_rhs('state', next_state)
        self.model.setup()

        pos_err = casadi.sumsqr(self.state[0:2] - ref[0:2])
        # Use cosine yaw cost to avoid angle wrap jump at +-pi.
        # yaw_err = 1.0 - casadi.cos(self.state[2] - ref[2])
        angle_diff = self.state[2] - ref[2]
        wrapped_angle_diff = casadi.atan2(casadi.sin(angle_diff), casadi.cos(angle_diff))


        # 直接使用平方误差
        yaw_err = casadi.sumsqr(wrapped_angle_diff)
        #结束代价，符号函数
        mterm = 8.0 * pos_err + 2.0 * yaw_err
        #过程代价
        lterm = 8.0 * pos_err + 2.0 * yaw_err 
       
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_rterm(u_input=np.array([50, 50, 50])) #直接用数值
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        set_up_settings={
            'n_horizon': 20,
            't_step': dt,
            'store_full_solution': False,
            'nlpsol_opts': {
                'ipopt.print_level': 2,  # 0 关闭 Ipopt 输出 (默认是 5)
                'print_time': False,     # 关闭 CasADi 的计时输出
            },
        }
        self.mpc.set_param(**set_up_settings)
        # Velocity bounds in body frame.
        self.mpc.bounds['lower', '_u', 'u_input'] = np.array([[-3.0], [-3.0], [-2.0]])
        self.mpc.bounds['upper', '_u', 'u_input'] = np.array([[3.0], [3.0], [2.0]])

        # 参数注册
        p_template=self.mpc.get_p_template(1)
        assert p_template is not None
        self.p_template = p_template
        def p_fun(_t_now: float):
            return self.p_template

        self.mpc.set_p_fun(p_fun)
        self.mpc.setup()
        from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
        self.pool= ThreadPoolExecutor(max_workers=1)
    def set_state_init(self, x0):
        '''设置 MPC 的初始状态'''
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
    @time_print(10)
    def update(self,x):
        assert isinstance(x, np.ndarray) and x.shape ==self.u_vec.shape
        '''根据当前状态 x 计算控制输入'''
        u = self.mpc.make_step(x)
        #u是二维数组，形状为 (3, 1)，我们需要将其转换为一维数组
        # print(self.mpc.data)
        return u.flatten()
    def set_target_point(self, target:np.ndarray):
        '''设置 MPC 的目标点'''
        assert len(target) == 3
        self.p_template['_p', 0, 'ref'] =target
    
    async def async_update(self,x):
        '''异步版本的 update 方法'''
        loop = asyncio.get_event_loop()
        u= await loop.run_in_executor(self.pool, self.update, x)
        return u
    
class MPCPathFollower:
    # 1. 切换为连续模型
    def __init__(self, dt):
        self.path_planner = SplinePlanner()
        self.ref_speed = 1.0  # 默认参考速度 (m/s)
        model_type = 'continuous' 
        self.model = do_mpc.model.Model(model_type)
        self.dt = dt
        self.n_horizon = 20
        self.end_point = np.array([0.0, 0.0, 0.0])
        self._step_count = 0
        # do-mpc 默认会持续记录运行历史，长时间运行会让 append/内存开销增长。
        self._history_reset_interval = 200
        self.enable_foxglove_stream = True

        # State: [x, y, theta]
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        # Input (极坐标): [v, alpha, vw]
        # v: 线速度大小, alpha: 运动方向(舵轮转角), vw: 角速度
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        
        ref = self.model.set_variable(var_type='_tvp', var_name='ref', shape=(3, 1))
        
        # 2. 定义连续系统动力学 (dx/dt)
        theta = self.state[2]
        v = self.u_vec[0]
        alpha = self.u_vec[1]
        vw = self.u_vec[2]

        # 连续时间下的导数项
        # 注意：这里不再乘以 dt，dt 在 mpc.set_param 中指定
        x_dot = v * cos(theta + alpha)
        y_dot = v * sin(theta + alpha)
        theta_dot = vw
        next_state=vertcat(x_dot, y_dot, theta_dot)
        assert isinstance(next_state, casadi.SX)
        self.model.set_rhs('state',next_state )
        self.model.setup()

        pos_err = casadi.sumsqr(self.state[0:2] - ref[0:2])
        # Use cosine yaw cost to avoid angle wrap jump at +-pi.
        # yaw_err = 1.0 - casadi.cos(self.state[2] - ref[2])
        angle_diff = self.state[2] - ref[2]
        wrapped_angle_diff = casadi.atan2(casadi.sin(angle_diff), casadi.cos(angle_diff))

        
        # 直接使用平方误差
        yaw_err = casadi.sumsqr(wrapped_angle_diff)
        #结束代价，符号函数
        mterm = 8.0 * pos_err + 1.0 * yaw_err 
        #过程代价
        lterm = 8.0 * pos_err + 1.0 * yaw_err   # 惩罚过小的速度，鼓励更快地到达目标
       
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_rterm(u_input=np.array([2, 5, 10])) #直接用数值
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        set_up_settings={
            'n_horizon': self.n_horizon,
            't_step': dt,
            'store_full_solution': False,
            'state_discretization': 'collocation', # 连续系统推荐使用正交胶合
            'collocation_deg': 2,
            'nlpsol_opts': {
                'ipopt.print_level': 0,      # 关闭 IPOPT 日志
                'print_time': False,
                'ipopt.max_iter': 30,        # 限制最大迭代次数提高实时性
                'ipopt.tol': 1e-3,           # 放宽收敛精度要求
                'ipopt.linear_solver': 'mumps', # 如果有 HSL 库，改为 'ma27' 或 'ma57' 会快很多
                'ipopt.mu_strategy': 'adaptive',
            },
        }
        self.mpc.set_param(**set_up_settings)
        # Velocity bounds in body frame.
        self.mpc.bounds['lower', '_u', 'u_input'] = np.array([[-3.0], [-3.0], [-2.0]])
        self.mpc.bounds['upper', '_u', 'u_input'] = np.array([[3.0], [3.0], [2.0]])

        # 参数注册
        tvp_template=self.mpc.get_tvp_template()
        assert tvp_template is not None
        self.tvp_template = tvp_template
        def tvp_fun(_t_now: float):
            return self.tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()
        from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
        self.pool= ThreadPoolExecutor(max_workers=1)

    def set_path(self,target_points: np.ndarray,target_yaw: float, ref_speed=None):
        self.is_following_path = True
        #传入的target_points是一个二维数组，形状为 (N, 2)，每行是一个路径点的 (x, y) 坐标
        x_pts = target_points[:, 0]
        y_pts = target_points[:, 1]
        self.path_planner.generate_path(x_pts, y_pts, step_cm=10.0)
        self.end_point=np.array([float(x_pts[-1]), float(y_pts[-1]), float(target_yaw)])
        if ref_speed is not None:
            self.ref_speed = float(ref_speed)
    def _update_prediction_reference(self, x: np.ndarray):
        if self.is_following_path==False:
            return
        # 根据当前位置找到路径上最近点，得到对应的虚拟弧长 s
        self.s = self.path_planner.get_nearest_s(float(x[0, 0]), float(x[1, 0]))
        # 在预测域内，s 按参考速度逐步向前推进，每步推进 ref_speed * dt
        for k in range(self.n_horizon + 1):
            s_k = self.s + self.ref_speed * self.dt * k
            # 插值得到第 k 步的参考点 (x_ref, y_ref, yaw_ref)；超出路径终点后自动钳位
            ref_k= self.path_planner.get_state_by_s(s_k)
            #构造 MPC 需要的参考值格式，并更新到 tvp_template 中
            ref_k[2]=self.end_point[2]  #保持角度参考不变，直接使用终点的角度参考
            #只更改位置参考，保持角度参考不变
            self.tvp_template['_tvp', k, 'ref'] =ref_k 
            # print(ref_k)
    def set_state_init(self, x0):
        '''设置 MPC 的初始状态'''
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
    @time_print(10)
    def update(self,x):
        assert isinstance(x, np.ndarray) and x.shape ==self.u_vec.shape
        '''根据当前状态 x 计算控制输入'''
        self._step_count += 1
        if self._step_count % self._history_reset_interval == 0:
            self.mpc.reset_history()

        self._update_prediction_reference(x)
        u = self.mpc.make_step(x)
        #u是二维数组，形状为 (3, 1)，我们需要将其转换为一维数组
        if self.enable_foxglove_stream:
            foxgloveTools.foxgloveViusalInstance.send(u.flatten(), topic="/mpc/control_input")
        vx_body=u[0]*cos(u[1])
        vy_body=u[0]*sin(u[1])
        u[0]=vx_body
        u[1]=vy_body
        # print(self.mpc.data)
        return u.flatten()
    def set_target_point(self, target:np.ndarray):
        '''设置 MPC 的目标点'''
        assert len(target) == 3
        self.is_following_path= False
        self.end_point=target
        for k in range(self.n_horizon + 1):
            self.tvp_template['_tvp', k, 'ref'] = target
    async def async_update(self,x):
        '''异步版本的 update 方法'''
        loop = asyncio.get_event_loop()
        u= await loop.run_in_executor(self.pool, self.update, x)
        return u