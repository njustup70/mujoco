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
        self.mpc.set_rterm(u_input=np.array([0.2, 0.2, 0.5])) #直接用数值
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
    def __init__(self, dt):
        model_type = 'discrete'
        self.model = do_mpc.model.Model(model_type)
        self.dt = dt
        self.n_horizon = 20
        self.s = 0.0
        self.ref_speed = 0.5
        self.path_planner = None
        self.path_total_length = 0.0

        # State: [x, y, theta]
        #self.state= mpc.x['state'] 等价写法
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        # Input (body frame): [vx, vy, vw]
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        # 参数,即参考轨迹点 [x_ref, y_ref, theta_ref]
        ref = self.model.set_variable(var_type='_tvp', var_name='ref', shape=(3, 1))
        
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
        self.mpc.set_rterm(u_input=np.array([0.2, 0.2, 0.5])) #直接用数值
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        set_up_settings={
            'n_horizon': self.n_horizon,
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
        tvp_template=self.mpc.get_tvp_template()
        assert tvp_template is not None
        self.tvp_template = tvp_template
        def tvp_fun(_t_now: float):
            return self.tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()
        from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
        self.pool= ThreadPoolExecutor(max_workers=1)

    def set_path(self, planner: SplinePlanner, ref_speed=None):
        if len(planner.s_samples) == 0 or len(planner.x_path) == 0 or len(planner.y_path) == 0:
            raise ValueError('Path planner has no sampled path, call generate_path() first.')
        self.path_planner = planner
        # 使用真实弧长数组的终点值作为路径总长
        self.path_total_length = float(planner.s_samples[-1])
        if ref_speed is not None:
            self.ref_speed = float(ref_speed)

    def set_reference_speed(self, ref_speed: float):
        self.ref_speed = float(ref_speed)

    def _sample_ref_by_s(self, s_query: float) -> np.ndarray:
        assert self.path_planner is not None
        if s_query >= self.path_total_length:
            x_ref = float(self.path_planner.x_path[-1])
            y_ref = float(self.path_planner.y_path[-1])
            yaw_ref = float(self.path_planner.yaw_path[-1])
            return np.array([x_ref, y_ref, yaw_ref])

        # 用真实弧长数组 s_samples 作为插值轴，保证弧长含义正确
        s_samples = self.path_planner.s_samples
        x_ref = float(np.interp(s_query, s_samples, self.path_planner.x_path))
        y_ref = float(np.interp(s_query, s_samples, self.path_planner.y_path))
        yaw_ref = float(np.interp(s_query, s_samples, self.path_planner.yaw_path))
        return np.array([x_ref, y_ref, yaw_ref])

    def _update_prediction_reference(self, x: np.ndarray):
        if self.path_planner is None:
            return

        idx, _, _, _, _ = self.path_planner.find_nearest_point(float(x[0, 0]), float(x[1, 0]))
        # 用真实弧长数组获取当前位置对应的弧长
        self.s = float(self.path_planner.s_samples[idx])

        for k in range(self.n_horizon + 1):
            s_k = self.s + self.ref_speed * self.dt * k
            ref_k = self._sample_ref_by_s(s_k)
            foxgloveTools.foxgloveViusalInstance.send(ref_k, topic="/mpc_ref")
            self.tvp_template['_tvp', k, 'ref'] = ref_k
            print(ref_k)
    def set_state_init(self, x0):
        '''设置 MPC 的初始状态'''
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
    @time_print(1)
    def update(self,x):
        assert isinstance(x, np.ndarray) and x.shape ==self.u_vec.shape
        '''根据当前状态 x 计算控制输入'''
        self._update_prediction_reference(x)
        u = self.mpc.make_step(x)
        #u是二维数组，形状为 (3, 1)，我们需要将其转换为一维数组
        # print(self.mpc.data)
        return u.flatten()
    def set_target_point(self, target:np.ndarray):
        '''设置 MPC 的目标点'''
        assert len(target) == 3
        for k in range(self.n_horizon + 1):
            self.tvp_template['_tvp', k, 'ref'] =target
    
    async def async_update(self,x):
        '''异步版本的 update 方法'''
        loop = asyncio.get_event_loop()
        u= await loop.run_in_executor(self.pool, self.update, x)
        return u