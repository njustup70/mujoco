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
        from concurrent.futures import ThreadPoolExecutor
        self.pool= ThreadPoolExecutor(max_workers=1)
    def set_state_init(self, x0):
        '''设置 MPC 的初始状态'''
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
    @time_print(10)
    def update(self,x):
        assert isinstance(x, np.ndarray) and x.shape == (3, 1)
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
    def __init__(self, dt,type='swerve'):
        assert type in ['swerve', 'omni'], "type must be 'swerve' or 'omni'"

        # --- 模型配置 ---
        self.drive_type = type
        self.dt = float(dt)
        self.n_horizon = 20
        model_type = 'continuous' if self.drive_type == 'swerve' else 'discrete'

        # --- 路径与运行状态 ---
        self.path_planner = SplinePlanner()
        self.is_following_path = False
        self.ref_speed = 1.0
        self.end_point = np.array([0.0, 0.0, 0.0])
        self.s = 0.0
        self._step_count = 0
        # do-mpc 默认会持续记录运行历史，长时间运行会让 append/内存开销增长。
        self._history_reset_interval = 200
        self.model = do_mpc.model.Model(model_type)
        # State: [x, y, theta, dx, dy, dtheta] (6维扩张状态，后3维为扰动项)
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(6, 1))
        # Input:
        # - swerve: [v, alpha, vw]
        # - omni:   [vx_body, vy_body, vw]
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        ref = self.model.set_variable(var_type='_tvp', var_name='ref', shape=(3, 1))

        # --- 按底盘类型定义动力学 ---
        # 提取原始状态与扰动项
        x, y, theta = self.state[0], self.state[1], self.state[2]
        dx, dy, dtheta = self.state[3], self.state[4], self.state[5]
        
        if self.drive_type == 'swerve':
            v = self.u_vec[0]
            alpha = self.u_vec[1]
            vw = self.u_vec[2]

            # 连续系统: x_dot = f(x,u) + 扰动项
            x_dot = v * cos(theta + alpha) + dx
            y_dot = v * sin(theta + alpha) + dy
            theta_dot = vw + dtheta
            # 扰动在预测域内认为是常数（衰减由观测器处理）
            dx_dot = 0.0
            dy_dot = 0.0
            dtheta_dot = 0.0
            rhs_state = vertcat(x_dot, y_dot, theta_dot, dx_dot, dy_dot, dtheta_dot)

            rterm = np.array([2.0, 5.0, 10.0])
            lower_u = np.array([[-3.0], [-np.pi], [-2.0]])
            upper_u = np.array([[3.0], [np.pi], [2.0]])
            set_up_settings = {
                'n_horizon': self.n_horizon,
                't_step': self.dt,
                'store_full_solution': False,
                'state_discretization': 'collocation',
                'collocation_deg': 2,
                'nlpsol_opts': {
                    'ipopt.print_level': 0,
                    'print_time': False,
                    'ipopt.max_iter': 30,
                    'ipopt.tol': 1e-3,
                    'ipopt.linear_solver': 'mumps',
                    'ipopt.mu_strategy': 'adaptive',
                },
            }
        else:
            vx_body = self.u_vec[0]
            vy_body = self.u_vec[1]
            vw = self.u_vec[2]

            # 离散系统: x(k+1) = f(x(k),u(k)) + 扰动项
            vx_world = cos(theta) * vx_body - sin(theta) * vy_body
            vy_world = sin(theta) * vx_body + cos(theta) * vy_body
            # 基本运动学 + 扰动（扰动在预测域内认为是常数）
            x_next = x + self.dt * vx_world + dx
            y_next = y + self.dt * vy_world + dy
            theta_next = theta + self.dt * vw + dtheta
            dx_next = dx
            dy_next = dy
            dtheta_next = dtheta
            rhs_state = vertcat(
                x_next,
                y_next,
                theta_next,
                dx_next,
                dy_next,
                dtheta_next,
            )

            rterm = np.array([50.0, 50.0, 50.0])
            lower_u = np.array([[-3.0], [-3.0], [-2.0]])
            upper_u = np.array([[3.0], [3.0], [2.0]])
            set_up_settings = {
                'n_horizon': self.n_horizon,
                't_step': self.dt,
                'store_full_solution': False,
                'nlpsol_opts': {
                    'ipopt.print_level': 0,
                    'print_time': False,
                    'ipopt.max_iter': 30,
                    'ipopt.tol': 1e-3,
                    'ipopt.linear_solver': 'mumps',
                    'ipopt.mu_strategy': 'adaptive',
                },
            }

        assert isinstance(rhs_state, casadi.SX)
        self.model.set_rhs('state', rhs_state)
        self.model.setup()

        # 成本函数只考虑位置/角度误差，不直接惩罚扰动
        # （扰动会通过其在动力学中的影响间接被优化）
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
        lterm = 8.0 * pos_err + 1.0 * yaw_err
       
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_rterm(u_input=rterm)
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_param(**set_up_settings)
        self.mpc.bounds['lower', '_u', 'u_input'] = lower_u
        self.mpc.bounds['upper', '_u', 'u_input'] = upper_u

        # 参数注册
        tvp_template=self.mpc.get_tvp_template()
        assert tvp_template is not None
        self.tvp_template = tvp_template
        def tvp_fun(_t_now: float):
            return self.tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()
        from concurrent.futures import ThreadPoolExecutor
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
        '''设置 MPC 初始状态：
        - 输入 (3,1): 扰动默认置零
        - 输入 (6,1): 直接使用外部传入扰动
        '''
        assert isinstance(x0, np.ndarray) and (x0.shape == (3, 1) or x0.shape == (6, 1)), "输入状态必须是 (3, 1) 或 (6, 1) 的 numpy 数组"
        if x0.shape == (3, 1):
            x0_expanded = np.array([[x0[0, 0]], [x0[1, 0]], [x0[2, 0]], [0.0], [0.0], [0.0]], dtype=float)
        else:
            x0_expanded = x0
        self.mpc.x0 = x0_expanded
        self.mpc.set_initial_guess()
    @time_print(10)
    def update(self,x):
        '''根据当前状态 x 计算控制输入：
        - 输入 (3,1): 扰动默认置零
        - 输入 (6,1): 直接使用外部传入扰动
        '''
        assert isinstance(x, np.ndarray) and (x.shape == (3, 1) or x.shape == (6, 1)), "输入状态必须是 (3, 1) 或 (6, 1) 的 numpy 数组"
        self._step_count += 1
        if self._step_count % self._history_reset_interval == 0:
            self.mpc.reset_history()

        self._update_prediction_reference(x)
        if x.shape == (3, 1):
            x_expanded = np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [0.0], [0.0], [0.0]], dtype=float)
        else:
            x_expanded = x
        u = self.mpc.make_step(x_expanded)
        #u是二维数组，形状为 (3, 1)，我们需要将其转换为一维数组
        # swerve 输出 [v, alpha, vw]，对外统一成 [vx_body, vy_body, vw]
        if self.drive_type == 'swerve':
            vx_body = u[0] * cos(u[1])
            vy_body = u[0] * sin(u[1])
            u[0] = vx_body
            u[1] = vy_body

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