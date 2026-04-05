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


class ChessicModel:
    def __init__(
        self,
        mass=50.0,
        iz=5.0,
        h_cg=0.15,
        dx_cg=0.05,
        dy_cg=0.02,
        wheel_base=0.4,
        wheel_width=0.4,
        mu=0.8,
        g=9.81,
    ):
        self.mass = float(mass)
        self.iz = float(iz)
        self.h_cg = float(h_cg)
        self.dx_cg = float(dx_cg)
        self.dy_cg = float(dy_cg)
        self.L = float(wheel_base)
        self.W = float(wheel_width)
        self.mu = float(mu)
        self.g = float(g)
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
        self.enable_foxglove_stream = True

        self.model = do_mpc.model.Model(model_type)

        # State: [x, y, theta]
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        # Input:
        # - swerve: [v, alpha, vw]
        # - omni:   [vx_body, vy_body, vw]
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        ref = self.model.set_variable(var_type='_tvp', var_name='ref', shape=(3, 1))

        # --- 按底盘类型定义动力学 ---
        theta = self.state[2]
        if self.drive_type == 'swerve':
            v = self.u_vec[0]
            alpha = self.u_vec[1]
            vw = self.u_vec[2]

            # 连续系统: x_dot = f(x,u)
            x_dot = v * cos(theta + alpha)
            y_dot = v * sin(theta + alpha)
            theta_dot = vw
            rhs_state = vertcat(x_dot, y_dot, theta_dot)

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

            # 离散系统: x(k+1) = f(x(k),u(k))
            vx_world = cos(theta) * vx_body - sin(theta) * vy_body
            vy_world = sin(theta) * vx_body + cos(theta) * vy_body
            rhs_state = vertcat(
                self.state[0] + self.dt * vx_world,
                self.state[1] + self.dt * vy_world,
                self.state[2] + self.dt * vw,
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
        '''设置 MPC 的初始状态'''
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
    @time_print(10)
    def update(self,x):
        assert isinstance(x, np.ndarray) and x.shape == (3, 1)
        '''根据当前状态 x 计算控制输入'''
        self._step_count += 1
        if self._step_count % self._history_reset_interval == 0:
            self.mpc.reset_history()

        self._update_prediction_reference(x)
        u = self.mpc.make_step(x)
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
class DynamicMPCPathFollower:
    def __init__(self, dt, chessic_model=None):
        if chessic_model is None:
            chessic_model = ChessicModel()
        self.chessic_model = chessic_model

        # --- 物理参数 ---
        self.mass = self.chessic_model.mass
        self.iz = self.chessic_model.iz
        self.h_cg = self.chessic_model.h_cg
        self.dx_cg = self.chessic_model.dx_cg
        self.dy_cg = self.chessic_model.dy_cg
        self.L = self.chessic_model.L
        self.W = self.chessic_model.W
        self.mu = self.chessic_model.mu
        self.g = self.chessic_model.g

        # --- 配置 ---
        self.dt = float(dt)
        self.n_horizon = 10
        self.model_type = 'continuous'
        
        # --- 状态与路径规划 ---
        self.path_planner = SplinePlanner() 
        self.is_following_path = False
        self.ref_speed = 1.0
        self.end_point = np.array([0.0, 0.0, 0.0]) # x, y, theta
        self._step_count = 0
        self._history_reset_interval = 200

        # --- 模型定义 ---
        self.model = do_mpc.model.Model(self.model_type)

        # 状态: [x, y, theta, vx, vy, omega] (均基于质心)
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(6, 1))
        # 输入: [Fx, Fy, Mz] (作用于质心)
        self.u_force = self.model.set_variable(var_type='_u', var_name='u_force', shape=(3, 1))
        # 参考: [x_ref, y_ref, theta_ref]
        ref = self.model.set_variable(var_type='_tvp', var_name='ref', shape=(3, 1))

        # --- 动力学方程 (考虑质心系) ---
        theta = self.state[2]
        vx = self.state[3]
        vy = self.state[4]
        omega = self.state[5]

        Fx = self.u_force[0]
        Fy = self.u_force[1]
        Mz = self.u_force[2]

        # 1. 运动学 (基于质心速度推导世界坐标系变化)
        dx = vx * cos(theta) - vy * sin(theta)
        dy = vx * sin(theta) + vy * cos(theta)
        dtheta = omega

        # 2. 动力学 (Newton-Euler)
        # 虽然质心偏移，但 F=ma 在质心系下依然成立
        dvx = Fx / self.mass + vy * omega
        dvy = Fy / self.mass - vx * omega
        domega = Mz / self.iz

        rhs_state = vertcat(dx, dy, dtheta, dvx, dvy, domega)
        assert isinstance(rhs_state, casadi.SX)
        self.model.set_rhs('state', rhs_state)
        self.model.setup()

        # --- 代价函数 (不包含速度误差) ---
        pos_err = casadi.sumsqr(self.state[0:2] - ref[0:2])
        angle_diff = self.state[2] - ref[2]
        wrapped_angle_diff = casadi.atan2(sin(angle_diff), cos(angle_diff))
        yaw_err = casadi.sumsqr(wrapped_angle_diff)

        # 仅优化位置和角度，让 MPC 自动决定 vx, vy 来达成目标
        self.lterm = 8.0 * pos_err + 2.0 * yaw_err
        self.mterm = 12.0 * pos_err + 5.0 * yaw_err

        # --- 在定义 MPC 控制器之后，setup 之前修改 ---
        self.mpc = do_mpc.controller.MPC(self.model)

        # 1. 明确获取 MPC 层级的控制变量符号 (非常重要)
        u = self.mpc.model.u
        Fx_sym = u['u_force', 0]
        Fy_sym = u['u_force', 1]

        # 2. 计算加速度符号
        ax_sym = Fx_sym / self.mass
        ay_sym = Fy_sym / self.mass

        # 3. 重新设置非线性约束
        # 摩擦圆
        # self.mpc.set_nl_cons('friction_circle', (Fx_sym**2 + Fy_sym**2), (self.mu * self.mass * self.g)**2)

        # 防翻转约束 (确保右侧是计算好的常数)
        limit_x = float(self.g * self.L / 2.0)
        limit_y = float(self.g * self.W / 2.0)

        # 注意：do_mpc 的 fabs 建议使用 casadi.fabs
        self.mpc.set_nl_cons('no_tip_over_x', casadi.fabs(ax_sym * self.h_cg), limit_x)
        self.mpc.set_nl_cons('no_tip_over_y', casadi.fabs(ay_sym * self.h_cg), limit_y)
        # --- MPC 运行参数 ---
        set_up_settings = {
            'n_horizon': self.n_horizon,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'store_full_solution': False,
            'nlpsol_opts': {
                'ipopt.print_level': 0,
                'print_time': False,
                'ipopt.max_iter': 40,
                'ipopt.linear_solver': 'mumps',
            },
        }
        self.mpc.set_param(**set_up_settings)
        
        # 这里的 Bounds 是虚拟力的物理极限
        self.mpc.bounds['lower', '_u', 'u_force'] = np.array([[-400.0], [-400.0], [-150.0]])
        self.mpc.bounds['upper', '_u', 'u_force'] = np.array([[400.0], [400.0], [150.0]])

        self.mpc.set_objective(mterm=self.mterm, lterm=self.lterm)
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(lambda _t: self.tvp_template)
        self.mpc.setup()
        
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1)

    def set_path(self, target_points: np.ndarray, target_yaw: float, ref_speed=None):
        self.is_following_path = True
        x_pts = target_points[:, 0]
        y_pts = target_points[:, 1]
        self.path_planner.generate_path(x_pts, y_pts, step_cm=10.0)
        self.end_point = np.array([float(x_pts[-1]), float(y_pts[-1]), float(target_yaw)])
        if ref_speed is not None:
            self.ref_speed = float(ref_speed)

    def _update_prediction_reference(self, x: np.ndarray):
        if not self.is_following_path:
            return
        
        # 根据质心当前位置查找最近点
        self.s = self.path_planner.get_nearest_s(float(x[0, 0]), float(x[1, 0]))
        
        for k in range(self.n_horizon + 1):
            # 这里的 s 前进依然参考 ref_speed，但 MPC 并没有速度误差代价
            # 这意味着 ref_speed 只是决定了“参考点”在路径上的演进速度
            s_k = self.s + self.ref_speed * self.dt * k
            ref_k_pos = self.path_planner.get_state_by_s(s_k)
            
            # 参考值: [x, y, theta]
            self.tvp_template['_tvp', k, 'ref'] = np.array([ref_k_pos[0], ref_k_pos[1], self.end_point[2]])

    def set_state_init(self, x0):
        """x0: [x, y, theta, vx, vy, omega]"""
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def update(self, x):
        assert x.shape == (6, 1)
        self._step_count += 1
        if self._step_count % self._history_reset_interval == 0:
            self.mpc.reset_history()

        self._update_prediction_reference(x)
        u = self.mpc.make_step(x)
        return u.flatten() # 返回 [Fx, Fy, Mz]

    def set_target_point(self, target: np.ndarray):
        """target: [x, y, theta]"""
        self.is_following_path = False
        self.end_point = target
        for k in range(self.n_horizon + 1):
            self.tvp_template['_tvp', k, 'ref'] = target
    @time_print(10)
    async def async_update(self, x):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, self.update, x)