import warnings
# 屏蔽所有 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)
import do_mpc
import casadi
from casadi import vertcat, cos, sin
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 假设你有一个 decorder 模块
# from decorder import time_print

def time_print(interval):
    # 临时定义一个装饰器，防止你本地运行报错
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ExtendedMPCModel:
    def __init__(self, dt):
        model_type = 'discrete'
        self.model = do_mpc.model.Model(model_type)

        # -----------------------------------------------------------------
        # 1. 状态量 (State): [x, y, theta] - 只有位置和方向
        # -----------------------------------------------------------------
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        
        # 提取各个状态以便后续书写公式
        x     = self.state[0]
        y     = self.state[1]
        theta = self.state[2]

        # -----------------------------------------------------------------
        # 2. 控制量 (Input): [V, alpha, omega] - 直接速度，不是加速度
        # -----------------------------------------------------------------
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        
        V     = self.u_vec[0]  # 合速度
        alpha = self.u_vec[1]  # 速度角（方向）
        omega = self.u_vec[2]  # 角速度

        # -----------------------------------------------------------------
        # 3. 参数 (Parameters): 目标点 [x_ref, y_ref, theta_ref]
        # -----------------------------------------------------------------
        ref = self.model.set_variable(var_type='_p', var_name='ref', shape=(3, 1))

        # -----------------------------------------------------------------
        # 4. 定义系统运动学 (Kinematics)
        # -----------------------------------------------------------------
        # 输入直接是速度，无需积分
        vx_body = V * cos(alpha)
        vy_body = V * sin(alpha)
        vw = omega

        # Convert body-frame velocity to world-frame
        vx_world = cos(theta) * vx_body - sin(theta) * vy_body
        vy_world = sin(theta) * vx_body + cos(theta) * vy_body

        # 下一时刻的状态 (只更新位置和航向，速度由输入直接给定)
        next_state = vertcat(
            x + dt * vx_world,
            y + dt * vy_world,
            theta + dt * vw
        )
        assert isinstance(next_state, casadi.SX)
        self.model.set_rhs('state', next_state)
        self.model.setup()

        # -----------------------------------------------------------------
        # 5. 定义代价函数 (Objective)
        # -----------------------------------------------------------------
        pos_err = casadi.sumsqr(self.state[0:2] - ref[0:2])
        
        # 处理航向角跳变（直接使用 self.state[2] 避免符号识别问题）
        angle_diff = self.state[2] - ref[2]
        wrapped_angle_diff = casadi.atan2(casadi.sin(angle_diff), casadi.cos(angle_diff))
        yaw_err = casadi.sumsqr(wrapped_angle_diff)

        # 终端代价和过程代价（只惩罚位置和航向偏差，让 MPC 自然地走向目标点）
        mterm = 8.0 * pos_err + 2.0 * yaw_err
        lterm = 2.0 * pos_err + 2.0 * yaw_err 
        
        self.mpc = do_mpc.controller.MPC(self.model)
        
        # 输入惩罚：[V, alpha, omega] 
        # V：稍微惩罚（0.2），鼓励较快但不过分的速度
        # alpha：温和惩罚（0.3），防止不必要的舵角转向导致终点震荡
        # omega：温和惩罚（0.1），避免过度旋转
        self.mpc.set_rterm(u_input=np.array([0.2, 0.3, 0.1])) 
        
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        set_up_settings = {
            'n_horizon': 20,
            't_step': dt,
            'nlpsol_opts': {
                'ipopt.print_level': 2,
                'print_time': False,
            },
        }
        self.mpc.set_param(**set_up_settings)

        # -----------------------------------------------------------------
        # 6. 设置边界条件 (Bounds)
        # -----------------------------------------------------------------
        # 6.1 状态边界 (State Bounds)：位置和方向没有约束
        # [x, y, theta]
        self.mpc.bounds['lower', '_x', 'state'] = np.array([-np.inf, -np.inf, -np.inf])
        self.mpc.bounds['upper', '_x', 'state'] = np.array([np.inf, np.inf, np.inf])

        # 6.2 输入边界 (Input Bounds)：约束速度
        # [V (合速度), alpha (速度角), omega (角速度)]
        self.mpc.bounds['lower', '_u', 'u_input'] = np.array([[0.0], [-np.pi], [-2.0]])
        self.mpc.bounds['upper', '_u', 'u_input'] = np.array([[3.0 * np.sqrt(2)], [np.pi], [2.0]])

        # -----------------------------------------------------------------
        # 7. 参数注册
        # -----------------------------------------------------------------
        p_template = self.mpc.get_p_template(1)
        assert p_template is not None
        self.p_template = p_template
        def p_fun(_t_now: float):
            return self.p_template

        self.mpc.set_p_fun(p_fun)
        self.mpc.setup()
        
        self.pool = ThreadPoolExecutor(max_workers=1)

    def set_state_init(self, x0):
        '''设置 MPC 的初始状态. 注意：现在的 x0 必须是 3 维的 [x, y, theta]'''
        assert isinstance(x0, np.ndarray) and x0.shape == (3, 1), "初始状态必须是形状为 (3, 1) 的 numpy 数组"
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    @time_print(10)
    def update(self, x):
        '''根据当前状态 x 计算控制输入'''
        assert isinstance(x, np.ndarray) and x.shape == (3, 1), "输入状态必须是形状为 (3, 1) 的 numpy 数组"
        
        # u 返回的是 [V, alpha, omega] - 直接速度
        u = self.mpc.make_step(x)
        
        return u.flatten()

    def set_target_point(self, target: np.ndarray):
        '''设置 MPC 的目标点'''
        assert len(target) == 3
        self.p_template['_p', 0, 'ref'] = target

    async def async_update(self, x):
        '''异步版本的 update 方法'''
        loop = asyncio.get_event_loop()
        u = await loop.run_in_executor(self.pool, self.update, x)
        return u