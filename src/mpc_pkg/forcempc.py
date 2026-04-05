import numpy as np
import do_mpc
from linear import SplinePlanner
from casadi import vertcat, sin, cos
import casadi
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AccMPCPathFollower:
    def __init__(
        self,
        dt,
        mass=22.0,
        iz=2.8,
        h_cg=0.15,
        wheel_base=0.4,
        wheel_width=0.4,
        mu=0.6,
        dx_cg=0.0,
        dy_cg=0.0,
        **_kwargs,
    ):
        # --- 物理参数 ---
        self.mass = float(mass)
        self.iz = float(iz)
        self.dx_cg = float(dx_cg)
        self.dy_cg = float(dy_cg)
        self.h_cg = float(h_cg)      # 质心高度
        self.L = float(wheel_base)    # 轴距 (前后)
        self.W = float(wheel_width)   # 轮距 (左右)
        self.mu = float(mu)           # 摩擦系数 (控制打滑)
        self.g = 9.81
        self.dt = float(dt)
        self.n_horizon = 10           # 加速度控制下，预测时域可以稍微长一点
        self.is_following_path = False
        self.ref_speed = 0.5
        self.end_point = np.array([0.0, 0.0, 0.0], dtype=float)
        self.s = 0.0
        self.int_vel_cmd = np.zeros(3, dtype=float)

        # --- 模型定义 ---
        self.model = do_mpc.model.Model('continuous')

        # 状态: [x, y, theta, vx, vy, omega] (Body frame 下的速度)
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(6, 1))
        
        # 输入: [ax, ay, alpha] (机体坐标系下的纵向、横向、角加速度)
        self.u_acc = self.model.set_variable(var_type='_u', var_name='u_acc', shape=(3, 1))
        
        # 参考值: [x_ref, y_ref, theta_ref]
        ref = self.model.set_variable(var_type='_tvp', var_name='ref', shape=(3, 1))

        # --- 动力学方程 ---
        theta = self.state[2]
        vx = self.state[3]
        vy = self.state[4]
        omega = self.state[5]

        ax = self.u_acc[0]
        ay = self.u_acc[1]
        alpha = self.u_acc[2]

        # 1. 位置变化 (世界坐标系)
        dx = vx * cos(theta) - vy * sin(theta)
        dy = vx * sin(theta) + vy * cos(theta)
        dtheta = omega

        # 2. 速度变化 (考虑向心力修正)
        # dv = a + (omega x v)
        dvx = ax 
        dvy = ay 
        domega = alpha

        rhs_state = vertcat(dx, dy, dtheta, dvx, dvy, domega)
        assert isinstance(rhs_state, casadi.SX) or isinstance(rhs_state, casadi.MX)
        self.model.set_rhs('state', rhs_state)
        self.model.setup()

        # --- MPC 控制器配置 ---
        self.mpc = do_mpc.controller.MPC(self.model)
        self.path_planner = SplinePlanner()
        self._history_reset_interval = 50
        self._step_count = 0
        # --- 约束条件 (防止倾倒与打滑) ---
        # 1. 摩擦圆约束: ax^2 + ay^2 <= (mu * g)^2
        # self.mpc.set_nl_cons('friction_circle', (ax**2 + ay**2), (self.mu * self.g)**2)

        # 2. 纵向防翻车: |ax| * h_cg <= g * L/2
        limit_ax_tip = (self.g * self.L / 2.0) / self.h_cg
        # 3. 横向防翻车: |ay| * h_cg <= g * W/2
        limit_ay_tip = (self.g * self.W / 2.0) / self.h_cg
        print(f"Calculated tip-over limits: ax <= {limit_ax_tip:.2f} m/s², ay <= {limit_ay_tip:.2f} m/s²")
        # 直接设置在 bounds 里比 nl_cons 更高效
        self.mpc.bounds['lower', '_u', 'u_acc'] = np.array([[-5], [-5], [-0.0]])
        self.mpc.bounds['upper', '_u', 'u_acc'] = np.array([[5], [5], [0.0]])

        # --- 代价函数 ---
        pos_err = casadi.sumsqr(self.state[0:2] - ref[0:2])
        angle_diff = self.state[2] - ref[2]
        yaw_err = casadi.sumsqr(casadi.atan2(sin(angle_diff), cos(angle_diff)))
        
        # 惩罚速度，防止无限制加速 (如果路径没有速度参考)
        self.mpc.set_objective(mterm=15.0 * pos_err + 5.0 * yaw_err, 
                               lterm=10.0 * pos_err + 2.0 * yaw_err )
        
        # 惩罚加速度的变化率，让动作更平滑
        self.mpc.set_rterm(u_acc=np.array([1e-1, 1e-1, 1e-1]))

        # --- 运行设置 ---
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
        
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(lambda _t: self.tvp_template)
        self.mpc.setup()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def reset_integrator(self):
        self.int_vel_cmd[:] = 0.0

    def set_path(self, target_points: np.ndarray, target_yaw: float, ref_speed=None):
        self.is_following_path = True
        x_pts = target_points[:, 0]
        y_pts = target_points[:, 1]
        self.path_planner.generate_path(x_pts, y_pts, step_cm=10.0)
        self.end_point = np.array([float(x_pts[-1]), float(y_pts[-1]), float(target_yaw)])
        if ref_speed is not None:
            self.ref_speed = float(ref_speed)

    def _update_prediction_reference(self, x: np.ndarray):
            """
            x: [x, y, theta, vx, vy, omega]
            根据当前位置 x[0:2] 在路径上插值，生成未来的参考轨迹
            """
            if not self.is_following_path:
                return
            
            # 1. 寻找当前在路径上的投影点 s
            self.s = self.path_planner.get_nearest_s(float(x[0, 0]), float(x[1, 0]))
            
            for k in range(self.n_horizon + 1):
                # 2. 预测时域内的参考点随时间(速度)向前推进
                # 这里的 ref_speed 决定了参考点在路径上“跑”多快
                s_k = self.s + self.ref_speed * self.dt * k
                ref_k_pos = self.path_planner.get_state_by_s(s_k)
                
                # 3. 构造参考向量 [x_ref, y_ref, theta_ref]
                # 注意：角度参考通常取终点角度，或者路径切线角，这里沿用你的逻辑取终点角度
                ref_val = np.array([
                    ref_k_pos[0],      # x_ref
                    ref_k_pos[1],      # y_ref
                    self.end_point[2]  # theta_ref (保持期望朝向)
                ])
                
                self.tvp_template['_tvp', k, 'ref'] = ref_val

    def update(self, x):
        """
        x: np.ndarray 形状为 (6, 1) -> [x, y, theta, vx, vy, omega]
          返回: 加速度控制量 [ax, ay, alpha]
        """
        assert x.shape == (6, 1), f"Input state must be (6, 1), got {x.shape}"
        
        self._step_count += 1
        # 定期重置历史，防止 do-mpc 内存占用过高
        if self._step_count % self._history_reset_interval == 0:
            self.mpc.reset_history()
            self.reset_integrator()

        # 更新参考轨迹
        self._update_prediction_reference(x)
        
        # 求解 MPC 得到当前的加速度输入 [ax, ay, alpha]
        u_acc = self.mpc.make_step(x)

        # 在 MPC 内部完成一次积分，输出速度指令。
        self.int_vel_cmd[0] += float(u_acc[0, 0]) * self.dt
        self.int_vel_cmd[1] += float(u_acc[1, 0]) * self.dt
        self.int_vel_cmd[2] += float(u_acc[2, 0]) * self.dt

        return self.int_vel_cmd.copy()

    def set_target_point(self, target: np.ndarray):
        """
        设置静态目标点（不走路径规划）
        target: [x, y, theta]
        """
        assert len(target) == 3
        self.is_following_path = False
        self.end_point = target
        
        # 将整个预测时域的参考点全部指向目标点
        for k in range(self.n_horizon + 1):
            self.tvp_template['_tvp', k, 'ref'] = target

    def set_state_init(self, x0):
        """
        初始化 MPC 状态
        x0: [x, y, theta, vx, vy, omega]
        """
        assert isinstance(x0, np.ndarray) and x0.shape == (6, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        self.reset_integrator()

    async def async_update(self, x):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, self.update, x)