import os
import numpy as np
from casadi import SX, vertcat, sin, cos
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from linear import SplinePlanner
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
        self.h_cg = float(h_cg)      
        self.L = float(wheel_base)    
        self.W_base = float(wheel_width)   
        self.mu = float(mu)           
        self.g = 9.81
        self.dt = float(dt)
        self.n_horizon = 100         
        self.is_following_path = False
        self.ref_speed = 1.5
        self.end_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float) # x, y, theta, vx, vy, omega
        self.s = 0.0
        self.int_vel_cmd = np.zeros(3, dtype=float)
        
        self.path_planner = SplinePlanner()
        self._step_count = 0

        # --- 1. 定义 acados 模型 ---
        self.model = self.setup_acados_model()
        
        # --- 2. 定义 OCP ---
        self.ocp = AcadosOcp()
        self.ocp.model = self.model
        #断言有效的状态和输入维度
        assert isinstance(self.model.x, SX) 
        assert isinstance(self.model.u, SX)
        nx = self.model.x.size1()
        nu = self.model.u.size1()
        
        # 使用新的 API 设置步数
        self.ocp.solver_options.N_horizon = self.n_horizon
        self.ocp.solver_options.tf = self.n_horizon * self.dt
        
        # --- 3. 约束 ---
        limit_ax_tip = (self.g * self.L / 2.0) / self.h_cg
        limit_ay_tip = (self.g * self.W_base / 2.0) / self.h_cg
        print(f"Calculated tip-over limits: ax <= {limit_ax_tip:.2f}, ay <= {limit_ay_tip:.2f}")

        self.ocp.constraints.idxbu = np.array([0, 1, 2])
        self.ocp.constraints.lbu = np.array([-5.0, -5.0, -5.0])
        self.ocp.constraints.ubu = np.array([5.0, 5.0, 5.0])
        self.ocp.constraints.x0 = np.zeros(nx)

        # --- 4. 代价函数 (Cost) ---
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        # 核心修复：确保权重矩阵严格正定（所有对角元素 > 0）
        # 顺序：[x, y, theta, vx, vy, omega, ax, ay, alpha]
        Q_diag = np.array([15.0, 15.0, 5.0, 1e-6, 1e-6, 1e-6]) 
        R_diag = np.array([0.01, 0.01, 0.01])
        self.ocp.cost.W = np.diag(np.concatenate([Q_diag, R_diag]))
        
        # 终端权重仅包含状态 [x, y, theta, vx, vy, omega]
        self.ocp.cost.W_e = np.diag(np.array([5.0, 5.0, 5.0, 1e-6, 1e-6, 1e-6]))

        # 定义输出表达式：y = [状态, 输入]
        self.ocp.model.cost_y_expr = vertcat(self.model.x, self.model.u)
        self.ocp.model.cost_y_expr_e = self.model.x
        
        # 外部参数
        self.ocp.parameter_values = np.array([0.0, 0.0, 0.0]) 

        # --- 5. 求解器设置 ---
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI' 
        # 设置初始 yref 占位符 (维度必须与 cost_y_expr 一致，即 nx + nu = 9)
        self.ocp.cost.yref = np.zeros(9)

        # 设置初始终端 yref_e 占位符 (维度必须与 cost_y_expr_e 一致，即 nx = 6)
        self.ocp.cost.yref_e = np.zeros(6)

        # 确保权重 W_0 也被初始化（通常等于 W）
        self.ocp.cost.W_0 = self.ocp.cost.W
        json_file = os.path.join(os.getcwd(), 'acados_ocp.json')
        self.solver = AcadosOcpSolver(self.ocp, json_file=json_file)
        
        self.pool = ThreadPoolExecutor(max_workers=1)

    def setup_acados_model(self):
        model = AcadosModel()
        model.name = "acc_mpc_model"

        x = SX.sym('x')
        y = SX.sym('y')
        theta = SX.sym('theta')
        vx = SX.sym('vx')
        vy = SX.sym('vy')
        omega = SX.sym('omega')
        state = vertcat(x, y, theta, vx, vy, omega)

        ax = SX.sym('ax')
        ay = SX.sym('ay')
        alpha = SX.sym('alpha')
        u = vertcat(ax, ay, alpha)

        x_ref = SX.sym('x_ref')
        y_ref_p = SX.sym('y_ref_p')
        theta_ref = SX.sym('theta_ref')
        p = vertcat(x_ref, y_ref_p, theta_ref)

        dx = vx * cos(theta) - vy * sin(theta)
        dy = vx * sin(theta) + vy * cos(theta)
        dtheta = omega
        dvx = ax + omega * vy
        dvy = ay + omega * (-vx)
        domega = alpha

        model.f_expl_expr = vertcat(dx, dy, dtheta, dvx, dvy, domega)
        model.x = state
        model.u = u
        model.p = p
        
        return model

    def set_path(self, target_points: np.ndarray, target_yaw: float, ref_speed=None):
        self.is_following_path = True
        x_pts = target_points[:, 0]
        y_pts = target_points[:, 1]
        self.path_planner.generate_path(x_pts, y_pts, step_cm=10.0)
        # 存储包含速度和角速度的终点状态参考（设为0）
        self.end_point = np.array([float(x_pts[-1]), float(y_pts[-1]), float(target_yaw), 0.0, 0.0, 0.0])
        if ref_speed is not None:
            self.ref_speed = float(ref_speed)

    def update(self, x_current):
        self._step_count += 1
        x_current_flat = x_current.flatten()
        
        if self.is_following_path:
            self.s = self.path_planner.get_nearest_s(x_current_flat[0], x_current_flat[1])
            for k in range(self.n_horizon):
                s_k = self.s + self.ref_speed * self.dt * k
                ref_pos = self.path_planner.get_state_by_s(s_k)
                
                # 1. 设置参数 p (用于模型中的参考)
                ref_p = np.array([ref_pos[0], ref_pos[1], self.end_point[2]])
                self.solver.set(k, "p", ref_p)
                
                # 2. 设置 yref: [x, y, theta, vx, vy, omega, ax, ay, alpha]
                # 必须是 9 维，且顺序与 cost_y_expr 一致
                y_ref = np.array([
                    ref_pos[0], ref_pos[1], self.end_point[2], # x, y, theta
                    0.0, 0.0, 0.0,                             # vx, vy, omega
                    0.0, 0.0, 0.0                              # ax, ay, alpha
                ])
                self.solver.set(k, "yref", y_ref)
            
            # 终端参考 yref_e (只有 6 维状态)
            self.solver.set(self.n_horizon, "yref", self.end_point)
        else:
            # 静态目标模式
            for k in range(self.n_horizon):
                self.solver.set(k, "p", self.end_point[:3])
                y_ref = np.concatenate([self.end_point, np.zeros(3)])
                self.solver.set(k, "yref", y_ref)
            self.solver.set(self.n_horizon, "yref", self.end_point)

        # 3. 设置当前状态约束并求解
        self.solver.set(0, "lbx", x_current_flat)
        self.solver.set(0, "ubx", x_current_flat)
        
        status = self.solver.solve()
        if status != 0:
            # 可以根据需要添加日志
            pass

        u_acc = self.solver.get(0, "u")

        # 4. 计算期望速度指令 (当前速度 + 加速度*dt)
        vx_cmd = float(x_current_flat[3] + u_acc[0] * self.dt)
        vy_cmd = float(x_current_flat[4] + u_acc[1] * self.dt)
        omega_cmd = float(x_current_flat[5] + u_acc[2] * self.dt)

        return np.array([vx_cmd, vy_cmd, omega_cmd])

    def set_target_point(self, target: np.ndarray):
        """target: [x, y, theta]"""
        self.is_following_path = False
        self.end_point = np.array([target[0], target[1], target[2], 0.0, 0.0, 0.0])

    def set_state_init(self, x0):
        x0_flat = x0.flatten()
        self.solver.set(0, "lbx", x0_flat)
        self.solver.set(0, "ubx", x0_flat)
        for k in range(self.n_horizon + 1):
            self.solver.set(k, "x", x0_flat)

    async def async_update(self, x):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, self.update, x)