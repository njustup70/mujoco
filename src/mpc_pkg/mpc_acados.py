import os
import numpy as np
import casadi
from casadi import SX, vertcat, cos, sin, atan2
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from linear import SplinePlanner

class AcadosMPC:
    def __init__(self, dt: float, model_type: str = 'omni', n_horizon: int = 20):
        self.dt, self.n_horizon, self.model_type = float(dt), n_horizon, model_type
        assert model_type in ['swerve', 'omni'], "type must be 'swerve' or 'omni'"
        # 1. 模型与约束定义
        model = AcadosModel()
        model.name = f"mpc_{model_type}"
        x, u = SX.sym('x', 3), SX.sym('u', 3) #type: ignore
        model.x, model.u = x, u
        
        theta = x[2]
        if model_type == 'swerve':
            v, alpha, vw = u[0], u[1], u[2]  # [速度, 舵角, 角速度]
            f_expl = vertcat(v * cos(theta + alpha), v * sin(theta + alpha), vw)
            lbu, ubu = np.array([-4.0, -np.pi, -2.0]), np.array([4.0, np.pi, 2.0])
        else: # omni
            vx, vy, vw = u[0], u[1], u[2]    # [vx, vy, vw] (车体系)
            f_expl = vertcat(vx * cos(theta) - vy * sin(theta), vx * sin(theta) + vy * cos(theta), vw)
            lbu, ubu = np.array([-3.0, -3.0, -2.0]), np.array([3.0, 3.0, 2.0])
            
        model.f_expl_expr = f_expl
        model.f_impl_expr = SX.sym('x_dot', 3) - f_expl # type: ignore
        p = SX.sym('p', 3) #type: ignore
        model.p = p
        # 2. OCP 求解器配置
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = n_horizon
        ocp.solver_options.tf = n_horizon * dt
        ocp.solver_options.qp_solver, ocp.solver_options.nlp_solver_type = 'PARTIAL_CONDENSING_HPIPM', 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4 # 龙格库塔的级数
        ocp.solver_options.sim_method_num_steps = 3  # 在一个 dt 内分 3 步走
        # 控制约束
        ocp.constraints.idxbu, ocp.constraints.lbu, ocp.constraints.ubu = np.arange(3), lbu, ubu
        ocp.constraints.x0 = np.zeros(3)

        # 3. 代价函数 (使用参数 p 传递参考点)
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # 与 forcempc 保持一致：使用 yref/yref_e 逐时刻赋值（线性时变参考）
        nx = 3
        nu = 3
        ny = nx + nu
        # 代价权重（状态 + 控制），按底盘类型拆分
        if self.model_type == 'swerve':
            state_w = np.array([20.0, 20.0, 12.0])
            input_w = np.array([0.35, 12, 0.75])
            terminal_w = np.array([12.0, 12.0, 24.0])
        else:
            state_w = np.array([20.0, 20.0, 10.0])
            input_w = np.array([0.50, 0.5, 0.50])
            terminal_w = np.array([10.0, 10.0, 20.0])
        theta_ref = SX.sym('theta_ref')#type: ignore
        # 计算角度误差的 wrapped 版本
        angle_diff = model.x[2] - theta_ref
        wrapped_angle_diff = atan2(sin(angle_diff), cos(angle_diff))

        ocp.cost.W = np.diag(np.concatenate([state_w, input_w]))
        ocp.cost.W_0 = ocp.cost.W
        ocp.cost.W_e = np.diag(terminal_w)
        # 使用与 forcempc 相同的 cost_y_expr 结构： [x; u]
        pos_err = x[0:2] - p[0:2]
        angle_diff = x[2] - p[2]
        wrapped_angle_diff = atan2(sin(angle_diff), cos(angle_diff))

        # cost_y_expr 定义了 [状态误差; 控制量]
        # 我们的目标是让这些表达式的结果趋于 yref (即 0)
        ocp.model.cost_y_expr = vertcat(pos_err, wrapped_angle_diff, u)
        ocp.model.cost_y_expr_e = vertcat(pos_err, wrapped_angle_diff)

        # 初始化 yref 缓冲，保持和 forcempc 一致的变量名
        self._yref_buffer = np.zeros(ny)
        self._yref_e_buffer = np.zeros(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)
        ocp.parameter_values = np.zeros(3)
        self.solver = AcadosOcpSolver(ocp, json_file=f"{model_type}_ocp.json",verbose=False)
        self.path_planner = SplinePlanner()

    def set_path(self, points: np.ndarray, target_yaw: float, ref_speed: float = 1.0):
        self.path_planner.generate_path(points[:, 0], points[:, 1])
        self.target_yaw, self.ref_speed, self.following = target_yaw, ref_speed, True

    def set_target_point(self, target: np.ndarray):
        self.following, self.target_point = False, target.flatten()

    def update(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten().astype(np.float64)
        # 强校验输入维度
        assert x.size == 3, f"Expected state size 3, got {x.size}"

        # 如果为路径跟踪模式，按时刻填充 yref（线性时变参考）
        if getattr(self, 'following', False):
            s_start = self.path_planner.get_nearest_s(x[0], x[1])
            s_queries = s_start + self.ref_speed * self.dt * np.arange(self.n_horizon + 1)

            # 批量获取参考状态 (N+1, 3)
            ref_states_all = self.path_planner.get_states_batch(s_queries)

            # 构建 yref 数据 (n_horizon, ny)
            ny = 3
            yref_data = np.zeros((self.n_horizon, ny))
            reference_yaw = ref_states_all[-1, 2]
            yref_data[:, 0:2] = ref_states_all[:self.n_horizon, 0:2]
            # 在此实现中，yaw 使用终点 yaw（与 forcempc 保持一致）
            yref_data[:, 2] = reference_yaw

            # 虽然 API 层面通常需要指定 k，但可以利用列表推导式配合 set
            # 这比手动在循环里做切片和赋值快
            [self.solver.set(k, "p", yref_data[k]) for k in range(self.n_horizon)]

            # 终端参考
            self._yref_e_buffer[0:3] = ref_states_all[-1, 0:3]
            self.solver.set(self.n_horizon, "p", self._yref_e_buffer)
        else:
            tp = getattr(self, 'target_point', x)
            # 构建单点 yref 并复制到所有时刻
            ny = 3
            yref_single = np.zeros(ny)
            yref_single[0:3] = tp.flatten()
            for k in range(self.n_horizon):
                self.solver.set(k, "p", yref_single)
            self._yref_e_buffer[0:3] = tp.flatten()
            self.solver.set(self.n_horizon, "p", self._yref_e_buffer)

        # 设置当前状态约束（第 0 时刻）
        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)

        status = self.solver.solve()
        if status != 0:
            print(f"Warning: acados solver returned status {status}")

        u = self.solver.get(0, "u")
        # 如果是 swerve，转为车体系 [vx, vy, vw] 以适配后文输出
        return np.array([u[0]*cos(u[1]), u[0]*sin(u[1]), u[2]]) if self.model_type == 'swerve' else u
    