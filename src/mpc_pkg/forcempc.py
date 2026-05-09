import os
import numpy as np
from typing import Optional, Union, Tuple
from casadi import SX, vertcat, sin, cos,horzcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from linear import SplinePlanner

class AccMPCPathFollower:
    def __init__(
        self,
        dt: float,
        h_cg: float = 0.15,
        wheel_base: float = 0.4,
        wheel_width: float = 0.4,
    ):
        # --- 参数校验与赋值 ---
        self.dt = float(dt)
        self.n_horizon: int = 100
        self.ref_speed: float = 1.5
        self.is_following_path: bool = False
        self.target_yaw: float = 0.0
        
        # 物理限制
        g = 9.81
        self.limit_ax = (g * float(wheel_base) / 2.0) / float(h_cg)
        self.limit_ay = (g * float(wheel_width) / 2.0) / float(h_cg)
        print(f"Acceleration limits set to ax: ±{self.limit_ax:.2f} m/s², ay: ±{self.limit_ay:.2f} m/s² based on h_cg={h_cg}, wheel_base={wheel_base}, wheel_width={wheel_width}")
        # 组件初始化
        self.path_planner = SplinePlanner()
        self.model: AcadosModel = self.setup_acados_model()
        self.solver: AcadosOcpSolver = self.setup_solver()
        self._yref_buffer = np.zeros(9)
        self._yref_e_buffer = np.zeros(6)
    def setup_acados_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = "acc_mpc_model"

        # 状态量: [x, y, theta, vx, vy, omega]
        x = SX.sym('x', 6)
        # 控制量: [ax, ay, alpha]
        u = SX.sym('u', 3)
        model.x = x
        model.u = u
        theta=x[2]
        model.f_expl_expr = vertcat(
            # --- 1. 位置导数 (Global Frame) ---
            cos(theta) * x[3] - sin(theta) * x[4],  # dx = cos(theta)*vx - sin(theta)*vy
            sin(theta) * x[3] + cos(theta) * x[4],  # dy = sin(theta)*vx + cos(theta)*vy
            # --- 2. 角度导数 ---
            x[5],                                 # dtheta = omega
            # --- 3. 车体坐标系速度导数 (加速度 + 科氏力) ---
            u[0] + x[5] * x[4],                   # dvx = ax + omega * vy
            u[1] - x[5] * x[3],                   # dvy = ay - omega * vx
            # --- 4. 角速度导数 ---
            u[2]                                  # domega = alpha
        )
        return model

    def setup_solver(self) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        ocp.model = self.model
        assert isinstance(self.model.x, SX) and isinstance(self.model.u, SX), "Model states and controls must be CasADi SX symbols"
        nx = self.model.x.size1()
        nu = self.model.u.size1()
        ny = nx + nu
        self.nx,self.nu,self.ny=nx,nu,ny
        # 求解器配置
        ocp.solver_options.N_horizon = self.n_horizon
        ocp.solver_options.tf = self.n_horizon * self.dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'

        # 控制约束
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.lbu = np.array([-self.limit_ax, -self.limit_ay, -5.0])
        ocp.constraints.ubu = np.array([self.limit_ax, self.limit_ay, 5.0])
        ocp.constraints.x0 = np.zeros(nx)

        # 代价函数权重
        Q_diag = np.array([20.0, 20.0, 10.0, 0.1, 0.1, 0.1])
        R_diag = np.array([0.01, 0.01, 0.01])
        ocp.cost.W = np.diag(np.concatenate([Q_diag, R_diag]))
        ocp.cost.W_0 = ocp.cost.W
        ocp.cost.W_e = np.diag(Q_diag)

        ocp.model.cost_y_expr = vertcat(self.model.x, self.model.u)
        ocp.model.cost_y_expr_e = self.model.x
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        json_file = os.path.join(os.getcwd(), 'acados_ocp.json')
        return AcadosOcpSolver(ocp, json_file=json_file,verbose=False)

    def update(self, x_current: np.ndarray) -> np.ndarray:
        """
        :param x_current: 形状为 (6,) 的当前状态
        :return: 形状为 (3,) 的速度指令 [vx, vy, omega]
        """
        # 强校验输入维度
        assert x_current.size == 6, f"Expected state size 6, got {x_current.size}"
        x_flat = x_current.flatten().astype(np.float64)
        
        if self.is_following_path:
            # 路径跟踪模式下的 yref 更新
            s_start = self.path_planner.get_nearest_s(x_flat[0], x_flat[1])
            
            s_queries = s_start + self.ref_speed * self.dt * np.arange(self.n_horizon + 1)
            
            # 2. 一次性拿到所有参考点 (N+1, 3)
            ref_states_all = self.path_planner.get_states_batch(s_queries)
            #全向移动,过程中保持目标航向
            yref_data = np.zeros((self.n_horizon, self.ny)) # ny 是控制周期内的参考输出维度
            refrence_yaw=ref_states_all[-1, 2]
            # 批量填充你的 buffer 数据
            yref_data[:, 0:2] = ref_states_all[:self.n_horizon, 0:2]
            yref_data[:, 2] = refrence_yaw # 这里假设所有时刻的 yaw 都是目标点的 yaw，你也可以根据需要设置不同的 yaw

            # 虽然 API 层面通常需要指定 k，但可以利用列表推导式配合 set
            # 这比手动在循环里做切片和赋值快
            [self.solver.set(k, "yref", yref_data[k]) for k in range(self.n_horizon)]
            # 4. 终端参考点
            self._yref_e_buffer[0:3] = ref_states_all[-1, 0:3]
            self.solver.set(self.n_horizon, "yref", self._yref_e_buffer)

        # 设置当前状态约束
        self.solver.set(0, "lbx", x_flat)
        self.solver.set(0, "ubx", x_flat)
        
        status = self.solver.solve()
        if status != 0:
            print(f"Warning: acados solver returned status {status}")

        # 获取当前最优控制输入 [ax, ay, alpha]
        u_res = self.solver.get(0, "u")
        assert u_res.shape == (3,), "Solver output 'u' shape mismatch"

        # 计算速度指令: v_next = v_curr + a * dt
        # x_flat[3:6] 是 [vx, vy, omega]
        v_cmd: np.ndarray = x_flat[3:6] + u_res * self.dt
        return v_cmd

    def set_path(self, target_points: np.ndarray, target_yaw: float, ref_speed: Optional[float] = None) -> None:
        """
        :param target_points: (N, 2) 的 NumPy 数组
        """
        assert target_points.ndim == 2 and target_points.shape[1] == 2, "Points must be (N, 2)"
        
        self.is_following_path = True
        self.target_yaw = float(target_yaw)
        if ref_speed is not None:
            self.ref_speed = float(ref_speed)
            
        self.path_planner.generate_path(target_points[:, 0], target_points[:, 1])

    def set_state_init(self, x0: np.ndarray) -> None:
        """初始化求解器内部状态轨迹"""
        assert x0.size == 6
        x0_flat = x0.flatten().astype(np.float64)
        for k in range(self.n_horizon + 1):
            self.solver.set(k, "x", x0_flat)