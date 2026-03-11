import do_mpc
import casadi
from casadi import vertcat, cos, sin
import numpy as np


class MPCModel:
    def __init__(self, dt):
        model_type = 'discrete'
        self.model = do_mpc.model.Model(model_type)

        # State: [x, y, theta]
        self.state = self.model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        # Input (body frame): [vx, vy, vw]
        self.u_vec = self.model.set_variable(var_type='_u', var_name='u_input', shape=(3, 1))
        # Reference: [x_ref, y_ref, theta_ref]
        self.ref = self.model.set_variable(var_type='_p', var_name='ref', shape=(3, 1))

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
        self.model.set_rhs('state', next_state)
        self.model.setup()

        pos_err = casadi.sumsqr(self.state[0:2] - self.ref[0:2])
        # Use cosine yaw cost to avoid angle wrap jump at +-pi.
        yaw_err = 1.0 - casadi.cos(self.state[2] - self.ref[2])
        mterm = 8.0 * pos_err + 2.0 * yaw_err
        lterm = 8.0 * pos_err + 2.0 * yaw_err

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_param(n_horizon=25, t_step=dt)
        # self.mpc.set_rterm(u_input=casadi.diag(vertcat(0.2, 0.2, 0.1)))

        # Velocity bounds in body frame.
        self.mpc.bounds['lower', '_u', 'u_input'] = np.array([[-1.2], [-1.2], [-2.0]])
        self.mpc.bounds['upper', '_u', 'u_input'] = np.array([[1.2], [1.2], [2.0]])

        self.p_template = self.mpc.get_p_template(1)

        def p_fun(_t_now):
            return self.p_template

        self.mpc.set_p_fun(p_fun)
        self.mpc.setup()

