#!/usr/bin/env python3
import threading
import numpy as np
import mujoco


class VirtualForceBodyController:
    """Receive local-frame virtual wrench command [Fx_local, Fy_local, Mz_local]."""

    def __init__(self, timeout_sec=0.2):
        self.timeout_sec = float(timeout_sec)
        self._cmd = np.zeros(3, dtype=float)
        self._has_cmd = False
        self._last_update = -1.0
        self._lock = threading.Lock()

    def update(self, msg_data, now_sec):
        if len(msg_data) < 3:
            return False
        with self._lock:
            self._cmd[0] = float(msg_data[0])
            self._cmd[1] = float(msg_data[1])
            self._cmd[2] = float(msg_data[2])
            self._has_cmd = True
            self._last_update = float(now_sec)
        return True

    def get_if_fresh(self, now_sec):
        with self._lock:
            if not self._has_cmd:
                return None
            if float(now_sec) - self._last_update > self.timeout_sec:
                return None
            return self._cmd.copy()


class BlockForceApplier:
    """Apply local virtual force/torque to chassis using MuJoCo generalized force."""

    def __init__(self, model, data, chassis_body_id):
        self.model = model
        self.data = data
        self.chassis_body_id = int(chassis_body_id)

    def apply_local_wrench(self, fx_local, fy_local, mz_local):
        # Local frame -> world frame.
        R_bw = self.data.xmat[self.chassis_body_id].reshape(3, 3)
        f_world = R_bw @ np.array([float(fx_local), float(fy_local), 0.0], dtype=float)
        t_world = R_bw @ np.array([0.0, 0.0, float(mz_local)], dtype=float)

        # Geometric center (body origin) and its ground projection.
        center_world = self.data.xpos[self.chassis_body_id].copy()
        contact_world = center_world.copy()
        contact_world[2] = 0.0

        # Force at contact projection, moment at geometric center.
        mujoco.mj_applyFT(
            self.model,
            self.data,
            f_world,
            np.zeros(3, dtype=float),
            contact_world,
            self.chassis_body_id,
            self.data.qfrc_applied,
        )
        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.zeros(3, dtype=float),
            t_world,
            center_world,
            self.chassis_body_id,
            self.data.qfrc_applied,
        )

        return f_world, t_world
