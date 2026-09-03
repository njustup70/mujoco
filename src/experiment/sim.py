#!/usr/bin/env python3
"""仿真节点：订阅 cmd_vel(父类)/reset(JSON)，发布 odom(父类)/state(JSON)。"""
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import rclpy
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'mujoco_ros2_bridge' / 'scripts'))

from mujoco_node import MujocoSimNode
from swerve_solver import SwerveSolver


def _euler(w, x, y, z):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class SimNode(MujocoSimNode):
    def __init__(self):
        self._lock = threading.Lock()
        self._reset_pending = False
        super().__init__()  # 父类线程会调用本类重写的 simulation_loop

        self.drive_dofadr = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}_drive')
            ] for i in range(4)
        ]
        self.state_pub = self.create_publisher(String, '/sim/state', 10)
        self.create_subscription(String, '/exp/reset', self._on_reset, 10)
        self.create_timer(0.01, self._publish_state)

    def simulation_loop(self):
        """无头仿真循环；复位在本线程内执行，避免与 mj_step 并发。"""
        while not self.stop_event.is_set():
            t0 = time.time()
            with self._lock:
                if self._reset_pending:
                    mujoco.mj_resetData(self.model, self.data)
                    self._reset_pending = False
                steer = [self.data.qpos[self.steer_qposadr[i]] for i in range(4)]
                cmds = self.swerve_solver.get_actuator_commands(
                    self.target_v_x, self.target_v_y, self.target_v_yaw, steer)
                for i, (s, d) in enumerate(cmds):
                    self.data.actuator(f'steer{i}').ctrl[0] = s
                    self.data.actuator(f'drive{i}').ctrl[0] = d
                mujoco.mj_step(self.model, self.data)
            rest = self.model.opt.timestep - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)

    def _publish_state(self):
        with self._lock:
            pos = self.data.body('chassis').xpos.copy()
            quat = self.data.body('chassis').xquat.copy()  # w,x,y,z
            steer = [float(self.data.qpos[a]) for a in self.steer_qposadr]
            wheel = [float(self.data.qvel[a]) for a in self.drive_dofadr]
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     self.chassis_body_id, vel, 1)
        roll, pitch, yaw = _euler(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        msg = String()
        msg.data = json.dumps({
            'x': float(pos[0]), 'y': float(pos[1]), 'yaw': yaw,
            'roll': roll, 'pitch': pitch,
            'vx': float(vel[3]), 'vy': float(vel[4]), 'omega': float(vel[2]),
            'steer': steer, 'wheel_w': wheel,
        })
        self.state_pub.publish(msg)

    def _on_reset(self, msg):
        self.target_v_x = self.target_v_y = self.target_v_yaw = 0.0
        self.swerve_solver = SwerveSolver(
            wheels_pos=self.wheels_pos, wheel_radius=self.wheel_radius,
            steer_lag_alpha=self.wheel_steer_lag_alpha,
            drive_lag_alpha=self.wheel_drive_lag_alpha,
            steer_noise_std=self.wheel_steer_noise_std,
            drive_noise_std=self.wheel_drive_noise_std)
        self._reset_pending = True


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
