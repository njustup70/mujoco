#!/usr/bin/env python3
"""MPC 控制节点：订阅 odom(父类)/path/state/save，发布 cmd_vel(父类)。"""
import json
import sys
from pathlib import Path

import numpy as np
import rclpy
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from mpc_pkg.control_node import MPCControlNode
from experiment.recorder import Recorder

SAMPLE_PERIOD = 0.1


class ControllerNode(MPCControlNode):
    def __init__(self):
        super().__init__()
        self.rec = Recorder()
        self.active = False
        self.ref = None
        self.create_subscription(String, '/exp/path', self._on_path, 10)
        self.create_subscription(String, '/sim/state', self._on_state, 10)
        self.create_subscription(String, '/exp/save', self._on_save, 10)
        self.create_timer(SAMPLE_PERIOD, self._sample)

    def _on_path(self, msg):
        d = json.loads(msg.data)
        wps = np.asarray(d['waypoints'], dtype=float)
        self.path_follwer.set_path(wps, target_yaw=float(d.get('target_yaw', 0.0)),
                                   ref_speed=float(d.get('ref_speed', 1.0)))
        self._publish_reference_path_once()
        # 复位控制器内部状态
        self.state_observer.reset()
        self.path_follwer.last_u = np.zeros(3)
        self.path_visual.path_cache.pop(self.tracked_path_topic, None)
        # 参考路径交给记录器
        pl = self.path_follwer.path_planner
        self.ref = {'x': np.asarray(pl.x_path), 'y': np.asarray(pl.y_path),
                    'yaw': np.asarray(pl.yaw_path), 's': np.asarray(pl.s_samples)}
        self.rec.reset()
        self.active = True
        print(f'[controller] set path {len(wps)} waypoints')

    def _on_state(self, msg):
        self.rec.on_state(msg.data)

    def _on_save(self, msg):
        d = json.loads(msg.data)
        ok = self.rec.save(d['path'], self.ref, d['end_reason'])
        print(f'[controller] save={ok} {d["path"]}')
        self.active = False

    def _sample(self):
        if self.active:
            self.rec.sample(self.ref)


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
