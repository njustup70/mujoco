#!/usr/bin/env python3
"""实验驱动：读配置 -> 唤醒 sim+controller -> 逐条路径下发/判断/保存。只用话题。"""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path as PathMsg
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'src' / 'experiment'


class Driver(Node):
    def __init__(self, cfg):
        super().__init__('experiment')
        self.cfg = cfg
        self.out = ROOT / 'data' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out.mkdir(parents=True, exist_ok=True)
        print('output dir:', self.out)

        self.path_pub = self.create_publisher(String, '/exp/path', 10)
        self.reset_pub = self.create_publisher(String, '/exp/reset', 10)
        self.save_pub = self.create_publisher(String, '/exp/save', 10)

        self.state = None
        self.ref_ready = False
        self.create_subscription(String, '/sim/state', self._on_state, 10)
        self.create_subscription(PathMsg, '/mpc/reference_path', self._on_ref, 10)

    def _on_state(self, msg):
        self.state = json.loads(msg.data)

    def _on_ref(self, msg):
        self.ref_ready = True

    def spin(self, t=0.01):
        rclpy.spin_once(self, timeout_sec=t)

    def sleep(self, s):
        end = time.monotonic() + s
        while time.monotonic() < end:
            self.spin(0.05)

    def judge(self, p):
        if self.state is None:
            return None
        tol = float(self.cfg.get('reach_tol', 0.2))
        fx, fy = p['waypoints'][-1]
        if math.hypot(self.state['x'] - fx, self.state['y'] - fy) < tol:
            return 'reached_end'
        if max(abs(math.degrees(self.state['roll'])), abs(math.degrees(self.state['pitch']))) \
                > float(self.cfg.get('flip_angle_deg', 80.0)):
            return 'flipped'
        return None

    def run(self):
        # 等 sim 状态 + controller 参考路径都出现（controller 的 acados 初始化约 2s）
        end = time.monotonic() + 15.0
        while time.monotonic() < end and not (self.state is not None and self.ref_ready):
            self.spin(0.05)
        self.sleep(1.0)  # 等 DDS 发现完成

        for i, p in enumerate(self.cfg['paths']):
            name = p.get('name', f'path_{i}')
            print(f'==== path {i} [{name}] ====')
            self.reset_pub.publish(String(data='{}'))
            self.path_pub.publish(String(data=json.dumps({
                'waypoints': p['waypoints'],
                'target_yaw': float(p.get('target_yaw', 0.0)),
                'ref_speed': float(p.get('ref_speed', 1.0)),
            })))
            self.sleep(float(self.cfg.get('settle_delay', 1.0)))

            reason = None
            while reason is None:
                self.spin(0.01)
                reason = self.judge(p)

            npz = self.out / f'path_{i:02d}_{name}.npz'
            self.save_pub.publish(String(data=json.dumps(
                {'path': str(npz), 'end_reason': reason})))
            self.sleep(0.3)
            print(f'path {i} [{name}] end: {reason}')
        print('experiment finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(EXP / 'config' / 'paths.json'))
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding='utf-8'))

    procs = [
        subprocess.Popen([sys.executable, str(EXP / 'sim.py')], cwd=str(ROOT)),
        subprocess.Popen([sys.executable, str(EXP / 'controller.py')], cwd=str(ROOT)),
    ]

    rclpy.init()
    driver = Driver(cfg)
    try:
        driver.run()
    finally:
        driver.destroy_node()
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=5.0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
