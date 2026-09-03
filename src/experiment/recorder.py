#!/usr/bin/env python3
"""记录器：纯 Python，无 ROS，由 controller 调用。"""
import json
import math
import time
from pathlib import Path

import numpy as np


def lateral_error(ref, x, y):
    """符号化横向误差（+ 为路径左侧），返回 (e_lat, s_ref)。"""
    rx, ry, ryaw, rs = ref['x'], ref['y'], ref['yaw'], ref['s']
    d2 = (rx - x) ** 2 + (ry - y) ** 2
    i = int(np.argmin(d2))
    tx, ty = math.cos(ryaw[i]), math.sin(ryaw[i])
    ex, ey = x - rx[i], y - ry[i]
    cross = tx * ey - ty * ex
    sign = 1.0 if cross >= 0 else -1.0
    return sign * math.hypot(ex, ey), float(rs[i])


class Recorder:
    def __init__(self):
        self.samples = []
        self.state = None
        self.t0 = None

    def reset(self):
        self.samples = []
        self.t0 = time.monotonic()

    def on_state(self, data_str):
        self.state = json.loads(data_str)

    def sample(self, ref):
        if self.state is None or self.t0 is None:
            return
        st = self.state
        x = float(st.get('x', float('nan')))
        y = float(st.get('y', float('nan')))
        e_lat, s_ref = lateral_error(ref, x, y)
        vx = float(st.get('vx', 0.0))
        vy = float(st.get('vy', 0.0))
        row = {
            't': time.monotonic() - self.t0,
            'x': x, 'y': y, 'yaw': float(st.get('yaw', float('nan'))),
            'vx': vx, 'vy': vy, 'omega': float(st.get('omega', float('nan'))),
            'v': math.hypot(vx, vy),
            'roll': float(st.get('roll', float('nan'))),
            'pitch': float(st.get('pitch', float('nan'))),
            'e_lat': e_lat, 's_ref': s_ref,
        }
        for i, a in enumerate(st.get('steer', [float('nan')] * 4)):
            row[f'steer{i}'] = float(a)
        for i, w in enumerate(st.get('wheel_w', [float('nan')] * 4)):
            row[f'wheel_w{i}'] = float(w)
        self.samples.append(row)

    def save(self, path, ref, end_reason):
        if not self.samples or ref is None:
            return False
        data = {k: np.asarray([r[k] for r in self.samples], dtype=float)
                for k in self.samples[0]}
        data['ref_x'] = ref['x']
        data['ref_y'] = ref['y']
        data['ref_yaw'] = ref['yaw']
        data['ref_s'] = ref['s']
        data['end_reason'] = np.asarray([end_reason])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
        return True

