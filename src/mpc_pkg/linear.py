import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def get_spline_path_with_yaw(x_points, y_points, start_yaw, end_yaw, num_samples=100):
    # 1. 计算参数 t (累计弧长)
    dx = np.diff(x_points)
    dy = np.diff(y_points)
    ds = np.sqrt(dx**2 + dy**2)
    t = np.concatenate(([0], np.cumsum(ds)))
    
    # 2. 定义起始和结束的一阶导数 (速度矢量)
    # 我们假设单位速度，或者根据相邻点距离调整以获得更自然的过渡
    v_start:tuple[float, float] = (np.cos(start_yaw), np.sin(start_yaw))
    v_end:tuple[float, float] = (np.cos(end_yaw), np.sin(end_yaw))

    # 3. 建立三次样条插值函数
    # bc_type=((1, val_start), (1, val_end)) 表示指定一阶导数
    from typing import Any, cast
    bc_type=cast(Any, ((1, v_start[0]), (1, v_end[0])))
    cs_x = CubicSpline(t, x_points, bc_type=bc_type)
    cs_y = CubicSpline(t, y_points, bc_type=bc_type)

    # 4. 生成采样
    t_fine = np.linspace(0, t[-1], num_samples)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    
    # 5. 计算 Yaw
    dx_dt = cs_x(t_fine, 1)
    dy_dt = cs_y(t_fine, 1)
    yaw = np.arctan2(dy_dt, dx_dt)
    
    return x_fine, y_fine, yaw
def test():

    # --- 测试 ---
    # x_pts = [0, 2, 4, 3, 6]
    # y_pts = [0, 1, 5, 8, 9]
    x_pts=[0,0.5]
    y_pts=[0,0.5]
    # 设定起始 Yaw 为 0° (向右), 终点 Yaw 为 180° (向左)
    s_yaw = 0.0
    e_yaw = np.pi 

    x_new, y_new, yaw_new = get_spline_path_with_yaw(x_pts, y_pts, s_yaw, e_yaw, num_samples=200)

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(x_pts, y_pts, 'ro', label='Control Points')
    plt.plot(x_new, y_new, 'b-', label='Clamped Spline')
    # 绘制方向箭头
    plt.quiver(x_new[::10], y_new[::10], np.cos(yaw_new[::10]), np.sin(yaw_new[::10]), 
            scale=15, color='green', alpha=0.6)
    # 突出显示起终点方向
    plt.arrow(x_new[0], y_new[0], np.cos(s_yaw), np.sin(s_yaw), color='orange', width=0.1, label='Start Goal Yaw')
    plt.arrow(x_new[-1], y_new[-1], np.cos(e_yaw), np.sin(e_yaw), color='orange', width=0.1)

    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    test()