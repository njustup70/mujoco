import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, List

class SplinePlanner:
    def __init__(self):
        self.x_path = np.array([])
        self.y_path = np.array([])
        self.yaw_path = np.array([])
        self.t_samples = np.array([])

    def generate_path(
        self, 
        x_pts, 
        y_pts, 
        start_yaw: Optional[float] = None, 
        end_yaw: Optional[float] = None, 
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成三次样条平滑路径
        :param x_pts: 控制点 X 列表
        :param y_pts: 控制点 Y 列表
        :param start_yaw: 起点航向角 (弧度), None 则自动计算
        :param end_yaw: 终点航向角 (弧度), None 则自动计算
        :param num_samples: 采样点数
        """
        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        
        # 1. 计算参数 t (基于累计弦长)
        ds = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
        t_pts = np.concatenate(([0], np.cumsum(ds)))
        
        # 2. 确定边界条件 (bc_type)
        # 如果不指定 yaw，使用 'not-a-knot' (默认)；如果指定，使用一阶导数约束
        bc_x = 'not-a-knot'
        bc_y = 'not-a-knot'
        
        # 这里的 scale 决定了“出弯/入弯”的力度，通常设为 1.0 或相邻两点间距
        v_scale = ds[0] if len(ds) > 0 else 1.0 

        if start_yaw is not None and end_yaw is not None:
            v_start = (v_scale * np.cos(start_yaw), v_scale * np.sin(start_yaw))
            v_end = (v_scale * np.cos(end_yaw), v_scale * np.sin(end_yaw))
            bc_x = ((1, v_start[0]), (1, v_end[0]))
            bc_y = ((1, v_start[1]), (1, v_end[1]))
        elif start_yaw is not None:
            v_start = (v_scale * np.cos(start_yaw), v_scale * np.sin(start_yaw))
            bc_x = ((1, v_start[0]), (2, 0.0)) # 起点一阶导，终点二阶导为0
            bc_y = ((1, v_start[1]), (2, 0.0))
        
        # 3. 拟合样条
        cs_x = CubicSpline(t_pts, x_pts, bc_type=bc_x) # type: ignore
        cs_y = CubicSpline(t_pts, y_pts, bc_type=bc_y) # type: ignore

        # 4. 插值采样
        self.t_samples = np.linspace(0, t_pts[-1], num_samples)
        self.x_path = cs_x(self.t_samples)
        self.y_path = cs_y(self.t_samples)
        
        # 5. 计算 Yaw 角
        dx = cs_x(self.t_samples, 1)        
        dy = cs_y(self.t_samples, 1)
        self.yaw_path = np.arctan2(dy, dx)
        
        return self.x_path, self.y_path, self.yaw_path

    def plot(self):
        """简单的可视化函数"""
        if len(self.x_path) == 0:
            print("No path to plot.")
            return
        
        plt.figure(figsize=(8, 8))
        plt.plot(self.x_path, self.y_path, 'b-', label="Spline Path")
        plt.quiver(self.x_path[::10], self.y_path[::10], 
                   np.cos(self.yaw_path[::10]), np.sin(self.yaw_path[::10]), 
                   color='green', scale=20, width=0.005)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

# --- 调用示例 ---
if __name__ == "__main__":
    planner = SplinePlanner()

    # 情况 1: 指定起终点 Yaw (强制掉头)
    print("Generating path with yaw constraints...")
    x1, y1, yaw1 = planner.generate_path(
        x_pts=[0, 2, 4], 
        y_pts=[0, 1, 0], 
        start_yaw=0.0,           # 水平向右出发
        end_yaw=np.deg2rad(180)  # 水平向左结束
    )
    planner.plot()

    # 情况 2: 不指定 Yaw (自由平滑插值)
    print("Generating free path...")
    x2, y2, yaw2 = planner.generate_path(
        x_pts=[0, 1, 3, 5], 
        y_pts=[0, 2, 1, 3]
    )
    planner.plot()