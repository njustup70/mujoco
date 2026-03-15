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
        # 真实弧长数组，对应每个采样点沿曲线的累计弧长
        self.s_samples = np.array([])

    def find_nearest_point(self, x: float, y: float) -> Tuple[int, float, float, float, float]:
        """Find the nearest sampled path point to the query position.

        Returns:
            (index, nearest_x, nearest_y, nearest_yaw, distance)
        """
        if len(self.x_path) == 0:
            raise ValueError("Path is empty. Call generate_path() first.")

        dx = self.x_path - x
        dy = self.y_path - y
        dist_sq = dx * dx + dy * dy

        idx = int(np.argmin(dist_sq))
        distance = float(np.sqrt(dist_sq[idx]))

        return (
            idx,
            float(self.x_path[idx]),
            float(self.y_path[idx]),
            float(self.yaw_path[idx]),
            distance,
        )

    def generate_path(
        self, 
        x_pts, 
        y_pts, 
        start_yaw: Optional[float] = None, 
        end_yaw: Optional[float] = None, 
        step_cm: float = 5.0  # 新增参数：固定间距（单位：厘米）
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        
        # 1. 计算参数 t (累计弦长)
        ds = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
        t_pts = np.concatenate(([0], np.cumsum(ds)))
        
        # 2. 确定边界条件 (保持原有逻辑)
        bc_x, bc_y = 'not-a-knot', 'not-a-knot'
        v_scale = ds[0] if len(ds) > 0 else 1.0 

        if start_yaw is not None and end_yaw is not None:
            v_start = (v_scale * np.cos(start_yaw), v_scale * np.sin(start_yaw))
            v_end = (v_scale * np.cos(end_yaw), v_scale * np.sin(end_yaw))
            bc_x = ((1, v_start[0]), (1, v_end[0]))
            bc_y = ((1, v_start[1]), (1, v_end[1]))
        elif start_yaw is not None:
            v_start = (v_scale * np.cos(start_yaw), v_scale * np.sin(start_yaw))
            bc_x = ((1, v_start[0]), (2, 0.0))
            bc_y = ((1, v_start[1]), (2, 0.0))
        
        # 3. 拟合样条
        cs_x = CubicSpline(t_pts, x_pts, bc_type=bc_x) # type: ignore
        cs_y = CubicSpline(t_pts, y_pts, bc_type=bc_y) # type: ignore

        # --- 核心修改部分：固定间距采样 ---
        total_length = t_pts[-1]           # 路径总长度（米）
        step_m = step_cm / 100.0           # 将厘米转换为米
        
        # 计算采样点数，确保覆盖终点
        num_samples = int(np.floor(total_length / step_m)) + 1
        # 使用 linspace 保证从 0 正好到终点，且间距几乎恒定为 step_m
        self.t_samples = np.linspace(0, total_length, num_samples)
        
        # 4. 插值采样
        self.x_path = cs_x(self.t_samples)
        self.y_path = cs_y(self.t_samples)
        
        # 5. 计算 Yaw 角
        dx = cs_x(self.t_samples, 1)        
        dy = cs_y(self.t_samples, 1)
        self.yaw_path = np.arctan2(dy, dx)

        # 6. 计算各采样点的真实弧长（相邻采样点间欧氏距离累积）
        ds_path = np.sqrt(np.diff(self.x_path)**2 + np.diff(self.y_path)**2)
        self.s_samples = np.concatenate(([0.0], np.cumsum(ds_path)))

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
        x_pts=[0, 1, 1, 8], 
        y_pts=[0, 2, 2.2, 5]
    )
    planner.plot()