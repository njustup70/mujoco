# experiment — 路径跟踪实验框架

一键同时唤醒 MPC 控制器与 MuJoCo 仿真，依次跟踪多条路径，数据存到 `data/<时间戳>/`。
不修改 `mpc_pkg` 与 `mujoco_ros2_bridge` 源码，均以子类方式扩展。

## 目录结构

```
src/experiment/
├── config/paths.json    # 路径与实验参数（JSON）
├── sim.py               # 仿真节点：发布 odom + state(JSON)，订阅 cmd_vel + reset
├── controller.py        # MPC 控制节点：订阅 odom/path/state/save，发布 cmd_vel
├── recorder.py          # 纯模块，无 ROS，由 controller 调用
├── experiment.py        # 驱动：读配置 -> 发路径 -> 判结束 -> 下一条
└── README.md
```

## 运行

```bash
cd /home/Elaina/ros2_ws
source install/setup.bash
python3 src/experiment/experiment.py
```

`experiment.py` 会自动唤醒 `sim.py` 与 `controller.py` 两个子进程。

## 配置（`config/paths.json`）

```json
{
  "reach_tol": 0.2,
  "flip_angle_deg": 80.0,
  "settle_delay": 1.0,
  "paths": [
    {"name": "straight", "waypoints": [[0, 0], [8, 8]], "target_yaw": 0.785, "ref_speed": 3.0}
  ]
}
```

- `reach_tol`：到末航点距离小于该值判定为到达终点（米）。
- `flip_angle_deg`：|roll| 或 |pitch| 超过该角度判定为翻车。
- `settle_delay`：复位+下发路径后的稳定等待（秒）。
- `paths`：多条路径，每条含 `name`、`waypoints`（航点 `[[x,y],...]`）、`target_yaw`（弧度）、`ref_speed`（m/s）。

## 一轮结束条件（在 experiment.py 的 judge 里，不用时间）

1. **到达终点**：真值位置到末航点距离 < `reach_tol`；
2. **翻车**：真值 roll/pitch 超过 `flip_angle_deg`。

## 话题（只用话题，复杂数据用 JSON 字符串）

| 名称 | 类型 | 说明 |
|---|---|---|
| `odom` | `nav_msgs/Odometry` | 仿真 → 控制器（带噪声里程计） |
| `cmd_vel` | `geometry_msgs/Twist` | 控制器 → 仿真 |
| `/sim/state` | `std_msgs/String`(JSON) | 仿真 → 控制器/experiment：x,y,yaw,roll,pitch,vx,vy,omega,steer[4],wheel_w[4] |
| `/exp/path` | `std_msgs/String`(JSON) | experiment → 控制器：waypoints,target_yaw,ref_speed |
| `/exp/reset` | `std_msgs/String`(JSON) | experiment → 仿真：复位 |
| `/exp/save` | `std_msgs/String`(JSON) | experiment → 控制器：path,end_reason |

## 输出（`data/<时间戳>/path_XX_<name>.npz`）

每条路径一个 npz，字段：

- `t`、`x`、`y`、`yaw`、`vx`、`vy`、`omega`、`v`（合速度 = sqrt(vx²+vy²)）
- `roll`、`pitch`
- `e_lat`（符号化横向误差，+ 为路径左侧）、`s_ref`（最近点弧长进度）
- `steer0..steer3`（4 个舵角）、`wheel_w0..wheel_w3`（4 个驱动轮角速度）
- `ref_x`、`ref_y`、`ref_yaw`、`ref_s`（参考路径）、`end_reason`
