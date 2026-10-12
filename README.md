# MuJoCo ROS 2 bridge · RC 2027 场地

本分支的 ROS 2 仿真默认加载 [`src/mujoco_ros2_bridge/model/rc2027/scene.xml`](src/mujoco_ros2_bridge/model/rc2027/scene.xml)：RC 2027 场地、53 个可动比赛用品和`main` 原有的四轮舵驱底盘。场地原始静态版 `field.xml`、可动版 `field_with_objects.xml` 及 OBJ 网格保存在同一目录，可单独加载。原始模型的说明、预览和许可也保存在 [`rc2027`](src/mujoco_ros2_bridge/model/rc2027/README.md)。

## 坐标与初始位置

- 世界坐标和 ROS `odom` 使用同一基准：单位米，场地中心为 `(0, 0, 0)`，`+x` 向东，`+y` 向北，`+z` 向上。红方在西侧，蓝方在东侧。原模型的场地及比赛用品坐标没有平移或旋转。
- 若外部坐标以西南角为原点：`x_world = x_corner - 5.5`，`y_world = y_corner - 5.5`。场地地面顶面为 `z=0.025`。
- `main` 分支的四轮舵驱底盘保持原有结构与轮组控制。其初始位姿为 `(4.75, -5.10, 0.235)`，偏航角 `+90°`，中心位于蓝方启动区，车头朝北、朝场心。底盘宽度超过启动区的南北宽度，前轮略伸出启动区，后轮与南侧围栏留有间隙。`cmd_vel.linear.x` 为车头方向，`linear.y` 为车体左侧方向。
- 12 个天空块位于南侧公共区，五色石位于 `(0, 4.25)` 基座，红/蓝各 20 个地块位于北侧围栏外。全部 53 个可动件在 `scene.xml` 中保留源模型的 `freejoint` 名称及初始位姿，未额外套用坐标变换。

## 加载与验证

```bash
python3 -m unittest discover -s tests -p 'test_rc2027_scene.py'
python3 src/mujoco_ros2_bridge/model/rc2027/example_load.py --steps 500
```

在具备 ROS 2 与 MuJoCo 的工作区中，构建后运行 `ros2 launch mujoco_ros2_bridge bridge.launch.py`。`CMakeLists.txt` 会把 `model/` 及所有网格一起安装到包的 share 目录。直接加载合并场景也可以：

```python
import mujoco
model = mujoco.MjModel.from_xml_path(
    "src/mujoco_ros2_bridge/model/rc2027/scene.xml"
)
```

场地模型来自同级目录保存的 `rc2027_field` 资源，许可及非官方模型声明见 [`LICENSE`](src/mujoco_ros2_bridge/model/rc2027/LICENSE)。`robot.xml` 保留 `main` 原有的单底盘场景供对照。
