import mujoco
import mujoco.viewer
import os
import time

def main():
    # 1. 替换为你的模型路径
    # 如果你在 robot.xml 目录下运行，直接写 'robot.xml'
    model_path = 'src/mujoco_ros2_bridge/model/robot.xml' 
    
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        return

    # 2. 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 3. 启动被动查看器
    # launch_passive 不会阻塞主线程，允许你在循环中做其他事
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo Viewer 已启动。")
        print("提示: ")
        print("- 在右侧 'Actuators' 栏拖动滑块控制关节")
        print("- 按住 'Ctrl + 鼠标左键' 可以在画面中拖拽机器人")
        print("- 按 '空格' 暂停/开始仿真")

        # 4. 保持窗口运行
        while viewer.is_running():
            step_start = time.time()

            # 只进行物理步进，没有任何 ctrl 覆盖逻辑
            mujoco.mj_step(model, data)

            # 同步渲染界面
            viewer.sync()

            # 维持仿真频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()