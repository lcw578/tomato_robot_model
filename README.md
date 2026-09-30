# 番茄采摘机器人整机 MuJoCo 模型

AUBO i5 机械臂 + 自研夹爪 + 轨道底盘的整机 MuJoCo 模型，含三台 RealSense 相机传感器帧。
主模型：[`model/tomato_picker.xml`](model/tomato_picker.xml)。

## 快速开始

```bash
# 3D 交互查看器（零位起，右侧滑块可摆臂姿）
python -m mujoco.viewer --mjcf=model/tomato_picker.xml

# 打开即为你设定的默认初始姿态（前伸 + 夹爪行程端点，伺服保持）
python3 -c "import mujoco, mujoco.viewer; m=mujoco.MjModel.from_xml_path('model/tomato_picker.xml'); d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); mujoco.viewer.launch(m,d)"

# 三台相机实时画面 + 3D 查看器（ESC 退出）
python3 live_monitor.py

# 三台相机 RGB + 深度图输出到 observations/
python3 observe_cameras.py
```

依赖：`mujoco>=3.3`、`opencv-python`（观测脚本用）。

## 项目结构

```
robot_assembly/
├── model/
│   ├── tomato_picker.xml      主模型（自包含，相对路径引用网格）
│   └── meshes_baked/          腕部相机/转接板烘焙网格（挂 base_link）
├── meshes/
│   ├── cad/                   整车 CAD 网格（307 零件中的 297 个被引用）
│   ├── gripper/               夹爪基线网格
│   └── aubo/                  AUBO i5 官方连杆网格（碰撞体）
├── dae_split/                 臂外观网格（官方 DAE 按材质拆分, 见 tools/split_dae2.py）
├── tools/                     历史构建管线（按当时绝对路径写死，仅供参考）
├── docs/                      模型技术笔记 + 相机视锥图
├── observe_cameras.py         三相机 RGB+深度观测
└── live_monitor.py            3D 查看器 + 三路相机实时监测（OpenCV）
```

## 模型概要

- **24 body / 19 关节 / 362 geom / 3 相机**，总质量 105.88 kg。
- 底盘（307 个 CAD 零件，静态焊死）+ 10 轨轮（2 主动 velocity 伺服 + 8 从动）+ AUBO i5 六连杆
  （官方运动学：连杆间距 122/121.5/408/376/102.5/94 mm，行程 ±175°）+ 齿条夹爪（equality 耦合）。
- **默认初始姿态**（home 关键帧，含 ctrl）：肩 −1.59 / 大臂 −0.061 / 前臂 1.5 / 腕1 −1.59 / 腕2 −1.65 /
  腕3 3.05 rad，夹爪齿轮 −0.474（行程端点）。Reset 即加载姿态+指令，伺服自动保持。
- **三台 RealSense 相机**（内参见各 fovy 由实测内参换算）：
  - `cam_wrist_d405`——腕部，沿工具轴向下（D405，7–50cm 近距抓取验证）
  - `cam_scene_yp_l515`——支架 +y 侧（L515，43.4°）
  - `cam_scene_yn_d435i`——支架 −y 侧（D435i，42.6°）
  - 两台场景相机水平朝外、向臂基座偏 16.8°（论文 Xu et al. JFR 2026 参数化装配思想）
- 地面为纯视觉（contype=0），整车焊死世界系——臂 teleop/感知仿真用，非移动底盘仿真。

## 观测脚本说明

- `observe_cameras.py`：`--steps N` 先仿真 N 步再出图；`--live N` 连续出 L515 序列。
  深度图为**米制**浮点（`_depth.npy`），背景=远平面 133m。
- `live_monitor.py`：`launch_passive` 查看器 + 三路 `Renderer` 渲染 + OpenCV 显示，
  ESC 退出。渲染器**必须在 launch_passive 之前创建**（GL 上下文顺序，已实测 120 帧无冲突）。
- **注意**：home 关键帧自带 ctrl，但如果你自己的仿真循环复位后不设 ctrl，
  位置伺服会把臂拉向零位（0.4s 内）——记得 `d.ctrl[3:9] = home 角`。

## 技术细节与决策记录

见 [`docs/model_notes.md`](docs/model_notes.md)（安装决策、坐标变换陷阱、配色方案、已验证项、已知限制）。
要点：CAD 装配倒置建模经根 body 翻转落地；夹爪经腕部打印件（轴对轴+面对平面）装真实法兰，
网格已烘入夹爪体系；臂外观=官方 DAE 按材质双色拆分；MuJoCo 会重定心网格——
**世界坐标 = geom_xpos + geom_xmat @ mesh_vert，勿再乘 mesh_pos/quat**。
