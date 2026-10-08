# 番茄采摘机器人整机 MuJoCo 模型

AUBO i5 机械臂 + 自研夹爪 + 轨道底盘的整机 MuJoCo 模型，含三台 RealSense 相机传感器帧、
温室环境与采摘闭环（真值选串 → 沿轨道行驶到站点 → 剪切 → 搬运 → 进筐，路径零穿透已验证）。
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

# 采摘链路：真值选最好摘的一串 → 驶到站点 → 剪切 → 搬运 → 进筐
python3 tools/harvest_demo.py --chain
```

依赖：`mujoco>=3.3`、`opencv-python`（观测/可视化脚本用）。

## 项目结构

```
robot_assembly/
├── model/
│   ├── tomato_picker.xml      主模型（自包含，相对路径引用网格）
│   └── meshes_baked/          腕部相机/转接板烘焙网格（挂 base_link）
├── meshes/
│   ├── cad/                   整车 CAD 网格（307 零件中的 297 个被引用）
│   ├── gripper/               夹爪基线网格
│   ├── aubo/                  AUBO i5 官方连杆网格（碰撞体）
│   └── env/                   环境资产：轨道钢管 + 10 株藤/叶/果串 OBJ + 贴图
├── dae_split/                 臂外观网格（官方 DAE 按材质拆分）
├── tools/                     构建与环境管线（phase_* 整机装配、patch_lift_* 升降补丁、
│                              build_environment 环境）+ 采摘 demo/可视化
├── docs/                      model_notes.md、相机视锥图、采摘视频（单串 + 链路）
├── observe_cameras.py         三相机 RGB+深度观测
└── live_monitor.py            3D 查看器 + 三路相机实时监测（OpenCV）
```

## 模型概要

- **75 body / 61 关节 / 1039 geom / 81 site / 3 相机 / 11 执行器**，82 条 equality
  （夹爪耦合 2 + 果串 weld 80）。
- 机器人：车体（CAD 零件整体静态）+ 10 轨轮（2 主动 velocity 伺服 + 8 从动）+ AUBO i5 六连杆
  （官方运动学：连杆间距 122/121.5/408/376/102.5/94 mm，行程 ±175°）+ 齿条夹爪（equality 耦合）。
- **行驶**（"走停摘"的底盘基础）：`machine` body 有 `machine_slide`（世界 x，±2.5 m）+
  `machine_servo` 位置伺服——沿轨道**运动学行驶**（轮子同步滚转仅视觉，轮轨无物理接触）。
- **升降平台**（`lift` body，slide 行程 0–0.5 m，位置伺服）：台面/收集筐/控制箱/传感器杆/
  场景相机/整条臂都挂在其上随之升降（论文工作高度 0.5–2.0 m 的仿真抽象）。
- **温室环境**（`tools/build_environment.py` 生成，标计区可幂等重跑）：
  - 每侧 5 株、株距 0.9 m、行距 1.6 m（y=±0.8）——论文 §2 种植参数；植株用
    aoc_tomato_farm 的 vine 随机变体（带贴图，果串按 marker 聚类，红/橙/青成熟度）。
  - **主茎 = 拟合的真实藤轴胶囊**（r=22 mm、透明，按资产生成的直圆柱规格拟合，与视觉藤
    贴合），参与碰撞：机械臂规划器的禁碰集 = 主茎 + 车体全部几何。
  - 40 条果串 = 两段式物理：`truss_p_k` 顶层 body（freejoint + 12 颗真尺寸果球碰撞）
    经 `hold_` weld 挂在植株（active），`grip_` weld 挂在夹爪 base_link（inactive，剪切后启用）。
    `cut_p{p}_t{t}` site = 果柄 55% 处剪切点（论文 §4.5 的"送刀口深处"；感知真值 + 采摘目标）。
  - 轨道：CAD 原件 part_110/149（Ø40 钢管）平铺拉长；碰撞圆柱已改纯视觉
    （曾因轮-轨持续穿透产生病态接触力污染求解器，表现为果串弹飞/全片抖动）。
  - 收集筐 `basket_site`（筐为凹腔，碰撞用 5 个隐形盒近似）。
- **默认初始姿态**（home 关键帧，含 11 维 ctrl：索引 9=lift、10=machine）：肩 −1.59 / 大臂 −0.061 /
  前臂 1.5 / 腕1 −1.59 / 腕2 −1.65 / 腕3 3.05 rad，夹爪齿轮 −0.474。另有 `lift_high`（升降 +0.45 m）。
- **三台 RealSense 相机**（内参见各 fovy 由实测内参换算）：
  - `cam_wrist_d405`——腕部，沿工具轴向下（D405，7–50cm 近距抓取验证）
  - `cam_scene_yp_l515`——支架 +y 侧（L515，43.4°）
  - `cam_scene_yn_d435i`——支架 −y 侧（D435i，42.6°）
  - 两台场景相机水平朝外、向臂基座偏 16.8°（论文 Xu et al. JFR 2026 参数化装配思想）
- 地面为纯视觉（contype=0）；z=−0.5 有一块只与果串碰撞的隐形接坠面（contype=2），
  掉出筐的果串不会无限下落。

## 采摘闭环演示

```bash
python3 tools/harvest_demo.py                  # 自动挑最近成熟串: 接近→剪切→搬运→筐上释放→进筐判定
python3 tools/harvest_demo.py --target 6_1     # 指定果串
python3 tools/harvest_demo.py --chain          # 链路: 真值选最好摘的串 → 驶到站点 → 采摘进筐
python3 tools/harvest_demo.py --reach          # 全部果串 IK 可达性抽检
```

- **剪切** = weld 状态机（`hold_` 断开、`grip_` 按当前相对位姿激活），物理上没有真"剪断"；
  成败由确定性几何判据给出：果柄线段 ∩ 剪切区盒（gear 系轴向 0.01–0.115 m、径向 ≤26 mm）
  命中 → 剪切成功，不命中 → 空剪 + `shear_miss`。
- **运动**：两阶段碰撞感知 IK（多种子，收敛后查碰撞）+ RRT-Connect 关节空间规划（直线优先）；
  抬升预检"能低不升"（0 → 0.25 → 0.5 逐档试解，全通过才升）。
- **当前状态**：链路全程 lift=0.00、in_basket，臂对主茎/筐/台面/风琴罩的穿透 >3 mm 采样 = 0；
  单串 demo 中 1_2 / 0_3 / 6_1 in_basket。

**可视化**：
```bash
python3 tools/harvest_view.py                    # 实时查看器, 相机跟随夹爪, --speed 0.5 慢放
python3 tools/harvest_view.py --chain            # 链路模式 (含底盘行驶)
python3 tools/harvest_view.py --mp4 docs/x.mp4   # 无显示器导出 MP4 (带阶段标签)
```
效果样例：[`docs/harvest_demo.mp4`](docs/harvest_demo.mp4)（单串 20 s）、
[`docs/harvest_chain.mp4`](docs/harvest_chain.mp4)（链路，含行驶）。

## 观测脚本说明

- `observe_cameras.py`：`--steps N` 先仿真 N 步再出图；`--live N` 连续出 L515 序列。
  深度图为**米制**浮点（`_depth.npy`），背景=远平面 133m。
- `live_monitor.py`：`launch_passive` 查看器 + 三路 `Renderer` 渲染 + OpenCV 显示，
  ESC 退出。渲染器**必须在 launch_passive 之前创建**（GL 上下文顺序，已实测 120 帧无冲突）。
- **注意**：home 关键帧自带 ctrl，但如果你自己的仿真循环复位后不设 ctrl，
  位置伺服会把臂拉向零位（0.4s 内）——记得 `d.ctrl[:] = home 姿态`。ctrl 共 11 维：
  `[gear, 左轮, 右轮, j1..j6, lift, machine]`（索引 9=lift、10=轨道行驶）。

## 技术细节与决策记录

见 [`docs/model_notes.md`](docs/model_notes.md)（安装决策、坐标变换陷阱、配色方案、
环境/升降/剪切状态机/控制栈/链路/规划器的逐轮记录与踩坑、已验证项、已知限制）。
要点：CAD 装配倒置建模经根 body 翻转落地；夹爪经腕部打印件（轴对轴+面对平面）装真实法兰，
网格已烘入夹爪体系；臂外观=官方 DAE 按材质双色拆分；MuJoCo 会重定心网格——
**世界坐标 = geom_xpos + geom_xmat @ mesh_vert，勿再乘 mesh_pos/quat**。
