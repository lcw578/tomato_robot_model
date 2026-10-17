# vision_pose_generate_client

用于单独调用 `vision_pose_generate` 提供的位姿生成服务：

- 服务名：`/vision/tomato/get_pick_poses`
- 服务类型：`vision_tomato_interfaces/srv/GetPickPoses`

## 1. 先启动位姿生成服务

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 launch vision_pose_generate pick_pose_service.launch.py
```

如果需要走 `yolo` 环境启动 `vision_pose_generate`：

```bash
MODE=conda /home/dong/codes_robot/robot_ws/scripts/run_vision_pose_generate.sh
```

## 2. 启动本功能包调用服务

使用 launch：

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 launch vision_pose_generate_client get_pick_poses_client.launch.py
```

或直接运行节点：

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 run vision_pose_generate_client get_pick_poses_client
```

## 3. 实时可视化节点

使用 launch：

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 launch vision_pose_generate_client pick_pose_visualizer.launch.py
```

或直接运行节点：

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 run vision_pose_generate_client pick_pose_visualizer
```

这个节点会：

- 实时显示裁剪后的彩色图
- 同时显示分割调试图和位姿调试图
- 按空格触发一次 `/vision/tomato/get_pick_poses`
- 在彩色图上叠加本次拾取点、像素位置和深度
- 在窗口底部叠加当前返回状态、分数、成熟度、像素位置、深度和位姿信息
- 可选打开 Open3D 窗口显示相机坐标系和返回位姿
- Open3D 中同时显示抓取点附近更完整的局部目标点云，并尽量带真实 RGB 颜色，便于验证果梗附近的三维几何

窗口按键：

- `Space`：触发一次位姿生成
- `q` 或 `Esc`：退出

Open3D 说明：

- 如果环境里安装了 `open3d`，会额外弹出一个 3D 窗口
- 3D 窗口中原点坐标系表示相机坐标系
- 返回的采摘位姿会显示为一个新的坐标系，并用线段连接到相机原点
- 节点会结合 `/vision/tomato/crop/depth/image_raw` 和 `/vision/tomato/crop/camera_info`，
  在抓取点周围生成一个局部目标点云
- 如果裁剪彩色图和深度图尺寸一致，点云会直接带 RGB 颜色
- 默认会把 Open3D 视角锁定为“从相机原点朝向目标位姿”的观察方向
- 如果没有安装 `open3d`，节点仍可运行，只是禁用 3D 位姿显示

## 4. 一次性调用节点

节点会：

- 等待服务 `/vision/tomato/get_pick_poses`
- 发起一次请求
- 打印返回结果
- 订阅调试图话题
- 将收到的调试图保存到本地
- 自动退出

输出内容包括：

- `success`
- `message`
- 最佳采摘位姿 `pose_xyzrpy_camera`
- 基坐标位姿 `pose_xyzrpy_base`
- `candidate_id`
- `peduncle_id`
- `cluster_id`
- `fruit_ids`
- `score`
- `maturity`
- `ripe_ratio`

默认调试图参数：

- `return_debug_images=true`
- `segment_debug_topic=/vision/tomato/pick_pose/segment_debug_image`
- `pose_debug_topic=/vision/tomato/pick_pose/debug_image`
- `debug_output_dir=/tmp/vision_pose_generate_client`
- `fail_on_unsuccessful_response=false`

说明：

- 当现场没有可用目标时，服务可能返回 `success=false` 和 `message=pose generation failed`
- 客户端现在默认把这种情况当作一次有效调用，不再以非零退出码退出
- 如果你希望保留“失败即退出码非零”，可将 `fail_on_unsuccessful_response` 设为 `true`

保存文件示例：

- `/tmp/vision_pose_generate_client/1781494495_123456789_segment_debug.png`
- `/tmp/vision_pose_generate_client/1781494495_123456789_pose_debug.png`

## 5. 前提条件

调用成功前，需要 `vision_pose_generate` 已经正常运行，并且这些输入话题有数据：

- `/camera/color/image_rect_raw`
- `/camera/aligned_depth_to_color/image_raw`
- `/camera/color/camera_info`

## 6. 自检命令

检查服务是否存在：

```bash
ros2 service list | grep /vision/tomato/get_pick_poses
```

检查相关话题：

```bash
ros2 topic list | grep /vision/tomato
```

查看调试图输出目录：

```bash
ls /tmp/vision_pose_generate_client
```
