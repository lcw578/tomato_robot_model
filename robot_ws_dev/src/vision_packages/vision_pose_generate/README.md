# vision_pose_generate

`vision_pose_generate` 用于基于裁剪后的 RGB-D 图像生成番茄采摘位姿。

它对外提供统一服务：

- `/vision/tomato/get_pick_poses`

内部由 4 个节点组成：

- `image_preprocess_node`
- `detection_service`
- `pose_estimator_service`
- `pick_pose_service_node`

## 1. 功能概览

整体处理链路：

1. 相机原始 RGB、深度、内参进入 `image_preprocess_node`
2. 预处理节点输出中心方形裁剪后的 RGB-D 与相机内参
3. `detection_service` 对裁剪后的彩色图做实例分割
4. `pose_estimator_service` 基于分割结果、深度图、相机内参生成采摘位姿
5. `pick_pose_service_node` 对外暴露统一服务 `/vision/tomato/get_pick_poses`

默认目标类别：

- `peduncle`
- `tomato_cluster`
- `tomato_fruit`

## 2. 节点与接口

### 2.1 image_preprocess_node

输入话题：

- `/camera/color/image_rect_raw`
- `/camera/aligned_depth_to_color/image_raw`
- `/camera/color/camera_info`

输出话题：

- `/vision/tomato/crop/color/image_raw`
- `/vision/tomato/crop/depth/image_raw`
- `/vision/tomato/crop/camera_info`

默认行为：

- `crop_mode=center_square`
- 按彩色图中心裁成正方形
- 同步修正 `CameraInfo.k` 和 `CameraInfo.p`

### 2.2 detection_service

服务名：

- `/vision/tomato/segment`

服务类型：

- `vision_tomato_interfaces/srv/SegmentTomato`

请求：

- `sensor_msgs/Image color_image`

返回：

- `InstanceMask[] instances`
- `sensor_msgs/Image debug_image`
- `bool success`
- `string message`

说明：

- 使用 YOLO 分割模型
- 默认模型路径：
  - `package://vision_pose_generate/weights/yolo11s_seg_peduncle_cluster_0601_640_best.pt`
- 运行时依赖 `torch` / `ultralytics`

### 2.3 pose_estimator_service

服务名：

- `/vision/tomato/generate_pick_poses`

服务类型：

- `vision_tomato_interfaces/srv/GeneratePickPoses`

请求：

- `sensor_msgs/Image color_image`
- `sensor_msgs/Image depth_image`
- `sensor_msgs/CameraInfo camera_info`
- `InstanceMask[] instances`

返回：

- `bool success`
- `PickPoseCandidate best`
- `sensor_msgs/Image debug_image`
- `string message`

说明：

- 当前只返回一个最佳候选 `best`
- 姿态来源是果梗局部切线拟合，不是完整表面法向估计
- 当前交接使用时只应关注相机坐标系位姿输出

### 2.4 pick_pose_service_node

服务名：

- `/vision/tomato/get_pick_poses`

服务类型：

- `vision_tomato_interfaces/srv/GetPickPoses`

请求：

- `bool return_debug_images`

返回：

- `bool success`
- `string message`
- `PickPoseCandidate best`

说明：

- 这是推荐给上层系统使用的统一入口
- 节点会自动读取最近一帧同步好的裁剪后 RGB、深度、相机内参
- 内部串行调用：
  - `/vision/tomato/segment`
  - `/vision/tomato/generate_pick_poses`

调试图发布：

- `/vision/tomato/pick_pose/segment_debug_image`
- `/vision/tomato/pick_pose/debug_image`

结果发布：

- `/vision/tomato/pick_pose/best`
- `/vision/tomato/pick_pose/markers`

## 3. 启动方式

### 3.0 先启动 RealSense 相机

`vision_pose_generate` 本身不包含相机驱动节点。

默认依赖这些上游话题：

- `/camera/color/image_rect_raw`
- `/camera/aligned_depth_to_color/image_raw`
- `/camera/color/camera_info`

如果使用 RealSense，相机可先这样启动：

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

如果你的工作区中使用的是定制 launch 或命名空间，请保证最终输出的话题名与上面 3 个默认输入一致。

启动后建议先检查：

```bash
ros2 topic hz /camera/color/image_rect_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info --once
```

### 3.1 推荐方式

如果 `detection_service` 依赖 `yolo` conda 环境，使用：

```bash
MODE=conda /home/dong/codes_robot/robot_ws/scripts/run_vision_pose_generate.sh
```

这个脚本会：

- 激活 `yolo` 环境
- 导出 `VISION_POSE_GENERATE_PYTHON`
- 仅让 `detection_service` 使用该解释器启动
- 其他 ROS 节点仍走系统 ROS 环境

### 3.2 直接启动 launch

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 launch vision_pose_generate pick_pose_service.launch.py
```

如果直接这样启动，前提是当前环境已经能满足：

- `rclpy`
- `cv_bridge`
- `torch`
- `ultralytics`

否则优先使用上面的脚本。

### 3.3 单独启动各节点

预处理：

```bash
ros2 launch vision_pose_generate preprocess.launch.py
```

检测：

```bash
ros2 launch vision_pose_generate detector.launch.py
```

位姿估计：

```bash
ros2 launch vision_pose_generate pose_estimator.launch.py
```

完整链路：

```bash
ros2 launch vision_pose_generate generate_pose.launch.py
```

## 4. 调用方式

### 4.1 直接调用 ROS service

```bash
ros2 service call /vision/tomato/get_pick_poses vision_tomato_interfaces/srv/GetPickPoses "{return_debug_images: true}"
```

如果不需要调试图：

```bash
ros2 service call /vision/tomato/get_pick_poses vision_tomato_interfaces/srv/GetPickPoses "{return_debug_images: false}"
```

### 4.2 使用包内示例客户端

```bash
source /opt/ros/humble/setup.bash
source /home/dong/codes_robot/robot_ws/install/setup.bash
ros2 run vision_pose_generate get_pick_poses_client_example
```

## 5. PickPoseCandidate 字段说明

核心字段：

- `candidate_id`
- `peduncle_id`
- `cluster_id`
- `fruit_ids`
- `image_u`
- `image_v`
- `depth`
- `pose_xyzrpy_camera`
- `pose_camera`
- `score`
- `maturity`
- `ripe_ratio`
- `truncated_by_crop`
- `reason`

说明：

- `pose_xyzrpy_camera = [x, y, z, roll, pitch, yaw]`
- 单位：
  - 位置：米
  - 角度：弧度
- 当前交接格式只使用相机坐标系位姿

## 6. 默认参数

### 6.1 preprocess

来源：[config/preprocess.yaml](/home/dong/codes_robot/robot_ws/src/vision_pose_generate/config/preprocess.yaml:1)

- `input_color_topic: /camera/color/image_rect_raw`
- `input_depth_topic: /camera/aligned_depth_to_color/image_raw`
- `input_camera_info_topic: /camera/color/camera_info`
- `output_color_topic: /vision/tomato/crop/color/image_raw`
- `output_depth_topic: /vision/tomato/crop/depth/image_raw`
- `output_camera_info_topic: /vision/tomato/crop/camera_info`
- `crop_mode: center_square`
- `publish_rate_limit_hz: 10.0`

### 6.2 detector

来源：[config/detector.yaml](/home/dong/codes_robot/robot_ws/src/vision_pose_generate/config/detector.yaml:1)

- `model_path: package://vision_pose_generate/weights/yolo11s_seg_peduncle_cluster_0601_640_best.pt`
- `image_size: 640`
- `conf_threshold: 0.35`
- `iou_threshold: 0.5`
- `device: cuda:0`

### 6.3 pose_estimator

来源：[config/pose_estimator.yaml](/home/dong/codes_robot/robot_ws/src/vision_pose_generate/config/pose_estimator.yaml:1)

- `min_depth_m: 0.15`
- `max_depth_m: 1.2`
- `depth_windows: [7, 11, 15, 21]`
- `min_peduncle_area_px: 20`
- `min_skeleton_points: 8`
- `pick_offset_px: 10`
- `pick_point_mode: skeleton_midpoint`
- `tangent_fit_radius_px: 18`
- `min_tangent_fit_points: 4`
- `crop_border_margin_px: 5`
- `enable_maturity: true`
- `hand_eye_enabled: false`
- `camera_frame: camera_color_optical_frame`

## 7. 自检命令

检查服务：

```bash
ros2 service list | grep /vision/tomato
```

检查话题：

```bash
ros2 topic list | grep /vision/tomato
```

检查裁剪图像是否有数据：

```bash
ros2 topic hz /vision/tomato/crop/color/image_raw
ros2 topic hz /vision/tomato/crop/depth/image_raw
```

检查检测节点是否真的走了 `yolo` 环境：

启动时日志应包含类似：

```text
[info] detection_python=/home/dong/miniconda3/envs/yolo/bin/python
```

## 8. 常见问题

### 8.1 `failed to import ultralytics` / `No module named 'torch'`

原因：

- `detection_service` 没跑在正确的 `conda` 环境里

处理：

- 优先使用：

```bash
MODE=conda /home/dong/codes_robot/robot_ws/scripts/run_vision_pose_generate.sh
```

### 8.2 `pose generation failed`

说明：

- 这是服务调用成功但没有生成出合法位姿

常见原因：

- 没有形成合法的 `peduncle + tomato_cluster` 配对
- 目标深度无效
- 目标被裁剪边界截断
- 果梗骨架不足

建议：

- 打开 `return_debug_images=true`
- 查看：
  - `/vision/tomato/pick_pose/segment_debug_image`
  - `/vision/tomato/pick_pose/debug_image`

### 8.3 `pitch` 经常为 0

原因：

- 当前姿态构造基于果梗切线和固定相机前向约束
- 属于受约束的姿态定义，不是完整自由 6D 姿态恢复

## 9. 当前已知限制

- 当前接口只返回一个最佳候选，没有返回完整候选列表
- 姿态估计对果梗分割质量较敏感
- 当前姿态中的 `pitch` 可能长期接近 0，这与当前旋转构造方法有关
- Open3D 调试可视化在客户端包 `vision_pose_generate_client` 中，不在本包内
