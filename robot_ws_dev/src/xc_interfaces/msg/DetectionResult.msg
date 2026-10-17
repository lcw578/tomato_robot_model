sensor_msgs/Image color_image
sensor_msgs/Image depth_image
string timestamp
uint32 handle_num 
uint32[] class_ids
string[] class_names
float32[] confidences
float32[] bboxes
xc_interfaces/Mask[] masks  # 修改为自定义Mask数组
uint32[] matched_rectangles
string[] growth_directions#朝前、朝左、朝右
float32[] picking_points_x#图像采摘点的x坐标
float32[] picking_points_y#图像采摘点的y坐标
float32[] picking_points_z#图像采摘点的对应的深度值
float32[] distances #图像采摘点的距离
float32[] target_arm_points_x#发生给机械臂的x坐标
float32[] target_arm_points_y#发生给机械臂的y坐标
float32[] target_arm_points_z#发生给机械臂的Z坐标
uint32[] watch_tomato_nums # 本次采摘的番茄编号  序号为1会放弃
uint32[] try_pick_times # 尝试采摘次数
uint32[] go_to_target_results #机械臂到达目标点的返回结果
uint32[] gripper_feedback_results # 末端反馈结果
uint32[] return_to_box_results #机械臂返回框的返回结果
uint32[] gripper_close_results #末端关闭结果
uint32[] gripper_open_results #末端打开结果






