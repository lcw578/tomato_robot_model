sensor_msgs/Image color_image
uint32[] class_ids
string[] class_names
float32[] confidences
float32[] bboxes
xc_interfaces/Mask[] masks  # 修改为自定义Mask数组
uint32[] matched_rectangles
string growth_directions#朝前、朝左、朝右
float32 picking_points_x#图像采摘点的x坐标
float32 picking_points_y#图像采摘点的y坐标
float32 picking_points_z#图像采摘点的对应的深度值
float32 distances #图像采摘点的距离
float64 mid2target_angle #果梗在机械臂XY平面的角度
float32 arm_ry #机械臂末端旋转角度,与采摘点切线角度有关系