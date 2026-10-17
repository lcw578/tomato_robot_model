from ultralytics import YOLO

model = YOLO('/home/xc/code/test/ros2/robot_ws/src/closed_loop_feedback_detection/weights/best20260422.pt')
model.model.names = ['stem', 'trunk']  # 按顺序，list 形式
model.save('/home/xc/code/test/ros2/robot_ws/src/closed_loop_feedback_detection/weights/best20260422_renamed.pt')
