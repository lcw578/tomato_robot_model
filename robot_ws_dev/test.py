from ultralytics import YOLO

import cv2

img=cv2.imread("test.jpg")
self.yolo_model = YOLO("/home/xucan/code/test/robot_ws/src/closed_loop_feedback_detection/weights/gripper_stem_20260222.pt")
results = self.yolo_model.predict(img, conf=0.5, device=self.device, verbose=False, iou=0.2)
annotated_img = results.plot()
cv2.write("test_annotated.jpg", annotated_img)
