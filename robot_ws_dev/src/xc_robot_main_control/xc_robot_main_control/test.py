from xc_robot_main_control.moge.moGeDepthEstimator import MoGeDepthEstimator

import cv2

import os
from ament_index_python.packages import get_package_share_directory

def main():
    # 创建深度估计器实例
    share_dir = get_package_share_directory('xc_robot_main_control')
    weight_path = os.path.join(share_dir, 'weights', 'moge-2-vits-normal.pt')

    estimator = MoGeDepthEstimator(model_path=weight_path)

    # 处理单张图像
    image_path = "/home/xc/code/test/ros2/robot_ws/20260130163654_D435_c.png"
    depth_map, inference_time = estimator.estimate_depth(image_path, return_time=True)
    print(f"推理时间: {inference_time:.4f} 秒")

    depth_map, inference_time = estimator.estimate_depth(image_path, return_time=True)
    print(f"推理时间: {inference_time:.4f} 秒")

if __name__ == '__main__':
    main()