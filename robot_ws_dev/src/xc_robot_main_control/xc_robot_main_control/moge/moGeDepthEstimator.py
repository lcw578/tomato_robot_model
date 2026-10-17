# 在文件开头添加
from pathlib import Path
import sys

# 添加moge根目录到搜索路径
if (moge_root := str(Path(__file__).parent.absolute())) not in sys.path:
    sys.path.insert(0, moge_root)

import cv2
import torch
import numpy as np
from moge.model.v2 import MoGeModel
import time

class MoGeDepthEstimator:
    """
    MoGe深度估计器类，用于从彩色图像生成深度图
    """
    
    def __init__(self, model_path="/home/xucan/code/test/robot_ws/src/xc_robot_main_control/xc_robot_main_control/moge/weights/moge-2-vits-normal.pt", device=None):
        """
        初始化深度估计器
        
        Args:
            model_path: 预训练模型路径
            device: 运行设备，默认为自动检测（优先使用CUDA）
        """
        # 检查CUDA是否可用
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型
        print("正在加载模型...")
        self.model = MoGeModel.from_pretrained(model_path).to(self.device)
        print("模型加载完成")
    
    def estimate_depth(self, image, return_time=False):
        """
        从彩色图像生成深度图
        
        Args:
            image: 输入彩色图像，可以是numpy数组（BGR或RGB格式）或图像路径
            return_time: 是否返回推理时间
        
        Returns:
            depth_map: 生成的深度图（numpy数组）
            inference_time: 推理时间（如果return_time=True）
        """
        # 如果输入是图像路径，则读取图像
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
        
        # 确保图像是BGR格式，转换为RGB
        if image.shape[-1] == 3:
            # 假设输入是BGR格式（OpenCV默认格式）
            input_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("输入图像必须是3通道彩色图像")
        
        # 转换为张量并归一化
        input_tensor = torch.tensor(input_image_rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        
        # 进行推理并计时
        start_time = time.time()
        output = self.model.infer(input_tensor)
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 获取深度图
        depth_map = output["depth"].cpu().numpy()
        
        if return_time:
            return depth_map, inference_time
        else:
            return depth_map
    
    def estimate_depth_batch(self, images, return_time=False):
        """
        批量处理多张图像
        
        Args:
            images: 图像列表，可以是numpy数组或图像路径
            return_time: 是否返回推理时间
        
        Returns:
            depth_maps: 深度图列表
            inference_times: 推理时间列表（如果return_time=True）
        """
        depth_maps = []
        inference_times = []
        
        for image in images:
            if return_time:
                depth_map, inference_time = self.estimate_depth(image, return_time=True)
                inference_times.append(inference_time)
            else:
                depth_map = self.estimate_depth(image)
            depth_maps.append(depth_map)
        
        if return_time:
            return depth_maps, inference_times
        else:
            return depth_maps

# 示例用法
if __name__ == "__main__":
    # 创建深度估计器实例
    estimator = MoGeDepthEstimator()
    
    # 处理单张图像
    image_path = "/home/xc_home_wsl/data/test/images/JinLingLong_0125_00090.jpg"  # 请替换为实际的图片路径
    depth_map, inference_time = estimator.estimate_depth(image_path, return_time=True)
    print(f"推理时间: {inference_time:.4f} 秒")
    
    # 输出100,100像素位置的深度值
    if depth_map.shape[0] > 100 and depth_map.shape[1] > 100:
        depth_value = depth_map[100, 100]
        print(f"100,100像素位置的深度值: {depth_value}")
    else:
        print("图像尺寸小于100x100，无法获取100,100位置的深度值")
    
    # 批量处理多张图像
    image_paths = [
        "/home/xc_home_wsl/data/test/images/JinLingLong_0125_00090.jpg",  # 请替换为实际的图片路径
        "/home/xc_home_wsl/data/test/images/yuekeda-193.jpg",
        "/home/xc_home_wsl/data/test/images/JinLingLong_ChengHong_12104.jpg"
        # 添加更多图片路径
        # "/path/to/second/image.jpg",
        # "/path/to/third/image.jpg",
    ]
    depth_maps, times = estimator.estimate_depth_batch(image_paths, return_time=True)
    for i, (depth_map, time) in enumerate(zip(depth_maps, times)):
        print(f"图像 {i+1} 推理时间: {time:.4f} 秒")
        if depth_map.shape[0] > 100 and depth_map.shape[1] > 100:
            print(f"图像 {i+1} 100,100像素位置的深度值: {depth_map[100, 100]}")
