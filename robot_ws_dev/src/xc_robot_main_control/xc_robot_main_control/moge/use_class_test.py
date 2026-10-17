from moGeDepthEstimator import MoGeDepthEstimator
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 创建深度估计器实例
estimator = MoGeDepthEstimator()

# 处理单张图像
image_path = "/home/xucan/code/test/moge/moge/example_images/02_Office.jpg"
depth_map, inference_time = estimator.estimate_depth(image_path, return_time=True)
print(f"推理时间: {inference_time:.4f} 秒")

# 读取原图
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 将深度图转换为彩色可视化
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_colormap = plt.cm.jet(depth_normalized)
depth_colormap = (depth_colormap[:, :, :3] * 255).astype(np.uint8)

# 创建并排显示
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 显示原图
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 显示彩色深度图
axes[1].imshow(depth_colormap)
axes[1].set_title('Depth Map (Colorized)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 或者直接传入图像数组
# image = cv2.imread(image_path)
# depth_map = estimator.estimate_depth(image)

# # 获取深度图后可以进行后续处理
# print(f"深度图形状: {depth_map.shape}")
# print(f"最小深度值: {depth_map.min()}")
# print(f"最大深度值: {depth_map.max()}")

# 批量处理多张图像
# image_paths = [
#     "path/to/image1.jpg",
#     "path/to/image2.jpg",
#     "path/to/image3.jpg"
# ]
# depth_maps, inference_times = estimator.estimate_depth_batch(image_paths, return_time=True)

# for i, (depth_map, time) in enumerate(zip(depth_maps, inference_times)):
#     print(f"图像 {i+1} 推理时间: {time:.4f} 秒")
#     print(f"深度图形状: {depth_map.shape}")