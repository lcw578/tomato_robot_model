#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def main():
    # 配置参数
    model_path = '/home/xucan/code/test/robot_ws/src/xc_robot_main_control/weights/tomato-seg2025063018-yolov8-xucan.pt'
    image_folder = '/mnt/e/xucan/Desktop/save_realsense/extract4'
    interval = 1.0  # 每隔2秒读取一张
    
    # 显示开关
    show_box = False      # 是否显示检测框
    show_mask = True     # 是否显示实例分割
    show_label = False    # 是否显示分类标签
    
    # 类别过滤（设为空列表表示显示所有类别）
    # 填写类别名称，例如: ['stem', 'tomato']
    filter_classes = ['stem']  # 只显示这些类别，空列表=显示全部
    
    # 保存选项
    save_images = True  # 是否保存识别后的图片
    save_folder = os.path.join(image_folder, 'yoloed')  # 保存文件夹
    # 仅当检测到指定类别时才保存（设为 None 表示不启用此条件）
    save_if_class = 'stem'  # 检测到该类别才保存，None=总是保存

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    # 检查图片文件夹
    if not os.path.isdir(image_folder):
        print(f"❌ 图片文件夹不存在: {image_folder}")
        return

    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        str(p) for p in Path(image_folder).glob('*')
        if p.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"❌ 文件夹中没有找到图片文件: {image_folder}")
        return

    print(f"✅ 加载了 {len(image_files)} 张图片")
    print(f"✅ 模型路径: {model_path}")
    if save_images:
        os.makedirs(save_folder, exist_ok=True)
        print(f"💾 保存文件夹: {save_folder}")

    # 加载YOLO模型
    print("🔄 正在加载YOLO模型...")
    model = YOLO(model_path)
    print("✅ YOLO模型加载完成")

    # 获取类别名称
    class_names = model.names
    print(f"📋 类别列表: {class_names}")

    # 创建显示窗口
    cv2.namedWindow('YOLOv8检测结果', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8检测结果', 1200, 600)

    # 循环处理图片
    image_index = 0
    while True:
        # 获取当前图片路径
        img_path = image_files[image_index % len(image_files)]
        image_index += 1

        # 读取图片
        print(f"\n📸 正在处理: {os.path.basename(img_path)} ({image_index}/{len(image_files)})")
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 无法读取图片: {img_path}")
            time.sleep(interval)
            continue

        # YOLO检测
        start_time = time.time()
        results = model(img)
        detect_time = time.time() - start_time

        # 获取检测结果
        result = results[0]
        boxes = result.boxes

        # 在原图上绘制检测框和实例分割
        img_detected = img.copy()
        
        # 为每个类别分配不同颜色（用于mask填充）
        class_colors = {
            0: (0, 255, 0),    # 绿色
            1: (0, 0, 255),    # 红色
            2: (255, 0, 0),    # 蓝色
            3: (0, 255, 255),  # 黄色
            4: (255, 0, 255),  # 品红
            5: (255, 255, 0),  # 青色
            6: (128, 0, 128),  # 紫色
            7: (0, 128, 0),    # 深绿
        }

        if boxes is not None and len(boxes) > 0:
            # 获取mask数据（如果是分割模型）
            masks = result.masks

            # 根据filter_classes过滤目标
            filtered_indices = []
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                if len(filter_classes) == 0 or class_name in filter_classes:
                    filtered_indices.append(i)
            
            if len(filtered_indices) == 0:
                print(f"ℹ️ 检测到 {len(boxes)} 个目标，但不在过滤类别中 {filter_classes}")
            else:
                print(f"✅ 检测到 {len(filtered_indices)} 个目标（已过滤）")

            for i, box in enumerate(boxes):
                # 获取边界框
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 获取类别和置信度
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                # 如果不在过滤列表中，跳过绘制
                if len(filter_classes) > 0 and class_name not in filter_classes:
                    continue
                # 获取颜色
                color = class_colors.get(cls, (0, 255, 0))

                # 绘制实例分割mask（如果有）
                if show_mask and masks is not None and i < len(masks):
                    mask = masks[i].data.cpu().numpy().astype(bool)[0]
                    # YOLO返回的mask是模型输入尺寸，需要resize到原图尺寸
                    mask = cv2.resize(mask.astype(np.uint8), 
                                       (img_detected.shape[1], img_detected.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    # 创建mask图像
                    mask_img = np.zeros_like(img_detected)
                    mask_img[mask] = color
                    # 只对mask区域做半透明叠加，非mask区域保持原图
                    mask_3ch = np.stack([mask]*3, axis=-1)
                    blended = cv2.addWeighted(mask_img, 0.5, img_detected, 0.5, 0)
                    img_detected = np.where(mask_3ch, blended, img_detected)
                    # 绘制mask轮廓
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(img_detected, contours, -1, color, 2)

                # 绘制矩形框
                if show_box:
                    cv2.rectangle(img_detected, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                if show_label:
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(img_detected, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            has_mask = masks is not None and len(masks) > 0
            print(f"   {'包含实例分割' if has_mask else '无实例分割'}")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                # 只打印过滤后的类别
                if len(filter_classes) > 0 and class_name not in filter_classes:
                    continue
                mask_info = " ✓mask" if (has_mask and i < len(masks)) else ""
                print(f"   - {class_name}: {conf:.2f}{mask_info}")
        else:
            print("ℹ️ 未检测到目标")

        # 保存识别后的图片
        if save_images:
            # 检查是否检测到指定类别
            should_save = True
            if save_if_class is not None:
                detected_classes = set()
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                        detected_classes.add(class_name)
                should_save = save_if_class in detected_classes

            if should_save:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(save_folder, f"{base_name}_yoloed.jpg")
                cv2.imwrite(save_path, img_detected)

        # 左右拼接：原图（左） + 检测图（右）
        height = max(img.shape[0], img_detected.shape[0])
        width = img.shape[1] + img_detected.shape[1]
        combined_img = cv2.resize(img, (640, 480))
        combined_detected = cv2.resize(img_detected, (640, 480))
        combined = cv2.hconcat([combined_img, combined_detected])

        # 添加标题
        cv2.putText(combined, '原图', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, f'检测结果 (耗时: {detect_time:.2f}s)', (660, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示图片
        cv2.imshow('YOLOv8检测结果', combined)

        # 等待按键或超时
        key = cv2.waitKey(int(interval * 1000))
        if key == 27:  # ESC键退出
            print("\n👋 用户按ESC退出")
            break
        elif key == ord('p'):  # P键暂停
            print("⏸️ 已暂停，按任意键继续...")
            cv2.waitKey(0)

    # 释放资源
    cv2.destroyAllWindows()
    print("✅ 程序结束")


if __name__ == '__main__':
    main()
