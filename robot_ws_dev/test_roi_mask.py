#!/usr/bin/env python3
import cv2
import os
import time
from pathlib import Path


def main():
    # 配置参数
    image_folder = '/mnt/f/AI_image/tomato_stem_gripper/20260727/origin'
    left_mask_ratio = 0.25   # 左边涂黑比例 (0.0 ~ 0.5)
    right_mask_ratio = 0.25  # 右边涂黑比例 (0.0 ~ 0.5)
    interval = 3.0           # 每隔几秒显示一张

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
    print(f"📐 左边涂黑比例: {left_mask_ratio*100:.0f}%, 右边涂黑比例: {right_mask_ratio*100:.0f}%")

    # 创建显示窗口
    cv2.namedWindow('ROI Mask Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Mask Preview', 1200, 600)

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

        h, w = img.shape[:2]

        # 创建涂黑后的图片
        masked_img = img.copy()

        # 左边涂黑
        left_width = int(w * left_mask_ratio)
        if left_width > 0:
            masked_img[:, :left_width] = 0

        # 右边涂黑
        right_width = int(w * right_mask_ratio)
        if right_width > 0:
            masked_img[:, w - right_width:] = 0

        # 左右拼接：原图（左） + 涂黑后（右）
        combined_img = cv2.resize(img, (640, 480))
        combined_masked = cv2.resize(masked_img, (640, 480))
        combined = cv2.hconcat([combined_img, combined_masked])

        # 添加标题
        cv2.putText(combined, '原图', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, f'涂黑后 (L:{left_mask_ratio*100:.0f}% R:{right_mask_ratio*100:.0f}%)', (660, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示图片
        cv2.imshow('ROI Mask Preview', combined)

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
