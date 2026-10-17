#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

def generate_coordinates():
    """
    生成符合以下范围的坐标点:
    -0.7 < x < 0.7
    0.25 < y < 0.7 或 -0.7 < y < -0.25
    0.2 < z < 0.7
    rx ry rz 固定为 90 0 180
    xyz 每隔 0.01 取一个坐标点
    """
    
    coordinates = []
    
    # X轴范围: -0.7 到 0.7，步长 0.01
    x_values = [round(-0.7 + i * 0.05, 3) for i in range(int((0.7 - (-0.7)) / 0.05) + 1)]
    
    # Y轴范围: 0.25 到 0.7 和 -0.7 到 -0.25，步长 0.01
    y_positive = [round(0.25 + i * 0.05, 3) for i in range(int((0.7 - 0.25) / 0.05) + 1)]
    y_negative = [round(-0.7 + i * 0.05, 3) for i in range(int((-0.25 - (-0.7)) / 0.05) + 1)]
    y_values = y_positive + y_negative
    
    # Z轴范围: 0.2 到 0.7，步长 0.01
    z_values = [round(0.2 + i * 0.05, 3) for i in range(int((0.7 - 0.2) / 0.05) + 1)]
    
    # 生成所有组合
    for x in x_values:
        for y in y_values:
            for z in z_values:
                ry = random.choice([-30, 0, 30])
                if y>0:
                    coordinates.append(f"{x:.3f} {y:.3f} {z:.3f} 90 {ry} 180")
                else:
                    coordinates.append(f"{x:.3f} {y:.3f} {z:.3f} 90 {ry} 0")
    
    return coordinates

def save_coordinates_to_file(coordinates, file_path):
    """将坐标点保存到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for coord in coordinates:
            f.write(coord + '\n')
    
    print(f"已生成 {len(coordinates)} 个坐标点，保存到 {file_path}")

if __name__ == "__main__":
    # 获取当前脚本所在目录的父目录下的config文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "config")
    
    # 确保config目录存在
    os.makedirs(config_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(config_dir, "target_positions_create.txt")
    
    # 生成坐标点
    coordinates = generate_coordinates()
    
    # 保存到文件
    save_coordinates_to_file(coordinates, output_file)