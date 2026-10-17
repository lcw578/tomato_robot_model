#!/usr/bin/env python3
import time
import re
import os

# ==================== 配置区域 ====================
# 在这里修改需要操作的文件列表
input_files = [
    # "/home/xucan/code/test/robot_ws/log_xc/2/motion.log",
    # "/home/xucan/code/test/robot_ws/log_xc/2/python3_1.log",
    # "/home/xucan/code/test/robot_ws/log_xc/2/python3_2.log",
    "/home/xucan/code/test/robot_ws/log_xc/7/python3_1.log",

    # "/home/xucan/code/test/robot_ws/log_xc/2/log_temp_txt.log",


    # 可以添加更多文件
]

# 输出文件保存目录，None表示与输入文件同目录
output_dir = None  # 例如: "/home/xucan/code/test/robot_ws/output"

# 输出文件名前缀
output_prefix = "converted_"
# ==================================================

# 正则表达式匹配ROS时间戳
ros_timestamp_pattern = r'\[(\d+\.\d+)\]'

def convert_file(input_file, output_file):
    """转换单个文件的时间戳"""
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in f:
            # 查找所有匹配的时间戳
            matches = re.findall(ros_timestamp_pattern, line)
            if matches:
                for timestamp in matches:
                    # 转换时间戳
                    timestamp_float = float(timestamp)
                    # 转换为本地时间
                    local_time = time.localtime(timestamp_float)
                    # 获取毫秒部分
                    ms = int((timestamp_float % 1) * 1000)
                    # 格式化时间字符串
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time) + f'.{ms:03d}'
                    # 替换原时间戳
                    line = line.replace(f'[{timestamp}]', f'[{formatted_time}]')
            # 写入处理后的行
            out_f.write(line)
    return output_file

def main():
    for input_file in input_files:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 文件 {input_file} 不存在，跳过处理")
            continue
        
        # 确定输出文件路径
        if output_dir:
            # 使用指定的输出目录
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(input_file)
            output_file = os.path.join(output_dir, output_prefix + base_name)
        else:
            # 使用默认输出路径（与输入文件同目录）
            dir_name = os.path.dirname(input_file)
            base_name = os.path.basename(input_file)
            output_file = os.path.join(dir_name, output_prefix + base_name)
        
        # 转换文件
        result_file = convert_file(input_file, output_file)
        print(f"转换完成，结果保存到: {result_file}")
        
        # 显示转换示例
        print(f"\n{input_file} 转换示例:")
        with open(result_file, 'r', encoding='utf-8') as out_f:
            for i, line in enumerate(out_f):
                if i < 5:  # 只显示前5行
                    print(line.strip())
                else:
                    break

if __name__ == "__main__":
    main()