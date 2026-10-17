#!/bin/bash
# CAN设备自动配置脚本
# 用途：在系统启动时自动配置CAN接口

# 等待USB设备就绪
sleep 2

# 查找第一个ttyACM设备
TTY_DEVICE=$(ls /dev/ttyACM* 2>/dev/null | head -n 1)

if [ -z "$TTY_DEVICE" ]; then
    echo "错误：未找到ttyACM设备"
    exit 1
fi

echo "找到CAN设备: $TTY_DEVICE"

# 启动slcand守护进程
echo "启动slcand..."
sudo slcand -o -c -s8 $TTY_DEVICE can0

# 等待接口创建
sleep 1

# 配置CAN接口参数
echo "配置CAN接口..."
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

echo "CAN接口配置完成！"
echo "CAN接口状态："
ip -details link show can0
