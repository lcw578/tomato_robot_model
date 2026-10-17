#!/bin/bash
# CAN设备自动配置脚本（支持多个ttyACM设备）
# 用途：在系统启动时自动配置CAN接口

# 等待USB设备就绪
sleep 2

# 查找所有ttyACM设备
TTY_DEVICES=$(ls /dev/ttyACM* 2>/dev/null)

if [ -z "$TTY_DEVICES" ]; then
    echo "错误：未找到ttyACM设备"
    echo "请检查："
    echo "  1. CAN设备是否已连接"
    echo "  2. 设备权限是否正确"
    exit 1
fi

echo "找到ttyACM设备："
echo "$TTY_DEVICES"

# 使用第一个设备
TTY_DEVICE=$(echo "$TTY_DEVICES" | head -n 1)
CAN_INTERFACE="can0"

echo "使用设备: $TTY_DEVICE"
echo "CAN接口: $CAN_INTERFACE"

# 关闭已存在的slcand进程（如果有）
sudo pkill -f "slcand.*$CAN_INTERFACE" 2>/dev/null

# 等待接口释放
sleep 1

# 启动slcand守护进程
echo "启动slcand..."
sudo slcand -o -c -s8 $TTY_DEVICE $CAN_INTERFACE

# 等待接口创建
sleep 1

# 检查接口是否存在
if ! ip link show $CAN_INTERFACE > /dev/null 2>&1; then
    echo "错误：CAN接口 $CAN_INTERFACE 未创建"
    exit 1
fi

# 配置CAN接口参数
echo "配置CAN接口..."
sudo ip link set $CAN_INTERFACE type can bitrate 1000000
sudo ip link set $CAN_INTERFACE up

# 检查配置结果
if ip link show $CAN_INTERFACE | grep -q "state UP"; then
    echo "=========================================="
    echo "CAN接口配置成功！"
    echo "=========================================="
    echo "CAN接口状态："
    ip -details link show $CAN_INTERFACE
    echo ""
    echo "CAN接口统计信息："
    ip -stats link show $CAN_INTERFACE
    exit 0
else
    echo "错误：CAN接口配置失败"
    exit 1
fi
