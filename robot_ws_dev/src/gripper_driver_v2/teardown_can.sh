#!/bin/bash
# 取消CAN接口配置
# 用途：手动停止CAN接口

CAN_INTERFACE="can0"

echo "停止CAN接口 $CAN_INTERFACE..."

# 关闭接口
sudo ip link set $CAN_INTERFACE down

# 停止slcand进程
sudo pkill -f "slcand.*$CAN_INTERFACE"

echo "CAN接口已停止"
