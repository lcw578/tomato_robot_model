#!/usr/bin/env python3
import pyrealsense2 as rs
import sys

print("测试不同RealSense后端在WSL中的表现")
print("=" * 50)

# 尝试不同的后端配置
test_configs = [
    ("默认后端", None),
    ("RSUSB后端", "RSUSB"),
    ("libuvc后端", "libuvc"),
]

for name, backend_env in test_configs:
    print(f"\n测试: {name}")
    
    if backend_env:
        import os
        os.environ['RS2_BACKEND'] = backend_env
    
    try:
        # 重新导入以应用环境变量
        import importlib
        import pyrealsense2
        importlib.reload(pyrealsense2)
        from pyrealsense2 import backend, context
        
        # 创建上下文
        ctx = context()
        devices = ctx.query_devices()
        
        if devices.size() > 0:
            print(f"  ✓ 找到 {devices.size()} 个设备")
            for i in range(devices.size()):
                dev = devices[i]
                print(f"     设备 {i}: {dev.get_info(rs.camera_info.name)}")
        else:
            print(f"  ✗ 未找到设备")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 清理环境变量
    if backend_env and 'RS2_BACKEND' in os.environ:
        del os.environ['RS2_BACKEND']