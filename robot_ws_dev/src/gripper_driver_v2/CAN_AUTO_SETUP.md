# CAN设备自动化配置说明

## 问题描述
每次电脑开机后需要手动执行以下步骤来连接CAN设备：
1. `ls /dev/ttyACM*` - 查找设备
2. `sudo slcand -o -c -s8 /dev/ttyACM0 can0` - 启动slcand（设备编号需要根据实际查找结果调整）
3. `sudo ip link set can0 type can bitrate 1000000` - 配置CAN接口
4. `sudo ip link set can0 up` - 启用CAN接口

## 解决方案
提供了两种自动化方案，自动查找ttyACM设备并使用正确的设备编号。

### 方案一：使用systemd服务（推荐）

#### 1. 复制服务文件到systemd目录
```bash
sudo cp /home/xucan/code/test/robot_ws/src/gripper_driver_v2/can-config.service /etc/systemd/system/
```

#### 2. 重新加载systemd配置
```bash
sudo systemctl daemon-reload
```

#### 3. 启用服务
```bash
sudo systemctl enable can-config.service
```

#### 4. 手动测试服务（可选）
```bash
# 启动服务
sudo systemctl start can-config.service

# 查看服务状态
sudo systemctl status can-config.service

# 查看服务日志
sudo journalctl -u can-config.service
```

#### 5. 重启测试
```bash
sudo reboot
```

重启后，系统会自动配置CAN接口。

---

### 方案二：使用自动配置脚本

#### 1. 赋予执行权限
```bash
chmod +x /home/xucan/code/test/robot_ws/src/gripper_driver_v2/setup_can_auto.sh
```

#### 2. 手动运行
```bash
sudo /home/xucan/code/test/robot_ws/src/gripper_driver_v2/setup_can_auto.sh
```

#### 3. 添加到开机启动
将以下行添加到 `/etc/rc.local` 文件的 `exit 0` 行之前：
```bash
sudo /home/xucan/code/test/robot_ws/src/gripper_driver_v2/setup_can_auto.sh
```

---

## 文件说明

### 提供的文件

1. **can-config.service** - systemd服务配置文件
   - 自动查找第一个ttyACM设备
   - 适用于Ubuntu 16.04+、Debian 8+等现代Linux系统
   - 系统启动时自动运行

2. **setup_can.sh** - 基础自动配置脚本
   - 自动查找第一个ttyACM设备
   - 配置CAN接口

3. **setup_can_auto.sh** - 增强版自动配置脚本
   - 支持多个ttyACM设备
   - 更详细的错误处理
   - 更好的状态反馈

4. **teardown_can.sh** - 停止CAN接口脚本
   - 用于手动停止CAN接口

5. **CAN_AUTO_SETUP.md** - 本说明文档

---

## 工作原理

### 自动设备查找
脚本使用以下命令自动查找ttyACM设备：
```bash
TTY_DEVICE=$(ls /dev/ttyACM* 2>/dev/null | head -n 1)
```

这会：
1. 列出所有 `/dev/ttyACM*` 设备
2. 取第一个设备作为CAN接口
3. 如果有多个ttyACM设备，会自动选择第一个

### systemd服务执行流程
1. 等待2秒让USB设备就绪
2. 自动查找第一个ttyACM设备
3. 使用查找到的设备启动slcand
4. 配置CAN接口参数（波特率1Mbps）
5. 启用CAN接口

---

## 故障排查

### 1. 服务无法启动
```bash
# 查看服务状态
sudo systemctl status can-config.service

# 查看日志
sudo journalctl -u can-config.service
```

### 2. 设备未找到
```bash
# 检查设备是否连接
ls /dev/ttyACM*

# 检查设备权限
ls -l /dev/ttyACM*
```

### 3. CAN接口未创建
```bash
# 检查接口
ip link show can0

# 检查slcand进程
ps aux | grep slcand
```

### 4. 权限问题
如果遇到权限问题，可以将用户添加到dialout组：
```bash
sudo usermod -a -G dialout $USER
```
然后重新登录。

---

## 自定义配置

### 修改CAN接口名称
如果系统已有其他CAN接口，可以修改接口名称：
1. 编辑服务文件：`sudo nano /etc/systemd/system/can-config.service`
2. 将 `can0` 修改为其他名称（如 `can1`）
3. 重新加载并启用服务

### 修改波特率
如果需要不同的波特率：
1. 编辑服务文件：`sudo nano /etc/systemd/system/can-config.service`
2. 修改 `ExecStartPost` 行中的 `bitrate 1000000`
3. 常用波特率：250000, 500000, 1000000

### 修改延迟时间
如果设备响应较慢，可以增加延迟时间：
1. 编辑服务文件：`sudo nano /etc/systemd/system/can-config.service`
2. 修改 `sleep 2` 和 `sleep 1` 的值

---

## 验证配置

配置完成后，可以使用以下命令验证：

```bash
# 查看CAN接口状态
ip link show can0

# 发送测试数据
cansend can0 001#FFFFFFFFFFFFFFFD

# 监控CAN通讯
sudo candump can0 -t a -x -e
```

---

## 注意事项

1. **设备识别**：如果系统有多个ttyACM设备，会自动选择第一个设备
2. **权限**：确保用户有权限访问CAN接口
3. **启动顺序**：systemd服务在系统启动时运行，可能需要等待USB设备就绪
4. **手动干预**：如果自动配置失败，可以使用手动配置脚本进行调试
5. **设备编号**：脚本会自动查找当前连接的ttyACM设备，不需要手动指定设备编号
