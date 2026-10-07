"""把升降平台从静态焊死改成可动 slide 关节 (论文 Table 4 工作高度 0.5-2.0m 的仿真抽象)。

手术内容 (对 model/tomato_picker.xml):
  1. 几何取证: 台面以上 (世界 z_min >= 0.73m) + 风琴罩 265 的 geom 移入新 body "lift"
     (台面 264 / 筐 330 / 控制箱 289+小件 / 传感器杆 328,329 / 杆顶相机件 / 风琴罩 265)
  2. 两台场景相机从 chassis 移入 lift (物理上立在台面杆顶, 必须随台升降)
  3. arm_base_mount 整条臂挂到 lift 下 (真实结构: 臂装在升降台上)
  4. lift = slide 关节, machine 系轴 (0,0,-1) = 世界 +z; 行程 [0, 0.5] m (0 = CAD 快照高度)
  5. 新增 lift_servo 位置执行器 + home/lift_high 关键帧补 qpos/ctrl
  6. 筐口 basket_site (定义在 lift 上, 随台走)
  7. chassis / lift 两体惯性按 parts.json 体积占比重新合成 (保持原塑料密度标定)

用法: python3 tools/patch_lift_body.py   (幂等性: 检测到 lift body 已存在则拒绝执行)
"""
import json, re, sys
import numpy as np
import mujoco

sys.path.insert(0, '/data/robot_assembly')

XML = '/data/robot_assembly/model/tomato_picker.xml'
TOMATO_DIR = '/data/robot_model_urdf/local_mukr9r1p_iz75v8_urdf_stl'

# ---------------------------------------------------------------- 1. 几何取证
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
chassis_id = m.body('chassis').id

def world_aabb(g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
    return w.min(0), w.max(0)

moved_parts = []          # 台面以上 + 风琴罩
for g in range(m.ngeom):
    if m.geom_bodyid[g] != chassis_id or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if not name.startswith('v_part_'):
        continue
    lo, hi = world_aabb(g)
    if lo[2] >= 0.73:
        moved_parts.append(name[2:])
moved_parts.sort()
print(f'移入 lift 的零件 {len(moved_parts)} 个: {moved_parts}')
moved_parts.append('part_265_up')   # 风琴罩上半段 (下半段留底盘, 见第 2 节)

# 轨道钢管 part_110/149 (906mm 长, Ø40, 沿 x, 驱动轮正下方): 从机器人移出,
# 由环境场景接管 (拉长平铺 + 圆柱碰撞, 见 tools/build_environment.py)
TRACK_PARTS = ['part_110', 'part_149']

# 留守底盘的 part 集合 (供惯性重算)
stay_parts = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] != chassis_id or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if name.startswith('v_part_') and name[2:] not in moved_parts \
            and name[2:] not in TRACK_PARTS and name[2:] != 'part_265':
        stay_parts.append(name[2:])
stay_parts.append('part_265_lo')            # 风琴罩下半段留在底盘
print(f'留在 chassis 的零件 {len(stay_parts)} 个')

# ---------------------------------------------------------------- 2. 惯性合成
# 与 phase_c_build.py 同法: 质量 = 标定塑料密度 x parts.json 体积, 质点置于 bbox 中心
tg = {p['part_id']: p for p in json.load(open(f'{TOMATO_DIR}/parts.json'))['parts']}
# 风琴罩 265 两段化: 下半段留底盘、上半段随台; 升起时中缝由合成伸缩立柱填补
# (CAD 的 418 件里层间只有风琴罩, 没有立柱/剪叉 —— 机构尚未设计, 见 split_bellows)
b265 = tg['part_265']['bbox']
vol265 = tg['part_265']['volume_estimate']
tg['part_265_lo'] = {'volume_estimate': vol265 / 2,
                     'bbox': {'min': [b265['min'][0], b265['min'][1], -564],
                              'max': [b265['max'][0], b265['max'][1], -428]}}
tg['part_265_up'] = {'volume_estimate': vol265 / 2,
                     'bbox': {'min': [b265['min'][0], b265['min'][1], -699],
                              'max': [b265['max'][0], b265['max'][1], -564]}}
um = json.load(open(f'{TOMATO_DIR}/user_model.json'))
ra = next(l for l in um['links'] if l['name'] == 'robot_arm')
RHO = ra['mass'] / sum(tg[p]['volume_estimate'] for p in ra['part_ids'])   # kg/mm^3
print(f'标定塑料密度 {RHO*1e6:.4f} g/cm3')

def composite(part_ids):
    M, c, I, ml = 0.0, np.zeros(3), np.zeros((3, 3)), []
    for pid in part_ids:
        p = tg[pid]
        mi = RHO * p['volume_estimate']
        ci = (np.array(p['bbox']['min']) + np.array(p['bbox']['max'])) / 2000.0
        ml.append((mi, ci)); M += mi; c += mi * ci
    c /= M
    for mi, ci in ml:
        dd = ci - c
        I += mi * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
    return M, c, I

M_stay, c_stay, I_stay = composite(stay_parts)
M_lift, c_lift, I_lift = composite(moved_parts)
print(f'chassis_new: {M_stay:.2f} kg   lift(含台/筐/箱/杆/罩): {M_lift:.2f} kg')

def inertial_line(M, c, I, indent):
    f = lambda a: f'{a:.8g}'
    return (f'{indent}<inertial pos="{f(c[0])} {f(c[1])} {f(c[2])}" mass="{f(M)}" '
            f'fullinertia="{f(I[0,0])} {f(I[1,1])} {f(I[2,2])} {f(I[0,1])} {f(I[0,2])} {f(I[1,2])}"/>')

# ---------------------------------------------------------------- 3. 行级手术
lines = open(XML).read().split('\n')
# 纹理目录: meshdir 只作用于网格, 纹理走 texturedir
for i, ln in enumerate(lines):
    if '<compiler' in ln:
        lines[i] = ln.replace('meshdir="../"', 'meshdir="../" texturedir="../"')
        break

# 3a. 从 chassis 抽走 moved geom 行与两台相机行
moved_lines, cam_lines = [], []
out = []
for ln in lines:
    mm = re.match(r'\s*<geom name="v_part_(\d+)" ', ln)
    if mm and ('part_' + mm.group(1)) in moved_parts:
        moved_lines.append(ln); continue
    if mm and ('part_' + mm.group(1)) in TRACK_PARTS:
        continue                      # 轨道件直接丢弃 (环境场景重建)
    if mm and mm.group(1) == '265':
        continue                      # 265 两段化: 由 3b 显式插入 lo/up 两 geom
    if '<mesh name="v_part_265" ' in ln:
        out.append('    <mesh name="v_part_265_lo" file="meshes/cad/part_265_lo.stl" scale="0.001 0.001 0.001" inertia="shell"/>')
        out.append('    <mesh name="v_part_265_up" file="meshes/cad/part_265_up.stl" scale="0.001 0.001 0.001" inertia="shell"/>')
        continue
    if '<mesh name="v_part_110" ' in ln or '<mesh name="v_part_149" ' in ln:
        continue                      # 对应 mesh 资产声明一并移除
    if 'name="cam_scene_' in ln:
        cam_lines.append(ln); continue
    if 'name="v_part_330" ' in ln:      # 筐网格改纯视觉 (碰撞由 crate_col_* 盒承担)
        ln = ln.replace(' group="1"/>', ' group="1" contype="0" conaffinity="0"/>')
    out.append(ln)
lines = out
assert len(moved_lines) == len(moved_parts), f'{len(moved_lines)} != {len(moved_parts)}'
assert len(cam_lines) == 2, cam_lines

# 3b. 定位 arm_base_mount 子树 (花括号深度计数) 并抽出
i_open = next(i for i, ln in enumerate(lines) if '<body name="arm_base_mount"' in ln)
depth = 0
i_close = None
for i in range(i_open, len(lines)):
    depth += lines[i].count('<body') - lines[i].count('</body>')
    if depth == 0:
        i_close = i
        break
arm_lines = lines[i_open:i_close + 1]
lines = lines[:i_open] + lines[i_close + 1:]
print(f'arm 子树: {len(arm_lines)} 行 (第 {i_open}~{i_close} 行)')

# 3c. 组装 lift body 并插到 chassis 关闭之后 (即被抽走行后的第一个 </body> 之后)
def indent(lns, pad):
    return [pad + ln if ln.strip() else ln for ln in lns]

# 筐口 site: 实测快照世界坐标筐 x[-0.706,-0.360] y[-0.238,0.218] z[0.771,0.921]
# machine 系: z_machine = -(z_world - 0.066379), y_machine = -y_world
site = ('    <site name="basket_site" pos="-0.533 0.010 -0.880" type="cylinder" '
        'size="0.14 0.02" rgba="0.2 0.9 0.2 0.25"/>')
# 筐的凹腔无法用凸包网格做碰撞 -> 5 个隐形碰撞盒 (四壁+底, group 3 不可见), 网格改纯视觉
# 世界系内腔: x[-0.700,-0.366] y[-0.232,0.212], 底 0.771, 壁顶 0.921, 壁厚 6mm
Z_BOT, Z_TOP, T = -(0.777 - 0.066379), -(0.846 - 0.066379), 0.006
crate_boxes = [
    ( -0.533,  0.010, Z_BOT,        0.170, 0.225, T     ),   # 底
    ( -0.703,  0.010, Z_TOP,        T,     0.225, 0.075 ),   # 世界 -x 壁
    ( -0.363,  0.010, Z_TOP,        T,     0.225, 0.075 ),   # 世界 +x 壁
    ( -0.533,  0.235, Z_TOP,        0.170, T,     0.075 ),   # 世界 -y 壁 -> machine +y
    ( -0.533, -0.215, Z_TOP,        0.170, T,     0.075 ),   # 世界 +y 壁 -> machine -y
]
crate_col = [f'    <geom name="crate_col_{i}" type="box" pos="{x} {y} {z}" '
             f'size="{a} {b} {c}" group="3" rgba="0 0 0 0" solref="0.04 1"/>'
             for i, (x, y, z, a, b, c) in enumerate(crate_boxes)]
lift_block = (['  <body name="lift" pos="0 0 0">',
               '    <joint name="lift_joint" type="slide" axis="0 0 -1" limited="true" '
               'range="0 0.5" damping="10"/>',
               inertial_line(M_lift, c_lift, I_lift, '    ')]
              + indent(moved_lines, '  ')
              + indent(cam_lines, '  ')
              + [site]
              + ['    <geom name="v_part_265_up" type="mesh" mesh="v_part_265_up" group="1"/>',
                 '    <geom name="lift_col_inner" type="cylinder" pos="0.002 0.000 -0.3536" '
                 'size="0.042 0.300" rgba="0.60 0.61 0.64 1" contype="0" conaffinity="0"/>']
              + crate_col
              + indent(arm_lines, '  ')
              + ['  </body>'])
# 3c'. 风琴罩下半段 + 伸缩立柱外管 归底盘 (插在下板 263 之后)
for i, ln in enumerate(lines):
    if '<geom name="v_part_263"' in ln:
        lines[i + 1:i + 1] = [
            '    <geom name="v_part_265_lo" type="mesh" mesh="v_part_265_lo" group="1"/>',
            '    <geom name="lift_col_outer" type="cylinder" pos="0.002 0.000 -0.4886" '
            'size="0.055 0.065" rgba="0.42 0.43 0.46 1" contype="0" conaffinity="0"/>']
        break
else:
    sys.exit('未找到 v_part_263 行')

# 插入点 = machine 的闭合 </body> 之前 (lift 必须是 machine 的子 body, 保持翻转坐标系)
for i, ln in enumerate(lines):
    if ln == '  </body>' and lines[i + 1].strip() == '</worldbody>':
        lines = lines[:i] + lift_block + [''] + lines[i:]
        break
else:
    sys.exit('未找到 machine 关闭行')

# 3d. chassis inertial 重算替换
for i, ln in enumerate(lines):
    if re.match(r'\s*<inertial pos=.* mass="85\.96', ln):
        lines[i] = inertial_line(M_stay, c_stay, I_stay, '    ')
        break
else:
    sys.exit('未找到 chassis inertial 行')

# 3e'. 排除 chassis<->lift 碰撞: 风琴罩 265 (随台) 底部插在下板 263 (留底盘) 里 35mm,
#      升降导视作理想连接, 整组排除 (与现有 exclude 风格一致)
for i, ln in enumerate(lines):
    if '<exclude body1="base_link" body2="wrist3_Link"/>' in ln:
        lines[i + 1:i + 1] = ['    <exclude body1="chassis" body2="lift"/>']
        break
else:
    sys.exit('未找到 contact 区末尾 exclude 行')

# 3e. 执行器: 追加 lift_servo
for i, ln in enumerate(lines):
    if '<position name="j6"' in ln:
        lines[i + 1:i + 1] = ['    <position name="lift_servo" joint="lift_joint" kp="20000" '
                              'kv="1400" ctrlrange="0 0.5" forcerange="-2500 2500"/>']
        break
else:
    sys.exit('未找到 j6 执行器行')

# 3f. 关键帧: home 补 lift 位 (joint 序: 10 轮 + lift + 6 臂 + gear + 2 指 = qpos 20 维)
def patch_key(ln, lift_q, lift_c):
    q = ln.split('qpos="')[1].split('"')[0].split()
    assert len(q) == 19, len(q)
    q = q[:10] + [f'{lift_q:.3f}'] + q[10:]
    c = ln.split('ctrl="')[1].split('"')[0].split()
    c = c + [f'{lift_c:.3f}']
    return ln.split('qpos="')[0] + 'qpos="' + ' '.join(q) + '" ctrl="' + ' '.join(c) + '"/>'

out = []
for ln in lines:
    if '<key name="home"' in ln:
        out.append(patch_key(ln, 0.0, 0.0))
    else:
        out.append(ln)
lines = out
# lift_high 关键帧: 在 </keyframe> 前插入 (qpos/ctrl 同 home, 仅 lift=0.45)
home_q = ['0'] * 10 + ['0.450'] + ['-1.590000', '-0.061100', '1.500000', '-1.590000',
           '-1.650000', '3.050000', '-0.474000', '0.009952', '0.009954']
home_c = ['-0.474000', '0.000000', '0.000000', '-1.590000', '-0.061100', '1.500000',
          '-1.590000', '-1.650000', '3.050000', '0.450000']
for i, ln in enumerate(lines):
    if ln.strip() == '</keyframe>':
        lines[i:i] = ['    <key name="lift_high" qpos="' + ' '.join(home_q)
                      + '" ctrl="' + ' '.join(home_c) + '"/>']
        break
else:
    sys.exit('未找到 </keyframe>')

open(XML, 'w').write('\n'.join(lines))
print(f'patch 完成: 共 {len(lines)} 行')
