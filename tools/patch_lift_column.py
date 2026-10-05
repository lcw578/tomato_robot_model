"""伸缩立柱 + 风琴罩两段化 (对"已打过 lift 补丁"的 tomato_picker.xml 操作, 幂等)。

背景: CAD 的 418 件里, 底盘顶板(263)与台面(264)之间只有风琴罩(265),
没有立柱/剪叉 —— 升降机构尚未设计。整段风琴罩挂台的结果是升起时裙罩
与底盘脱开, 视觉上台面悬空。

本补丁:
  1. 风琴罩网格按世界 z=0.6305 切成两半 (tools 生成的 part_265_lo/up.stl):
     下半段挂 chassis, 上半段挂 lift (升起时裙罩从中缝分开);
  2. 合成两级伸缩立柱: 外管 Ø110 挂 chassis (世界 0.49~0.62),
     内管 Ø85 挂 lift (世界 0.12~0.72, 全行程下探入外管, 不悬空);
  3. 按体积对半重分 chassis/lift 两体惯性 (265 质量各半, 质心用半件 bbox)。

用法: python3 tools/patch_lift_column.py   (幂等: 检测 lift_col_inner 已存在则跳过)
前置: meshes/cad/part_265_lo.stl, part_265_up.stl (由 part_265 按半切分, 已入库)
"""
import json, re, sys
import numpy as np

sys.path.insert(0, '/data/robot_assembly')

XML = '/data/robot_assembly/model/tomato_picker.xml'
TOMATO_DIR = '/data/robot_model_urdf/local_mukr9r1p_iz75v8_urdf_stl/'
SPLIT_Z = -564.1          # V 系切面 = 世界 0.6305

# 伸缩立柱 (machine 系坐标): 内管挂 lift、足够长, 全行程 [0,0.5]m 下端始终插在外管内
COL_OUTER = '    <geom name="lift_col_outer" type="cylinder" pos="0.002 0.000 -0.4886" ' \
            'size="0.055 0.065" rgba="0.42 0.43 0.46 1" contype="0" conaffinity="0"/>'
COL_INNER = '    <geom name="lift_col_inner" type="cylinder" pos="0.002 0.000 -0.3536" ' \
            'size="0.042 0.300" rgba="0.60 0.61 0.64 1" contype="0" conaffinity="0"/>'

src = open(XML).read()
if 'lift_col_inner' in src:
    print('已打过补丁, 跳过')
    sys.exit(0)

# ---------------------------------------------------------------- 惯性重算所需
tg = {p['part_id']: p for p in json.load(open(f'{TOMATO_DIR}parts.json'))['parts']}
um = json.load(open(f'{TOMATO_DIR}user_model.json'))
ra = next(l for l in um['links'] if l['name'] == 'robot_arm')
RHO = ra['mass'] / sum(tg[p]['volume_estimate'] for p in ra['part_ids'])   # kg/mm^3
b265 = tg['part_265']['bbox']
vol265 = tg['part_265']['volume_estimate']
halves = {
    'part_265_lo': {'volume_estimate': vol265 / 2,
                    'bbox': {'min': [b265['min'][0], b265['min'][1], -564],
                             'max': [b265['max'][0], b265['max'][1], -428]}},
    'part_265_up': {'volume_estimate': vol265 / 2,
                    'bbox': {'min': [b265['min'][0], b265['min'][1], -699],
                             'max': [b265['max'][0], b265['max'][1], -564]}},
}
tg.update(halves)

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

def inertial_line(M, c, I, indent):
    f = lambda a: f'{a:.8g}'
    return (f'{indent}<inertial pos="{f(c[0])} {f(c[1])} {f(c[2])}" mass="{f(M)}" '
            f'fullinertia="{f(I[0,0])} {f(I[1,1])} {f(I[2,2])} '
            f'{f(I[0,1])} {f(I[0,2])} {f(I[1,2])}"/>')

# 从当前 XML 归属读取两体的 v_part 集合 (chassis 段 vs lift 段)
lift_i = src.index('<body name="lift"')
chassis_sect = src[:lift_i]
lift_sect = src[lift_i:src.index('</equality>')]
stay = re.findall(r'<geom name="v_part_(\d+)" ', chassis_sect)
lifted = re.findall(r'<geom name="v_part_(\d+)" ', lift_sect)
stay_ids = ['part_' + s for s in stay] + ['part_265_lo']
lift_ids = ['part_' + s for s in lifted if s != '265'] + ['part_265_up']
M_s, c_s, I_s = composite(stay_ids)
M_l, c_l, I_l = composite(lift_ids)
print(f'chassis: {M_s:.2f} kg (含罩下半)   lift: {M_l:.2f} kg (含罩上半)')

# ---------------------------------------------------------------- 行级手术
lines = src.split('\n')

# 1) mesh 资产: v_part_265 -> 265_lo/265_up
out = []
for ln in lines:
    if '<mesh name="v_part_265" ' in ln:
        out.append('    <mesh name="v_part_265_lo" file="meshes/cad/part_265_lo.stl" '
                   'scale="0.001 0.001 0.001" inertia="shell"/>')
        out.append('    <mesh name="v_part_265_up" file="meshes/cad/part_265_up.stl" '
                   'scale="0.001 0.001 0.001" inertia="shell"/>')
        continue
    out.append(ln)
lines = out

# 2) lift body 里的整段 265 geom 行 -> 换成上半段 geom + 内管
out = []
for ln in lines:
    if '<geom name="v_part_265" ' in ln:
        out.append('    <geom name="v_part_265_up" type="mesh" mesh="v_part_265_up" group="1"/>')
        out.append(COL_INNER)
        continue
    out.append(ln)
lines = out

# 3) chassis: 下板 263 之后插入 下半段罩 + 外管
for i, ln in enumerate(lines):
    if '<geom name="v_part_263"' in ln:
        lines[i + 1:i + 1] = ['    <geom name="v_part_265_lo" type="mesh" mesh="v_part_265_lo" group="1"/>',
                              COL_OUTER]
        break
else:
    sys.exit('未找到 v_part_263 行')

# 4) 两侧 inertial 重写: 只替换 chassis 自身与 lift 自身的第一条 <inertial>
out = []
lift_zone = False
done_ch = done_lf = False
depth = 0
for ln in lines:
    if '<body name="lift"' in ln:
        lift_zone = True
    if ln.strip().startswith('<inertial ') and not (done_ch and done_lf):
        if lift_zone and not done_lf:
            out.append(inertial_line(M_l, c_l, I_l, '    '))
            done_lf = True
            continue
        if not lift_zone and not done_ch:
            out.append(inertial_line(M_s, c_s, I_s, '    '))
            done_ch = True
            continue
    out.append(ln)
assert done_ch and done_lf, 'inertial 替换不完整'

open(XML, 'w').write('\n'.join(out))
print(f'伸缩立柱 + 风琴罩两段化完成: chassis {len(stay_ids)} 件 / lift {len(lift_ids)} 件')
