"""生成番茄温室环境并注入整机模型 tomato_picker.xml (幂等, 可反复重跑)。

内容:
  1. 植株: aoc_tomato_farm 的 vine 随机变体 (asset_extract/out/variants/vN), 每侧 5 株,
     行距 1.6m (y=+-0.8), 株距 0.9m —— 论文 §2 种植参数。
     - assimp DAE->OBJ, 按 usemtl 材质拆分 (vine/leaf/blossom/red/org/grn)
     - 果串: 用 markers.json 的 FRUIT 标记 (truss_id + ripeness) 把果面三角形聚类成串
     - 两段式物理: truss body (freejoint, 果面视觉 + 2 球碰撞 + 人工果柄)
       <weld> 挂在植株上 (active) + <weld> 挂在夹爪 base_link (active=false, 供剪切后搬运)
     - cut_point_{p}_{k} site = 果柄剪切点 (感知定位真值 + 采摘目标)
     - 主茎碰撞圆柱 + 吊蔓线 (纯视觉)
  2. 轨道: CAD 原件 part_110/149 (906mm Ø40 沿 x) 平铺拉长, 圆柱碰撞
  3. 关键帧 home/lift_high 的 qpos 扩写 (追加果串 freejoint 状态)

用法: python3 tools/build_environment.py [--flat-colors]
  --flat-colors: 材质不带纹理 (UV 异常时的退路)
"""
import json, os, re, shutil, struct, subprocess, sys
import numpy as np

sys.path.insert(0, '/data/robot_assembly/tools')
from geomlib import load_stl

XML = '/data/robot_assembly/model/tomato_picker.xml'
OUT_DIR = '/data/robot_assembly/meshes/env'
TEX_DIR = f'{OUT_DIR}/tex'
PLANT_DIR = f'{OUT_DIR}/plants'
FARM = '/home/lcw/aoc_tomato_farm'
CAD_MESH = '/data/robot_model_urdf/local_mukr9r1p_iz75v8_urdf_stl/meshes'
TEX_SRC = f'{FARM}/asset_extract/out'

VARIANTS = [f'v{i}' for i in range(10)]          # 10 株 = 每侧 5 株
X_POS = [-1.8, -0.9, 0.0, 0.9, 1.8]              # 株距 0.9 m
ROW_Y = [0.8, -0.8]                              # 行距 1.6 m
FRUIT_R = 0.014
TRUSS_MASS = 0.30                                # 串重 (论文 250~500g 取中)
PIPE_STL = [('pipe_a', 'part_110'), ('pipe_b', 'part_149')]
PIPE_Y = [0.323, -0.342]                         # 驱动轮接地点世界 y
PIPE_LEN, PIPE_N = 0.906, 9
RIP_TEX = {'red': 'frt2', 'orange': 'frt1', 'green': 'frt4'}
RIP_RGBA = {'red': '0.75 0.09 0.05 1', 'orange': '0.92 0.45 0.10 1',
            'green': '0.35 0.60 0.15 1'}
FLAT = '--flat-colors' in sys.argv

os.makedirs(TEX_DIR, exist_ok=True)
os.makedirs(PLANT_DIR, exist_ok=True)
for t in ['frt1', 'frt2', 'frt4', 'brn1', 'lef1', 'blo1']:
    shutil.copy(f'{TEX_SRC}/{t}.png', TEX_DIR)

# ---------------------------------------------------------------- OBJ 工具
def parse_obj(path):
    vs, vts, mats, cur = [], [], {}, 'default'
    for ln in open(path):
        if ln.startswith('v '):
            vs.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('vt '):
            vts.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith('usemtl '):
            cur = ln.split()[1]
        elif ln.startswith('f '):
            mats.setdefault(cur, []).append(
                [tuple(int(q) if q else 0 for q in tok.split('/')) for tok in ln.split()[1:]])
    return np.array(vs), np.array(vts), mats

def emit_obj(path, vs, vts, faces):
    used_v, used_t = {}, {}
    v_out, f_out = [], []
    for f in faces:
        trip = []
        for vi, ti, ni in f:
            if vi not in used_v:
                used_v[vi] = len(used_v) + 1
                v_out.append('v %f %f %f' % tuple(vs[vi - 1]))
            if ti > 0:
                if ti not in used_t:
                    used_t[ti] = len(used_t) + 1
                    v_out.append('vt %f %f' % tuple(vts[ti - 1]))
                trip.append('%d/%d/%d' % (used_v[vi], used_t[ti], ni))
            else:
                trip.append('%d//%d' % (used_v[vi], ni))
        f_out.append('f ' + ' '.join(trip))
    open(path, 'w').write('\n'.join(v_out + f_out) + '\n')

def face_verts_idx(faces):
    return np.array([[tok[0] for tok in f] for f in faces]) - 1

def face_centroids(vs, faces):
    return vs[face_verts_idx(faces)].mean(1)

def quat_z_to(d):
    """(0,0,1)->d 的最小旋转四元数 wxyz"""
    d = d / np.linalg.norm(d)
    ax = np.cross([0, 0, 1], d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return '1 0 0 0' if d[2] > 0 else '0 1 0 0'
    ax /= s
    ang = np.arccos(np.clip(d[2], -1, 1))
    w, h = np.cos(ang / 2), np.sin(ang / 2)
    return '%.5f %.5f %.5f %.5f' % (float(w), float(ax[0]) * float(h),
                                    float(ax[1]) * float(h), float(ax[2]) * float(h))

# ---------------------------------------------------------------- 单株处理
def prep_variant(vname):
    vdir = f'{FARM}/asset_extract/out/variants/{vname}'
    os.makedirs('/tmp/mjobj', exist_ok=True)
    obj = f'/tmp/mjobj/{vname}.obj'
    subprocess.run(['assimp', 'export', f'{vdir}/tomato_vine_repo_{vname}.dae', obj],
                   check=True, capture_output=True)
    vs, vts, mats = parse_obj(obj)
    markers = json.load(open(f'{vdir}/markers.json'))
    trusses = {}
    for mk in markers:
        trusses.setdefault(mk['truss_id'], []).append(mk)
    for k in trusses:                       # 先存原始 dae 系 marker, 坐标映射后再算质心
        ms = trusses[k]
        trusses[k] = {'raw': np.array([m['translation'] for m in ms]),
                      'rip': ms[0]['ripeness']}
    # 坐标映射自动判定 (dae Z-up vs assimp 导出的轴序): marker 与果面顶点最近邻
    fruit_mats = [m for m in mats if m in RIP_TEX or m in ('red', 'org', 'grn')]
    fv = np.concatenate([vs[face_verts_idx(mats[m])].reshape(-1, 3) for m in fruit_mats])[::37]
    allmk = np.array([m['translation'] for m in markers])
    cand = {'xyz': lambda p: p,
            'xzy': lambda p: np.array([p[0], p[2], p[1]]),
            'x-z_y': lambda p: np.array([p[0], p[2], -p[1]]),
            'x_y-z': lambda p: np.array([p[0], -p[1], p[2]])}
    best_name, best_d = None, 1e9
    for nm, fn in cand.items():
        q = np.array([fn(p) for p in allmk])
        d = np.min(np.linalg.norm(fv[None, :] - q[:, None], axis=2), axis=1).mean()
        if d < best_d:
            best_name, best_d = nm, d
    print(f'{vname}: 坐标映射 {best_name}, marker 偏差 {best_d*1000:.1f} mm')
    # 顶点保持 OBJ 系 (Y-up) 不动; marker 映射到 OBJ 系参与聚类/主茎拟合。
    # 植株 body 用 quat 把 OBJ 系转到世界系: world = R @ obj, R = 绕x +90° -> (x, -z, y)
    for k in trusses:
        P = np.array([cand[best_name](p) for p in trusses[k]['raw']])
        trusses[k]['markers'] = P
        trusses[k]['centroid'] = P.mean(0)
        trusses[k]['radius'] = np.linalg.norm(P - P.mean(0), axis=1).max() + FRUIT_R

    # 果面 -> 最近果串
    flist, ftag = [], []
    for m in fruit_mats:
        flist += mats[m]
        ftag += [m] * len(mats[m])
    fcent = face_centroids(vs, flist)
    ctd = np.array([trusses[k]['centroid'] for k in sorted(trusses)])
    tr_of_face = np.argmin(np.linalg.norm(fcent[:, None] - ctd[None], axis=2), axis=1)

    # 主茎轴 (vine 材质 PCA, 3x3 协方差特征分解)
    vv = vs[face_verts_idx(mats['vine'])].reshape(-1, 3)
    c0 = vv.mean(0)
    w = np.linalg.eigh((vv - c0).T @ (vv - c0))[1][:, -1]
    s = (vv - c0) @ w
    axis_a, axis_b = c0 + w * np.percentile(s, 1), c0 + w * np.percentile(s, 99)

    out = {'geoms': [], 'trusses': [], 'axis': (axis_a, axis_b)}
    for mname in ['vine', 'leaf', 'blossom']:
        if mname not in mats:
            continue
        emit_obj(f'{PLANT_DIR}/{vname}_{mname}.obj', vs, vts, mats[mname])
        out['geoms'].append((f'{vname}_{mname}', mname))
    for ki, k in enumerate(sorted(trusses)):
        t = trusses[k]
        sel = tr_of_face == ki
        faces = [f for f, s2 in zip(flist, sel) if s2]
        ab = axis_b - axis_a
        ctr = t['centroid']
        tt = np.clip(np.dot(ctr - axis_a, ab) / np.dot(ab, ab), 0, 1)
        attach = axis_a + tt * ab
        dirc = ctr - attach
        dist = max(np.linalg.norm(dirc), 1e-9)
        dirc /= dist
        cut = attach + dirc * max(dist * 0.30, 0.03)   # 果柄内 1/3 处 (论文式剪切点)
        # 果串 body 原点 = 剪切点 (torquescale=0 点抓的枢轴): 网格/球均相对剪切点,
        # 否则 MuJoCo 的"按授权位置补偿"会把渲染/碰撞放到双倍偏移处
        emit_obj(f'{PLANT_DIR}/{vname}_truss{k}.obj', vs - cut, vts, faces)
        # 碰撞: 每颗果实一个真尺寸球 (按 marker), 直径小于指间距, 不会被指笼关住
        # 碰撞球 0.75x 果实尺寸: 减小指/果接触卡滞 (仿真标准: 碰撞体小于视觉体),
        # 真实果实柔性可让位; 全尺寸刚球会把夹爪卡在果簇外
        spheres = [(rel, FRUIT_R * 1.45) for rel in (t['markers'] - cut)]
        out['trusses'].append(dict(k=k, pos=cut, mesh=f'{vname}_truss{k}', rip=t['rip'],
                                   spheres=spheres, cut=cut, attach=attach, dirc=dirc,
                                   ctr_rel=ctr - cut))
    return out

# ---------------------------------------------------------------- 材质声明
def mat_block(name, png, rgba):
    if FLAT:
        return [f'    <material name="{name}" rgba="{rgba}"/>']
    return [f'    <texture name="tex_{png}" type="2d" file="meshes/env/tex/{png}.png"/>',
            f'    <material name="{name}" texture="tex_{png}"/>']

assets = []
assets += mat_block('mat_red', 'frt2', RIP_RGBA['red'])
assets += mat_block('mat_orange', 'frt1', RIP_RGBA['orange'])
assets += mat_block('mat_green', 'frt4', RIP_RGBA['green'])
assets += mat_block('mat_vine', 'brn1', '0.36 0.25 0.12 1')
assets += mat_block('mat_leaf', 'lef1', '0.22 0.45 0.15 1')
assets += mat_block('mat_blossom', 'blo1', '0.95 0.85 0.30 1')

# ---------------------------------------------------------------- 植株/果串
plant_bodies, truss_bodies, welds, truss_qpos = [], [], [], []
truss_idx = 0
for p, vname in enumerate(VARIANTS):
    px, py = X_POS[p % 5], ROW_Y[p // 5]
    out = prep_variant(vname)
    g_lines = []
    for mesh, mat in out['geoms']:
        assets.append(f'    <mesh name="m_{mesh}" file="meshes/env/plants/{mesh}.obj"/>')
        g_lines.append(f'      <geom name="g_{mesh}" type="mesh" mesh="m_{mesh}" '
                       f'material="mat_{mat}" contype="0" conaffinity="0"/>')
    a_a, a_b = out['axis']
    mid, half = (a_a + a_b) / 2, np.linalg.norm(a_b - a_a) / 2
    q = quat_z_to(a_b - a_a)
    g_lines.append(f'      <geom name="g_{vname}_stem" type="cylinder" pos="{mid[0]:.3f} {mid[1]:.3f} {mid[2]:.3f}" '
                   f'quat="{q}" size="0.02 {half:.3f}" rgba="0.36 0.25 0.12 1" contype="0" conaffinity="0"/>')
    top = a_b if a_b[1] > a_a[1] else a_a            # OBJ 系里 "上" 是 +y
    if top[1] < 2.35:                                # 藤顶低于吊线设计高才补吊蔓线
        g_lines.append(f'      <geom name="g_{vname}_line" type="cylinder" pos="{top[0]:.3f} '
                       f'{top[1] + (2.4 - top[1]) / 2:.3f} {top[2]:.3f}" size="0.002 {(2.4 - top[1]) / 2:.3f}" '
                       f'quat="{quat_z_to([0, 1, 0])}" rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>')
    print(f'  {vname}: 藤顶 OBJ y={top[1]:.2f} (世界高≈{top[1]:.2f}m)')
    # 植株 body: quat 把 OBJ 系 (Y-up) 转到世界系 (Z-up), world = R@obj, R(x,y,z)=(x,-z,y)
    body_lines = [f'  <body name="plant_{p}" pos="{px} {py} 0" quat="0.70710678 0.70710678 0 0">'] + g_lines
    for t in out['trusses']:
        sp = '\n      '.join(
            f'<geom name="g_tr{truss_idx}_s{s}" type="sphere" pos="{o[0]:.3f} {o[1]:.3f} {o[2]:.3f}" '
            f'size="{r:.3f}" contype="2" conaffinity="2" rgba="0 0 0 0"/>'
            for s, (o, r) in enumerate(t['spheres']))
        ped_a = np.zeros(3)                     # 柄根 = 剪切点 (body 原点)
        ped_b = t['ctr_rel']                    # 柄梢: 果簇质心方向
        if np.linalg.norm(ped_a - ped_b) < 0.02:   # 极短果柄: 保证胶囊最小长度
            ped_a = -t['dirc'] * 0.02
        assets.append(f'    <mesh name="m_{t["mesh"]}" file="meshes/env/plants/{t["mesh"]}.obj"/>')
        tx, ty, tz = t['pos']
        # freejoint 必须是顶层 body: 世界位姿 = plant_pos + R@obj_pos, 姿态 = R
        truss_bodies.append(
            f'  <body name="truss_{p}_{t["k"]}" pos="{px + tx:.4f} {py - tz:.4f} {ty:.4f}" '
            f'quat="0.70710678 0.70710678 0 0">\n'
            f'    <joint name="j_truss_{p}_{t["k"]}" type="free" damping="0.02"/>\n'
            f'    <inertial pos="{t["ctr_rel"][0]:.4f} {t["ctr_rel"][1]:.4f} {t["ctr_rel"][2]:.4f}" '
            f'mass="{TRUSS_MASS}" diaginertia="0.0015 0.0015 0.0008"/>\n'
            f'    <geom name="g_tr{truss_idx}_mesh" type="mesh" mesh="m_{t["mesh"]}" '
            f'material="mat_{t["rip"]}" '
            f'contype="0" conaffinity="0"/>\n'
            f'    {sp}\n'
            f'    <geom name="g_tr{truss_idx}_ped" type="capsule" '
            f'fromto="{ped_a[0]:.3f} {ped_a[1]:.3f} {ped_a[2]:.3f} '
            f'{ped_b[0]:.3f} {ped_b[1]:.3f} {ped_b[2]:.3f}" size="0.004" '
            f'rgba="0.36 0.25 0.12 1" contype="0" conaffinity="0"/>\n'
            f'  </body>')
        cut = t['cut']
        body_lines.append(f'    <site name="cut_p{p}_t{t["k"]}" pos="{cut[0]:.4f} {cut[1]:.4f} '
                          f'{cut[2]:.4f}" size="0.01" rgba="1 0 1 0.25"/>')
        att = t['attach']
        body_lines.append(f'    <site name="attach_p{p}_t{t["k"]}" pos="{att[0]:.4f} {att[1]:.4f} '
                          f'{att[2]:.4f}" size="0.008" rgba="0 1 1 0.25"/>')
        aa = t['attach'] - t['dirc'] * 0.015
        ab2 = t['cut'] + t['dirc'] * 0.005
        body_lines.append(f'    <geom name="g_{vname}_stub_{t["k"]}" type="capsule" '
                          f'fromto="{aa[0]:.3f} {aa[1]:.3f} {aa[2]:.3f} '
                          f'{ab2[0]:.3f} {ab2[1]:.3f} {ab2[2]:.3f}" size="0.004" '
                          f'rgba="0.36 0.25 0.12 1" contype="0" conaffinity="0"/>')
        welds.append(f'  <weld name="hold_{p}_{t["k"]}" body1="plant_{p}" body2="truss_{p}_{t["k"]}" '
                     f'active="true" solref="0.005 1"/>')
        welds.append(f'  <weld name="grip_{p}_{t["k"]}" body1="base_link" body2="truss_{p}_{t["k"]}" '
                     f'active="false" solref="0.01 1"/>')
        # freejoint 关键帧要世界位姿: pos_w = plant_pos + R@obj_pos, quat_w = R
        truss_qpos.append((f'{px + tx:.4f}', f'{py - tz:.4f}', f'{ty:.4f}',
                           '0.7071068 0.7071068 0 0'))
        truss_idx += 1
    plant_bodies.append('\n'.join(body_lines) + '\n  </body>')

# ---------------------------------------------------------------- 轨道
for nm, pid in PIPE_STL:
    # 预居中: 顶点减去顶点质心 -> 编译后 mesh_pos≈0, geom pos 即为管子中心
    # (V 系授权偏移若不消除, 平铺会整体埋入地下 0.4m 且 y 镜像)
    v, faces = load_stl(f'{CAD_MESH}/{pid}_solid_{pid[5:]}.stl')
    ctr = v[np.unique(faces.ravel())].mean(0)
    vc = v - ctr
    tris = vc[faces]
    with open(f'{OUT_DIR}/{nm}.stl', 'wb') as fh:
        fh.write(b'\0' * 80)
        fh.write(struct.pack('<I', len(tris)))
        for tri in tris:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.zeros(3)
            fh.write(struct.pack('<3f', *nn.astype(np.float32)))
            for pnt in tri:
                fh.write(struct.pack('<3f', *pnt.astype(np.float32)))
            fh.write(struct.pack('<H', 0))
    assets.append(f'    <mesh name="m_{nm}" file="meshes/env/{nm}.stl" scale="0.001 0.001 0.001"/>')
pipe_lines = []
pipe_lines.append('  <geom name="g_catch_plane" type="plane" pos="0 0 -0.5" size="12 12 0.1" '
                  'contype="2" conaffinity="2" rgba="0 0 0 0"/>')   # 只接果串 (contype 2)
for side, (nm, pyw) in enumerate(zip([p[0] for p in PIPE_STL], PIPE_Y)):
    for k in range(PIPE_N):
        xk = -PIPE_LEN * PIPE_N / 2 + PIPE_LEN * (k + 0.5)
        pipe_lines.append(f'  <geom name="g_{nm}_{k}" type="mesh" mesh="m_{nm}" '
                          f'pos="{xk:.4f} {pyw} -0.02" rgba="0.55 0.56 0.60 1" '
                          f'contype="0" conaffinity="0"/>')
    L = PIPE_LEN * PIPE_N
    pipe_lines.append(f'  <geom name="g_pipe_col_{side}" type="cylinder" pos="0 {pyw} -0.035" '
                      f'size="0.02 {L / 2:.3f}" euler="0 90 0" rgba="0 0 0 0" contype="0" conaffinity="0"/>')

# ---------------------------------------------------------------- 注入
src = open(XML).read()

def inject(text, tag, anchor, payload):
    B, E = f'<!-- ENV-{tag}-BEGIN (tools/build_environment.py 生成, 勿手改) -->', f'<!-- ENV-{tag}-END -->'
    blk = f'{B}\n{payload}\n{E}'
    if B in text:
        s, e = text.index(B), text.index(E) + len(E)
        return text[:s] + blk + text[e:]
    return text.replace(anchor, blk + '\n' + anchor, 1)

src = inject(src, 'ASSET', '  </asset>', '\n'.join(assets))
src = inject(src, 'WORLD', '  <body name="machine"',
             '\n'.join(pipe_lines + plant_bodies + truss_bodies))
src = inject(src, 'EQ', '  </equality>', '\n'.join(welds))

def extend_keys(text):
    def fix(m):
        head, q, ctrl = m.group(1), m.group(2).split(), m.group(3).split()
        # 展平: 每串 = pos(3) + quat 字符串再拆 4 个 token (共 7)
        tvals = []
        for vals in truss_qpos:
            tvals += [vals[0], vals[1], vals[2]] + vals[3].split()
        if len(q) == 20:
            # 环境 WORLD 块注入在 machine 之前: 果串 freejoint 的 qpos 排在最前 280 位
            q = tvals + q
        elif len(q) == 20 + len(tvals):
            # 已扩写过 (幂等重跑): 原位更新果串段 (几何变了必须刷新)
            q[:len(tvals)] = tvals
        return f'{head}qpos="{" ".join(q)}" ctrl="{" ".join(ctrl)}"/>'
    return re.sub(r'(<key name="(?:home|lift_high)" )qpos="([^"]*)" ctrl="([^"]*)"/>', fix, text)

src = extend_keys(src)
open(XML, 'w').write(src)
print(f'环境已注入: {truss_idx} 果串 / {len(VARIANTS)} 株 / 轨道 {PIPE_N * 2} 段')
