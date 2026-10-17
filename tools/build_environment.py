"""生成番茄温室环境并注入整机模型 tomato_picker.xml (幂等, 可反复重跑)。

内容:
  1. 植株: aoc_tomato_farm 的 vine 随机变体 (asset_extract/out/variants/vN), 每侧 5 株,
     行距 1.6m (y=+-0.8), 株距 0.9m —— 论文 §2 种植参数。
     - assimp DAE->OBJ, 按 usemtl 材质拆分 (vine/leaf/blossom/red/org/grn)
     - 果串: 用 markers.json 的 FRUIT 标记 (truss_id + ripeness) 把果面三角形聚类成串
     - 两段式物理: truss body (freejoint, 果面视觉 + 2 球碰撞 + 人工果柄)
       <weld> 挂在植株上 (active) + <weld> 挂在夹爪 base_link (active=false, 供剪切后搬运)
     - cut_p{p}_t{k} site = 果柄剪切点 (感知定位真值 + 采摘目标)。
       规则 v2 (2026-10-12): 以"簇顶挂结点"为锚 (真实果柄从簇顶接出), 沿其主茎
       投影方向停在果簇网格之外 (到果面净距 >=1cm, 离主茎表面 >=2cm); 顶果贴茎
       的串退回旧 55% 规则 (资产受限例外, 见 model_notes)
     - 主茎碰撞圆柱 + 吊蔓线 (纯视觉)
  2. 轨道: CAD 原件 part_110/149 (906mm Ø40 沿 x) 平铺拉长, 圆柱碰撞
  3. 关键帧 home/lift_high 的 qpos 扩写 (追加果串 freejoint 状态)

用法: python3 tools/build_environment.py [--flat-colors]
  --flat-colors: 材质不带纹理 (UV 异常时的退路)
"""
import heapq, json, os, re, shutil, struct, subprocess, sys
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
STEM_R = 0.022                   # 主茎碰撞胶囊半径 (藤轴拟合)
CUT_FRUIT_CLEAR = 0.010          # 剪切停点到果面网格净距目标 (扫描)
CUT_STEM_CLEAR = 0.020           # 剪切停点离主茎表面下限 (硬约束)
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


def flood_rachis(vine_faces, vine_cent, seeds, blocked, budget):
    """簇内穗轴面选取 (方案 B, 2026-10-17): 在 vine 面邻接图上从种子 Dijkstra 泛洪,
    权重 = 相邻面质心距, 超出测地预算停止; blocked 面 (贴主干) 不收不扩。
    返回选中的 vine 面下标集合 —— 这些面从植株网格迁出、随果串剪走 (用户语义:
    "连接一串番茄中间的长柄"要跟着果串进筐)。"""
    nf = len(vine_faces)
    edge2face = {}
    for fi, f in enumerate(vine_faces):
        iv = [tok[0] for tok in f]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (iv[a], iv[b]) if iv[a] < iv[b] else (iv[b], iv[a])
            edge2face.setdefault(key, []).append(fi)
    nbrs = [[] for _ in range(nf)]
    for fl in edge2face.values():
        for i in fl:
            for j in fl:
                if i != j:
                    nbrs[i].append(j)
    dist = {s: 0.0 for s in seeds}
    heap = [(0.0, s) for s in seeds]
    heapq.heapify(heap)
    taken = set()
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist.get(u, np.inf) or blocked[u]:
            continue
        taken.add(u)
        cu = vine_cent[u]
        for v in nbrs[u]:
            if v in taken or blocked[v]:
                continue
            nd = du + float(np.linalg.norm(vine_cent[v] - cu))
            if nd <= budget and nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return taken



def fit_cane(vname, px, py, theta_deg, R=0.06):
    """拟合真实藤轴: 过基点假设线 + 管内点无中心 SVD 精修 (build_vine 的藤为
    从基点出发的直圆柱, theta 来自 params.json; 见 model_notes 2026-10-01)。
    返回 (base, dir, z_top) 世界系; 失败返回 None。"""
    vs = []
    for ln in open(f'{PLANT_DIR}/{vname}_vine.obj'):
        if ln.startswith('v '):
            vs.append([float(x) for x in ln.split()[1:4]])
    if len(vs) < 100:
        return None
    V = np.array(vs)
    W = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)   # OBJ->世界: (x, -z, y)
    W[:, 0] += px
    W[:, 1] += py
    base = np.array([px, py, 0.0])
    th = np.radians(theta_deg)
    d_dir = np.array([np.sin(th), 0.0, np.cos(th)])
    for _ in range(3):
        t = (W - base) @ d_dir
        perp = np.linalg.norm(W - base - np.outer(t, d_dir), axis=1)
        inl = (perp < R) & (t > 0.05)
        if inl.sum() < 100:
            return None
        _, _, vt = np.linalg.svd(W[inl] - base, full_matrices=False)
        d_dir = vt[0]
        if d_dir[2] < 0:
            d_dir = -d_dir
    t = (W - base) @ d_dir
    perp = np.linalg.norm(W - base - np.outer(t, d_dir), axis=1)
    inl = (perp < R) & (t > 0.05)
    z_top = float(W[inl][:, 2].max())
    return base, d_dir, z_top


def pick_cut(markers, faces, vs, cane, px, py, cut_legacy):
    """剪切点 v2 (2026-10-12, 用户语义): 锚定"簇顶挂结点" —— 真实果柄从簇顶接出。
    沿"最顶端果实 -> 真实藤轴"方向, 从 5mm 起扫描, 取首个到果面网格净距 >=
    CUT_FRUIT_CLEAR 的停点; 上界 = 离藤轴表面 CUT_STEM_CLEAR (扫描全程不达标则取
    区间内净距最大点)。顶果贴茎 (无可行段) 或藤轴拟合失败 -> 退回旧 55% 规则。
    参数均在 OBJ 系 (脚注: 世界系 = (px+ox, py-oz, oy)); cane 为世界系 (base,dir,z_top)。
    返回 (cut, mode, clearance): mode ∈ {top, top_best, legacy55}。"""
    if cane is None:
        return cut_legacy, 'legacy55', float('nan')
    m_top = markers[np.argmax(markers[:, 1])]            # obj 系 y = 世界竖直向上
    base, dirv, z_top = cane
    top_w = base + dirv * (z_top / dirv[2])              # 世界系藤顶
    a_o = np.array([base[0] - px, base[2], py - base[1]])   # 藤基 -> obj
    b_o = np.array([top_w[0] - px, top_w[2], py - top_w[1]])
    ab = b_o - a_o
    tproj = np.clip(np.dot(m_top - a_o, ab) / np.dot(ab, ab), 0, 1)
    att_top = a_o + tproj * ab                           # 簇顶在藤轴上的投影
    vec = att_top - m_top
    L = np.linalg.norm(vec)
    upper = L - STEM_R - CUT_STEM_CLEAR
    if upper >= 0.005:
        u = vec / L
        V = vs[face_verts_idx(faces)].reshape(-1, 3)     # 该串果面顶点 (压平!)
        if os.environ.get('CUTDBG'):
            print('  [cutdbg] m_top=%s u=%s L=%.3f upper=%.3f |V|=%s len(vs)=%d len(faces)=%d' %
                  (np.round(m_top, 3), np.round(u, 3), L, upper, V.shape, len(vs), len(faces)))
        best = None
        for it, tv in enumerate(np.arange(0.005, upper + 1e-9, 0.005)):
            p = m_top + u * tv
            c = float(np.linalg.norm(V - p, axis=1).min())
            if os.environ.get('CUTDBG') and it == 0:
                j = int(np.argmin(np.linalg.norm(V - p, axis=1)))
                print('     t=0.005 p=%s 最近顶点 idx=%d pos=%s d=%.4f' %
                      (np.round(p, 3), j, np.round(V[j], 3), c))
                print('     V bbox min=%s max=%s' % (np.round(V.min(0), 3), np.round(V.max(0), 3)))
            if os.environ.get('CUTDBG') and (it % 20 == 0 or c >= CUT_FRUIT_CLEAR):
                print('     t=%.3f c=%.4f' % (tv, c))
            if c >= CUT_FRUIT_CLEAR:
                return p, 'top', c
            if best is None or c > best[1]:
                best = (p, c)
        if best is not None and best[1] >= 0.0:
            return best[0], 'top_best', best[1]
    return cut_legacy, 'legacy55', float('nan')          # 顶果贴茎 (资产受限)


# ---------------------------------------------------------------- 单株处理
def prep_variant(vname, px, py):
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
    # 真实藤轴 (世界系; 上面 vine obj 已落盘): 剪切点 v2 规则用它做"离主茎"约束
    _pr = json.load(open(f'{vdir}/params.json'))
    cane = fit_cane(vname, px, py, _pr['theta_deg'])
    if cane is None:
        print(f'  {vname}: 藤轴拟合失败, 剪切点退回 55% 规则')
    out['cane'] = cane
    # 簇内穗轴迁移 (方案 B) 上下文: vine 面质心 + 主干屏蔽掩码 (离藤轴表面
    # <6mm 的面不收不扩 —— 贴茎例外族因此自然少迁/不迁, 主干永不被搬走)
    vine_faces = mats.get('vine', [])
    vine_sel_all = set()
    vine_ctx = None
    if vine_faces and cane is not None:
        vine_cent = face_centroids(vs, vine_faces)
        base, dirv, z_top = cane
        top_w = base + dirv * (z_top / dirv[2])
        a_o = np.array([base[0] - px, base[2], py - base[1]])
        b_o = np.array([top_w[0] - px, top_w[2], py - top_w[1]])
        ab = b_o - a_o
        tt = np.clip((vine_cent - a_o) @ ab / (ab @ ab), 0, 1)
        d_cane = np.linalg.norm(vine_cent - (a_o + np.outer(tt, ab)), axis=1)
        vine_ctx = (vine_cent, d_cane < STEM_R + 0.006)
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
        cut_legacy = attach + dirc * max(dist * 0.55, 0.03)   # 旧规则: 55% (贴茎例外族兜底)
        cut, cut_mode, cut_clear = pick_cut(t['markers'], faces, vs, cane,
                                            px, py, cut_legacy)   # v2: 簇顶锚点
        if os.environ.get('CUTDBG'):
            print('  [cutdbg] %s_t%d: mode=%s clear=%s' % (
                vname, k, cut_mode,
                'nan' if cut_clear != cut_clear else '%.4f' % cut_clear))
        # 簇内穗轴面选取 (方案 B): 种子 = 簇球内 ∪ 果柄走廊 (cut->质心 12mm 内),
        # 均须在切点靠果簇一侧; 泛洪测地预算 = |质心-cut| + 4cm; 跨串去重。
        n_rachis = 0
        if vine_ctx is not None:
            vine_cent, cane_blocked = vine_ctx
            R_cl = t['radius']
            ctr = t['centroid']
            d_ctr = np.linalg.norm(vine_cent - ctr, axis=1)
            side = (vine_cent - cut) @ (ctr - cut) > 0
            seg = ctr - cut
            L_seg = float(np.linalg.norm(seg))
            tt2 = np.clip((vine_cent - cut) @ seg / max(L_seg ** 2, 1e-12), 0, 1)
            d_seg = np.linalg.norm(vine_cent - (cut + np.outer(tt2, seg)), axis=1)
            seeds = np.where(((d_ctr < R_cl + 0.03) | (d_seg < 0.012)) & side)[0]
            if len(seeds):
                taken = flood_rachis(vine_faces, vine_cent,
                                     [int(s) for s in seeds], cane_blocked,
                                     L_seg + 0.04)
                taken -= vine_sel_all
                vine_sel_all |= taken
                if taken:
                    idx = sorted(taken)
                    emit_obj(f'{PLANT_DIR}/{vname}_rachis{k}.obj', vs - cut, vts,
                             [vine_faces[i] for i in idx])
                    n_rachis = len(idx)
        # 果串 body 原点 = 剪切点 (torquescale=0 点抓的枢轴): 网格/球均相对剪切点,
        # 否则 MuJoCo 的"按授权位置补偿"会把渲染/碰撞放到双倍偏移处
        emit_obj(f'{PLANT_DIR}/{vname}_truss{k}.obj', vs - cut, vts, faces)
        # 碰撞: 每颗果实一个真尺寸球 (按 marker), 直径小于指间距, 不会被指笼关住
        # 碰撞球 0.75x 果实尺寸: 减小指/果接触卡滞 (仿真标准: 碰撞体小于视觉体),
        # 真实果实柔性可让位; 全尺寸刚球会把夹爪卡在果簇外
        spheres = [(rel, FRUIT_R * 1.45) for rel in (t['markers'] - cut)]
        out['trusses'].append(dict(k=k, pos=cut, mesh=f'{vname}_truss{k}', rip=t['rip'],
                                   spheres=spheres, cut=cut, attach=attach, dirc=dirc,
                                   ctr_rel=ctr - cut, cut_mode=cut_mode,
                                   cut_clear=cut_clear, rachis=n_rachis))
    if vine_sel_all:
        keep = [f for i, f in enumerate(vine_faces) if i not in vine_sel_all]
        emit_obj(f'{PLANT_DIR}/{vname}_vine.obj', vs, vts, keep)
        print(f'  {vname}: 簇内穗轴面迁出 {len(vine_sel_all)}/{len(vine_faces)} ('
              + ', '.join(f"t{t['k']}:{t['rachis']}" for t in out['trusses'] if t['rachis'])
              + ')')
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
cut_log = []                      # (串名, 规则模式, 到果面净距)
world_extra = []   # 世界级: 藤胶囊 + 吊蔓线
truss_idx = 0
for p, vname in enumerate(VARIANTS):
    px, py = X_POS[p % 5], ROW_Y[p // 5]
    out = prep_variant(vname, px, py)
    cut_log += [(f'{p}_{t["k"]}', t['cut_mode'], t['cut_clear']) for t in out['trusses']]
    # 真实藤轴 (prep_variant 内已拟合) -> 世界级碰撞胶囊 (r=0.022, 透明) + 吊蔓线
    import json as _json2
    _pr = _json2.load(open(f'{FARM}/asset_extract/out/variants/{vname}/params.json'))
    fit = out['cane']
    if fit is not None:
        base, dirv, z_top = fit
        top_pt = base + dirv * (z_top / dirv[2])
        world_extra.append(
            f'  <geom name="g_v{p}_stem" type="capsule" '
            f'fromto="{base[0]:.4f} {base[1]:.4f} 0 {top_pt[0]:.4f} {top_pt[1]:.4f} {top_pt[2]:.4f}" '
            f'size="{STEM_R}" rgba="0 0 0 0" contype="4" conaffinity="1"/>')
        if z_top < 2.35:
            world_extra.append(
                f'  <geom name="g_v{p}_line" type="cylinder" '
                f'pos="{top_pt[0]:.3f} {top_pt[1]:.3f} {(z_top + 2.4) / 2:.3f}" '
                f'size="0.002 {(2.4 - z_top) / 2:.3f}" rgba="0.25 0.25 0.25 1" '
                f'contype="0" conaffinity="0"/>')
        print(f'  {vname}: 藤轴拟合 theta_fit={np.degrees(np.arccos(dirv[2])):.1f}°(参数{_pr["theta_deg"]:.1f}°) z_top={z_top:.2f}')
    else:
        print(f'  {vname}: 藤轴拟合失败! (无碰撞胶囊)')
    g_lines = []
    for mesh, mat in out['geoms']:
        assets.append(f'    <mesh name="m_{mesh}" file="meshes/env/plants/{mesh}.obj"/>')
        g_lines.append(f'      <geom name="g_{mesh}" type="mesh" mesh="m_{mesh}" '
                       f'material="mat_{mat}" contype="0" conaffinity="0"/>')
    a_a, a_b = out['axis']
    # 主茎碰撞胶囊与世界系吊蔓线: 移到主循环按"真实藤轴"拟合生成 (见 fit_cane)
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
        # 簇内穗轴网格 (方案 B): 随果串剪走, 纯视觉 (mat_vine), 不参与碰撞
        if t.get('rachis'):
            assets.append(f'    <mesh name="m_{vname}_rachis{t["k"]}" '
                          f'file="meshes/env/plants/{vname}_rachis{t["k"]}.obj"/>')
        rachis_geom = (f'    <geom name="g_tr{truss_idx}_rachis" type="mesh" '
                       f'mesh="m_{vname}_rachis{t["k"]}" material="mat_vine" '
                       f'contype="0" conaffinity="0"/>\n') if t.get('rachis') else ''
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
            + rachis_geom +
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
                          f'rgba="0.36 0.25 0.12 1" contype="4" conaffinity="1"/>')
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
                      f'size="0.02 {L / 2:.3f}" quat="0.707107 0 0.707107 0" '
                      f'rgba="0 0 0 0" contype="0" conaffinity="0"/>')

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
             '\n'.join(pipe_lines + plant_bodies + truss_bodies + world_extra))
src = inject(src, 'EQ', '  </equality>', '\n'.join(welds))

def extend_keys(text):
    def fix(m):
        head, q, ctrl = m.group(1), m.group(2).split(), m.group(3).split()
        # 展平: 每串 = pos(3) + quat 字符串再拆 4 个 token (共 7)
        tvals = []
        for vals in truss_qpos:
            tvals += [vals[0], vals[1], vals[2]] + vals[3].split()
        if len(q) == 21:
            # 环境 WORLD 块注入在 machine 之前: 果串 freejoint 的 qpos 排在最前 280 位
            q = tvals + q
        elif len(q) == 21 + len(tvals):
            # 已扩写过 (幂等重跑): 原位更新果串段 (几何变了必须刷新)
            q[:len(tvals)] = tvals
        return f'{head}qpos="{" ".join(q)}" ctrl="{" ".join(ctrl)}"/>'
    return re.sub(r'(<key name="(?:home|lift_high)" )qpos="([^"]*)" ctrl="([^"]*)"/>', fix, text)

src = extend_keys(src)
open(XML, 'w').write(src)
print(f'环境已注入: {truss_idx} 果串 / {len(VARIANTS)} 株 / 轨道 {PIPE_N * 2} 段')
from collections import Counter
cnt = Counter(mode for _, mode, _ in cut_log)
print('剪切点规则: %s' % ', '.join(f'{k}x{v}' for k, v in sorted(cnt.items())))
lg = [nm for nm, mode, _ in cut_log if mode == 'legacy55']
if lg:
    print('  贴茎例外 (退回55%%): %d 串: %s' % (len(lg), ', '.join(lg)))
