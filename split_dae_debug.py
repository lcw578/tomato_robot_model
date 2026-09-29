"""Split AUBO visual DAEs into material groups (m1/m2/none) as STLs."""
import xml.etree.ElementTree as ET
import numpy as np, struct, os, json, sys

AUBO_DIR = '/home/lcw/VR_teleoperation/_refs/Auboi5_Scan_Simulator'
OUT = '/data/robot_assembly/dae_split'
os.makedirs(OUT, exist_ok=True)

def parse_dae_fixed(path):
    t = ET.parse(path); r = t.getroot()
    unit = r.find('.//{*}unit')
    scale = float(unit.get('meter')) if unit is not None else 1.0
    geoms = {}
    global_dbg = {'nsrc':0}
    _dbg_n=[0]
    for g in r.iter():
        if not g.tag.endswith('geometry'):
            continue
        gid = g.get('id') or 'noname'
        mesh = g.find('{*}mesh')
        if mesh is None:
            print('  [dbg] mesh None:', gid[:40]); continue
        global_dbg['nsrc']=0
        raw = {}; stride = {}
        for s in mesh.findall('{*}source'):
            fa = s.find('{*}float_array')
            if fa is None:
                continue
            raw['#' + s.get('id')] = np.fromstring(fa.text, sep=' ')
            st = 3
            tc = s.find('{*}technique_common')
            if tc is not None:
                inp = tc.find('{*}input')
                params = inp.findall('{*}param') if inp is not None else []
                if params:
                    st = len(params)
            stride['#' + s.get('id')] = st
        vsrc = mesh.find('{*}vertices')
        if vsrc is None:
            continue
        verts = None; vnorm = None
        _dbg_n[0]+=1
        if _dbg_n[0]>3: return geoms, scale
        print('  [dbg]', gid[:40], 'sources:', len(raw), 'sizes:', {k[1:20]: len(v) for k,v in list(raw.items())[:3]}, 'strides:', {k[1:20]: v for k,v in list(stride.items())[:3]})
        for inp in vsrc.findall('{*}input'):
            src = inp.get('source'); sem = inp.get('semantic')
            if src not in raw:
                print('  [dbg]   missing source for', sem, src[:40] if src else None); continue
            arr = raw[src]; st = stride[src]
            if arr.size % st:
                print('  [dbg]   size%stride fail', sem, arr.size, st); continue
            arr = arr.reshape(-1, st)[:, :3] * scale
            if sem == 'VERTEX':
                verts = arr
            if sem == 'NORMAL':
                vnorm = arr
        if verts is None:
            continue
        tris = {}
        for tri in list(mesh.findall('{*}triangles')) + list(mesh.findall('{*}polylist')):
            mat = tri.get('material')
            offs = {}
            for inp in tri.findall('{*}input'):
                offs.setdefault(inp.get('semantic'), int(inp.get('offset')))
            stride_t = max(offs.values()) + 1
            p = tri.find('{*}p')
            if p is None or 'VERTEX' not in offs:
                continue
            idx = np.fromstring(p.text, sep=' ', dtype=int).reshape(-1, stride_t)
            vidx = idx[:, offs['VERTEX']].reshape(-1, 3)
            nidx = idx[:, offs['NORMAL']].reshape(-1, 3) if 'NORMAL' in offs and vnorm is not None else None
            key = 'm1' if mat == '_01 - Default' else ('m2' if mat == '_02 - Default' else 'none')
            tris.setdefault(key, []).append((vidx, nidx))
        geoms[gid] = (verts, vnorm, tris)
    return geoms, scale

def write_split(geoms, out_prefix):
    out = {}
    for k in ['m1', 'm2', 'none']:
        alltris = []
        for gid, (verts, vnorm, tris) in geoms.items():
            for vidx, nidx in tris.get(k, []):
                T = verts[vidx]
                if nidx is not None and vnorm is not None and len(vnorm):
                    N = vnorm[np.clip(nidx, 0, len(vnorm) - 1)]
                    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
                    ln = np.linalg.norm(fn, axis=1); ln[ln == 0] = 1
                    fn = (fn.T / ln).T
                    nn = N[:, 0] + N[:, 1] + N[:, 2]
                    ln2 = np.linalg.norm(nn, axis=1); ln2[ln2 == 0] = 1
                    nn = (nn.T / ln2).T
                    flip = (fn * nn).sum(1) < 0
                    T = T.copy(); T[flip] = T[flip][:, [0, 2, 1]]
                alltris.append(T)
        if not alltris:
            continue
        T = np.vstack(alltris)
        fn2 = f'{out_prefix}_{k}.stl'
        with open(fn2, 'wb') as fh:
            fh.write(b'\x00' * 80)
            fh.write(struct.pack('<I', len(T)))
            for tri in T:
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                nn = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.zeros(3)
                fh.write(struct.pack('<3f', *nn.astype(np.float32)))
                for pnt in tri:
                    fh.write(struct.pack('<3f', *pnt.astype(np.float32)))
                fh.write(struct.pack('<H', 0))
        out[k] = len(T)
    return out

LINKS = ['base_link', 'shoulder_Link', 'upperArm_Link', 'foreArm_Link',
         'wrist1_Link', 'wrist2_Link', 'wrist3_Link']
summary = {}
for lk in LINKS:
    g, s = parse_dae_fixed(f'{AUBO_DIR}/models/{lk}.DAE')
    ntri = {k: sum(len(v) for _, v in tris.items()) for gid, (v, vn, tris) in g.items() for k in ['m1','m2','none'] for v in [tris.get(k,[])] if v}
    st = write_split(g, f'{OUT}/{lk}')
    summary[lk] = st
    print(lk, st)
json.dump(summary, open(f'{OUT}/groups.json', 'w'), indent=1)
