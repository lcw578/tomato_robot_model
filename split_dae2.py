"""Split AUBO visual DAEs into material groups, honoring the full node transform hierarchy."""
import xml.etree.ElementTree as ET
import numpy as np, struct, os, json

AUBO_DIR = '/home/lcw/VR_teleoperation/_refs/Auboi5_Scan_Simulator'
OUT = '/data/robot_assembly/dae_split'
os.makedirs(OUT, exist_ok=True)

def parse_geoms(path):
    t = ET.parse(path); r = t.getroot()
    geoms = {}
    for g in r.iter():
        if not g.tag.endswith('geometry'):
            continue
        gid = g.get('id')
        if not gid:
            continue
        mesh = g.find('{*}mesh')
        if mesh is None:
            continue
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
        for inp in vsrc.findall('{*}input'):
            src = inp.get('source'); sem = inp.get('semantic')
            if src not in raw:
                continue
            arr = raw[src]; st = stride[src]
            if arr.size % st:
                continue
            arr = arr.reshape(-1, st)[:, :3]
            if sem == 'POSITION' or sem == 'VERTEX':
                verts = arr
            if sem == 'NORMAL':
                vnorm = arr
        if verts is None:
            continue
        tris = {}
        for tri in mesh.findall('{*}triangles'):
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
        geoms['#' + gid] = (verts, vnorm, tris)
    return geoms

def node_matrix(n):
    """Compose translate/rotate/scale/matrix children into a 4x4."""
    M = np.eye(4)
    for c in n:
        tag = c.tag.split('}')[-1]
        if tag == 'matrix':
            v = np.fromstring(c.text, sep=' ')
            if v.size == 16:
                M = M @ v.reshape(4, 4)
        elif tag == 'translate':
            T = np.eye(4); T[:3, 3] = np.fromstring(c.text, sep=' ')
            M = M @ T
        elif tag == 'rotate':
            v = np.fromstring(c.text, sep=' ')
            ax = v[:3] / np.linalg.norm(v[:3]); ang = np.radians(v[3])
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            RM = np.eye(4); RM[:3, :3] = R
            M = M @ RM
        elif tag == 'scale':
            S = np.eye(4)
            sv = np.fromstring(c.text, sep=' ')
            S[0, 0], S[1, 1], S[2, 2] = sv[0], sv[1], sv[2]
            M = M @ S
    return M

# simpler: reuse parse per file
def split_link2(dae_path, out_prefix, unit_scale=1.0, geoms=None):
    if geoms is None:
        geoms = parse_geoms(dae_path)
    t = ET.parse(dae_path); r = t.getroot()
    unit = r.find('.//{*}unit')
    scale = float(unit.get('meter')) if unit is not None else 1.0
    vs = r.find('.//{*}visual_scene')
    pergeom = {}
    def walk(n, parent_M):
        M = parent_M @ node_matrix(n)
        for c in n:
            tag = c.tag.split('}')[-1]
            if tag == 'instance_geometry':
                url = c.get('url')
                if url in geoms:
                    pergeom.setdefault(url, []).append(M)
            elif tag == 'node':
                walk_scene_node(c, M)
    def walk_scene_node(n, parent_M):
        M = parent_M @ node_matrix(n)
        for c in n:
            tag = c.tag.split('}')[-1]
            if tag == 'instance_geometry':
                url = c.get('url')
                if url in geoms:
                    pergeom.setdefault(url, []).append(M)
            elif tag == 'node':
                walk_scene_node(c, M)
    for top in vs:
        if top.tag.split('}')[-1] in ('node',):
            walk_scene_node(top, np.eye(4))
    # merge: for each geometry, all instances share verts; transform verts by each M
    groups = {'m1': [], 'm2': [], 'none': []}
    for url, Ms in pergeom.items():
        verts, vnorm, tris = geoms[url]
        for M in Ms:
            for key, chunks in tris.items():
                for vidx, nidx in chunks:
                    V = verts[vidx] * scale
                    W = (M[:3, :3] @ V.T).T + M[:3, 3]
                    groups[key].append(W)
    return groups

def write_stls(groups, out_prefix):
    out = {}
    for k in ['m1', 'm2', 'none']:
        if not groups.get(k): continue
        alltris = []
        for Wpts in groups[k]:
            for i in range(0, len(Wpts), 3):
                T = Wpts[i:i + 3]
                if len(T) < 3: continue
                alltris.append(T)
        if not alltris: continue
        T = np.vstack(alltris).reshape(-1, 3, 3)
        # only drop nonfinite vertices (keep all geometry otherwise)
        good = np.isfinite(T).all(axis=(1,2))
        if (~good).any():
            print(f"  {out_prefix}_{k}: dropped {int((~good).sum())} nonfinite tris")
        T = T[good]
        fn = f'{out_prefix}_{k}.stl'
        with open(fn, 'wb') as fh:
            fh.write(b'\x00' * 80); fh.write(struct.pack('<I', len(T)))
            for tri in T:
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                nn = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.zeros(3)
                fh.write(struct.pack('<3f', *nn.astype(np.float32)))
                for pnt in tri: fh.write(struct.pack('<3f', *pnt.astype(np.float32)))
                fh.write(struct.pack('<H', 0))
        out[k] = len(T)
    return out

if __name__ == '__main__':
    import sys
    LINKS = ['base_link', 'shoulder_Link', 'upperArm_Link', 'foreArm_Link',
             'wrist1_Link', 'wrist2_Link', 'wrist3_Link']
    summary = {}
    for lk in LINKS:
        t = ET.parse(f'{AUBO_DIR}/models/{lk}.DAE'); r = t.getroot()
        unit = r.find('.//{*}unit')
        scale = float(unit.get('meter')) if unit is not None else 1.0
        geoms = parse_geoms(f'{AUBO_DIR}/models/{lk}.DAE')
        vs = r.find('.//{*}visual_scene')
        pergeom = {}
        def walk(n, parent_M):
            M = parent_M @ node_matrix(n)
            for c in n:
                tag = c.tag.split('}')[-1]
                if tag == 'instance_geometry':
                    url = c.get('url')
                    if url in geoms:
                        pergeom.setdefault(url, []).append(M)
                elif tag == 'node':
                    walk(c, M)
        for top in vs:
            if top.tag.split('}')[-1] == 'node':
                walk(top, np.eye(4))
        groups = {'m1': [], 'm2': [], 'none': []}
        for url, Ms in pergeom.items():
            verts, vnorm, tris = geoms[url]
            for M in Ms:
                for key, chunks in tris.items():
                    for vidx, nidx in chunks:
                        V = verts[vidx] * scale
                        W = (M[:3, :3] @ V.T).T + M[:3, 3]
                        groups[key].append(W)
        st = write_stls(groups, f'{OUT}/{lk}')
        summary[lk] = st
        print(lk, st)
    json.dump(summary, open(f'{OUT}/groups.json', 'w'), indent=1)
