"""Verify the rebuilt mount: print disc face on the real flange, axis<->axis."""
import sys
sys.path.insert(0, '/data/robot_assembly')
import mujoco, numpy as np
from geomlib import load_stl, AUBO_DIR
from scipy.spatial import cKDTree

m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
print(f'compile OK, ncon={d.ncon}')

def W(g):
    mid = m.geom_dataid[g]
    v0, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[v0:v0 + vn].astype(float)
    return (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]

def gid(name):
    return [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '') == name]

groups = {'print': [], 'cam': [], 'wrist3': [], 'gripper': []}
for g in range(m.ngeom):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    mid = m.geom_dataid[g]
    mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(mid)) if mid >= 0 else ''
    if n.startswith('ad_'):
        groups['print'].append(W(g))
    elif n.startswith('cam_'):
        groups['cam'].append(W(g))
    elif (n or '').startswith('a_wrist3'):
        groups['wrist3'].append(W(g))
    elif mn and mn.startswith('g_'):
        groups['gripper'].append(W(g))
P = {k: np.vstack(v) for k, v in groups.items()}
for k, v in P.items():
    print(f'{k:8s} n={len(v):6d} world bbox {np.round(v.min(0), 4)} .. {np.round(v.max(0), 4)}')
T = {k: cKDTree(v) for k, v in P.items()}

# flange plane (real wrist3, axis<->axis): face plane y = mean of tight band
v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
yf = v3[v3[:, 1] > v3[:, 1].max() - 0.0003][:, 1].mean()
w3b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_Link')
Rw3, pw3 = d.xmat[w3b].reshape(3, 3), d.xpos[w3b]
fw = pw3 + Rw3 @ np.array([0.0, yf, 0.0])
nrm = Rw3 @ np.array([0.0, 1.0, 0.0])
print(f'\nflange plane y={yf*1000:.2f} mm, world center {np.round(fw, 4)}')

# print top face vs flange plane (should mate: signed dist ~ 0)
gp = gid('ad_part_362')[0]
Pw = W(gp)
Rbody = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')].reshape(3, 3)
zdir = Rbody @ np.array([0.0, 0.0, 1.0])          # world direction of file +z
tproj = (Pw - d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')]) @ zdir
top_world = Pw[tproj > tproj.max() - 0.0003]
sd = (top_world - fw) @ nrm
print(f'print top face signed dist to flange plane: {sd.min()*1000:.2f}..{sd.max()*1000:.2f} mm (0 = mated)')
perp = (top_world - fw) - np.outer((top_world - fw) @ nrm, nrm)
rr = np.linalg.norm(perp, axis=1)
print(f'  print top-face radii from tool axis: {rr.min()*1000:.1f}..{rr.max()*1000:.1f} mm (real annulus 18.1..31.9)')

print('\nclearances (mm):')
for a, b in [('print', 'wrist3'), ('cam', 'wrist3'), ('gripper', 'wrist3'),
             ('print', 'gripper'), ('cam', 'print')]:
    dd, _ = T[b].query(P[a][::2])
    print(f'  {a:8s} <-> {b:8s} min {dd.min()*1000:7.2f}')
