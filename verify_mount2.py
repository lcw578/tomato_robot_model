import mujoco, numpy as np
from geomlib import load_stl, AUBO_DIR
from scipy.spatial import cKDTree
from PIL import Image

m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
print(f'compile OK  ncon={d.ncon}')

def W(g):
    mid = m.geom_dataid[g]; v0, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[v0:v0 + vn].astype(float)
    return (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]   # formula A (verified)

grp = {'print': [], 'cam': [], 'wrist3': [], 'gripper': []}
for g in range(m.ngeom):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    mid = m.geom_dataid[g]
    if mid < 0: continue
    mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(mid)) or ''
    if n.startswith('ad_'): grp['print'].append(W(g))
    elif n.startswith('cam_'): grp['cam'].append(W(g))
    elif n.startswith('a_wrist3'): grp['wrist3'].append(W(g))
    elif mn.startswith('g_'): grp['gripper'].append(W(g))
P = {k: np.vstack(v) for k, v in grp.items()}
for k, v in P.items():
    print(f'{k:8s} n={len(v):6d} world bbox {np.round(v.min(0), 4)}..{np.round(v.max(0), 4)}')
T = {k: cKDTree(v) for k, v in P.items()}
print('clearances (mm):')
for a, b in [('print', 'wrist3'), ('cam', 'wrist3'), ('gripper', 'wrist3'),
             ('print', 'gripper'), ('cam', 'print'), ('gripper', 'cam')]:
    dd, _ = T[b].query(P[a])
    print(f'  {a:8s} <-> {b:8s} min {dd.min()*1000:7.2f}')
# flange mating check: print top face vs flange plane
B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_Link')
Rw3, pw3 = d.xmat[B].reshape(3, 3), d.xpos[B]
v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
yf = v3[v3[:, 1] > v3[:, 1].max() - 0.0003][:, 1].mean()
nrm = Rw3 @ np.array([0, 1.0, 0])
fw = pw3 + Rw3 @ np.array([0, yf, 0])
sp = (P['print'] - fw) @ nrm
print(f'print verts signed dist to flange plane: min {sp.min()*1000:.2f} max {sp.max()*1000:.2f} mm  (0 = mated)')
rr = np.hypot(*(P['print'] - fw)[:, [0, 2]].T)
on_plane = P['print'][np.abs(sp) < 0.0005]
if len(on_plane):
    r2 = np.hypot(*(on_plane - fw)[:, [0, 2]].T)
    print(f'  verts on flange plane: n={len(on_plane)} radii {r2.min()*1000:.1f}..{r2.max()*1000:.1f} mm (real annulus 18.1..31.9)')

# render
r = mujoco.Renderer(m, 1000, 1333)
gp = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')]
def shot(az, el, dist, path, lookat):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.azimuth = az; cam.elevation = el; cam.distance = dist
    r.update_scene(d, camera=cam); Image.fromarray(r.render()).save(path); print('saved', path)
shot(20, -12, 0.5, 'fix17_gripper.png', gp + [0, 0, 0.07])
shot(-70, -6, 0.55, 'fix17_side.png', gp + [0, 0, 0.07])
shot(35, -25, 0.6, 'fix17_back.png', gp + [0, 0, 0.07])
shot(35, -80, 0.45, 'fix17_top.png', gp + [0, 0, 0.03])
