import numpy as np, sys
sys.path.insert(0, '/data/robot_assembly')
import mujoco
from geomlib import load_stl, AUBO_DIR

# 1) real flange face plane (tight band)
v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
ymax = v3[:, 1].max()
for band in (0.3, 0.6, 1.5):
    sel = v3[v3[:, 1] > ymax - band / 1000]
    print(f'wrist3 face band -{band}mm: n={len(sel)} y mean {sel[:,1].mean()*1000:.2f} min {sel[:,1].min()*1000:.2f} max {sel[:,1].max()*1000:.2f}')

# 2) 362 top face plane (tight band)
f362, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_362_baked.stl')
zmax = f362[:, 2].max()
for band in (0.3, 0.6):
    sel = f362[f362[:, 2] > zmax - band / 1000]
    print(f'362 top band -{band}mm: n={len(sel)} z mean {sel[:,2].mean()*1000:.2f} min {sel[:,2].min()*1000:.2f} max {sel[:,2].max()*1000:.2f}')

# 3) gripper baseline topmost point in file frame (formula A, world = geom_xpos + geom_xmat@V)
m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
Rg, pg = d.xmat[G].reshape(3, 3), d.xpos[G]
ztop = -1e9; zbot = 1e9
for g in range(m.ngeom):
    mid = m.geom_dataid[g]
    if mid < 0: continue
    mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(mid)) or ''
    if not mn.startswith('g_'): continue
    v0, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[v0:v0 + vn].astype(float)
    L = (Rg.T @ (((d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]) - pg).T).T
    ztop = max(ztop, L[:, 2].max()); zbot = min(zbot, L[:, 2].min())
print(f'\ngripper baseline in file frame: z {zbot*1000:.1f}..{ztop*1000:.1f} mm  (362 top = 51.1)')
