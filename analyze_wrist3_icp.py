"""Register the CAD's stylized wrist replica (part_361) onto the official wrist3_Link mesh.

Ground truth: part_361/358 are the CAD's stylized copies of the arm's wrist3.
The print (362 bracket + 371 ring) bolts to that replica in the CAD, so the
ICP transform of 361 -> real wrist3 IS the correct mount of the whole
gripper+print stack onto the real flange.

All quantities in meters unless noted.  File frame = gripper-body (baked) frame.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import load_stl, AUBO_DIR
from scipy.spatial import cKDTree

W3 = f'{AUBO_DIR}/models/wrist3_Link.STL'
B361 = '/data/robot_assembly/model/meshes_baked/part_361_baked.stl'
B362 = '/data/robot_assembly/model/meshes_baked/part_362_baked.stl'
B371 = '/data/robot_assembly/model/meshes_baked/part_371_baked.stl'

v3, _ = load_stl(W3)
f361, _ = load_stl(B361)
f362, _ = load_stl(B362)
f371, _ = load_stl(B371)

print('== real wrist3 (m) ==')
ymax = v3[:, 1].max()
ann = v3[v3[:, 1] > ymax - 0.0015]
ra = np.hypot(ann[:, 0], ann[:, 2])
print(f'  y-max {ymax*1000:.2f} mm; annulus band n={len(ann)} radii {ra.min()*1000:.1f}..{ra.max()*1000:.1f} mm')
face_plane_y = ann[:, 1].mean()
print(f'  flange annulus plane y = {face_plane_y*1000:.2f} mm (center on axis (0,*,0))')

print('== part_361 (stylized wrist copy, file frame, m) ==')
zmin = f361[:, 2].min()
bot = f361[f361[:, 2] < zmin + 0.0008]
rb = np.hypot(bot[:, 0] - bot[:, 0].mean(), bot[:, 1] - bot[:, 1].mean())
rb0 = np.hypot(bot[:, 0], bot[:, 1])
print(f'  bottom face z={zmin*1000:.2f} mm n={len(bot)} center=({bot[:,0].mean()*1000:.2f},{bot[:,1].mean()*1000:.2f}) mm')
print(f'  radii from own centroid {rb.min()*1000:.1f}..{rb.max()*1000:.1f} mm; from (0,0) {rb0.min()*1000:.1f}..{rb0.max()*1000:.1f} mm')

print('== part_362 (L-bracket) top region ==')
zmax = f362[:, 2].max()
top = f362[f362[:, 2] > zmax - 0.0008]
rt = np.hypot(top[:, 0], top[:, 1])
near = top[rt < 0.040]
print(f'  top z={zmax*1000:.2f} mm; verts near axis(r<40mm) n={len(near)} center=({near[:,0].mean()*1000:.1f},{near[:,1].mean()*1000:.1f}) mm '
      f'radii {np.hypot(near[:,0],near[:,1]).min()*1000:.1f}..{np.hypot(near[:,0],near[:,1]).max()*1000:.1f} mm')

print('== part_371 (ring) ==')
r371 = np.hypot(f371[:, 0], f371[:, 1])
print(f'  radii {r371.min()*1000:.1f}..{r371.max()*1000:.1f} mm; z {f371[:,2].min()*1000:.1f}..{f371[:,2].max()*1000:.1f} mm')

# ---------------- ICP 361 -> wrist3 ----------------
# init: tool axis alignment  file +z -> -y_w3, file x -> x_w3, file y -> z_w3
#   flange planes coincident: file (0,0,zmin) must land on (0, face_plane_y, 0)
#   => t0 = (0, face_plane_y + zmin, 0)  [file origin 95 mm along +y_w3, tool side]
R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
t0 = np.array([0.0, face_plane_y + zmin, 0.0])
print(f'\nICP init: t0={np.round(t0*1000,2)} mm   (file origin {t0[1]*1000:.1f} mm beyond flange plane along +y_w3)')

src = f361[::2]
tgt_tree = cKDTree(v3)
R, t = R0.copy(), t0.copy()
prev = 1e9
for it in range(80):
    P = src @ R.T + t
    dd, idx = tgt_tree.query(P)
    keep = dd < np.percentile(dd, 80)
    A, Bm = src[keep], v3[idx[keep]]
    ca, cb = A.mean(0), Bm.mean(0)
    H = (A - ca).T @ (Bm - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    Rn = Vt.T @ D @ U.T
    tn = cb - Rn @ ca
    rms = np.sqrt((dd[keep] ** 2).mean())
    if np.abs(Rn - R).max() < 1e-9 and np.abs(tn - t).max() < 1e-9:
        R, t = Rn, tn
        break
    R, t = Rn, tn
    if abs(prev - rms) < 1e-7:
        break
    prev = rms
print(f'ICP done: rms(all)={np.sqrt((dd**2).mean())*1000:.2f} mm  trimmed rms={np.sqrt((dd[keep]**2).mean())*1000:.2f} mm  iters={it}')
print('R (file->w3):\n', np.round(R, 4))
print('t (file origin in w3, mm):', np.round(t * 1000, 2))
dR0 = R @ R0.T
ang = np.degrees(np.arccos(np.clip((np.trace(dR0) - 1) / 2, -1, 1)))
dt0 = t - t0
print(f'delta vs axis-aligned init: rotation {ang:.2f} deg, translation {np.round(dt0*1000,2)} mm')

# where does 361's bottom face land?
bw = f361[f361[:, 2] < zmin + 0.0008] @ R.T + t
print(f'361 bottom face after ICP: y {bw[:,1].min()*1000:.2f}..{bw[:,1].max()*1000:.2f} mm (real flange plane {face_plane_y*1000:.2f})')
rc = np.hypot(bw[:, 0], bw[:, 2])
print(f'  radii from w3 axis: {rc.min()*1000:.1f}..{rc.max()*1000:.1f} mm (real annulus {ra.min()*1000:.1f}..{ra.max()*1000:.1f})')

# full-part clearances after ICP: 362/371 vs real wrist3
for nm, f in [('362', f362), ('371', f371)]:
    P = f[::2] @ R.T + t
    dd2, _ = tgt_tree.query(P)
    print(f'clearance print {nm} -> real wrist3: min {dd2.min()*1000:.2f} mm')

np.save('/data/robot_assembly/icp_361_R.npy', R)
np.save('/data/robot_assembly/icp_361_t.npy', t)
print('\nsaved icp_361_R.npy / icp_361_t.npy')
