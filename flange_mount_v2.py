"""Determine the correct mount of the print (362+371) + camera on the real wrist3 flange.

Established ground truth:
- real wrist3 flange: annulus centered on the tool axis (0,*,0) in the w3 frame,
  plane y=44.53 mm, radii 18.1..31.9 mm.
- part_361 (the CAD's stylized wrist3 copy) bottom face: annulus centered on the
  file axis (0,0), plane z=50.48 mm, radii 15.7..31.6 mm  -> same feature.
So the mount is axis<->axis, plane<->plane; the only free DOF is the roll about
the tool axis.  This script (a) runs a guarded ICP of 361 onto the real wrist3
to recover the roll, (b) checks bolt holes on both annuli as a cross-check.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import load_stl, AUBO_DIR
from scipy.spatial import cKDTree

v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
f361, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_361_baked.stl')
f362, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_362_baked.stl')
f371, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_371_baked.stl')

ymax = v3[:, 1].max()
ann = v3[v3[:, 1] > ymax - 0.0015]
face_plane_y = ann[:, 1].mean()
zmin = f361[:, 2].min()
print(f'real annulus plane y={face_plane_y*1000:.2f} mm, radii {np.hypot(ann[:,0],ann[:,2]).min()*1000:.1f}..{np.hypot(ann[:,0],ann[:,2]).max()*1000:.1f} mm')
print(f'361 bottom plane z={zmin*1000:.2f} mm, radii from (0,0) {np.hypot(*(f361[f361[:,2]<zmin+0.0008][:,:2]).T).min()*1000:.1f}..{np.hypot(*(f361[f361[:,2]<zmin+0.0008][:,:2]).T).max()*1000:.1f} mm')

# --- angular occupancy of both annuli (hole detection) ---
def angular_hist(P, nb=72):
    th = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360
    h, _ = np.histogram(th, bins=nb, range=(0, 360))
    return h
h_real = angular_hist(ann)
bot = f361[f361[:, 2] < zmin + 0.0008]
h_361 = angular_hist(bot)
print(f'\nreal annulus empty {5}-deg bins: {np.where(h_real==0)[0]*5}')
print(f'361  bottom  empty {5}-deg bins: {np.where(h_361==0)[0]*5}')

# --- guarded ICP: 361 -> real wrist3 ---
R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
t0 = np.array([0.0, face_plane_y + zmin, 0.0])
print(f'\nICP init t0 = {np.round(t0*1000,2)} mm')
src = f361[::2]
tree = cKDTree(v3)
R, t = R0.copy(), t0.copy()
prev = 1e9
for it in range(100):
    P = src @ R.T + t
    dd, idx = tree.query(P)
    keep = dd < np.percentile(dd, 75)
    A, Bm = src[keep], v3[idx[keep]]
    ca, cb = A.mean(0), Bm.mean(0)
    H = (A - ca).T @ (Bm - cb)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rn = Vt.T @ D @ U.T
    tn = cb - Rn @ ca
    rms = np.sqrt((dd[keep] ** 2).mean())
    conv = np.abs(Rn - R).max() < 1e-10 and np.abs(tn - t).max() < 1e-10
    R, t = Rn, tn
    if conv or abs(prev - rms) < 1e-8:
        break
    prev = rms
print(f'ICP: trimmed rms={np.sqrt((dd[keep]**2).mean())*1000:.2f} mm  iters={it}')
print('R:\n', np.round(R, 4))
print('t (mm):', np.round(t * 1000, 2))
dR = R @ R0.T
ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
if np.linalg.norm(ax) > 1e-9:
    ax = ax / np.linalg.norm(ax)
print(f'roll correction vs init: {ang:.2f} deg about {np.round(ax,3)};  dt = {np.round((t-t0)*1000,2)} mm')

bw = f361[f361[:, 2] < zmin + 0.0008] @ R.T + t
print(f'361 bottom after ICP: y {bw[:,1].min()*1000:.2f}..{bw[:,1].max()*1000:.2f} mm (plane {face_plane_y*1000:.2f}); '
      f'radii {np.hypot(bw[:,0],bw[:,2]).min()*1000:.1f}..{np.hypot(bw[:,0],bw[:,2]).max()*1000:.1f} mm')
for nm, f in [('362', f362), ('371', f371)]:
    P = f[::2] @ R.T + t
    dd2, _ = tree.query(P)
    print(f'clearance {nm} -> real wrist3: min {dd2.min()*1000:.2f} mm')
np.save('/data/robot_assembly/icp_361_R.npy', R)
np.save('/data/robot_assembly/icp_361_t.npy', t)
print('saved icp_361_R.npy / icp_361_t.npy')
