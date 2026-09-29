"""Determine the correct print/gripper mount on the real wrist3 flange.

Facts established:
- real wrist3 flange = annular face, centered on the w3 axis (0,*,0), plane y=y_f,
  radii ~18..32 mm.
- part_361 (CAD's stylized wrist3 copy) bottom face = annulus centered on the
  file axis (0,0), plane z=z_b, radii ~15.7..31.6 mm -> same feature.
=> mount = axis<->axis, plane<->plane; only the ROLL about the tool axis is free.
Determine roll by 1-DOF scan of 361's shell against the real wrist3 surface,
cross-checked with bolt holes on both annuli.
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

def face_plane(P, axis, hi, tol):
    sel = P[P[:, axis] > hi - tol]
    return sel[:, axis].mean(), sel

yf, annr = face_plane(v3, 1, v3[:, 1].max(), 0.0004)
zb, bot = face_plane(f361, 2, f361[:, 2].min(), 0.0004)
print(f'real flange face plane y = {yf*1000:.2f} mm (n={len(annr)})')
print(f'361 bottom face plane z = {zb*1000:.2f} mm (n={len(bot)})')

def holes(P, ax, ctr_ax, rlo, rhi, nb=72):
    """angular occupancy histogram of verts within a radial band (hole detection)"""
    c = np.delete(P, ax, axis=1) - 0.0   # coords in the face plane
    r = np.hypot(c[:, 0], c[:, 1])
    m = (r > rlo) & (r < rhi)
    th = np.degrees(np.arctan2(c[m, 1], c[m, 0])) % 360
    h, _ = np.histogram(th, bins=nb, range=(0, 360))
    return h, m.sum()

h_real, n_real = holes(annr, 1, 0, 0.020, 0.030)
h_361, n_361 = holes(bot, 2, 0, 0.016, 0.032)
print(f'\nreal flange annulus band r20-30mm: n={n_real}, empty bins(72): {(h_real==0).sum()}')
print(f'361 bottom band r16-32mm:          n={n_361}, empty bins(72): {(h_361==0).sum()}')
if (h_real == 0).any():
    print('  real empty-angle bins:', np.round(np.where(h_real == 0)[0] * 5, 1))
if (h_361 == 0).any():
    print('  361  empty-angle bins:', np.round(np.where(h_361 == 0)[0] * 5, 1))

# ---- fixed mount: axis<->axis, plane<->plane ----
R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
t0 = np.array([0.0, yf + zb, 0.0])
print(f'\naxis/plane mount: t0 = {np.round(t0*1000,2)} mm')

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

tree3 = cKDTree(v3)
src = f361[::2]
band = src[:, 2] < zb + 0.035          # flange-end shell region only
src_b = src[band]
print(f'roll scan source: {len(src_b)}/{len(src)} verts (z < {zb*1000+35:.0f} mm)')
best = None
for roll in np.arange(0, 360, 2.0):
    R = rot_y(np.radians(roll)) @ R0
    P = src_b @ R.T + t0
    dd, _ = tree3.query(P)
    keep = dd < np.percentile(dd, 70)
    rms = np.sqrt((dd[keep] ** 2).mean())
    if best is None or rms < best[0]:
        best = (rms, roll, R)
rms, roll, R = best
print(f'best roll = {roll:.0f} deg, trimmed rms = {rms*1000:.2f} mm')
# refine +-3 deg at 0.25 deg
for roll_f in np.arange(roll - 3, roll + 3.01, 0.25):
    Rf = rot_y(np.radians(roll_f)) @ R0
    P = src_b @ Rf.T + t0
    dd, _ = tree3.query(P)
    keep = dd < np.percentile(dd, 70)
    r2 = np.sqrt((dd[keep] ** 2).mean())
    if r2 < rms:
        rms, roll, R = r2, roll_f, Rf
print(f'refined roll = {roll:.2f} deg, trimmed rms = {rms*1000:.2f} mm')

# clearances of the print after mount
for nm, f in [('362', f362), ('371', f371)]:
    P = f @ R.T + t0
    dd, _ = tree3.query(P)
    print(f'clearance {nm} -> real wrist3: min {dd.min()*1000:.2f} mm')
# 361 vs real wrist3 (should hug it)
P = f361[::2] @ R.T + t0
dd, _ = tree3.query(P)
print(f'361 -> real wrist3: median {np.median(dd)*1000:.2f} mm, p90 {np.percentile(dd,90)*1000:.2f} mm')

np.save('/data/robot_assembly/mount_R.npy', R)
np.save('/data/robot_assembly/mount_t.npy', t0)
print(f'\nsaved mount_R.npy / mount_t.npy  (roll {roll:.2f} deg, t0={np.round(t0*1000,2)} mm)')
print('R (file->w3):\n', np.round(R, 5))
