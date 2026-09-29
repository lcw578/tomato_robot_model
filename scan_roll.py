"""1-DOF roll scan: stylized wrist copy (part_361) vs real wrist3.

Mount is fully determined except the roll about the tool axis:
  R = R0 @ Rz(roll),  t = (0, face_plane_y + zmin_361, 0)
For each roll, trimmed nearest-neighbor distance of 361's verts to the real
wrist3 surface.  The stylized copy hugs the real wrist's flange-end shell, so
the correct roll minimizes that distance.
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
face_plane_y = v3[v3[:, 1] > ymax - 0.0015][:, 1].mean()
zmin = f361[:, 2].min()
R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
t0 = np.array([0.0, face_plane_y + zmin, 0.0])
print(f'flange plane y={face_plane_y*1000:.2f} mm, 361 bottom z={zmin*1000:.2f} mm -> t0={np.round(t0*1000,2)} mm')

tree = cKDTree(v3)
src = f361[::2]          # subsample
best = None
prof = []
for roll in np.arange(0, 360, 2.0):
    a = np.radians(roll)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    R = R0 @ Rz
    P = src @ R.T + t0
    dd, _ = tree.query(P)
    keep = dd < np.percentile(dd, 70)
    rms = np.sqrt((dd[keep] ** 2).mean())
    prof.append((roll, rms, dd.min()))
    if best is None or rms < best[1]:
        best = (roll, rms, dd.min())
prof = np.array(prof)
print('\nroll profile (deg, trimmed rms mm, min mm):')
for i in range(0, len(prof), 6):
    print(f'  {prof[i,0]:5.0f}  {prof[i,1]*1000:6.2f}  {prof[i,2]*1000:6.2f}')
print(f'\nBEST roll = {best[0]:.0f} deg  trimmed rms {best[1]*1000:.2f} mm  min {best[2]*1000:.2f} mm')

# clearance of the whole print at the best roll
a = np.radians(best[0])
Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
R = R0 @ Rz
for nm, f in [('362', f362), ('371', f371)]:
    P = f @ R.T + t0
    dd, _ = tree.query(P)
    print(f'clearance {nm} -> real wrist3: min {dd.min()*1000:.2f} mm')
np.save('/data/robot_assembly/roll_best.npy', np.array([best[0]]))
print('saved roll_best.npy')
