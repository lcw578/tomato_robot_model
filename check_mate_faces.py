import numpy as np, sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import load_stl

f362, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_362_baked.stl')
f371, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_371_baked.stl')
f361, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_361_baked.stl')

def face_stats(P, tag):
    r = np.hypot(P[:, 0], P[:, 1])
    th = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360
    h, _ = np.histogram(th, bins=36, range=(0, 360))
    print(f'{tag}: n={len(P)} center=({P[:,0].mean()*1000:.1f},{P[:,1].mean()*1000:.1f}) mm')
    print(f'   radii from (0,0): {r.min()*1000:.1f}..{r.max()*1000:.1f} mm')
    print(f'   angle occupancy(10deg bins): {h}')
    hr, _ = np.histogram(r, bins=16, range=(0, 0.040))
    print(f'   radius hist 0..40mm (2.5mm bins): {hr}')

z362 = f362[:, 2].max()
face_stats(f362[f362[:, 2] > z362 - 0.0003], '362 top face (-0.3mm band)')
face_stats(f362[f362[:, 2] > z362 - 0.0015], '362 top face (-1.5mm band)')
z371 = f371[:, 2].max()
face_stats(f371[f371[:, 2] > z371 - 0.0003], '371 top face')
zb = f361[:, 2].min()
face_stats(f361[f361[:, 2] < zb + 0.0003], '361 bottom face')
