import sys
sys.path.insert(0, '/data/robot_assembly')
import numpy as np
from geomlib import load_stl, AUBO_DIR

v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
zmax = v3[:, 2].max()
print(f'wrist3 z-extent: {v3[:,2].min()*1000:.2f}..{zmax*1000:.2f} mm')

# +z flange face: plane + bolt holes
for band in (0.3, 0.8):
    sel = v3[v3[:, 2] > zmax - band / 1000]
    if not len(sel): continue
    r = np.hypot(sel[:, 0], sel[:, 1])
    print(f'+z band -{band}mm: n={len(sel)} z mean {sel[:,2].mean()*1000:.2f} min {sel[:,2].min()*1000:.2f} max {sel[:,2].max()*1000:.2f}')
    hr, _ = np.histogram(r, bins=32, range=(0.010, 0.042))
    print('   radius hist 10..42mm (1mm bins):', hr)
    th = np.degrees(np.arctan2(sel[:, 1], sel[:, 0])) % 360
    # hole detection: verts near r=25mm, angular occupancy
    ring = sel[np.abs(r - 0.025) < 0.004]
    if len(ring):
        h, _ = np.histogram(np.degrees(np.arctan2(ring[:, 1], ring[:, 0])) % 360, bins=36, range=(0, 360))
        print(f'   bolt-circle band r=21..29mm: n={len(ring)} angle occupancy(10deg): {h}')

# is the annulus face planar? z-spread of verts at r in [17,30]
ann = v3[(np.hypot(v3[:, 0], v3[:, 1]) > 0.017) & (np.hypot(v3[:, 0], v3[:, 1]) < 0.030) & (v3[:, 2] > -0.006)]
print(f'\nannulus region r17-30, z>-6mm: n={len(ann)} z values (mm): {np.round(np.unique(np.round(ann[:,2]*1000,1)),1)}')
