import sys
sys.path.insert(0, '/data/robot_assembly')
import numpy as np, re
from geomlib import load_stl, AUBO_DIR

v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
zmin, zmax = v3[:, 2].min(), v3[:, 2].max()
print('wrist3 STL bbox (mm):', np.round(v3.min(0) * 1000, 1), np.round(v3.max(0) * 1000, 1))

# z-histogram: where is the mass near the two z ends?
h, edges = np.histogram(v3[:, 2], bins=40, range=(zmin, zmax))
print('\nz-profile (2mm bins, count):')
for i in range(40):
    if h[i]:
        print(f'  z {edges[i]*1000:7.1f}..{edges[i+1]*1000:7.1f}: n={h[i]}')

for tag, lo, hi in [('-z end', zmin, zmin + 0.003), ('+z end', zmax - 0.003, zmax)]:
    sel = v3[(v3[:, 2] >= lo) & (v3[:, 2] <= hi)]
    if len(sel) == 0:
        print(f'\n{tag}: EMPTY')
        continue
    r = np.hypot(sel[:, 0], sel[:, 1])
    th = np.degrees(np.arctan2(sel[:, 1], sel[:, 0])) % 360
    h36, _ = np.histogram(th, bins=36, range=(0, 360))
    print(f'\n{tag}: n={len(sel)}  radii from z-axis(0,0): {r.min()*1000:.1f}..{r.max()*1000:.1f} mm')
    print(f'   angle occupancy(10deg): {h36}')

# +y face for comparison
sel = v3[v3[:, 1] > v3[:, 1].max() - 0.0003]
r = np.hypot(sel[:, 0], sel[:, 2])
th = np.degrees(np.arctan2(sel[:, 2], sel[:, 0])) % 360
h36, _ = np.histogram(th, bins=36, range=(0, 360))
print(f'\n+y face: n={len(sel)} radii from y-axis(0,0): {r.min()*1000:.1f}..{r.max()*1000:.1f} mm')
print(f'   angle occupancy(10deg): {h36}')

xml = open(f'{AUBO_DIR}/mjcf/aubo_i5.xml').read()
print()
for mo in re.finditer(r'<site[^>]*/>', xml):
    print('SITE:', mo.group(0))
for mo in re.finditer(r'<body name="wrist[23]_Link"[^>]*>', xml):
    print('BODY:', mo.group(0))
