"""Phase A: register gripper assembly into vehicle CAD frame -> T_V_from_G."""
import sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import *
import numpy as np, json

tg = json.load(open(f'{TOMATO_DIR}/parts.json'))['parts']
gp = json.load(open(f'{GRIPPER_DIR}/parts.json'))['parts']

# --- 1. volume fingerprint candidates (tolerance 2% volume, 3% diag) ---
cands = []
for g in gp:
    gv = g['volume_estimate']
    gd = float(np.linalg.norm(bbox_dims(g['bbox'])))
    if gv <= 1.0:
        continue  # specks: unusable fingerprint
    for t in tg:
        tv = t['volume_estimate']
        td = float(np.linalg.norm(bbox_dims(t['bbox'])))
        if abs(tv - gv) / gv < 0.02 and abs(td - gd) / max(gd, 1) < 0.03:
            cands.append((g['part_id'], t['part_id'], gv))
print(f"volume-fingerprint candidate pairs: {len(cands)}")
for c in cands:
    print("  ", c)

# --- 2. initial T from Kabsch over candidate bbox centers (all of them, robust check after) ---
src = np.array([bbox_center(next(p for p in gp if p['part_id'] == a)['bbox']) for a, b, v in cands])
dst = np.array([bbox_center(next(p for p in tg if p['part_id'] == b)['bbox']) for a, b, v in cands])
T0 = kabsch(src, dst)
res = np.linalg.norm(apply(T0, src) - dst, axis=1)
print("\nKabsch-on-all-candidates residuals (mm):", np.round(res, 2))

# --- 3. point clouds ---
def cloud(parts, d, per=2500, cap=None):
    pts = []
    for p in parts:
        try:
            v, f = load_stl(f"{d}/{p['mesh_file']}")
        except Exception:
            continue
        if len(f) == 0:
            continue
        n = min(per, max(200, len(f)))
        pts.append(sample_mesh(v, f, n=n))
    P = np.vstack(pts)
    return P

G = cloud(gp, GRIPPER_DIR)
print("gripper cloud:", G.shape)

# target: tomato parts whose bbox center is within the tail window (generous)
tw = lambda p: bbox_center(p['bbox'])
tail_parts = [p for p in tg
              if -650 < tw(p)[0] < -230 and 30 < tw(p)[1] < 280 and -1420 < tw(p)[2] < -1050]
print("tail-window tomato parts:", len(tail_parts))
V = cloud(tail_parts, TOMATO_DIR, per=1200)
print("tail cloud:", V.shape)
tree = cKDTree(V)

# --- 4. ICP refine (coarse -> fine) ---
T, rms, nin = icp(G, tree, V, T0, iters=40, max_dist=25.0)
print(f"ICP pass1: rms={rms:.2f}mm inliers={nin}/{len(G)}")
T, rms, nin = icp(G, tree, V, T, iters=40, max_dist=6.0)
print(f"ICP pass2: rms={rms:.2f}mm inliers={nin}/{len(G)}")
T, rms, nin = icp(G, tree, V, T, iters=60, max_dist=2.5)
print(f"ICP pass3: rms={rms:.2f}mm inliers={nin}/{len(G)}")

# residual distribution
d, _ = tree.query(apply(T, G), distance_upper_bound=3.0)
print("residual percentiles (mm): 50%=%.2f 80%=%.2f 95%=%.2f" % tuple(
    np.percentile(d[np.isfinite(d)], [50, 80, 95])))

R = T[:3, :3]
print("\nT_V_from_G (cad mm):")
print("R=", np.round(R, 5), " rpy(deg)=", np.round(rpy_deg(R), 2))
print("t=", np.round(T[:3, 3], 2))

np.save('/data/robot_assembly/T_V_from_G.npy', T)
print("saved T_V_from_G.npy")
