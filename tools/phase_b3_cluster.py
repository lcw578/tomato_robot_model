"""Phase B v4: contact-based connectivity clustering of robot_arm parts.

Bolted contact = min surface distance < 2.5mm. Cluster anchored at part_370
(stylized wrist, unambiguous) = arm+gripper+end-effector cluster.
Gripper solids identified via T_VG proximity; end-effector furniture = cluster
parts within 130mm of the gripper mount face (kept, attached to gripper frame);
everything else in the cluster = stylized arm (removed, replaced by official i5).
"""
import sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import *
import numpy as np, json, os

OUT = '/data/robot_assembly'
tg = json.load(open(f'{TOMATO_DIR}/parts.json'))['parts']
um = json.load(open(f'{TOMATO_DIR}/user_model.json'))
gp = json.load(open(f'{GRIPPER_DIR}/parts.json'))['parts']
T_VG = np.load(f'{OUT}/T_V_from_G.npy')

# parts belonging to separate links (wheels/act/base_link) — excluded from clustering
separate = set()
for l in um['links']:
    if l['name'] != 'robot_arm':
        separate.update(l['part_ids'])
ra = [l for l in um['links'] if l['name'] == 'robot_arm'][0]
ra_ids = set(ra['part_ids'])
print(f"robot_arm parts: {len(ra_ids)}, separate-link parts: {len(separate)}")

# sample each robot_arm part (bbox-overlap pairs only, so cache meshes)
CACHE = f'{OUT}/part_samples.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    samples = z['samples'].item()
else:
    samples = {}
    for pid in ra_ids:
        p = next(q for q in tg if q['part_id'] == pid)
        try:
            v, f = load_stl(f"{TOMATO_DIR}/{p['mesh_file']}")
        except Exception:
            continue
        if len(f) == 0: continue
        samples[pid] = sample_mesh(v, f, n=min(500, max(100, len(f))), seed=5)
    np.savez(CACHE, samples=samples)
print("sampled parts:", len(samples))

pids = list(samples)
bbs = {}
for pid in pids:
    P = samples[pid]
    bbs[pid] = (P.min(0) - 3.0, P.max(0) + 3.0)

# union-find
parent = {pid: pid for pid in pids}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a, b):
    ra_, rb_ = find(a), find(b)
    if ra_ != rb_: parent[rb_] = ra_

pairs_checked = 0
for i in range(len(pids)):
    for j in range(i + 1, len(pids)):
        a, b = pids[i], pids[j]
        (amn, amx), (bmn, bmx) = bbs[a], bbs[b]
        if np.any(amn > bmx) or np.any(bmn > amx):
            continue
        pairs_checked += 1
        Pa, Pb = samples[a], samples[b]
        t = cKDTree(Pb)
        d, _ = t.query(Pa, distance_upper_bound=2.5)
        if np.isfinite(d).any():
            union(a, b)
print(f"bbox-adjacent pairs checked: {pairs_checked}")

clusters = {}
for pid in pids:
    clusters.setdefault(find(pid), []).append(pid)
clusters = sorted(clusters.values(), key=len, reverse=True)
print(f"clusters: {len(clusters)}, sizes: {[len(c) for c in clusters[:8]]}")

arm_cluster = None
for c in clusters:
    if 'part_370' in c:
        arm_cluster = c
        break
print(f"arm+ee cluster: {len(arm_cluster)} parts")

# gripper solids among cluster (proximity to transformed gripper cloud)
G_pts = []
for p in gp:
    v, f = load_stl(f"{GRIPPER_DIR}/{p['mesh_file']}")
    if len(f): G_pts.append(sample_mesh(v, f, n=min(800, max(150, len(f)))))
G = apply(T_VG, np.vstack(G_pts))
tglob = cKDTree(V if False else np.vstack([samples[p] for p in arm_cluster]))
# per-part: fraction of its points within 2.5mm of G
grip_parts, ee_parts, arm_parts = [], [], []
gtree = cKDTree(G)
for pid in arm_cluster:
    P = samples[pid]
    d, _ = gtree.query(P, distance_upper_bound=2.5)
    frac = float(np.isfinite(d).mean())
    if frac > 0.5:
        grip_parts.append(pid)
    else:
        # distance to gripper mount face
        face_c_V = np.array([-342.5, 147.5, -1186.8])
        c = P.mean(0)
        (ee_parts if np.linalg.norm(c - face_c_V) < 130 else arm_parts).append(pid)
print(f"gripper solids: {len(grip_parts)}, end-effector furniture: {len(ee_parts)}, stylized arm: {len(arm_parts)}")

static = [p for p in pids if p not in set(arm_cluster)]
print(f"static vehicle parts: {len(static)}")
json.dump({'gripper': sorted(grip_parts), 'end_effector': sorted(ee_parts),
           'arm': sorted(arm_parts), 'static': sorted(static)},
          open(f'{OUT}/part_classes.json', 'w'), indent=1)
print("saved part_classes.json")
print("arm parts:", sorted(arm_parts))
print("ee parts:", sorted(ee_parts))
