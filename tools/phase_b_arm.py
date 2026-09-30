"""Phase B v2: seed at AUBO base (part_331 located), walk DOWN the chain with
constrained 1-DOF (joint-angle) registration. Cross-check wrist3 against the
gripper mount face from Phase A."""
import sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import *
import numpy as np, json, os

OUT = '/data/robot_assembly'
tg = json.load(open(f'{TOMATO_DIR}/parts.json'))['parts']
gp = json.load(open(f'{GRIPPER_DIR}/parts.json'))['parts']
T_VG = np.load(f'{OUT}/T_V_from_G.npy')

# ---------------- vehicle cloud ----------------
cache = f'{OUT}/veh_cloud.npz'
z = np.load(cache)
V, vpart = z['V'], z['vpart']
print("vehicle cloud:", V.shape)

# ---------------- gripper cloud in V (for claiming) ----------------
G_pts = []
for p in gp:
    v, f = load_stl(f"{GRIPPER_DIR}/{p['mesh_file']}")
    if len(f): G_pts.append(sample_mesh(v, f, n=min(1500, max(150, len(f)))))
G = apply(T_VG, np.vstack(G_pts))
claimed = np.zeros(len(V), bool)
d, _ = cKDTree(V).query(G, distance_upper_bound=3.0)
claimed[np.where(np.isfinite(d))[0]] = True
for i, p in enumerate(tg):
    if int(p['part_id'].split('_')[1]) in {410, 389, 402, 418}:
        claimed[vpart == i] = True
print("claimed by gripper:", claimed.sum())

# ---------------- AUBO chain (mm) ----------------
CHAIN = [
    ('shoulder_Link', (0, 0, 122.0),   (-3.67321e-06, 0, 0, 1)),
    ('upperArm_Link', (0, 121.5, 0),   (0.499998, -0.5, -0.5, -0.500002)),
    ('foreArm_Link',  (408.0, 0, 0),   (-3.67321e-06, -1, 0, 0)),
    ('wrist1_Link',   (376.0, 0, 0),   (-2.59734e-06, 0.707105, 0.707108, -2.59735e-06)),
    ('wrist2_Link',   (0, 102.5, 0),   (0.707105, -0.707108, 0, 0)),
    ('wrist3_Link',   (0, -94.0, 0),   (0.707105, 0.707108, 0, 0)),
]
MESH = {}
for name in ['base_link'] + [c[0] for c in CHAIN]:
    v, f = load_stl(f'{AUBO_DIR}/models/{name}.STL')
    MESH[name] = (v, f)

def T_from_pq(pos, quat):
    T = np.eye(4); T[:3, :3] = R_from_quat(np.array(quat)); T[:3, 3] = pos
    return T

def Tz(q):
    T = np.eye(4); c, s = np.cos(q), np.sin(q)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T

def child_pose(T_parent, pos, quat, q):
    return T_parent @ T_from_pq(pos, quat) @ Tz(q)

def joint_q(T_parent, T_child, pos, quat):
    M = np.linalg.inv(T_from_pq(pos, quat)) @ np.linalg.inv(T_parent) @ T_child
    return np.arctan2(M[1, 0], M[0, 0])

def capped_score(pc, tree, T, cap):
    d, _ = tree.query(apply(T, pc), distance_upper_bound=cap)
    d = np.where(np.isfinite(d), d, cap)
    return d.mean()

def sample_link(name, n, seed=1):
    v, f = MESH[name]
    return sample_mesh(v, f, n=n, seed=seed) * 1000.0

def window_mask(center, radius):
    return (np.abs(V - center) < radius).all(1) & ~claimed

# ---------------- seed: base_link from part_331 ----------------
# part_331 bbox z [-754,-704]; mounted UNDER platform -> base +z points DOWN in V.
# Mount face (mesh z=0) touches platform underside at z=-704. Yaw absorbed into q1.
Rx = np.eye(4); Rx[:3, :3] = np.diag([1.0, -1.0, -1.0])   # 180 deg about x: z down
T_base = Rx  # rotation part
T_base[:3, 3] = [2.0, 4.0, -704.0]
poses = {'base_link': T_base}
print(f"base_link seed : center={np.round(T_base[:3,3],1)} (z-down, from part_331 bbox)")
np.save(f'{OUT}/T_V_from_base_link.npy', T_base)

# ---------------- walk DOWN ----------------
for idx in range(len(CHAIN)):
    name, pos, quat = CHAIN[idx]
    T_parent = poses['base_link' if idx == 0 else CHAIN[idx - 1][0]]
    pc = sample_link(name, 4000, seed=idx + 11)
    v, f = MESH[name]
    link_r = 1.2 * np.linalg.norm(v.max(0) - v.min(0)) / 2 * 1000
    off = np.linalg.norm(pos)
    m = window_mask(T_parent[:3, 3], off + link_r + 60)
    tgt = V[m]; tt = cKDTree(tgt)

    def refine(T_init):
        T1, _, _ = icp(pc, tt, tgt, T_init, iters=25, max_dist=12.0)
        M = np.linalg.inv(T_from_pq(pos, quat)) @ np.linalg.inv(T_parent) @ T1
        q0 = np.arctan2(M[1, 0], M[0, 0])
        best = None
        for dq in np.radians([-3, -1.5, 0, 1.5, 3]):
            Tp = child_pose(T_parent, pos, quat, q0 + dq)
            s3 = capped_score(pc, tt, Tp, 3.0)
            if best is None or s3 < best[0]:
                best = (s3, q0 + dq, Tp)
        return best

    if idx == 1:
        # upperArm level: (q1, q2) joint 2-DOF scan (q1 = shoulder yaw, unknown)
        name_s, pos_s, quat_s = CHAIN[0]
        qs1 = np.arange(-np.pi, np.pi, np.radians(10))
        qs2 = np.arange(-np.pi, np.pi, np.radians(6))
        cands = []
        T_sh0 = child_pose(T_parent, pos_s, quat_s, 0.0)
        for q1 in qs1:
            T_sh = child_pose(T_parent, pos_s, quat_s, q1)
            for q2 in qs2:
                Tp = child_pose(T_sh, pos, quat, q2)
                cands.append((capped_score(pc, tt, Tp, 20.0), q1, q2))
        cands.sort(key=lambda x: x[0])
        best = None
        for _, q1, q2 in cands[:6]:
            Tp0 = child_pose(child_pose(T_parent, pos_s, quat_s, q1), pos, quat, q2)
            s3, q2r, Tp = refine(Tp0)
            if best is None or s3 < best[0]:
                best = (s3, q1, q2r, Tp)
        s3, q1, q2, T1 = best
        poses['shoulder_Link'] = child_pose(T_parent, pos_s, quat_s, q1)
        np.save(f'{OUT}/T_V_from_shoulder_Link.npy', poses['shoulder_Link'])
        d3, _ = tt.query(apply(T1, pc), distance_upper_bound=3.0)
        frac = float(np.isfinite(d3).mean())
        poses[name] = T1
        print(f"shoulder_Link  : q1={np.degrees(q1):7.1f} deg (from joint scan)")
        print(f"{name:14s}: q2={np.degrees(q2):7.1f} deg  score3={s3:.2f}mm inliers={frac*100:.0f}% "
              f"center={np.round(T1[:3,3],1)}")
        np.save(f'{OUT}/T_V_from_{name}.npy', T1)
        d, _ = cKDTree(V).query(apply(T1, pc), distance_upper_bound=4.0)
        claimed[np.where(np.isfinite(d))[0]] = True
        continue

    qs = np.arange(-np.pi, np.pi, np.radians(2))
    scores = [capped_score(pc, tt, child_pose(T_parent, pos, quat, q), 20.0) for q in qs]
    order = np.argsort(scores)
    best = None
    for k in order[:5]:
        s3, q, Tp = refine(child_pose(T_parent, pos, quat, qs[k]))
        if best is None or s3 < best[0]:
            best = (s3, q, Tp)
    s3, q, T1 = best
    d3, _ = tt.query(apply(T1, pc), distance_upper_bound=3.0)
    frac = float(np.isfinite(d3).mean())
    poses[name] = T1
    flag = "" if frac > 0.25 else "  <<< LOW INLIERS"
    print(f"{name:14s}: q={np.degrees(q):7.1f} deg  score3={s3:.2f}mm inliers={frac*100:.0f}% "
          f"center={np.round(T1[:3,3],1)}{flag}")
    np.save(f'{OUT}/T_V_from_{name}.npy', T1)
    d, _ = cKDTree(V).query(apply(T1, pc), distance_upper_bound=4.0)
    claimed[np.where(np.isfinite(d))[0]] = True

np.save(f'{OUT}/claimed.npy', claimed)
qpos = {}
for idx in range(len(CHAIN)):
    nm, ps, qt = CHAIN[idx]
    Tp = poses['base_link' if idx == 0 else CHAIN[idx - 1][0]]
    qpos[nm] = float(joint_q(Tp, poses[nm], ps, qt))
json.dump(qpos, open(f'{OUT}/q_cad.json', 'w'), indent=1)
print("\nCAD snapshot joint angles (deg):", {k: round(np.degrees(v), 1) for k, v in qpos.items()})

# ---------------- cross-check: wrist3 vs gripper mount face ----------------
T_w3 = poses['wrist3_Link']
# flange face: wrist3 mesh z=-0.0395 plane -> flange origin in wrist3 frame
v3, f3 = MESH['wrist3_Link']
face_c_local = v3[v3[:, 2] < v3[:, 2].min() + 0.002].mean(0) * 1000
flange_c_V = apply(T_w3, face_c_local[None])[0]
flange_axis_V = T_w3[:3, 2]  # wrist3 +z points from flange back into wrist
face_axis_V = -flange_axis_V  # tool direction
print("\nflange face center from chain:", np.round(flange_c_V, 1))
print("flange tool axis from chain  :", np.round(face_axis_V, 3))
gp_base_ids = {'part_003','part_010','part_011','part_012','part_013','part_027','part_028',
               'part_029','part_030','part_031','part_032','part_033','part_034','part_035',
               'part_036','part_037','part_038','part_039','part_040','part_041','part_042',
               'part_043','part_044','part_045','part_046','part_047','part_001','part_009'}
base_pts = np.vstack([load_stl(f"{GRIPPER_DIR}/{p['mesh_file']}")[0]
                      for p in gp if p['part_id'] in gp_base_ids])
zmax = base_pts[:, 2].max()
face_c_G = base_pts[base_pts[:, 2] > zmax - 1.5].mean(0)
face_c_V = apply(T_VG, face_c_G[None])[0]
tool_V = apply(T_VG, (face_c_G + np.array([0, 0, -10.0]))[None])[0] - face_c_V
tool_V /= np.linalg.norm(tool_V)
print("gripper mount face center    :", np.round(face_c_V, 1))
print("gripper tool axis            :", np.round(tool_V, 3))
print("face center discrepancy (mm) :", np.round(flange_c_V - face_c_V, 1))
print("axis angle discrepancy (deg) :", round(np.degrees(np.arccos(np.clip(face_axis_V @ tool_V, -1, 1))), 2))
