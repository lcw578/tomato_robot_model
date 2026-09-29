"""Phase B v3: seed upperArm on the observed tube (part_344), then walk down."""
import sys
sys.path.insert(0, '/data/robot_assembly')
from geomlib import *
import numpy as np, json

OUT = '/data/robot_assembly'
tg = json.load(open(f'{TOMATO_DIR}/parts.json'))['parts']
gp = json.load(open(f'{GRIPPER_DIR}/parts.json'))['parts']
T_VG = np.load(f'{OUT}/T_V_from_G.npy')

cache = f'{OUT}/veh_cloud.npz'
z = np.load(cache); V, vpart = z['V'], z['vpart']

G_pts = []
for p in gp:
    v, f = load_stl(f"{GRIPPER_DIR}/{p['mesh_file']}")
    if len(f): G_pts.append(sample_mesh(v, f, n=min(1200, max(150, len(f)))))
G = apply(T_VG, np.vstack(G_pts))
claimed = np.zeros(len(V), bool)
d, _ = cKDTree(V).query(G, distance_upper_bound=3.0)
claimed[np.where(np.isfinite(d))[0]] = True
for i, p in enumerate(tg):
    if int(p['part_id'].split('_')[1]) in {410, 389, 402, 418, 331, 344}:
        claimed[vpart == i] = True          # gripper solids + base + upper-arm tube seed part
print("claimed:", claimed.sum())

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
    MESH[name] = load_stl(f'{AUBO_DIR}/models/{name}.STL')

def T_from_pq(pos, quat):
    T = np.eye(4); T[:3, :3] = R_from_quat(np.array(quat)); T[:3, 3] = pos; return T

def Tz(q):
    T = np.eye(4); c, s = np.cos(q), np.sin(q)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; return T

def child_pose(Tp, pos, quat, q):
    return Tp @ T_from_pq(pos, quat) @ Tz(q)

def capped_score(pc, tree, T, cap):
    d, _ = tree.query(apply(T, pc), distance_upper_bound=cap)
    return float(np.where(np.isfinite(d), d, cap).mean())

def sample_link(name, n, seed=1):
    v, f = MESH[name]
    return sample_mesh(v, f, n=n, seed=seed) * 1000.0

def window_mask(center, radius):
    return (np.abs(V - center) < radius).all(1) & ~claimed

# base from part_331 (z-down, flange face at platform underside)
Rx = np.diag([1.0, -1.0, -1.0])
T_base = np.eye(4); T_base[:3, :3] = Rx; T_base[:3, 3] = [2.0, 4.0, -704.0]
poses = {'base_link': T_base}
print("base seed:", np.round(T_base[:3, 3], 1))

# ---------- upperArm seeded on part_344 tube ----------
pv, pf = load_stl(f"{TOMATO_DIR}/meshes/part_344_solid_344.stl")
# tube axis by PCA
c_part = pv.mean(0)
u, s, vt = np.linalg.svd(pv - c_part, full_matrices=False)
axis_part = vt[0]
t = (pv - c_part) @ axis_part
print(f"part_344 axial extent: {t.min():.0f}..{t.max():.0f} mm, cross std: {s[1]/np.sqrt(len(pv)):.1f}/{s[2]/np.sqrt(len(pv)):.1f}")

mv, mf = MESH['upperArm_Link']
mv_mm = mv * 1000.0
c_mesh = mv_mm.mean(0)
um, sm, vtm = np.linalg.svd(mv_mm - c_mesh, full_matrices=False)
axis_mesh = vtm[0]

pc = sample_link('upperArm_Link', 5000, seed=3)
m = window_mask(c_part, 500)
tgt = V[m]; tt = cKDTree(tgt)

best = None
for sign in (+1, -1):
    b = sign * axis_mesh
    for roll in np.arange(0, 2 * np.pi, np.pi / 12):
        # rotation mapping mesh axis b -> part axis, with roll about it
        tmp = np.array([0, 0, 1.0]) if abs(b @ np.array([0, 0, 1.0])) < 0.9 else np.array([1.0, 0, 0])
        x = np.cross(tmp, b); x /= np.linalg.norm(x)
        y = np.cross(b, x)
        R0 = np.stack([x, y, b], axis=1)      # maps e_z -> b
        cr, sr = np.cos(roll), np.sin(roll)
        R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        R_mesh = R0 @ R_roll                  # mesh frame orientation giving axis b
        # target orientation: maps mesh +x (approx axis) -> part axis a
        # build orientation mapping b -> axis_part with same roll parameterization
        tmp2 = np.array([0, 0, 1.0]) if abs(axis_part @ np.array([0, 0, 1.0])) < 0.9 else np.array([1.0, 0, 0])
        x2 = np.cross(tmp2, axis_part); x2 /= np.linalg.norm(x2)
        y2 = np.cross(axis_part, x2)
        R0t = np.stack([x2, y2, axis_part], axis=1)   # maps e_z -> axis_part
        R_t = R0t @ R_roll
        R = R_t @ R_mesh.T
        T0 = np.eye(4); T0[:3, :3] = R; T0[:3, 3] = c_part - R @ c_mesh
        s20 = capped_score(pc, tt, T0, 20.0)
        T1, rms, nin = icp(pc, tt, tgt, T0, iters=40, max_dist=15.0)
        T1, rms, nin = icp(pc, tt, tgt, T1, iters=30, max_dist=6.0)
        s3 = capped_score(pc, tt, T1, 3.0)
        if best is None or s3 < best[0]:
            best = (s3, rms, T1)
s3, rms, T_upper = best
d3, _ = tt.query(apply(T_upper, pc), distance_upper_bound=3.0)
frac = float(np.isfinite(d3).mean())
print(f"upperArm tube-seed: rms={rms:.2f} score3={s3:.2f} inliers={frac*100:.0f}% center={np.round(T_upper[:3,3],1)}")

# ---------- extract (q1,q2) from T_upper ----------
def family_err(q1, q2):
    T_sh = T_base @ T_from_pq(CHAIN[0][1], CHAIN[0][2]) @ Tz(q1)
    Tu = T_sh @ T_from_pq(CHAIN[1][1], CHAIN[1][2]) @ Tz(q2)
    Rm = Tu[:3, :3] @ T_upper[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    pos = np.linalg.norm(Tu[:3, 3] - T_upper[:3, 3])
    return ang, pos

best = None
for q1 in np.arange(-np.pi, np.pi, np.radians(4)):
    for q2 in np.arange(-np.pi, np.pi, np.radians(4)):
        a, p = family_err(q1, q2)
        e = a + p * 0.5   # deg + mm/2 weighting
        if best is None or e < best[0]:
            best = (e, q1, q2)
e, q1, q2 = best
from scipy.optimize import minimize
res = minimize(lambda x: sum(family_err(x[0], x[1])), [q1, q2], method='Nelder-Mead')
q1, q2 = res.x % (2 * np.pi)
a, p = family_err(q1, q2)
print(f"projection: q1={np.degrees(q1):.1f} q2={np.degrees(q2):.1f}  (ang err {a:.2f} deg, pos err {p:.2f} mm)")

T_sh = T_base @ T_from_pq(CHAIN[0][1], CHAIN[0][2]) @ Tz(q1)
T_upper = T_sh @ T_from_pq(CHAIN[1][1], CHAIN[1][2]) @ Tz(q2)
poses['shoulder_Link'] = T_sh
poses['upperArm_Link'] = T_upper
np.save(f'{OUT}/T_V_from_base_link.npy', T_base)
np.save(f'{OUT}/T_V_from_shoulder_Link.npy', T_sh)
np.save(f'{OUT}/T_V_from_upperArm_Link.npy', T_upper)

# ---------- walk down: foreArm .. wrist3 (1-DOF constrained scans) ----------
for idx in range(2, len(CHAIN)):
    name, pos, quat = CHAIN[idx]
    T_parent = poses[CHAIN[idx - 1][0]]
    pc = sample_link(name, 4000, seed=idx + 21)
    v, f = MESH[name]
    link_r = 1.2 * np.linalg.norm(v.max(0) - v.min(0)) / 2 * 1000
    m = window_mask(T_parent[:3, 3], np.linalg.norm(pos) + link_r + 60)
    tgt = V[m]; tt = cKDTree(tgt)
    qs = np.arange(-np.pi, np.pi, np.radians(2))
    scores = [capped_score(pc, tt, child_pose(T_parent, pos, quat, q), 20.0) for q in qs]
    order = np.argsort(scores)
    best = None
    for k in order[:5]:
        T1, _, _ = icp(pc, tt, tgt, child_pose(T_parent, pos, quat, qs[k]), iters=25, max_dist=12.0)
        M = np.linalg.inv(T_from_pq(pos, quat)) @ np.linalg.inv(T_parent) @ T1
        q0 = np.arctan2(M[1, 0], M[0, 0])
        for dq in np.radians([-3, -1.5, 0, 1.5, 3]):
            Tp = child_pose(T_parent, pos, quat, q0 + dq)
            s3 = capped_score(pc, tt, Tp, 3.0)
            if best is None or s3 < best[0]:
                best = (s3, q0 + dq, Tp)
    s3, q, T1 = best
    d3, _ = tt.query(apply(T1, pc), distance_upper_bound=3.0)
    frac = float(np.isfinite(d3).mean())
    poses[name] = T1
    flag = "" if frac > 0.2 else "  <<< LOW"
    print(f"{name:14s}: q={np.degrees(q):7.1f}  score3={s3:.2f}mm in={frac*100:.0f}% c={np.round(T1[:3,3],1)}{flag}")
    np.save(f'{OUT}/T_V_from_{name}.npy', T1)
    d, _ = cKDTree(V).query(apply(T1, pc), distance_upper_bound=4.0)
    claimed[np.where(np.isfinite(d))[0]] = True

np.save(f'{OUT}/claimed.npy', claimed)
qpos = {}
for idx in range(len(CHAIN)):
    nm, ps, qt = CHAIN[idx]
    Tp = poses['base_link' if idx == 0 else CHAIN[idx - 1][0]]
    M = np.linalg.inv(T_from_pq(ps, qt)) @ np.linalg.inv(Tp) @ poses[nm]
    qpos[nm] = float(np.arctan2(M[1, 0], M[0, 0]))
json.dump(qpos, open(f'{OUT}/q_cad.json', 'w'), indent=1)
print("q:", {k: round(np.degrees(v), 1) for k, v in qpos.items()})

# cross-check flange vs gripper
T_w3 = poses['wrist3_Link']
v3, _ = MESH['wrist3_Link']
face_c_local = v3[v3[:, 2] < v3[:, 2].min() + 0.002].mean(0) * 1000
flange_c_V = apply(T_w3, face_c_local[None])[0]
flange_axis_V = -T_w3[:3, 2]
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
print("\nflange center (chain):", np.round(flange_c_V, 1), " axis:", np.round(flange_axis_V, 3))
print("grip face  (phaseA) :", np.round(face_c_V, 1), " axis:", np.round(tool_V, 3))
print("dc:", np.round(flange_c_V - face_c_V, 1), " axis deg:",
      round(np.degrees(np.arccos(np.clip(flange_axis_V @ tool_V, -1, 1))), 2))
