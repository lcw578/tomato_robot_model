"""Geometry helpers for tomato-picker model assembly.

Frames:
  G = gripper CAD frame (mm)      — gripper_v1 parts.json / meshes
  V = vehicle CAD frame (mm)      — tomato parts.json / meshes  (z_up_x_forward)
All transforms 4x4 homogeneous, acting on mm coordinates.
"""
import json
import struct
import numpy as np
from scipy.spatial import cKDTree

TOMATO_DIR = '/data/robot_model_urdf/local_mukr9r1p_iz75v8_urdf_stl'
GRIPPER_DIR = '/data/gripper_v1'
AUBO_DIR = '/home/lcw/VR_teleoperation/_refs/Auboi5_Scan_Simulator'


def load_stl(path):
    """Binary STL -> (vertices Nx3 float64, triangles Mx3 int). Merges duplicate verts."""
    with open(path, 'rb') as f:
        header = f.read(80)
        if header[:5] == b'solid' and b'facet' in header:
            return _load_ascii_stl(path)
        n = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
        tris = data[:, 12:48].copy().view('<f4').reshape(n, 3, 3).astype(np.float64)
    verts, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    return verts, faces


def _load_ascii_stl(path):
    verts = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            s = line.split()
            if s and s[0] == 'vertex':
                verts.append([float(s[1]), float(s[2]), float(s[3])])
    tris = np.array(verts, dtype=np.float64).reshape(-1, 3, 3)
    v, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def sample_mesh(verts, faces, n=4000, seed=0):
    """Area-weighted surface sampling."""
    rng = np.random.default_rng(seed)
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    w = area / area.sum()
    idx = rng.choice(len(faces), size=n, p=w)
    u = rng.random(n)[:, None]
    v = rng.random(n)[:, None]
    flip = (u + v > 1)
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    p = verts[faces[idx, 0]] + u * e1[idx] + v * e2[idx]
    return p


def bbox_center(bbox):
    return (np.array(bbox['min']) + np.array(bbox['max'])) / 2.0


def bbox_dims(bbox):
    return np.array(bbox['max']) - np.array(bbox['min'])


def kabsch(src, dst):
    """Rigid transform mapping src -> dst (Nx3)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cd - R @ cs
    return T


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def rpy_deg(R):
    """Rotation matrix -> ZYX euler degrees (yaw/pitch/roll) for reporting."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1, 1)
    pitch = np.degrees(np.arcsin(sy))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def icp(src, dst_tree, dst_pts, T_init, iters=30, max_dist=8.0, tol=1e-5):
    """Point-to-point ICP, src Nx3 -> dst (point cloud + prebuilt tree)."""
    T = T_init.copy()
    prev = 1e9
    for it in range(iters):
        s = apply(T, src)
        dist, idx = dst_tree.query(s, distance_upper_bound=max_dist)
        ok = np.isfinite(dist)
        if ok.sum() < 50:
            break
        T_step = kabsch(s[ok], dst_pts[idx[ok]])
        T = T_step @ T
        mean = dist[ok].mean()
        if abs(prev - mean) < tol:
            break
        prev = mean
    s = apply(T, src)
    dist, _ = dst_tree.query(s, distance_upper_bound=max_dist)
    ok = np.isfinite(dist)
    return T, float(dist[ok].mean()) if ok.sum() else float('inf'), int(ok.sum())


def quat_from_R(R):
    """Rotation matrix -> quaternion (w,x,y,z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def R_from_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
