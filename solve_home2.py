import sys
sys.path.insert(0, '/data/robot_assembly')
import mujoco, numpy as np

m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m)
qa = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{n}_joint')]
      for n in ['shoulder', 'upperArm', 'foreArm', 'wrist1', 'wrist2', 'wrist3']}
W3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_Link')
LIM = 3.0543
for n, v in [('shoulder', 1.570796), ('upperArm', -0.802851), ('foreArm', 0.802851),
             ('wrist3', 0.0)]:
    d.qpos[qa[n]] = v

def err(j4, j5):
    d.qpos[qa['wrist1']] = j4
    d.qpos[qa['wrist2']] = j5
    mujoco.mj_forward(m, d)
    z3 = d.xmat[W3].reshape(3, 3)[:, 2]
    return np.linalg.norm(z3 - np.array([0, 0, -1.0]))

best = None
g = np.radians(np.arange(-175, 175.1, 5.0))
for j4 in g:
    for j5 in g:
        e = err(j4, j5)
        if best is None or e < best[0]:
            best = (e, j4, j5)
for _ in range(4):
    e0, j4, j5 = best
    for dj4 in np.radians(np.arange(-4, 4.01, 0.5)):
        for dj5 in np.radians(np.arange(-4, 4.01, 0.5)):
            a4, a5 = np.clip(j4 + dj4, -LIM, LIM), np.clip(j5 + dj5, -LIM, LIM)
            e = err(a4, a5)
            if e < best[0]:
                best = (e, a4, a5)
e, j4, j5 = best
d.qpos[qa['wrist1']], d.qpos[qa['wrist2']] = j4, j5
mujoco.mj_forward(m, d)
G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
print(f'wrist1 = {np.degrees(j4):8.3f} deg ({j4: .6f} rad)   margin {np.degrees(LIM-abs(j4)):6.2f} deg')
print(f'wrist2 = {np.degrees(j5):8.3f} deg ({j5: .6f} rad)   margin {np.degrees(LIM-abs(j5)):6.2f} deg')
print(f'residual |z_w3-(0,0,-1)| = {e*1000:.3f}e-3   z_w3 = {np.round(d.xmat[W3].reshape(3,3)[:,2],5)}')
print(f'gripper pos = {np.round(d.xpos[G], 4)}   ncon = {d.ncon}')
