"""Re-solve home wrist joints so the NEW tool axis (+z_w3) points straight down.

The mount moved from the +y bulge to the +z tool flange, so the old keyframe
(wrist1=-90, wrist2=-92, tuned for -y_w3 down) is 90 deg off.  Keep
J1/J2/J3 (the reach, mast avoidance) and J6=0; solve J4/J5 (2-DOF can point
z_w3 anywhere) so that +z_w3 = (0,0,-1) in the world.
"""
import sys
sys.path.insert(0, '/data/robot_assembly')
import mujoco, numpy as np

m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m)
qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{n}_joint')]
        for n in ['shoulder', 'upperArm', 'foreArm', 'wrist1', 'wrist2', 'wrist3']}
W3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_Link')
G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
HOME = dict(shoulder=1.570796, upperArm=-0.802851, foreArm=0.802851)

def zdir(j4, j5, j6=0.0):
    for n, v in [('wrist1', j4), ('wrist2', j5), ('wrist3', j6)]:
        d.qpos[qadr[n]] = v
    for n, v in HOME.items():
        d.qpos[qadr[n]] = v
    mujoco.mj_forward(m, d)
    return d.xmat[W3].reshape(3, 3)[:, 2].copy(), d.xpos[G].copy()

best = None
LIM = 3.04   # official aubo_i5.xml wrist range ±3.04 rad
sols = []
for j4 in np.radians(np.arange(-174, 174.1, 2.0)):
    for j5 in np.radians(np.arange(-174, 174.1, 2.0)):
        z3, _ = zdir(j4, j5)
        err = np.linalg.norm(z3 - np.array([0, 0, -1.0]))
        if err < 0.02:
            sols.append((err, j4, j5))
        if best is None or err < best[0]:
            best = (err, j4, j5)
# refine each coarse hit, keep all in-range solutions
fine = []
for e0, j4, j5 in sols:
    for _ in range(3):
        for dj4 in np.radians(np.arange(-2, 2.01, 0.25)):
            for dj5 in np.radians(np.arange(-2, 2.01, 0.25)):
                z3, _ = zdir(j4 + dj4, j5 + dj5)
                err = np.linalg.norm(z3 - np.array([0, 0, -1.0]))
                if err < e0:
                    e0, j4, j5 = err, j4 + dj4, j5 + dj5
    if e0 < 0.005 and abs(j4) < LIM and abs(j5) < LIM:
        fine.append((e0, j4, j5))
print('in-range solutions (err, J4 deg, J5 deg, limit margin deg):')
for e0, j4, j5 in sorted(fine):
    marg = min(LIM - abs(j4), LIM - abs(j5))
    print(f'  err {e0*1000:6.3f}   J4 {np.degrees(j4):8.2f}  J5 {np.degrees(j5):8.2f}   margin {np.degrees(marg):6.1f}')
best = min(fine, key=lambda s: -min(LIM - abs(s[1]), LIM - abs(s[2]))) if fine else best
err, j4, j5 = best
z3, gp = zdir(j4, j5)
print(f'solved: wrist1={np.degrees(j4):.2f} deg ({j4:.6f} rad)  wrist2={np.degrees(j5):.2f} deg ({j5:.6f} rad)')
print(f'residual |z_w3-(0,0,-1)| = {err*1000:.3f} mm-equivalent (unit vec, x1000)')
print(f'z_w3 world = {np.round(z3, 5)}   gripper pos = {np.round(gp, 4)}')
