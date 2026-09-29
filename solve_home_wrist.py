import sys
sys.path.insert(0, '/data/robot_assembly')
import numpy as np
from geomlib import load_stl, AUBO_DIR

v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
zmin = v3[:, 2].min()
rim = v3[v3[:, 2] < zmin + 0.0008]
print(f'-z face plane: z={rim[:,2].mean()*1000:.3f} mm (n={len(rim)}, spread {(rim[:,2].max()-rim[:,2].min())*1000:.3f} mm)')
r = np.hypot(rim[:, 0], rim[:, 1])
print(f'  rim radius: {r.min()*1000:.2f}..{r.max()*1000:.2f} mm  -> flange disc D={r.max()*2000:.1f} mm, coaxial with J6 (z_w3)')

# roll evidence: stylized copy 361 vs real wrist3 bbox (x/y swap = +-90deg roll about tool axis)
f361, _ = load_stl('/data/robot_assembly/model/meshes_baked/part_361_baked.stl')
print(f'\nreal wrist3 : x[{v3[:,0].min()*1000:.1f},{v3[:,0].max()*1000:.1f}] y[{v3[:,1].min()*1000:.1f},{v3[:,1].max()*1000:.1f}] z[{v3[:,2].min()*1000:.1f},{v3[:,2].max()*1000:.1f}]')
print(f'copy 361    : x[{f361[:,0].min()*1000:.1f},{f361[:,0].max()*1000:.1f}] y[{f361[:,1].min()*1000:.1f},{f361[:,1].max()*1000:.1f}] z[{f361[:,2].min()*1000:.1f},{f361[:,2].max()*1000:.1f}]')
print('-> 361 z-span 40mm beyond print disc matches real z-span 39.5mm from flange plane;')
print('   361 x/y extents are the real x/y SWAPPED -> file frame = w3 frame rolled +-90 deg about tool axis')

# solve wrist2/wrist3 joint angles so the TOOL axis (-z_w3) points straight down, J4 fixed at -90deg
import mujoco
m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
W3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_Link')
j4 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'wrist1_joint')
j5 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'wrist2_joint')
j6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'wrist3_joint')
qadr4, qadr5, qadr6 = m.jnt_qposadr[j4], m.jnt_qposadr[j5], m.jnt_qposadr[j6]
print(f'\ncurrent home: J4={d.qpos[qadr4]:.3f} J5={d.qpos[qadr5]:.3f} J6={d.qpos[qadr6]:.3f}')
mujoco.mj_forward(m, d)
z3 = d.xmat[W3].reshape(3, 3)[:, 2]
print(f'current z_w3 (world) = {np.round(z3, 3)}   -> tool dir (-z_w3) = {np.round(-z3, 3)}  (should be 0,0,-1)')

best = None
for a4 in np.radians(np.arange(-180, 180, 5.0)):
    for a5 in np.arange(-3.05, 3.05, 0.05):
        d.qpos[qadr4], d.qpos[qadr5] = a4, a5
        mujoco.mj_kinematics(m, d)
        z3 = d.xmat[W3].reshape(3, 3)[:, 2]
        err = np.linalg.norm(z3 - np.array([0, 0, 1.0]))   # want z_w3 = +up (tool = -z = down)
        if best is None or err < best[0]:
            best = (err, a4, a5)
print(f'\ngrid best: err={best[0]*1000:.2f} mJ  J4={np.degrees(best[1]):.1f} deg  J5={np.degrees(best[2]):.1f} deg')
# refine
for a4 in np.radians(np.arange(np.degrees(best[1]) - 6, np.degrees(best[1]) + 6.01, 1.0)):
    for a5 in np.arange(best[2] - 0.06, best[2] + 0.061, 0.01):
        d.qpos[qadr4], d.qpos[qadr5] = a4, a5
        mujoco.mj_kinematics(m, d)
        z3 = d.xmat[W3].reshape(3, 3)[:, 2]
        err = np.linalg.norm(z3 - np.array([0, 0, 1.0]))
        if err < best[0]:
            best = (err, a4, a5)
err, a4, a5 = best
print(f'refined: |z_w3 - up| = {err*1000:.2f} mm-equivalent   J4 = {np.degrees(a4):.2f} deg ({a4:.6f} rad)   J5 = {np.degrees(a5):.2f} deg ({a5:.6f} rad)')
d.qpos[qadr4], d.qpos[qadr5] = a4, a5
mujoco.mj_kinematics(m, d)
G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
print(f'gripper world pos at new wrist angles: {np.round(d.xpos[G], 4)}')
print(f'tool dir = {np.round(-d.xmat[W3].reshape(3,3)[:,2], 4)}')
np.save('/data/robot_assembly/home_wrist_fix.npy', np.array([a4, a5]))
