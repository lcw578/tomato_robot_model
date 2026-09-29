"""Phase D v2: verification on the clean model."""
import sys
sys.path.insert(0, '/data/robot_assembly')
import mujoco, numpy as np
import re
from PIL import Image

MODEL = '/data/robot_assembly/model/tomato_picker.xml'
m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

ARM = ['shoulder_joint','upperArm_joint','foreArm_joint','wrist1_joint','wrist2_joint','wrist3_joint']
J = {jn: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in
     ARM + ['gear_joint','right_finger_joint','left_finger_joint','act_left_joint','act_right_joint']}

# 1) servo hold at zero pose (arm hanging), 3 s
HOME = {'shoulder_joint': np.pi/2, 'upperArm_joint': -0.8, 'foreArm_joint': 1.4, 'wrist2_joint': -0.6}
def set_home(dd):
    dd.qpos[:] = 0
    for jn, v in HOME.items():
        dd.qpos[J[jn]] = v
d1 = mujoco.MjData(m); set_home(d1); mujoco.mj_forward(m, d1)
for a in ['j1','j2','j3','j4','j5','j6']:
    d1.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] = 0.0
for _ in range(1500):
    mujoco.mj_step(m, d1)
err = max(abs(d1.qpos[J[jn]]) for jn in ARM)
print(f"[hold] 3s at home: max arm joint drift = {err:.5f} rad, ncon={d1.ncon}, max|qacc|={np.abs(d1.qacc).max():.3f}")

# 2) smooth arm motion sweep (all servos), 6 s sinusoid
d2 = mujoco.MjData(m); set_home(d2); mujoco.mj_forward(m, d2)
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ['j1','j2','j3','j4','j5','j6']]
amp = [0.8, 0.5, 0.6, 0.4, 0.4, 0.4]
maxerr = 0.0
for k in range(3000):
    t = k * m.opt.timestep
    for i, a in enumerate(ids):
        d2.ctrl[a] = amp[i] * np.sin(2*np.pi*0.25*t + i*0.7)
    mujoco.mj_step(m, d2)
    if k % 100 == 0:
        for i, jn in enumerate(ARM):
            maxerr = max(maxerr, abs(d2.ctrl[ids[i]] - d2.qpos[J[jn]]))
print(f"[sweep] 6s sinusoid: max tracking err = {maxerr:.3f} rad, ncon={d2.ncon}")

# 3) gripper coupling (settled at several gear targets)
res = []
for tgt in [-0.30, 0.0, 0.30]:
    d3 = mujoco.MjData(m); set_home(d3); mujoco.mj_forward(m, d3)
    d3.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gear_servo')] = tgt
    for _ in range(2500):
        mujoco.mj_step(m, d3)
    qg = d3.qpos[J['gear_joint']]; qr = d3.qpos[J['right_finger_joint']]; ql = d3.qpos[J['left_finger_joint']]
    res.append((tgt, qg, qr/qg if abs(qg)>1e-6 else float('nan'), ql/qg if abs(qg)>1e-6 else float('nan')))
for tgt, qg, rr, rl in res:
    print(f"[grip] ctrl={tgt:+.2f} q_gear={qg:+.4f} ratio_r={rr:+.5f} ratio_l={rl:+.5f} (baseline -0.020995/-0.020999)")

# 4) wheels
d4 = mujoco.MjData(m); set_home(d4); mujoco.mj_forward(m, d4)
for w in ['act_left_servo','act_right_servo']:
    d4.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, w)] = 6.0
for _ in range(500):
    mujoco.mj_step(m, d4)
print(f"[wheels] 1s @6: L={d4.qvel[J['act_left_joint']]:.2f} R={d4.qvel[J['act_right_joint']]:.2f} rad/s")

# 5) keyframe (home = zero pose) + renders
qfull = np.zeros(m.nq)
for jn, v in HOME.items(): qfull[J[jn]] = v
qstr = ' '.join(f'{x:.6f}' for x in qfull)
kf = '  <keyframe>\n    <key name="home" qpos="' + qstr + '"/>\n  </keyframe>\n'
xml = open(MODEL).read()
if '<keyframe>' in xml:
    xml = re.sub(r'  <keyframe>.*?</keyframe>\n', kf, xml, flags=re.S)
else:
    xml = xml.replace('</mujoco>', kf + '</mujoco>')
open(MODEL,'w').write(xml)
m2 = mujoco.MjModel.from_xml_path(MODEL)
d5 = mujoco.MjData(m2)
mujoco.mj_resetDataKeyframe(m2, d5, 0)
mujoco.mj_forward(m2, d5)
r = mujoco.Renderer(m2, 1080, 1920)
def shot(p_, lookat, path, dist):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist
    vec = np.array(p_) - np.array(lookat); dxy = np.hypot(vec[0], vec[1])
    cam.azimuth = np.degrees(np.arctan2(vec[1], vec[0])); cam.elevation = np.degrees(np.arctan2(-vec[2], dxy))
    r.update_scene(d5, camera=cam); Image.fromarray(r.render()).save(path); print("saved", path)
ctr = [0.0, 0.0, -0.75]
shot([1.7, -1.7, -0.1], ctr, 'final_iso.png', 2.4)
shot([0, -2.6, -0.8], ctr, 'final_front.png', 2.4)
shot([2.6, 0, -0.8], ctr, 'final_side.png', 2.4)
shot([0, 0, 0.4], [0, 0, -0.9], 'final_top.png', 2.2)
