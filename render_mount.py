import sys
sys.path.insert(0, '/data/robot_assembly')
import mujoco, numpy as np
from PIL import Image

m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 1000, 1333)
G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
gp = d.xpos[G].copy()

def shot(az, el, dist, path, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(path)
    print('saved', path)

# junction close-ups: real flange -> printed adapter -> gripper (+ camera on bracket arm)
shot(20, -12, 0.50, 'mnt_gripper.png', gp + [0, 0, 0.07])
shot(-70, -6, 0.55, 'mnt_side.png', gp + [0, 0, 0.07])
shot(35, -80, 0.45, 'mnt_top.png', gp + [0, 0, 0.03])
shot(150, -25, 0.60, 'mnt_back.png', gp + [0, 0, 0.07])
# whole machine, home pose
mb = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'machine')]
shot(-60, -12, 1.6, 'mnt_full.png', mb + [0, 0, 0.55])
