"""Render the mount junction from the TOOL side (below the flange) - the side the
user photographs from.  Home pose: tool axis points down, so the tool side is UP
is wrong - camera goes BELOW the gripper looking up (el ~ -85).
"""
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
W3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_Link')
gp, wp = d.xpos[G].copy(), d.xpos[W3].copy()
print('gripper pos', np.round(gp, 4), ' wrist3 pos', np.round(wp, 4))
mid = (gp + wp) / 2

def shot(az, el, dist, path, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(f'/data/robot_assembly/{path}')
    print('saved', path)

# from tool side (below), looking up at the junction
shot(90, -86, 0.55, 'ts_axis.png', mid)
shot(90, -80, 0.55, 'ts_axis2.png', mid)
shot(60, -82, 0.60, 'ts_a60.png', mid)
shot(120, -82, 0.60, 'ts_a120.png', mid)
shot(20, -70, 0.60, 'ts_a20.png', mid)
shot(160, -70, 0.60, 'ts_a160.png', mid)
# closer on the flange-adapter interface
shot(90, -86, 0.35, 'ts_close.png', gp + [0, 0, 0.02])
shot(270, -86, 0.55, 'ts_back.png', mid)
