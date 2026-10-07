"""实时三相机监测: 3D 交互查看器 + 三路相机画面 (OpenCV)
用法: cd /data/robot_assembly && python3 live_monitor.py
退出: ESC 或关闭 MuJoCo 查看器窗口
"""
import mujoco
import mujoco.viewer
import cv2

XML = '/data/robot_assembly/model/tomato_picker.xml'

# 相机名 -> (竖直分辨率, 水平分辨率)  = 各自 color 流原生分辨率
CAMS = {
    'cam_scene_yp_l515': (540, 960),
    'cam_scene_yn_d435i': (480, 640),
    'cam_wrist_d405': (720, 1280),
}
TITLES = {
    'cam_scene_yp_l515': 'L515 (+y side)',
    'cam_scene_yn_d435i': 'D435i (-y side)',
    'cam_wrist_d405': 'D405 (wrist, down)',
}

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = [-0.474, 0.0, 0.0, -1.59, -0.0611, 1.5, -1.59, -1.65, 3.05, 0.0, 0.0]  # 末位=lift
mujoco.mj_forward(m, d)

renderers = {c: mujoco.Renderer(m, h, w) for c, (h, w) in CAMS.items()}

with mujoco.viewer.launch_passive(m, d) as v:
    k = 0
    while v.is_running():
        mujoco.mj_step(m, d)
        if k % 5 == 0:
            v.sync()                       # 物理状态推给 3D 窗口
        if k % 2 == 0:                     # 相机画面隔步渲染 (省算力)
            for c, (h, w) in CAMS.items():
                r = renderers[c]
                r.update_scene(d, camera=c)
                f = r.render()             # RGB
                cv2.imshow(TITLES[c], cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == 27:    # ESC 退出
            break
        k += 1

for r in renderers.values():
    r.close()
cv2.destroyAllWindows()
print('monitor closed after %d steps' % k)
