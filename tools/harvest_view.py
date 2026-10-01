"""采摘过程可视化 (两种模式):

  1. 实时查看器 (需要显示器):
       python3 tools/harvest_view.py                 # MuJoCo 查看器 + 跟随视角实时看
       python3 tools/harvest_view.py --target 7_1    # 指定果串
       python3 tools/harvest_view.py --speed 0.5     # 0.5 倍速慢放
     鼠标可自由旋转缩放, 相机 lookat 自动跟随夹爪 (--no-follow 关闭跟随)。

  2. MP4 导出 (无显示器也能跑):
       python3 tools/harvest_view.py --mp4 docs/harvest.mp4
     相机自动跟随夹爪, 画面左上角叠加阶段标签。

流程与 tools/harvest_demo.py 完全同源 (直接调用 Picker.harvest, 仅注入可视化钩子)。
"""
import argparse, sys, time
import numpy as np
import cv2
import mujoco
import mujoco.viewer

sys.path.insert(0, '/data/robot_assembly/tools')
from harvest_demo import Picker, GEAR_HOME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default=None)
    ap.add_argument('--speed', type=float, default=1.0, help='实时回放倍率')
    ap.add_argument('--no-follow', action='store_true')
    ap.add_argument('--mp4', default=None, help='导出 MP4 (无显示器模式)')
    args = ap.parse_args()

    pk = Picker()
    mujoco.mj_resetDataKeyframe(pk.m, pk.d, 0)
    pk.d.ctrl[:] = pk.home_ctrl
    mujoco.mj_forward(pk.m, pk.d)
    base = pk.d.xpos[pk.m.body('arm_base_mount').id]
    if args.target:
        target = args.target
    else:
        ripe = sorted((t for t in pk.trusses if t['ripe']),
                      key=lambda t: np.linalg.norm(pk.cut_site_pos(t['site']) - base))
        target = ripe[0]['name']
    print('目标果串:', target)

    if args.mp4:
        W, H, FPS = 960, 540, 10
        vw = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
        r = mujoco.Renderer(pk.m, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 1.5, -60, -15
        lookat = pk.tcp_pos().copy()

        def tick():
            nonlocal lookat
            tcp = pk.tcp_pos()
            lookat += 0.15 * (tcp - lookat)          # 平滑跟随
            cam.lookat[:] = lookat
            r.update_scene(pk.d, camera=cam)
            img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, pk.stage, (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 255), 2, cv2.LINE_AA)
            vw.write(img)
        tick.n = 0
        pk.tick = tick
        log = pk.harvest(target)
        r.close()
        vw.release()
        print('MP4 已保存:', args.mp4)
        print('log:', log)
        return

    # 实时查看器模式
    with mujoco.viewer.launch_passive(pk.m, pk.d) as v:
        t0 = time.time()
        lookat = pk.tcp_pos().copy()

        def tick():
            nonlocal lookat
            if not args.no_follow:
                lookat += 0.15 * (pk.tcp_pos() - lookat)
                v.cam.lookat[:] = lookat
            v.sync()
            if args.speed > 0:                       # 实时回放节流
                late = pk.d.time / args.speed - (time.time() - t0)
                if late > 0:
                    time.sleep(min(late, 0.05))
        pk.tick = tick
        log = pk.harvest(target)
        print('log:', log)
        print('流程结束, 查看器保持打开, Ctrl+C 退出')
        try:
            while v.is_running():
                v.sync()
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
