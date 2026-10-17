"""采摘过程可视化 (两种模式):

  1. 实时查看器 (需要显示器):
       python3 tools/harvest_view.py                 # MuJoCo 查看器 + 跟随视角实时看
       python3 tools/harvest_view.py --target 7_1    # 指定果串
       python3 tools/harvest_view.py --speed 0.5     # 0.5 倍速慢放
       python3 tools/harvest_view.py --chain --cams  # 链路演示 + 三路 RealSense 相机窗口同屏
     鼠标可自由旋转缩放, 相机 lookat 自动跟随夹爪 (--no-follow 关闭跟随);
     --cams 另开三台相机实时窗口 (ESC 退出演示), 分辨率/刷新率用 --cam-scale/--cam-fps 调;
     运行的同时三路相机画面自动录制到 docs/cams_<target>/{l515_yp,l515_yn,d405_wrist}.mp4
     (录的就是窗口里的画面, --no-cam-record 可关闭)。

  2. MP4 导出 (无显示器也能跑):
       python3 tools/harvest_view.py --mp4 docs/harvest.mp4
     相机自动跟随夹爪, 画面左上角叠加阶段标签。

流程与 tools/harvest_demo.py 完全同源 (直接调用 Picker.harvest, 仅注入可视化钩子)。
"""
import argparse, os, sys, time
import numpy as np
import cv2
import mujoco
import mujoco.viewer

sys.path.insert(0, '/data/robot_assembly/tools')
from harvest_demo import Picker, GEAR_HOME, run_chain

# --cams 三路相机: 名 -> 原生 (竖直, 水平) 分辨率 (与 live_monitor.py 同源)
CAMS = {
    'cam_scene_yp_l515': (1080, 1920),
    'cam_scene_yn_l515': (1080, 1920),
    'cam_wrist_d405': (720, 1280),
}
CAM_TITLES = {
    'cam_scene_yp_l515': 'L515 (+y side)',
    'cam_scene_yn_l515': 'L515 (-y side)',
    'cam_wrist_d405': 'D405 (wrist, down)',
}


class YoloOverlay:
    """相机窗口的 YOLO 实时标注 (按相机缓存, 限频)."""
    def __init__(self, weights, fps=5.0):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.period = 1.0 / max(fps, 0.5)
        self.last = {}

    def annotate(self, cam, bgr):
        now = time.time()
        hit = self.last.get(cam)
        if hit is not None and now - hit[0] < self.period:
            return hit[1]
        res = self.model.predict(bgr, imgsz=640, conf=0.35, iou=0.5,
                                 device='cuda', verbose=False)[0]
        ann = res.plot()
        self.last[cam] = (now, ann)
        return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default=None)
    ap.add_argument('--speed', type=float, default=1.0, help='实时回放倍率')
    ap.add_argument('--no-follow', action='store_true')
    ap.add_argument('--follow', action='store_true', help='MP4 也用跟随视角 (默认固定机位)')
    ap.add_argument('--mp4', default=None, help='导出 MP4 (无显示器模式, 默认垄道端点固定机位)')
    ap.add_argument('--chain', action='store_true', help='链路模式: 驶到最佳站点摘一串 (含底盘移动)')
    ap.add_argument('--percept', action='store_true',
                    help='混合感知: stem=分割真值, tomato/trunk=真机 YOLO (需 conda env tomato_sim)')
    ap.add_argument('--weights', default='/home/lcw/robot_ws_dev/tomato-seg2025063018-yolov8-xucan.pt')
    ap.add_argument('--cams', action='store_true', help='另开三路 RealSense 相机实时窗口 (仅实时模式, ESC 退出)')
    ap.add_argument('--cam-scale', type=float, default=0.5, help='相机窗口渲染缩放 (1.0=原生分辨率)')
    ap.add_argument('--cam-fps', type=float, default=10.0, help='相机窗口刷新率上限 (Hz)')
    ap.add_argument('--no-cam-record', action='store_true',
                    help='--cams 时不录制三路相机 MP4 (默认录制到 docs/cams_<target>/)')
    ap.add_argument('--yolo-overlay', action='store_true',
                    help='--cams 时在相机画面上叠加真机 YOLO 的框/掩码')
    ap.add_argument('--yolo-fps', type=float, default=5.0,
                    help='YOLO 叠加推理频率上限 (Hz)')
    ap.add_argument('--sync-every', type=int, default=2, help='查看器同步间隔 (步; 调大可加快实时模式)')
    ap.add_argument('--azim', type=float, default=0.0)
    ap.add_argument('--elev', type=float, default=-14.0)
    ap.add_argument('--dist', type=float, default=4.2)
    args = ap.parse_args()

    pk = Picker()
    mujoco.mj_resetDataKeyframe(pk.m, pk.d, 0)
    pk.d.ctrl[:] = pk.home_ctrl
    mujoco.mj_forward(pk.m, pk.d)
    base = pk.d.xpos[pk.m.body('arm_base_mount').id]
    if args.target:
        target = args.target
    elif args.mp4:
        target = '1_2'          # 固定机位默认: +y 近侧行高处串, 顺带展示升降
    else:
        ripe = sorted((t for t in pk.trusses if t['ripe']),
                      key=lambda t: np.linalg.norm(pk.cut_site_pos(t['site']) - base))
        target = ripe[0]['name']
    print('目标果串:', target)

    if args.mp4:
        if args.cams:
            print('--cams 仅用于实时模式, 已忽略')
        if args.chain:                 # 链路含行驶与长路径: 降分辨率+抽帧+提播放率
            W, H, FPS, STRIDE = 640, 360, 90, 6
        else:
            W, H, FPS, STRIDE = 960, 540, 60, 1
        vw = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
        r = mujoco.Renderer(pk.m, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        lookat = np.array([0.0, 0.0, 0.95])
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = args.dist, args.azim, args.elev

        def tick():
            tick.n += 1
            if tick.n % STRIDE:
                return
            if args.follow:                          # 跟随视角 (--follow)
                lookat += 0.15 * (pk.tcp_pos() - lookat)
                cam.lookat[:] = lookat
            r.update_scene(pk.d, camera=cam)
            img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, pk.stage, (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 255), 2, cv2.LINE_AA)
            vw.write(img)
        tick.n = 0
        pk.tick = tick
        log = run_chain(pk, target=args.target, percept=args.percept, weights=args.weights) if args.chain else pk.harvest(target)
        r.close()
        vw.release()
        print('MP4 已保存:', args.mp4)
        print('log:', log)
        return

    # 实时查看器模式
    # 相机 Renderer 必须在 launch_passive 之前创建 (GL 上下文顺序, 同 live_monitor.py)
    renderers = {}
    cam_writers, cam_out = {}, None
    if args.cams:
        for c, (h, w) in CAMS.items():
            renderers[c] = mujoco.Renderer(
                pk.m, max(1, int(h * args.cam_scale)), max(1, int(w * args.cam_scale)))
        if not args.no_cam_record:
            cam_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'docs', 'cams_%s' % (target if args.target else 'best'))
            os.makedirs(cam_out, exist_ok=True)
            for c, (h, w) in CAMS.items():
                fn = {'cam_scene_yp_l515': 'l515_yp.mp4',
                      'cam_scene_yn_l515': 'l515_yn.mp4',
                      'cam_wrist_d405': 'd405_wrist.mp4'}[c]
                hh, ww = max(1, int(h * args.cam_scale)), max(1, int(w * args.cam_scale))
                cam_writers[c] = cv2.VideoWriter(
                    os.path.join(cam_out, fn), cv2.VideoWriter_fourcc(*'mp4v'),
                    max(1.0, float(args.cam_fps)), (ww, hh))
            print('三路相机录制 ->', cam_out)
    overlay = None
    if args.cams and args.yolo_overlay:
        try:
            overlay = YoloOverlay(args.weights, fps=args.yolo_fps)
            print('YOLO 实时标注已启用 (weights=%s, %.1f Hz)' % (args.weights, args.yolo_fps))
        except Exception as exc:
            print('YOLO 标注启用失败 (需要 conda env tomato_sim):', exc)

    try:
        with mujoco.viewer.launch_passive(pk.m, pk.d) as v:
            t0 = time.time()
            lookat = pk.tcp_pos().copy()
            last_cam = [0.0]

            def show_cams():
                nonlocal overlay
                for c, r in renderers.items():
                    r.update_scene(pk.d, camera=c)
                    frame = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
                    if overlay is not None:
                        try:
                            frame = overlay.annotate(c, frame)
                        except Exception as exc:
                            print('YOLO 标注失败:', exc)
                            overlay = None
                    if c in cam_writers:
                        cam_writers[c].write(frame)     # 与窗口同帧录制
                    cv2.imshow(CAM_TITLES[c], frame)
                if cv2.waitKey(1) & 0xFF == 27:         # ESC: 退出演示
                    raise KeyboardInterrupt

            def maybe_show_cams():
                if not renderers:
                    return
                now = time.time()
                if now - last_cam[0] >= 1.0 / args.cam_fps:
                    last_cam[0] = now
                    show_cams()

            def tick():
                nonlocal lookat
                tick.n += 1
                if not args.no_follow:
                    lookat += 0.15 * (pk.tcp_pos() - lookat)
                    v.cam.lookat[:] = lookat
                if tick.n % args.sync_every == 0:
                    v.sync()
                maybe_show_cams()
                if args.speed > 0:                       # 实时回放节流
                    late = pk.d.time / args.speed - (time.time() - t0)
                    if late > 0:
                        time.sleep(min(late, 0.05))
            tick.n = 0
            pk.tick = tick
            aborted = False
            try:
                log = run_chain(pk, target=args.target, percept=args.percept, weights=args.weights) if args.chain else pk.harvest(target)
                print('log:', log)
            except KeyboardInterrupt:
                aborted = True
                print('提前退出演示 (ESC / Ctrl+C)')
            if not aborted:
                print('流程结束, 查看器保持打开, Ctrl+C 退出')
                try:
                    while v.is_running():
                        v.sync()
                        maybe_show_cams()
                        time.sleep(0.03)
                except KeyboardInterrupt:
                    pass
    finally:
        for r in renderers.values():
            r.close()
        if cam_writers:
            for w_ in cam_writers.values():
                w_.release()
            print('三路相机视频已保存至:', cam_out)
        if renderers:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:   # 兜底: 二次 SIGINT 竞态 (GUI 后端下偶发) 不外漏 traceback
        print('\n已中断退出')
