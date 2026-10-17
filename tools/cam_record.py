"""三路 RealSense 相机独立录制 (链路全过程, 无窗口).

对 --chain/--target 指定的采摘链路, 分别用每台相机渲染并写成独立 MP4:
  cam_scene_yp_l515 -> l515_yp.mp4
  cam_scene_yn_l515 -> l515_yn.mp4
  cam_wrist_d405    -> d405_wrist.mp4
分辨率 = 各自原生 (L515 1920x1080, D405 1280x720), 可用 --scale 缩放.
帧率按仿真时间 (规划停顿不产生帧, 视频只含运动过程), --fps 默认 30.

用法:
  python3 tools/cam_record.py --chain --target 9_1
  python3 tools/cam_record.py --chain --target 9_1 --out docs/cams_9_1 --fps 30
"""
import argparse
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_demo import Picker, run_chain, XML  # noqa: E402

# 相机名 -> (原生竖直, 原生水平, 输出文件名)
CAMS = {
    'cam_scene_yp_l515': (1080, 1920, 'l515_yp.mp4'),
    'cam_scene_yn_l515': (1080, 1920, 'l515_yn.mp4'),
    'cam_wrist_d405': (720, 1280, 'd405_wrist.mp4'),
}
# tick 钩子约每 2 个物理步触发一次 (dt=0.002) -> 250 Hz 仿真时间
TICK_HZ = 250.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default=None, help='果串名, 如 9_1; 默认真值挑最佳串')
    ap.add_argument('--chain', action='store_true', help='链路模式 (含底盘行驶)')
    ap.add_argument('--out', default=None, help='输出目录 (默认 docs/cams_<target>)')
    ap.add_argument('--fps', type=float, default=30.0, help='视频帧率 (仿真时间)')
    ap.add_argument('--scale', type=float, default=1.0, help='渲染缩放 (1.0=原生)')
    ap.add_argument('--no-label', action='store_true', help='不在画面上叠加阶段标签')
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'docs', 'cams_%s' % (args.target or 'best'))
    os.makedirs(out_dir, exist_ok=True)

    pk = Picker()
    mujoco.mj_resetDataKeyframe(pk.m, pk.d, 0)
    pk.d.ctrl[:] = pk.home_ctrl
    mujoco.mj_forward(pk.m, pk.d)

    stride = max(1, int(round(TICK_HZ / max(args.fps, 1e-6))))
    fps_out = TICK_HZ / stride
    writers, renderers, sizes = {}, {}, {}
    for name, (h, w, fn) in CAMS.items():
        hh, ww = max(1, int(h * args.scale)), max(1, int(w * args.scale))
        renderers[name] = mujoco.Renderer(pk.m, hh, ww)
        writers[name] = cv2.VideoWriter(os.path.join(out_dir, fn),
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        fps_out, (ww, hh))
        sizes[name] = (hh, ww)
        print('  %-22s -> %-14s %dx%d @ %.1f fps' % (name, fn, ww, hh, fps_out))

    state = {'n': 0, 'frames': {k: 0 for k in CAMS}}

    def tick():
        state['n'] += 1
        if state['n'] % stride:
            return
        for name in CAMS:
            renderers[name].update_scene(pk.d, camera=name)
            img = cv2.cvtColor(renderers[name].render(), cv2.COLOR_RGB2BGR)
            if not args.no_label:
                cv2.putText(img, pk.stage, (24, 56), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 2, cv2.LINE_AA)
            writers[name].write(img)
            state['frames'][name] += 1

    pk.tick = tick
    print('录制中 (链路%s%s)...' % (
        ' target=' + args.target if args.target else '', ''))
    log = run_chain(pk, target=args.target) if args.chain else pk.harvest(args.target)

    for name in CAMS:
        writers[name].release()
        renderers[name].close()
    print('\n完成:')
    for name, (h, w, fn) in CAMS.items():
        path = os.path.join(out_dir, fn)
        mb = os.path.getsize(path) / 1e6
        print('  %-22s %6d 帧  %dx%d  %.1f MB  %s' % (
            name, state['frames'][name], sizes[name][1], sizes[name][0], mb, path))
    if log:
        print('  链路结果: %s | shear_hit %s' % (log.get('result'), log.get('shear_hit')))


if __name__ == '__main__':
    main()
