"""番茄采摘机器人 —— 三相机观测脚本（MuJoCo 3.3.5）

用法:
  python3 observe_cameras.py                # 复位到 home，三台相机各出一张 RGB + 深度
  python3 observe_cameras.py --steps 500    # 先用伺服保持 home 仿真 500 步，再观测
  python3 observe_cameras.py --live 300     # 连续观测 300 帧（每步渲染 +y 侧 L515）

输出: /data/robot_assembly/observations/<相机名>_{rgb,depth}.png / .npy
"""
import argparse
import os

import numpy as np
import mujoco
from PIL import Image

XML = '/data/robot_assembly/model/tomato_picker.xml'
OUT = '/data/robot_assembly/observations'

# 三台相机: 名字 -> (竖直分辨率, 水平分辨率)  = 各自 color 流原生分辨率
CAMS = {
    'cam_wrist_d405':   (720, 1280),   # D405  腕部, 朝下沿工具轴
    'cam_scene_yp_l515': (540, 960),   # L515  +y 侧 (半分辨率, fovy 不变)
    'cam_scene_yn_d435i': (480, 640),  # D435i −y 侧
}

# 默认初始指令 (与 keyframe "home" 的 ctrl 一致)
CTRL_INIT = [-0.474, 0.0, 0.0, -1.59, -0.0611, 1.5, -1.59, -1.65, 3.05, 0.0]  # 末位=lift


def hold_home(m, d):
    """复位到默认初始状态 (关键帧自带 ctrl, 这里显式再设一遍兜底)."""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = CTRL_INIT
    mujoco.mj_forward(m, d)


def render_cam(m, d, name, h, w, depth=False):
    r = mujoco.Renderer(m, h, w)
    if depth:
        r.enable_depth_rendering()
    r.update_scene(d, camera=name)
    img = r.render()
    r.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=0,
                    help='观测前先仿真多少步 (伺服保持 home)')
    ap.add_argument('--live', type=int, default=0,
                    help='连续观测 N 帧 (每 10 步渲染一次 +y 侧 L515)')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    hold_home(m, d)

    if args.live:
        for k in range(args.live):
            mujoco.mj_step(m, d)
            if k % 10 == 0:
                mujoco.mj_forward(m, d)
                h, w = CAMS['cam_scene_yp_l515']
                rgb = render_cam(m, d, 'cam_scene_yp_l515', h, w)
                Image.fromarray(rgb).save('%s/live_%04d.png' % (OUT, k))
        print('live 序列已存到 %s/live_*.png' % OUT)
        return

    for _ in range(args.steps):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    for name, (h, w) in CAMS.items():
        rgb = render_cam(m, d, name, h, w)
        Image.fromarray(rgb).save('%s/%s_rgb.png' % (OUT, name))

        dep = render_cam(m, d, name, h, w, depth=True).squeeze()
        np.save('%s/%s_depth.npy' % (OUT, name), dep)
        # 深度可视化 (近黑远白)
        dvis = (dep - dep.min()) / max(dep.ptp(), 1e-9)
        Image.fromarray((dvis * 255).astype(np.uint8)).save(
            '%s/%s_depth.png' % (OUT, name))
        print('%-20s rgb=%s  depth范围=%.3f..%.3f m'
              % (name, rgb.shape, dep.min(), dep.max()))

    print('全部输出在 %s/' % OUT)


if __name__ == '__main__':
    main()
