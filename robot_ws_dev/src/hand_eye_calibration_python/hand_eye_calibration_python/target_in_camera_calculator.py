#!/usr/bin/env python3

import numpy as np
import re
import ast
from scipy.spatial.transform import Rotation as R


def euler_to_matrix(euler_xyz):
    r = R.from_euler('xyz', euler_xyz)
    return r.as_matrix()


def make_transform(rmat, tvec):
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = np.array(tvec, dtype=np.float64).flatten()
    return T


def matrix_to_euler(rmat):
    return R.from_matrix(rmat).as_euler('xyz')


def compute_target_in_camera(
    target_tvec_base,
    target_euler_base,
    arm_tvec_end2base,
    arm_euler_end2base,
    T_cam2end
):
    R_target_base = euler_to_matrix(target_euler_base)
    T_target_base = make_transform(R_target_base, target_tvec_base)

    R_end_base = euler_to_matrix(arm_euler_end2base)
    T_end_base = make_transform(R_end_base, arm_tvec_end2base)

    T_end2base_inv = np.linalg.inv(T_end_base)
    T_cam2end_inv = np.linalg.inv(T_cam2end)

    T_target_cam = T_cam2end_inv @ T_end2base_inv @ T_target_base

    t_target_cam = T_target_cam[:3, 3]
    R_target_cam = T_target_cam[:3, :3]
    rvec_target_cam = R.from_matrix(R_target_cam).as_rotvec()
    euler_target_cam = matrix_to_euler(R_target_cam)

    return {
        'tvec': t_target_cam,
        'rvec': rvec_target_cam.reshape(3, 1),
        'euler_xyz': euler_target_cam,
        'rmat': R_target_cam,
        'transform': T_target_cam,
    }


def load_arm_poses_from_file(filepath):
    R_end2base_Euler = []
    T_end2base = []

    with open(filepath, 'r') as f:
        content = f.read()

    sections = re.split(r'\n\n(?=(?:R_end2base_Euler|T_end2base|R_cam2target|T_cam2target):)', content)
    sections = [section for section in sections if section.strip()]

    data_sections = {}
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        header = lines[0]
        if header.startswith('R_end2base_Euler:'):
            data_sections['R_end2base_Euler'] = lines[1:]
        elif header.startswith('T_end2base:'):
            data_sections['T_end2base'] = lines[1:]

    if 'R_end2base_Euler' in data_sections:
        for line in data_sections['R_end2base_Euler']:
            if ':' not in line:
                continue
            try:
                value_str = line.split(':', 1)[1].strip()
                value_str = value_str.strip('[]')
                values = [float(x) for x in value_str.split()]
                if len(values) == 3:
                    R_end2base_Euler.append(np.array([values[0], values[1], values[2]]))
            except Exception:
                pass

    if 'T_end2base' in data_sections:
        for line in data_sections['T_end2base']:
            if ':' not in line:
                continue
            try:
                value_str = line.split(':', 1)[1].strip()
                t_list = ast.literal_eval(value_str)
                if isinstance(t_list, list) and len(t_list) == 3:
                    T_end2base.append(t_list)
            except Exception:
                pass

    return R_end2base_Euler, T_end2base


def save_results_to_file(filepath, results):
    with open(filepath, 'w') as f:
        f.write('R_cam2target_pred:\n')
        for i, r in enumerate(results):
            f.write(f'{i}: {r["rvec"]}\n')
        f.write('\nT_cam2target_pred:\n')
        for i, r in enumerate(results):
            f.write(f'{i}: {r["tvec"].reshape(3, 1)}\n')


def main():
    config_path = '/home/xucan/code/test/robot_ws/src/hand_eye_calibration_python/config/hand/'
    data_file = config_path + 'arcuo_end_3d_results.txt'
    output_file = config_path + 'target_in_camera_results.txt'

    T_cam2end = np.array([
        [-0.93697984, -0.34321276, -0.06537419, 0.01031222],
        [0.34879741, -0.90805463, -0.23189903, 0.00514664],
        [0.02022737, -0.24008707, 0.9705406, -0.01286485],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    target_tvec_base = [0.5, 0.0, 0.2]
    target_euler_base = [0.0, 0.0, 0.0]

    R_end2base_Euler, T_end2base = load_arm_poses_from_file(data_file)
    print(f"加载了 {len(R_end2base_Euler)} 组 R_end2base_Euler")
    print(f"加载了 {len(T_end2base)} 组 T_end2base")

    if len(R_end2base_Euler) != len(T_end2base):
        print(f"警告：数据数量不匹配！R_end2base_Euler={len(R_end2base_Euler)}, T_end2base={len(T_end2base)}")

    n = min(len(R_end2base_Euler), len(T_end2base))
    results = []

    print("=" * 70)
    print("目标在相机坐标系下的位姿计算结果")
    print("=" * 70)
    print(f"目标在基系下 - 平移: {target_tvec_base}")
    print(f"目标在基系下 - 欧拉角(xyz): {target_euler_base}")
    print()

    for i in range(n):
        result = compute_target_in_camera(
            target_tvec_base,
            target_euler_base,
            T_end2base[i],
            R_end2base_Euler[i],
            T_cam2end,
        )
        results.append(result)
        print(f"--- 第 {i} 组机械臂位姿 ---")
        print(f"  末端位姿 - 平移: {T_end2base[i]}")
        print(f"  末端位姿 - 欧拉角(xyz): {R_end2base_Euler[i]}")
        print(f"  目标在相机系 - 平移 (x, y, z): {result['tvec']}")
        print(f"  目标在相机系 - 欧拉角(xyz, rad): {result['euler_xyz']}")
        print()

    save_results_to_file(output_file, results)
    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
