#!/usr/bin/env python3

import numpy as np
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


def compute_target_in_base(
    target_tvec_cam,
    target_euler_cam,
    arm_tvec_end2base,
    arm_euler_end2base,
    T_cam2end
):
    R_target_cam = euler_to_matrix(target_euler_cam)
    T_target_cam = make_transform(R_target_cam, target_tvec_cam)

    R_end_base = euler_to_matrix(arm_euler_end2base)
    T_end_base = make_transform(R_end_base, arm_tvec_end2base)

    T_target_base = T_end_base @ T_cam2end @ T_target_cam

    t_target_base = T_target_base[:3, 3]
    R_target_base = T_target_base[:3, :3]
    euler_target_base = matrix_to_euler(R_target_base)

    return {
        'tvec': t_target_base,
        'euler_xyz': euler_target_base,
        'rmat': R_target_base,
        'transform': T_target_base,
    }


def main():
    T_cam2end = np.array([
        [-0.93697984, -0.34321276, -0.06537419, 0.01031222],
        [0.34879741, -0.90805463, -0.23189903, 0.00514664],
        [0.02022737, -0.24008707, 0.9705406, -0.01286485],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    target_tvec_cam = [0.04587435, 0.0147061, 0.23650242]
    target_euler_cam = [1.08143359, -2.79936905, 0.5500928]

    arm_poses = [
        {
            'tvec': [-0.03291614416350688, -0.34382048988310177, 0.456157175484822],
            'euler': [2.72055888, 0.50161266, 0.20679568],
        },
        {
            'tvec': [0.0, 0.0, 0.0],
            'euler': [0.0, 0.0, 0.0],
        },
    ]

    print("=" * 70)
    print("目标在基坐标系下的位姿计算结果")
    print("=" * 70)
    print(f"目标在相机系下 - 平移: {target_tvec_cam}")
    print(f"目标在相机系下 - 欧拉角(xyz): {target_euler_cam}")
    print()

    for i, pose in enumerate(arm_poses):
        result = compute_target_in_base(
            target_tvec_cam,
            target_euler_cam,
            pose['tvec'],
            pose['euler'],
            T_cam2end,
        )
        print(f"--- 第 {i} 组机械臂位姿 ---")
        print(f"  末端位姿 - 平移: {pose['tvec']}")
        print(f"  末端位姿 - 欧拉角(xyz): {pose['euler']}")
        print(f"  目标在基系 - 平移 (x, y, z): {result['tvec']}")
        print(f"  目标在基系 - 欧拉角(xyz, rad): {result['euler_xyz']}")
        print(f"  目标在基系 - 旋转矩阵:\n{result['rmat']}")
        print(f"  目标在基系 - 齐次变换矩阵:\n{result['transform']}")
        print()


if __name__ == '__main__':
    main()
