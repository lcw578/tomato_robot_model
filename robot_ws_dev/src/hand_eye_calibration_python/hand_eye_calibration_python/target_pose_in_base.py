#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def rotvec_to_matrix(rotvec):
    rotvec = np.array(rotvec, dtype=np.float64).reshape(3, 1)
    rmat, _ = cv2.Rodrigues(rotvec)
    return rmat


def euler_to_matrix(euler_xyz):
    r = R.from_euler('xyz', euler_xyz)
    return r.as_matrix()


def make_transform(rmat, tvec):
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = np.array(tvec, dtype=np.float64).flatten()
    return T


def compute_target_in_base(
    target_rvec_cam,
    target_tvec_cam,
    arm_euler_end2base,
    arm_tvec_end2base,
    T_cam2end
):
    R_target_cam = rotvec_to_matrix(target_rvec_cam)
    T_target_cam = make_transform(R_target_cam, target_tvec_cam)

    R_end_base = euler_to_matrix(arm_euler_end2base)
    T_end_base = make_transform(R_end_base, arm_tvec_end2base)

    T_target_base = T_end_base @ T_cam2end @ T_target_cam

    R_target_base = T_target_base[:3, :3]
    t_target_base = T_target_base[:3, 3]

    r_target_base, _ = cv2.Rodrigues(R_target_base)
    euler_target_base = R.from_matrix(R_target_base).as_euler('xyz')

    return {
        'tvec': t_target_base,
        'rmat': R_target_base,
        'rvec': r_target_base.flatten(),
        'euler_xyz': euler_target_base,
        'transform': T_target_base,
    }


def main():
    T_cam2end = np.array([
        [-0.93697984, -0.34321276, -0.06537419, 0.01031222],
        [0.34879741, -0.90805463, -0.23189903, 0.00514664],
        [0.02022737, -0.24008707, 0.9705406, -0.01286485],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    data = [
        {
            'target_rvec_cam': [1.08143359, -2.79936905, 0.5500928],
            'target_tvec_cam': [0.04587435, 0.0147061, 0.23650242],
            'arm_euler_end2base': [2.72055888, 0.50161266, 0.20679568],
            'arm_tvec_end2base': [-0.03291614416350688, -0.34382048988310177, 0.456157175484822],
        },
        {
            'target_rvec_cam': [-2.45201469, 0.93998852, 0.16673807],
            'target_tvec_cam': [0.03798534, -0.01526096, 0.20896953],
            'arm_euler_end2base': [-2.90029383, 0.04264487, 2.25450206],
            'arm_tvec_end2base': [0.18700943699183958, -0.3909996706014678, 0.3912183702267598],
        },
        {
            'target_rvec_cam': [-0.09470312, 2.94854223, -0.4900507],
            'target_tvec_cam': [-0.09647335, -0.0335526, 0.26851452],
            'arm_euler_end2base': [2.61177588, 0.14794129, -0.10179684],
            'arm_tvec_end2base': [-0.07614503726519076, -0.2013707355000593, 0.5981518189357212],
        },
    ]

    print("=" * 60)
    print("目标点在机械臂基坐标系下的位姿计算结果")
    print("=" * 60)

    for i, d in enumerate(data):
        result = compute_target_in_base(
            d['target_rvec_cam'],
            d['target_tvec_cam'],
            d['arm_euler_end2base'],
            d['arm_tvec_end2base'],
            T_cam2end,
        )
        print(f"\n--- 第 {i} 组数据 ---")
        print(f"  平移向量 (x, y, z): {result['tvec']}")
        print(f"  旋转矩阵:\n{result['rmat']}")
        print(f"  旋转向量: {result['rvec']}")
        print(f"  欧拉角 (xyz, rad): {result['euler_xyz']}")
        print(f"  齐次变换矩阵:\n{result['transform']}")


if __name__ == '__main__':
    main()
