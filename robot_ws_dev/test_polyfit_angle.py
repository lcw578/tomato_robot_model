import numpy as np
import math


def test_polyfit_angle(xs, ys):
    """
    测试 np.polyfit 线性回归 + arctan 计算切线角度
    """
    if len(xs) <= 1:
        print("  点数不足，需要至少2个点")
        return None, None, None

    coefficients = np.polyfit(xs, ys, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    angle_rad = np.arctan(slope)
    angle_deg = math.degrees(angle_rad)

    print(f"  拟合结果: y = {slope:.6f} * x + {intercept:.6f}")
    print(f"  斜率 slope = {slope:.6f}")
    print(f"  角度 angle = {angle_rad:.6f} rad = {angle_deg:.3f}°")

    print(f"\n  各点拟合残差（y实际 - y拟合）:")
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        y_fit = slope * xi + intercept
        residual = yi - y_fit
        print(f"    点{i}: x={xi}, y_actual={yi}, y_fit={y_fit:.4f}, 残差={residual:.4f}")

    return slope, intercept, angle_rad


def run_test(name, xs, ys):
    print(f"\n{'='*60}")
    print(f"测试用例: {name}")
    print(f"  xs = {xs}")
    print(f"  ys = {ys}")
    print(f"  点数: {len(xs)}")
    print(f"{'-'*60}")
    test_polyfit_angle(xs, ys)


if __name__ == "__main__":
    # # 用例1: 水平直线 y=0
    # run_test(
    #     "水平直线 y=0",
    #     xs=[0, 1, 2, 3, 4],
    #     ys=[0, 0, 0, 0, 0],
    # )

    # # 用例2: 45度上升 y=x
    run_test(
        "45度上升 y=x",
        xs=[0, 1, 2, 3, 4],
        ys=[0, -1, -2, -3, -4],
    )

    # # 用例3: 45度下降 y=-x
    run_test(
        "45度下降 y=-x",
        xs=[-4, -3, -2, -1, 0],
        ys=[-4, -3, -2, -1, 0],
    )

    # # 用例4: 30度 y = tan(30°)*x
    # angle_30 = math.tan(math.radians(30))
    # run_test(
    #     "30度上升",
    #     xs=[0, 1, 2, 3, 4],
    #     ys=[0, angle_30*1, angle_30*2, angle_30*3, angle_30*4],
    # )

    # # 用例5: 带噪声的近似直线
    # run_test(
    #     "带噪声的近似直线",
    #     xs=[0, 1, 2, 3, 4, 5],
    #     ys=[0.1, 0.9, 2.2, 2.8, 4.1, 5.0],
    # )

    # # 用例6: 只有2个点
    # run_test(
    #     "只有2个点",
    #     xs=[10, 20],
    #     ys=[5, 15],
    # )

    # # 用例7: 垂直方向（x相同，y变化）—— polyfit会有问题！
    # run_test(
    #     "接近垂直 x变化很小",
    #     xs=[0, 0.1, 0.2, 0.3],
    #     ys=[0, 1, 2, 3],
    # )

    print(f"\n{'='*60}")
    print("测试完成")
