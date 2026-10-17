import math


def create_mid2target_angle(base_point_left, base_point_right, base_point_top, growth_direction, Y):
    # tangent_angle_by_base 末端旋转的角度 范围在-90-90度 逆时针为正,顺时针为负
    # mid2target_angle 过渡点与目标点之间的角度 范围在-180-180度 逆时针为正 顺时针为负

    leftOrRight = 0 #0是前，1是左，2是右
    tangent_angle_by_base = 90.0

    distance_top2left = math.sqrt((base_point_top[0] - base_point_left[0])**2 + (base_point_top[2] - base_point_left[2])**2)
    distance_top2right = math.sqrt((base_point_top[0] - base_point_right[0])**2 + (base_point_top[2] - base_point_right[2])**2)
    if distance_top2left > distance_top2right:
        base_point_bottom = base_point_left
        print(f"  base_point_bottom: left")
    else:
        base_point_bottom = base_point_right
        print(f"  base_point_bottom: right")
        
    if abs(base_point_bottom[2] - base_point_top[2]) > 0.005:
        if Y > 0:
            tangent_angle_by_base = math.atan2(
                base_point_top[2] - base_point_bottom[2],
                base_point_top[0] - base_point_bottom[0]
            )
        else:
            tangent_angle_by_base = math.atan2(
                base_point_top[2] - base_point_bottom[2],
                base_point_bottom[0] - base_point_top[0]
            )
        print(f"  1111 tangent_angle_by_base: {tangent_angle_by_base*180/math.pi:.3f}")
        
        w2hRatio = (base_point_top[0] - base_point_bottom[0]) / (base_point_top[2] - base_point_bottom[2])
        print(f"  w2hRatio: {w2hRatio}")
        if abs(w2hRatio) > 0.9:
            if Y > 0:
                leftOrRight = 1 if w2hRatio > 0 else 2
            else:
                leftOrRight = 2 if w2hRatio > 0 else 1

        if tangent_angle_by_base < -90 * math.pi / 180:
            tangent_angle_by_base = tangent_angle_by_base + 180 * math.pi / 180
        elif tangent_angle_by_base > 90 * math.pi / 180:
            tangent_angle_by_base = tangent_angle_by_base - 180 * math.pi / 180

    if leftOrRight != 0:
        mid2target_angle = math.atan2(
            base_point_bottom[1] - base_point_top[1],
            base_point_bottom[0] - base_point_top[0]
        )
        print(f"  xz朝向判断: {leftOrRight}, Y: {Y}, mid2target_angle: {mid2target_angle}")
    else:
        mid2target_angle = 0.0

    return mid2target_angle, tangent_angle_by_base, leftOrRight


def run_test(name, base_point_left, base_point_right, base_point_top, growth_direction, Y):
    print(f"\n{'='*60}")
    print(f"测试用例: {name}")
    print(f"  base_point_left  = {base_point_left}")
    print(f"  base_point_right = {base_point_right}")
    print(f"  base_point_top   = {base_point_top}")
    print(f"  growth_direction = {growth_direction}")
    print(f"  Y                = {Y}")
    print(f"{'-'*60}")

    mid2target_angle, tangent_angle_by_base, leftOrRight = create_mid2target_angle(
        base_point_left, base_point_right, base_point_top, growth_direction, Y
    )

    print(f"{'-'*60}")
    print(f"  返回结果:")
    print(f"    mid2target_angle    = {mid2target_angle:.6f} rad  = {math.degrees(mid2target_angle):.3f}°")
    print(f"    tangent_angle_by_base = {tangent_angle_by_base:.6f} rad  = {math.degrees(tangent_angle_by_base):.3f}°")
    print(f"    leftOrRight         = {leftOrRight}  (0=朝前, 1=左边, 2=右边)")


if __name__ == "__main__":
    # 用例1: 
    run_test(
        "朝左前 -140 +40 Y>0",
        base_point_left=[-1.0, 0.2, 0.2],
        base_point_right=[0.0, 0.7, 0.7],
        base_point_top=[0.0, 0.7, 0.7],
        growth_direction="朝左",
        Y=0.45,
    )

    # 用例2: 朝右 -40 -40 Y>0
    run_test(
        "朝右前 -40 -40 Y>0",
        base_point_left=[0.0, 0.7, 0.7],
        base_point_right=[1.0, 0.2, 0.2],
        base_point_top=[0.0, 0.7, 0.7],
        growth_direction="朝右",
        Y=0.05,
    )

    # # 用例3: 朝左（左边远点在左上方）
    run_test(
        "朝左后 +140  +40 Y>0",
        base_point_left=[-1.0, 1.2, 0.2],
        base_point_right=[0.0, 0.7, 0.7],
        base_point_top=[0.0, 0.7, 0.7],
        growth_direction="朝左",
        Y=1.45,
    )

    # # 用例4: 朝右
    run_test(
        "朝右后 +40 -40 Y>0",
        base_point_left=[0.0, 0.7, 0.7],
        base_point_right=[1.0, 1.2, 0.2],
        base_point_top=[0.0, 0.7, 0.7],
        growth_direction="朝右",
        Y=1.45,
    )

    # 用例5: 接近垂直（z差很小，应该leftOrRight=0）
    run_test(
        "朝左前 +40 +40 Y<0",
        base_point_left=[1.0, -0.2, 0.2],
        base_point_right=[0.0, -0.7, 0.7],
        base_point_top=[0.0, -0.7, 0.7],
        growth_direction="朝左",
        Y=-0.45,
    )

    # # 用例6: 斜着长（45度方向）
    run_test(
        "朝右前  +140 -40 Y<0",
        base_point_left=[0.0, -0.7, 0.7],
        base_point_right=[-1.0, -0.2, 0.2],
        base_point_top=[0.0, -0.7, 0.7],
        growth_direction="朝右",
        Y=-0.45,
    )
    # # 用例7: 
    run_test(
        "朝左后  -40  +40 Y<0",
        base_point_left=[1.0, -1.2, 0.2],
        base_point_right=[0.0, -0.7, 0.7],
        base_point_top=[0.0, -0.7, 0.7],
        growth_direction="朝左",
        Y=-0.45,
    )
    # # 用例8: 
    run_test(
        "朝右后  -140 -40Y<0",
        base_point_left=[0.0, -0.7, 0.7],
        base_point_right=[-1.0, -1.2, 0.2],
        base_point_top=[0.0, -0.7, 0.7],
        growth_direction="朝右",
        Y=-0.45,
    )

    print(f"\n{'='*60}")
    print("测试完成")
