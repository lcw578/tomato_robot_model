import cv2
import numpy as np
import time
import os

class StemGrowthAndPoints:
    def __init__(self):
        self.save_img = True
    
    def calculate_skeleton(self, mask):
        """计算掩码的骨骼"""
        # 性能与算法配置
        CONFIG = {
            'enable_downsampling': False,      # 启用下采样 (加速大图处理)
            'downsample_ratio': 1.0,          # 下采样比例 (0.5表示长宽各缩小一半)
            'max_image_size': 400,            # 触发下采样的最小边长阈值
            'min_skeleton_points': 6,         # 最小骨架点数 (设置较低以适应短果梗)
            'trim_ratio': 0.15,               # 统计直径时，去除两端极大/极小值的比例 (15%)
        }
        try:
            # --- 1. 预处理 ---
            if len(mask_data.shape) > 2:
                mask_data = mask_data[:, :, 0]
                
            original_shape = mask_data.shape
            scale_factor = 1.0
            
            # 智能下采样：如果图像够大，则缩小以加速计算
            if (CONFIG['enable_downsampling'] and 
                max(original_shape) > CONFIG['max_image_size']):
                
                scale_factor = CONFIG['downsample_ratio']
                new_w = int(original_shape[1] * scale_factor)
                new_h = int(original_shape[0] * scale_factor)
                # 使用最近邻插值保持二值特性（0或255）
                mask_working = cv2.resize(mask_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_working = mask_data.copy()

            # 形态学操作：平滑边缘，填充内部小孔
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # 先闭运算(连通断开区域) 再开运算(去噪)
            mask_cleaned = cv2.morphologyEx(mask_working.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            # 安全检查
            if np.count_nonzero(mask_cleaned) < 10:
                return {'success': False, 'msg': '有效像素过少'}
            # 骨架化：提取单像素宽度的中心线
            skeleton = cv2.ximgproc.thinning(mask_cleaned, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            
            # 清理骨架：只保留最大的连通分支，去除毛刺
            skeleton = self._keep_largest_skeleton(skeleton)
            
            # 获取骨架坐标点 (y, x)
            skeleton_points = np.column_stack(np.where(skeleton > 0))
            
            if len(skeleton_points) < CONFIG['min_skeleton_points']:
                return {'success': False, 'msg': f'骨架点不足({len(skeleton_points)})'}
            else:
                return {'success': True, 'points': skeleton_points}

            
        except Exception as e:
            print(f"骨架提取或直径计算失败: {e}")
            return None 
    
    def _keep_largest_skeleton(self,skel_img):
        """清理骨架，只保留最大连通域"""
        num, labels = cv2.connectedComponents(skel_img)
        if num <= 2: return skel_img # 0是背景，1是骨架，若只有1个或没有则直接返回
        
        max_area = 0
        max_idx = 1
        for i in range(1, num):
            area = np.sum(labels == i)
            if area > max_area:
                max_area = area
                max_idx = i
                
        res = np.zeros_like(skel_img)
        res[labels == max_idx] = 255
        return res 


    def image_thin(self, src):
        """图像细化算法，模拟C++中的ImageThin_y函数"""
        try:
            # 确保输入是二值图像
            if len(src.shape) == 3:
                src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            
            # 二值化处理
            _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
            
            # # 显示二值图像
            # cv2.imshow('Binary Image', binary)
            # cv2.waitKey(0)  # 等待按键
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # 先闭运算(连通断开区域) 再开运算(去噪)
            mask_working = binary.copy()
            mask_cleaned = cv2.morphologyEx(mask_working.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            # 安全检查
            if np.count_nonzero(mask_cleaned) < 10:
                return {'success': False, 'msg': '有效像素过少'}
            # 骨架化：提取单像素宽度的中心线
            skeleton = cv2.ximgproc.thinning(mask_cleaned, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            
            # 清理骨架：只保留最大的连通分支，去除毛刺
            skeleton = self._keep_largest_skeleton(skeleton)
            # 转换为0-1二值图像
            skeleton = skeleton // 255
            print(f"图像细化完成------")
            
            return skeleton
        except Exception as e:
            print(f"图像细化时出错: {e}")
            # 确保关闭窗口
            try:
                cv2.destroyAllWindows()
            except:
                pass
            return src
    
    def image_thin_y(self, src, max_iterations=-1):
        """图像细化算法，基于C++的ImageThin_y函数实现
        暂时没有用上
        参数:
            src: 输入二值图像
            max_iterations: 最大迭代次数，-1表示无限制
        返回:
            细化后的二值图像
        """
        try:
            # 确保输入是二值图像
            if len(src.shape) == 3:
                src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            
            # 二值化处理，确保像素值为0或1
            _, binary = cv2.threshold(src, 127, 1, cv2.THRESH_BINARY)
            
            # 创建输出图像
            dst = binary.copy()
            height, width = dst.shape
            count = 0
            
            while True:
                count += 1
                if max_iterations != -1 and count > max_iterations:
                    break
                
                # 第一阶段标记
                m_flag = []
                for i in range(height):
                    for j in range(width):
                        p1 = dst[i, j]
                        if p1 != 1:
                            continue
                        
                        # 获取邻域像素值，边界处理
                        p4 = 0 if j == width - 1 else dst[i, j + 1]
                        p8 = 0 if j == 0 else dst[i, j - 1]
                        p2 = 0 if i == 0 else dst[i - 1, j]
                        p3 = 0 if (i == 0 or j == width - 1) else dst[i - 1, j + 1]
                        p9 = 0 if (i == 0 or j == 0) else dst[i - 1, j - 1]
                        p6 = 0 if i == height - 1 else dst[i + 1, j]
                        p5 = 0 if (i == height - 1 or j == width - 1) else dst[i + 1, j + 1]
                        p7 = 0 if (i == height - 1 or j == 0) else dst[i + 1, j - 1]
                        
                        # 条件1: 邻域非零像素数在2-6之间
                        neighbor_sum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                        if 2 <= neighbor_sum <= 6:
                            # 条件2: 计算AP值（相邻像素变化次数）
                            ap = 0
                            if p2 == 0 and p3 == 1:
                                ap += 1
                            if p3 == 0 and p4 == 1:
                                ap += 1
                            if p4 == 0 and p5 == 1:
                                ap += 1
                            if p5 == 0 and p6 == 1:
                                ap += 1
                            if p6 == 0 and p7 == 1:
                                ap += 1
                            if p7 == 0 and p8 == 1:
                                ap += 1
                            if p8 == 0 and p9 == 1:
                                ap += 1
                            if p9 == 0 and p2 == 1:
                                ap += 1
                            
                            # 条件3: AP值为1
                            # 条件4: p2*p4*p6 == 0 且 p4*p6*p8 == 0
                            if ap == 1 and (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                                m_flag.append((i, j))
                
                # 删除标记的点
                for (i, j) in m_flag:
                    dst[i, j] = 0
                
                if not m_flag:
                    break
                
                # 第二阶段标记
                m_flag.clear()
                for i in range(height):
                    for j in range(width):
                        p1 = dst[i, j]
                        if p1 != 1:
                            continue
                        
                        # 获取邻域像素值，边界处理
                        p4 = 0 if j == width - 1 else dst[i, j + 1]
                        p8 = 0 if j == 0 else dst[i, j - 1]
                        p2 = 0 if i == 0 else dst[i - 1, j]
                        p3 = 0 if (i == 0 or j == width - 1) else dst[i - 1, j + 1]
                        p9 = 0 if (i == 0 or j == 0) else dst[i - 1, j - 1]
                        p6 = 0 if i == height - 1 else dst[i + 1, j]
                        p5 = 0 if (i == height - 1 or j == width - 1) else dst[i + 1, j + 1]
                        p7 = 0 if (i == height - 1 or j == 0) else dst[i + 1, j - 1]
                        
                        # 条件1: 邻域非零像素数在2-6之间
                        neighbor_sum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                        if 2 <= neighbor_sum <= 6:
                            # 条件2: 计算AP值（相邻像素变化次数）
                            ap = 0
                            if p2 == 0 and p3 == 1:
                                ap += 1
                            if p3 == 0 and p4 == 1:
                                ap += 1
                            if p4 == 0 and p5 == 1:
                                ap += 1
                            if p5 == 0 and p6 == 1:
                                ap += 1
                            if p6 == 0 and p7 == 1:
                                ap += 1
                            if p7 == 0 and p8 == 1:
                                ap += 1
                            if p8 == 0 and p9 == 1:
                                ap += 1
                            if p9 == 0 and p2 == 1:
                                ap += 1
                            
                            # 条件3: AP值为1
                            # 条件4: p2*p4*p8 == 0 且 p2*p6*p8 == 0
                            if ap == 1 and (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                                m_flag.append((i, j))
                
                # 删除标记的点
                for (i, j) in m_flag:
                    dst[i, j] = 0
                
                if not m_flag:
                    break
            
            # 将结果转换为0-255范围
            dst = dst * 255
            return dst
        except Exception as e:
            print(f"图像细化算法出错: {e}")
            # 如果出错，返回原始二值图像
            return src

    def get_skeleton_break_points(self, skeleton):
        """获取骨骼的端点"""
        try:
            points = np.column_stack(np.where(skeleton > 0))
            if len(points) < 2:
                return []
            
            # 找到最小和最大的y坐标点（假设y轴向下）
            top_idx = np.argmin(points[:, 0])
            bottom_idx = np.argmax(points[:, 0])
            
            # 转换为(x, y)格式
            top_point = (points[top_idx, 1], points[top_idx, 0])
            bottom_point = (points[bottom_idx, 1], points[bottom_idx, 0])
            
            return [top_point, bottom_point]
        except Exception as e:
            print(f"获取骨骼端点时出错: {e}")
            return []
    
    def find_nearest_point_on_skeleton(self, skeleton, line_type, line_pos):
        """寻找骨骼线上离中线最近的点
        参数:
            skeleton: 骨骼图像
            line_type: 线类型 ('horizontal' 或 'vertical')
            line_pos: 线的位置
        返回:
            tuple: (x, y) 最近点的坐标
        """
        try:
            # 提取骨骼点
            skeleton_points = np.column_stack(np.where(skeleton > 0))
            if len(skeleton_points) == 0:
                return (0, 0)
            
            min_dist = float('inf')
            nearest_point = (0, 0)
            
            for point in skeleton_points:
                y, x = point
                if line_type == 'horizontal':
                    # 水平线，计算y方向的距离
                    dist = abs(y - line_pos)
                else:  # vertical
                    # 垂直线，计算x方向的距离
                    dist = abs(x - line_pos)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = (x, y)
            
            return nearest_point
        except Exception as e:
            print(f"寻找最近点时出错: {e}")
            return (0, 0)
    
    def calculate_tangent_angle(self, skeleton, point, stem_box, tomato_box, radius=20):
        """计算采摘点的切线角度
        参数:
            skeleton: 骨骼图像
            point: 采摘点坐标 (x, y)
            radius: 搜索半径
        返回:
            float: 切线角度（弧度）
        """
        try:
            # 将点坐标转换为整数
            x, y = map(int, point)
            height, width = skeleton.shape
            
            # 收集采摘点附近的骨骼点
            nearby_points = []
            
            for i in range(max(0, y - radius), min(height, y + radius + 1)):
                for j in range(max(0, x - radius), min(width, x + radius + 1)):
                    if skeleton[i, j] == 1:
                        nearby_points.append((j, i))  # (x, y)格式
            
            if len(nearby_points) < 2:
                # 点数不足，返回0
                print(f"计算切线角度时，点数量不足: {len(nearby_points)}")
                return 0.0, skeleton  # 点数不足，返回0
            
            # 将点按x坐标排序（如果是水平方向）或y坐标排序（如果是垂直方向）
            # 这里我们简单按x坐标排序
            nearby_points.sort(key=lambda p: p[0])
            
            # 提取x和y坐标
            xs = [p[0] for p in nearby_points]
            ys = [p[1] for p in nearby_points]
            
            # 使用线性回归计算切线角度
            if len(xs) > 1:
                # 计算斜率
                coefficients = np.polyfit(xs, ys, 1)
                slope = coefficients[0]
                intercept = coefficients[1]
                # 计算角度
                angle = np.arctan(slope)
                
                # 创建彩色图像以绘制切线和采摘点
                if len(skeleton.shape) == 2:
                    # 如果是灰度图像，转换为彩色
                    skeleton_color = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)
                else:
                    skeleton_color = skeleton.copy()
                # 绘制采摘点（红色）
                cv2.circle(skeleton_color, (x, y), 5, (0, 0, 255), -1)
                
                # 计算切线的起点和终点
                # 为了使切线可见，我们计算图像边界上的点
                if abs(slope) < 1:
                    # 水平线为主
                    x1 = 0
                    y1 = int(slope * x1 + intercept)
                    x2 = width - 1
                    y2 = int(slope * x2 + intercept)
                else:
                    # 垂直线为主
                    y1 = 0
                    x1 = int((y1 - intercept) / slope) if slope != 0 else x
                    y2 = height - 1
                    x2 = int((y2 - intercept) / slope) if slope != 0 else x
                
                # 确保点在图像范围内
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                # 绘制切线（绿色）
                cv2.line(skeleton_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 在左上角显示单行文字
                text = f"angle: {np.degrees(angle):.2f}deg"    
                position = (10, 30)  # (x, y) 坐标，从左上角开始
                font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
                font_scale = 1.0  # 字体大小
                color = (0, 255, 0)  # 颜色 (B, G, R)
                thickness = 2  # 线条粗细
                line_type = cv2.LINE_AA  # 抗锯齿线条

                # 添加文字到图片
                cv2.putText(skeleton_color, text, position, font, font_scale, color, thickness, line_type)
                cv2.rectangle(skeleton_color, (int(stem_box[0]), int(stem_box[1])), (int(stem_box[2]), int(stem_box[3])), (0, 0, 255), 2)
                cv2.rectangle(skeleton_color, (int(tomato_box[0]), int(tomato_box[1])), (int(tomato_box[2]), int(tomato_box[3])), (255, 0, 0), 2)
                return angle, skeleton_color
            else:
                # 只有一个点，无法计算角度
                print(f"计算切线角度时，只有一个点: {nearby_points[0]}")
                return 0.0, skeleton
        except Exception as e:
            print(f"计算切线角度时出错: {e}")
            return 0.0, skeleton

    def calculate_tangent_angle_with_abc(self, skeleton, point, stem_box, tomato_box):
        """计算采摘点的切线角度（最高点+最左/最右点法）
        逻辑：
        1. 从骨骼图像中找最高点、最左点、最右点
        2. 比较最高点到最左点距离（dis_left）和最高点到最右点距离（dis_right）
        3. 距离大的那个点作为最低点
        4. 连接最高点和最低点，计算斜率和角度（arctan）
        5. 在skeleton上画一条过point、斜率相同的线段
        """
        try:
            x_p, y_p = map(int, point)
            height, width = skeleton.shape

            ys, xs = np.where(skeleton == 1)

            if len(xs) < 2:
                print(f"骨骼点数量不足: {len(xs)}")
                if len(skeleton.shape) == 2:
                    skeleton_color = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)
                else:
                    skeleton_color = skeleton.copy()
                return 0.0, skeleton_color

            top_idx = np.argmin(ys)
            top_point = (int(xs[top_idx]), int(ys[top_idx]))

            left_idx = np.argmin(xs)
            left_point = (int(xs[left_idx]), int(ys[left_idx]))

            right_idx = np.argmax(xs)
            right_point = (int(xs[right_idx]), int(ys[right_idx]))

            dis_left = np.sqrt(
                (top_point[0] - left_point[0]) ** 2 +
                (top_point[1] - left_point[1]) ** 2
            )
            dis_right = np.sqrt(
                (top_point[0] - right_point[0]) ** 2 +
                (top_point[1] - right_point[1]) ** 2
            )

            if dis_left >= dis_right:
                bottom_point = left_point
                side_label = "left"
            else:
                bottom_point = right_point
                side_label = "right"

            dx = bottom_point[0] - top_point[0]
            dy = bottom_point[1] - top_point[1]

            if dx == 0:
                slope = float('inf')
                angle = np.pi / 2
            else:
                slope = dy / dx
                angle = np.arctan(slope)

            if len(skeleton.shape) == 2:
                skeleton_color = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)
            else:
                skeleton_color = skeleton.copy()

            cv2.circle(skeleton_color, top_point, 5, (0, 0, 255), -1)
            cv2.putText(skeleton_color, "top",
                        (top_point[0] + 8, top_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.circle(skeleton_color, left_point, 5, (255, 0, 0), -1)
            cv2.putText(skeleton_color, "left",
                        (left_point[0] + 8, left_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.circle(skeleton_color, right_point, 5, (255, 255, 0), -1)
            cv2.putText(skeleton_color, "right",
                        (right_point[0] + 8, right_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.circle(skeleton_color, bottom_point, 6, (0, 165, 255), -1)
            cv2.putText(skeleton_color, "bottom(" + side_label + ")",
                        (bottom_point[0] + 8, bottom_point[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.line(skeleton_color, top_point, bottom_point, (255, 0, 255), 2)

            cv2.circle(skeleton_color, (x_p, y_p), 5, (0, 255, 0), -1)

            if np.isfinite(slope):
                intercept = y_p - slope * x_p
                if abs(slope) < 1:
                    x1_line = 0
                    y1_line = int(slope * x1_line + intercept)
                    x2_line = width - 1
                    y2_line = int(slope * x2_line + intercept)
                else:
                    y1_line = 0
                    x1_line = int((y1_line - intercept) / slope)
                    y2_line = height - 1
                    x2_line = int((y2_line - intercept) / slope)
            else:
                x1_line = x_p
                y1_line = 0
                x2_line = x_p
                y2_line = height - 1

            x1_line = max(0, min(x1_line, width - 1))
            y1_line = max(0, min(y1_line, height - 1))
            x2_line = max(0, min(x2_line, width - 1))
            y2_line = max(0, min(y2_line, height - 1))

            cv2.line(skeleton_color, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 0), 2)

            text = f"angle: {np.degrees(angle):.2f}deg dis_L={dis_left:.1f} dis_R={dis_right:.1f} {side_label}"
            cv2.putText(skeleton_color, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.rectangle(skeleton_color,
                          (int(stem_box[0]), int(stem_box[1])),
                          (int(stem_box[2]), int(stem_box[3])),
                          (0, 0, 255), 2)
            cv2.rectangle(skeleton_color,
                          (int(tomato_box[0]), int(tomato_box[1])),
                          (int(tomato_box[2]), int(tomato_box[3])),
                          (255, 0, 0), 2)

            return angle, skeleton_color
        except Exception as e:
            print(f"计算切线角度时出错: {e}")
            if len(skeleton.shape) == 2:
                skeleton_color = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)
            else:
                skeleton_color = skeleton.copy()
            return 0.0, skeleton_color


    def detection_growth_direction(self, skeleton_points, stem_center_x, tomato_center_x, stem_aspect_ratio):
        """判断番茄相对于果梗的位置和果梗生长方向，并计算采摘点
        参数:
            skeleton_points: 果梗骨架点
            stem_center_x: 果梗中心x坐标
            tomato_center_x: 番茄中心x坐标
            stem_aspect_ratio: 果梗宽高比
            
        返回:
           growth_direction: 番茄相对于果梗的位置（朝左、朝右、朝前）
        """
        #修改初步判断左右的代码
        if len(skeleton_points)>2:                   
            # 找到最小行(y坐标)和最大行(y坐标)的索引
            top_y = np.argmin(skeleton_points[:, 0])  # 最小行索引
            bottom_y = np.argmax(skeleton_points[:, 0])  # 最大行索引
            # 获取对应的列坐标(x坐标)
            top_y_x = skeleton_points[top_y, 1]  # 最小行对应的列坐标
            bottom_y_x = skeleton_points[bottom_y, 1]  # 最大行对应的列坐标
            position = "左边" if top_y_x > bottom_y_x else "右边"
            print(f"骨架法：果梗顶部y坐标: {top_y_x}, 果梗底部y坐标: {bottom_y_x}")

        else:  #保底计算左右，只使用矩形框判断      
            print(f"果框比较法：番茄中心x坐标: {tomato_center_x}, 果梗中心x坐标: {stem_center_x}")
            position = "左边" if tomato_center_x < stem_center_x else "右边"
        growth_direction = "朝前"        
        print(f"果梗宽高比: {stem_aspect_ratio}, 果梗初步判断位置：{position}")
        # 根据位置和宽高比确定生长方向和采摘点
        if position == "左边":
            if stem_aspect_ratio < 0.6:
                growth_direction = "朝前"
            else:
                growth_direction = "朝左"
        else:  # position == "右边"
            if stem_aspect_ratio < 0.9:
                growth_direction = "朝前"

            else:
                growth_direction = "朝右"               
        print(f"果梗最终判断朝向：{growth_direction}")
        return growth_direction




    
    def determine_growth_direction_and_picking_point(self, pair_idx, stem, tomato, stem_cv_mask, tomato_cv_mask, cv_color,stem_center_x, tomato_center_x, stem_aspect_ratio, w, h):
        """判断番茄相对于果梗的位置和果梗生长方向，并计算采摘点
        参数:
            pair_idx: 匹配对的索引
            stem: 果梗数据 [x1, y1, x2, y2]
            stem_cv_mask: 果梗掩码
            tomato_cv_mask: 番茄掩码
            cv_color: 原始彩色图像
            stem_center_x: 果梗中心x坐标
            tomato_center_x: 番茄中心x坐标
            stem_aspect_ratio: 果梗宽高比
            w, h: 图像宽度和高度
            tomato: 番茄数据 [x1, y1, x2, y2]
        返回:
            tuple: (stemROI_Location_Relative_tomatoROI, angle, image_pick_point, rad, skeleton_leftmost_point, skeleton_rightmost_point, tangent_angle)
            stemROI_Location_Relative_tomatoROI: 果梗相对于番茄的位置 (-1: 向左, 0: 垂直, 1: 向右)
            angle: 生长角度类型 (0: 垂直, 1: 左高右低, 2: 右高左低)
            image_pick_point: 采摘点图像坐标 (x, y)
            rad: 果梗生长的实际角度
            skeleton_leftmost_point: 细化骨骼的最左边点坐标
            skeleton_rightmost_point: 细化骨骼的最右边点坐标
            tangent_angle: 采摘点的切线角度
        """
        # 初始化返回值
        stemROI_Location_Relative_tomatoROI = "朝前"
        angle = 0  # 0: 垂直, 1: 朝下生长, 2: 朝上生长 即翘起
        is_growth_down = 0
        image_pick_point = (0, 0)
        rad = 0.0
        skeleton_leftmost_point = (0, 0)
        skeleton_rightmost_point = (0, 0)
        skeleton_topmost_point = (0, 0)
        tangent_angle = 0.0
        mind_stem_1=(0, 0)
        mind_stem_2=(0, 0)

        
        # 1. 检测掩膜的轮廓数量
        try:
            # 确保输入是二值图像
            if len(stem_cv_mask.shape) == 3:
                stem_cv_mask_gray = cv2.cvtColor(stem_cv_mask, cv2.COLOR_BGR2GRAY)
            else:
                stem_cv_mask_gray = stem_cv_mask.copy()
            
            # 2. 查找轮廓
            contours, _ = cv2.findContours(stem_cv_mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. 如果掩膜数量不是1，对掩膜进行膨胀操作
            if len(contours) != 1:
                print(f"果梗掩膜不连续，执行膨胀操作...")
                # 创建膨胀核
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                # 执行膨胀操作
                dilated_mask = cv2.dilate(stem_cv_mask_gray, kernel)
                # 转换为0-1二值图像
                dilated_mask = dilated_mask // 255
            else:
                # 直接使用原始掩码
                dilated_mask = stem_cv_mask_gray // 255
                print(f"果梗掩膜是连续的，不需要执行膨胀操作...")

            
            # 4. 对膨胀的果梗进行细化提取骨骼
            thin = self.image_thin(dilated_mask * 255)
            print(f"---------------细化骨骼提取完成----------------")
            # print(f"细化图像的尺寸: {thin.shape}")
            
            
            # 5. 计算果梗相对于番茄的位置 stemROI_Location_Relative_tomatoROI
            # 果梗和番茄的边界框
            stem_x1=stem[0]
            stem_y1=stem[1]
            stem_x2=stem[2]
            stem_y2=stem[3]
            stem_w = stem_x2 - stem_x1
            stem_h = stem_y2 - stem_y1
            print(f"---------------计算果梗宽高完成----------------")
            print(f"果梗宽度: {stem_w}")
            print(f"果梗高度: {stem_h}")
            
            
            # 计算番茄的边界框
            tomato_x1=tomato[0]
            tomato_y1=tomato[1]
            tomato_x2=tomato[2]
            tomato_y2=tomato[3]
            tomato_w = tomato_x2 - tomato_x1
            tomato_h = tomato_y2 - tomato_y1
            print(f"---------------计算番茄宽高完成----------------")
            print(f"番茄宽度: {tomato_w}")
            print(f"番茄高度: {tomato_h}")
            # 提取骨骼点
            skeleton_points = np.column_stack(np.where(thin > 0))
            print(f"---------------提取骨骼点完成----------------")
            print(f"提取到的骨骼点数量: {len(skeleton_points)}")
            
            # 获取细化骨骼的最左边、最右边、最高点和最低点
            if len(skeleton_points) > 10:
                # 最左边点（最小x坐标）
                leftmost_idx = np.argmin(skeleton_points[:, 1])
                skeleton_leftmost_point = (skeleton_points[leftmost_idx, 1], skeleton_points[leftmost_idx, 0])
                
                # 最右边点（最大x坐标）
                rightmost_idx = np.argmax(skeleton_points[:, 1])
                skeleton_rightmost_point = (skeleton_points[rightmost_idx, 1], skeleton_points[rightmost_idx, 0])
                
                # 最高点（最小y坐标，图像坐标系中y越小表示越高）
                topmost_idx = np.argmin(skeleton_points[:, 0])
                skeleton_topmost_point = (skeleton_points[topmost_idx, 1], skeleton_points[topmost_idx, 0])
                
                # 最低点（最大y坐标，图像坐标系中y越大表示越低）
                bottommost_idx = np.argmax(skeleton_points[:, 0])
                skeleton_bottommost_point = (skeleton_points[bottommost_idx, 1], skeleton_points[bottommost_idx, 0])
                
                print(f"---------------获取骨骼点完成----------------")
                print(f"最左边点坐标: {skeleton_leftmost_point}")
                print(f"最右边点坐标: {skeleton_rightmost_point}")
                print(f"最高点坐标: {skeleton_topmost_point}")
                print(f"最低点坐标: {skeleton_bottommost_point}")
            else:
                print(f"提取到的骨骼点数量: {len(skeleton_points)},细化后骨骼数量小于10，放弃这串")
                return False, "朝前", 0, (0, 0), 0.0, (0, 0), (0, 0), (0, 0), 0.0
            
            # # 计算果梗相对于番茄的位置 判断左右 方法1：
            # if stem_x1 - 0.1 * stem_w < tomato_x1:
            #     # 果梗左边在番茄左边
            #     stemROI_Location_Relative_tomatoROI = "朝右"  # 向右生长
            # elif stem_x2 + 0.1 * stem_w > tomato_x2:
            #     # 果梗右边在番茄右边
            #     stemROI_Location_Relative_tomatoROI = "朝左"  # 向左生长
            # else:
            #     # 果梗在番茄垂直上方
            #     stemROI_Location_Relative_tomatoROI = "朝前"  # 垂直生长
            
            
            #  计算果梗相对于番茄的位置，判断左右，方法2：
            stemROI_Location_Relative_tomatoROI=self.detection_growth_direction(skeleton_points,stem_center_x, tomato_center_x,stem_aspect_ratio)
            print(f"---------------果梗生长方向判断完毕----------------")
            print(f"果梗生长方向是: {stemROI_Location_Relative_tomatoROI}")
            # 6. 根据果梗的宽高比判断生长方向并计算采摘点
            if stem_h > 0.8 * stem_w:
                # 垂直生长
                mind = int(stem_h * 0.5+stem_y1)
                mind_stem_1=(stem_x1, mind) # 中线上的点  opencv格式,即右上角为(0,0)
                mind_stem_2=(stem_x2, mind) # 中线上的点  opencv格式,即右上角为(0,0)


                cut_x_sum = 0
                cout_cut = 0
                
                # 查找中线上与骨骼的交点
                for x1 in range(thin.shape[1]):
                    if thin[mind, x1] == 1:
                        cut_x_sum += x1
                        cout_cut += 1
                
                if cout_cut > 0:
                    cut_x = cut_x_sum // cout_cut
                    # 计算采摘点在原始图像中的坐标
                    image_pick_point = (cut_x, mind)
                else:
                    # 如果没有找到交点，寻找骨骼线离中线最近的点
                    nearest_point = self.find_nearest_point_on_skeleton(thin, 'horizontal', mind)
                    # 转换为原始图像坐标
                    image_pick_point = (nearest_point[0], nearest_point[1])
                    print(f"未找到中线与骨骼的交点，使用最近点: {image_pick_point}")
            else:
                # 水平生长
                mind = int(stem_w * 0.5+stem_x1)
                mind_stem_1=(mind, stem_y1) # 中线上的点  opencv格式,即右上角为(0,0)
                mind_stem_2=(mind, stem_y2) # 中线上的点  opencv格式,即右上角为(0,0)

                cut_y_sum = 0
                cout_cut = 0
                
                # 查找中线上与骨骼的交点
                for y1 in range(thin.shape[0]):
                    if thin[y1, mind] == 1:
                        cut_y_sum += y1
                        cout_cut += 1
                
                if cout_cut > 0:
                    cut_y = cut_y_sum // cout_cut
                    # 计算采摘点在原始图像中的坐标
                    image_pick_point = (mind, cut_y)
                else:
                    # 如果没有找到交点，寻找骨骼线离中线最近的点
                    nearest_point = self.find_nearest_point_on_skeleton(thin, 'vertical', mind)
                    # 转换为原始图像坐标
                    image_pick_point = (nearest_point[0], nearest_point[1])
                    print(f"未找到中线与骨骼的交点，使用最近点: {image_pick_point}")
            #打印采摘点
            print(f"---------------计算采摘点完成----------------")
            print(f"计算到的采摘点坐标: {image_pick_point}")
            # 7. 计算采摘点的切线角度
            # 将采摘点转换为骨骼图像中的坐标
            skeleton_pick_point = (image_pick_point[0], image_pick_point[1])
            # tangent_angle, skeleton_color = self.calculate_tangent_angle(thin, skeleton_pick_point, stem, tomato, radius=10)
            tangent_angle, skeleton_color = self.calculate_tangent_angle_with_abc(thin, skeleton_pick_point, stem, tomato)

            
            #tangent_angle角度说明,为正值时表示切线斜向下,为负值时表示切线斜向上
            
            print(f"---------------计算采摘点切线角度完成----------------")
            print(f"采摘点的切线角度: {tangent_angle} 弧度, 角度: {tangent_angle * 180 / np.pi} 度")
            
            # 8. 计算果梗生长的实际角度
            skeleton_endpoints = self.get_skeleton_break_points(thin)
            if len(skeleton_endpoints) == 2:
                # 计算角度
                dy = skeleton_endpoints[1][1] - skeleton_endpoints[0][1]
                dx = skeleton_endpoints[1][0] - skeleton_endpoints[0][0]
                
                if dx != 0:
                    # rad = np.arctan2(dy, dx)
                    rad=tangent_angle
                    deg = rad * 180 / np.pi
                    print(f"弧度rad: {rad}, 角度deg: {deg}")
                    print(f"果梗生长的实际角度: {deg} 度")
                    # 根据角度和相对位置确定生长角度类型 1代表朝上生长 2代表朝下生长 0代表垂直生长
                    if -70 <= deg <= 0:
                        if stemROI_Location_Relative_tomatoROI == "朝右":   
                            angle = 1  # 右高左低,向右生长
                        elif stemROI_Location_Relative_tomatoROI == "朝左":
                            angle = 2  # 右高左低，向左生长
                        else:
                            angle = 0  # 垂直
                    elif 0 < deg <= 70:
                        if stemROI_Location_Relative_tomatoROI == "朝左":
                            angle = 2  # 左高右低，向左生长
                        elif stemROI_Location_Relative_tomatoROI == "朝右":
                            angle = 1  # 左高右低，向右生长
                        else:
                            angle = 0  # 垂直
                    else:
                        angle = 0  # 垂直生长
                else:
                    angle = 0  # 垂直生长
                    print("dx为0，将此果梗归为垂直生长!")
            else:
                angle = 0  # 垂直生长
                print("骨骼端点不足，将此果梗归为垂直生长!")
            # 在左上角显示单行文字
            if stemROI_Location_Relative_tomatoROI=="朝右":
                location="right"
            elif stemROI_Location_Relative_tomatoROI=="朝左":
                location="left"
            else:
                location="forward"
            text = f"left_or_right: {location},angle: {angle}"
            position = (30, 300)  # (x, y) 坐标，从左上角开始
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
            font_scale = 1.0  # 字体大小
            color = (0, 255, 0)  # 颜色 (B, G, R)
            thickness = 2  # 线条粗细
            line_type = cv2.LINE_AA  # 抗锯齿线条

            # 添加文字到图片
            cv2.putText(skeleton_color, text, position, font, font_scale, color, thickness, line_type)


            # 9. 如果连通区域数量大于1，使用最小外接矩形方法
            num_labels, labels = cv2.connectedComponents(dilated_mask * 255)
            actual_regions = num_labels - 1
            print(f"---------------最终连通区域数量判断完成----------------")
            print(f"最终连通区域数量: {actual_regions}")
            if actual_regions > 1:
                coords = np.column_stack(np.where(dilated_mask > 0))
                if len(coords) > 0:
                    # OpenCV的坐标是(y,x)，需要转换为(x,y)
                    coords = coords[:, ::-1]
                    # 计算最小外接矩形
                    rect = cv2.minAreaRect(coords.astype(np.float32))
                    # 获取矩形的中心点作为采摘点
                    picking_point = (int(rect[0][0]), int(rect[0][1]))
                    # 转换为原始图像坐标 这个需要加上
                    image_pick_point = (picking_point[0], picking_point[1])
                    print(f"使用最小外接矩形方法确定采摘点: {image_pick_point}")
                    # 重新计算切线角度
                    skeleton_pick_point = (image_pick_point[0], image_pick_point[1])
                    tangent_angle, skeleton_color = self.calculate_tangent_angle(thin, skeleton_pick_point, stem, tomato, radius=10)
                    # tangent_angle, skeleton_color = self.calculate_tangent_angle_with_abc(thin, skeleton_pick_point, stem, tomato)
            
            print(f"果梗相对于番茄的位置: {stemROI_Location_Relative_tomatoROI}")
            print(f"生长角度类型: {angle}")
            print(f"采摘点: {image_pick_point}")
            print(f"细化骨骼的最左边点: {skeleton_leftmost_point}")
            print(f"细化骨骼的最右边点: {skeleton_rightmost_point}")
            x_left  , y_left  = map(int, skeleton_leftmost_point)
            cv2.circle(skeleton_color, (x_left, y_left), 5, (255, 0, 0), -1)
            x_right , y_right = map(int, skeleton_rightmost_point)
            cv2.circle(skeleton_color, (x_right, y_right), 5, (0, 255, 0), -1)
            cv2.line(skeleton_color, (int(mind_stem_1[0]), int(mind_stem_1[1])), (int(mind_stem_2[0]), int(mind_stem_2[1])), (255, 255, 0), 2)
            try:
                # 创建保存目录（如果不存在）
                # 自动获取主目录，避免硬编码用户名
                home_dir = os.path.expanduser("~")
                # 构建目标文件夹路径（示例：主目录下的test/temp_test/my_folder）
                save_dir = os.path.join(home_dir, "code", "test", "temp_img")
                # save_dir = "/home/xc_home_wsl/code/test/temp"
                os.makedirs(save_dir, exist_ok=True)
                
                # 生成唯一的文件名
                timestamp = int(time.time() * 1000)
                filename = f"{save_dir}/tangent_line_{timestamp}.jpg"
                filename_color = f"{save_dir}/tangent_line_{timestamp}_color.jpg"

                
                # 保存图像
                if self.save_img==True:
                    cv2.imwrite(filename, skeleton_color)
                    cv2.imwrite(filename_color, cv_color)
                    # 保存果梗掩码图像
                    filename_stem_mask = f"{save_dir}/tangent_line_{timestamp}_stem_mask.jpg"
                    cv2.imwrite(filename_stem_mask, stem_cv_mask)
                    print(f"切线和采摘点图像已保存到: {filename}")
                # cv2.imshow("Tangent Line with Pick Point", skeleton_color)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            except Exception as e:
                print(f"保存图像时出错: {e}")
            # 保存图像到文件，避免窗口管理问题
            
                # 确保关闭所有窗口（如果有）
            # 为了与调用代码兼容，将angle作为is_growth_down返回
            is_growth_down = angle
            
        except Exception as e:
            print(f"处理过程中出错: {e}")
            # 使用默认值
            image_pick_point = (stem_center_x, int((stem[1] + stem[3]) / 2))
        image_pick_point = (float(image_pick_point[0]), float(image_pick_point[1]))
        return True,stemROI_Location_Relative_tomatoROI, is_growth_down, image_pick_point, rad, skeleton_leftmost_point, skeleton_rightmost_point, skeleton_topmost_point, tangent_angle

# 示例用法
if __name__ == "__main__":
    system = StemGrowthAndPoints()
    
    # 示例数据
    pair_idx = 0
    stem = [100, 50, 200, 150]  # [x1, y1, x2, y2]
    tomato = [120, 150, 220, 250]  # [x1, y1, x2, y2]
    w, h = 640, 480
    
    # 创建示例掩码
    stem_cv_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(stem_cv_mask, (int(stem[0]), int(stem[1])), (int(stem[2]), int(stem[3])), 255, -1)
    
    tomato_cv_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(tomato_cv_mask, (int(tomato[0]), int(tomato[1])), (int(tomato[2]), int(tomato[3])), 255, -1)
    
    cv_color = np.zeros((h, w, 3), dtype=np.uint8)
    stem_center_x = (stem[0] + stem[2]) // 2
    tomato_center_x = (tomato[0] + tomato[2]) // 2
    stem_aspect_ratio = (stem[2] - stem[0]) / (stem[3] - stem[1])
    
    # 调用函数
    result = system.determine_growth_direction_and_picking_point(
        pair_idx, stem, stem_cv_mask, tomato_cv_mask, cv_color, 
        stem_center_x, tomato_center_x, stem_aspect_ratio, w, h, tomato
    )
    
    print("\n最终结果:")
    print(f"stemROI_Location_Relative_tomatoROI: {result[0]}")
    print(f"angle: {result[1]}")
    print(f"image_pick_point: {result[2]}")
    print(f"rad: {result[3]}")
    print(f"skeleton_leftmost_point: {result[4]}")
    print(f"skeleton_rightmost_point: {result[5]}")
    print(f"tangent_angle: {result[6]} 弧度")
