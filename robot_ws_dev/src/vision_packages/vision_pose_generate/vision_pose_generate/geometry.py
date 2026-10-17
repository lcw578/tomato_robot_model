import math
from collections import deque

import cv2
import numpy as np


def mask_msg_to_bool(mask_msg, bridge):
    mask = bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
    return mask > 0


def bool_mask_to_msg(mask, bridge, header):
    msg = bridge.cv2_to_imgmsg((mask.astype(np.uint8) * 255), encoding='mono8')
    msg.header = header
    return msg


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def centroid_from_mask(mask):
    moments = cv2.moments(mask.astype(np.uint8))
    if moments['m00'] <= 0:
        return None
    return np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']], dtype=np.float32)


def touches_border(mask, margin):
    if mask.size == 0:
        return False
    h, w = mask.shape[:2]
    margin = max(0, int(margin))
    left = mask[:, :margin + 1].any()
    right = mask[:, max(0, w - margin - 1):].any()
    top = mask[:margin + 1, :].any()
    bottom = mask[max(0, h - margin - 1):, :].any()
    return bool(left or right or top or bottom)


def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def skeletonize(mask):
    binary = (mask.astype(np.uint8) * 255)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(binary) > 0

    # Fallback morphological skeletonization when opencv-contrib is unavailable.
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded.copy()
        if cv2.countNonZero(work) == 0:
            break
    return skel > 0


def skeleton_endpoints(skeleton):
    points = np.column_stack(np.where(skeleton))
    endpoints = []
    for y, x in points:
        y0, y1 = max(0, y - 1), min(skeleton.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(skeleton.shape[1], x + 2)
        neighbors = int(skeleton[y0:y1, x0:x1].sum()) - 1
        if neighbors == 1:
            endpoints.append((x, y))
    return np.array(endpoints, dtype=np.float32)


def ordered_skeleton_from_endpoint(skeleton, start_xy):
    points_yx = np.column_stack(np.where(skeleton))
    if len(points_yx) == 0:
        return np.empty((0, 2), dtype=np.float32)

    skel_set = {(int(x), int(y)) for y, x in points_yx}
    start = min(skel_set, key=lambda p: (p[0] - start_xy[0]) ** 2 + (p[1] - start_xy[1]) ** 2)
    visited = {start}
    ordered = [start]
    queue = deque([start])

    while queue:
        current = queue.popleft()
        cx, cy = current
        neighbors = []
        for ny in range(cy - 1, cy + 2):
            for nx in range(cx - 1, cx + 2):
                if (nx, ny) == current:
                    continue
                if (nx, ny) in skel_set and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
        neighbors.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        for n in neighbors:
            visited.add(n)
            ordered.append(n)
            queue.append(n)

    return np.array(ordered, dtype=np.float32)


def min_mask_distance(mask_a, mask_b):
    ys_a, xs_a = np.where(mask_a)
    ys_b, xs_b = np.where(mask_b)
    if len(xs_a) == 0 or len(xs_b) == 0:
        return float('inf')
    pts_a = np.column_stack((xs_a, ys_a)).astype(np.float32)
    pts_b = np.column_stack((xs_b, ys_b)).astype(np.float32)
    if len(pts_a) > 800:
        pts_a = pts_a[np.linspace(0, len(pts_a) - 1, 800).astype(int)]
    if len(pts_b) > 800:
        pts_b = pts_b[np.linspace(0, len(pts_b) - 1, 800).astype(int)]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


def robust_depth_at(depth_image, mask, u, v, windows, min_depth, max_depth):
    h, w = depth_image.shape[:2]
    for win in windows:
        radius = int(win) // 2
        x0, x1 = max(0, int(round(u)) - radius), min(w, int(round(u)) + radius + 1)
        y0, y1 = max(0, int(round(v)) - radius), min(h, int(round(v)) + radius + 1)
        roi = depth_image[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1] if mask is not None else np.ones_like(roi, dtype=bool)
        values = roi[mask_roi]
        values = values[np.isfinite(values)]
        values = values[(values > min_depth) & (values < max_depth)]
        if values.size:
            return float(np.median(values))
    return 0.0


def direct_depth_at(depth_image, u, v, min_depth, max_depth):
    h, w = depth_image.shape[:2]
    x = int(round(u))
    y = int(round(v))
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    value = float(depth_image[y, x])
    if np.isfinite(value) and min_depth < value < max_depth:
        return value
    return 0.0


def normalize_depth_image(depth):
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def deproject_pixel(u, v, depth_m, camera_info):
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]
    if fx == 0 or fy == 0 or depth_m <= 0:
        return None
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def fit_3d_tangent_from_pixels(pixel_points, depth_image, camera_info, min_depth, max_depth, reference_direction=None):
    if pixel_points is None or len(pixel_points) < 2:
        return None

    h, w = depth_image.shape[:2]
    points_3d = []
    for u, v in np.asarray(pixel_points, dtype=np.float64):
        x = int(round(u))
        y = int(round(v))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        z = float(depth_image[y, x])
        if not np.isfinite(z) or not (min_depth < z < max_depth):
            continue
        p = deproject_pixel(float(u), float(v), z, camera_info)
        if p is not None:
            points_3d.append(p)

    if len(points_3d) < 2:
        return None

    points = np.asarray(points_3d, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    tangent = vh[0]
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return None
    tangent = tangent / norm
    if reference_direction is not None:
        ref = np.asarray(reference_direction, dtype=np.float64)
        ref_norm = np.linalg.norm(ref)
        if ref_norm > 1e-6 and float(np.dot(tangent, ref / ref_norm)) < 0:
            tangent = -tangent
    return tangent


def quaternion_from_matrix(rotation):
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def rpy_from_matrix(rotation):
    m = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(m[2, 1], m[2, 2])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def rotation_from_tangent(tangent_3d):
    tangent = np.asarray(tangent_3d, dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6:
        tangent = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / tangent_norm

    # Tool frame convention for picking:
    #   tool_z: forward, close to camera optical +Z
    #   tool_y: fitted peduncle direction
    #   tool_x: left side in the image, close to optical -X
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    left = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    tool_y = tangent
    tool_z = forward - tool_y * float(np.dot(forward, tool_y))
    z_norm = np.linalg.norm(tool_z)
    if z_norm < 1e-6:
        tool_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        tool_z = tool_z / z_norm

    tool_x = np.cross(tool_y, tool_z)
    x_norm = np.linalg.norm(tool_x)
    if x_norm < 1e-6:
        tool_x = left.copy()
    else:
        tool_x = tool_x / x_norm

    if float(np.dot(tool_x, left)) < 0:
        tool_y = -tool_y
        tool_x = -tool_x

    tool_z = np.cross(tool_x, tool_y)
    tool_z = tool_z / max(np.linalg.norm(tool_z), 1e-6)
    if float(np.dot(tool_z, forward)) < 0:
        tool_x = -tool_x
        tool_z = -tool_z
    tool_y = np.cross(tool_z, tool_x)
    tool_y = tool_y / max(np.linalg.norm(tool_y), 1e-6)

    rotation = np.column_stack((tool_x, tool_y, tool_z))
    return rotation


def pose_orientation_from_tangent(tangent_3d):
    return quaternion_from_matrix(rotation_from_tangent(tangent_3d))


def pose_orientation_rpy_from_tangent(tangent_3d):
    return rpy_from_matrix(rotation_from_tangent(tangent_3d))
