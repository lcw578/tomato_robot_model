import cv2
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray


CLASS_COLORS = {
    'peduncle': (255, 0, 0),
    'tomato_cluster': (180, 220, 0),
    'tomato_fruit': (95, 0, 0),
}


def ensure_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def overlay_mask(image, mask, color, alpha=0.35):
    out = image.copy()
    if mask is None or not mask.any():
        return out
    color_img = np.zeros_like(out)
    color_img[:] = color
    out[mask.astype(bool)] = cv2.addWeighted(out, 1.0 - alpha, color_img, alpha, 0)[mask.astype(bool)]
    return out


def draw_instance_debug(image, instances):
    out = ensure_bgr(image)
    for inst in instances:
        mask = inst['mask']
        name = inst['class_name']
        color = CLASS_COLORS.get(name, (255, 255, 255))
        if name == 'peduncle':
            out = overlay_mask(out, mask, color, 0.45)
            continue

        if name == 'tomato_cluster':
            out = overlay_mask(out, mask, color, 0.32)
            x1, y1, x2, y2 = [int(round(v)) for v in inst['bbox_xyxy']]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            continue

        if name == 'tomato_fruit':
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, color, 1)
            continue

        out = overlay_mask(out, mask, color)
    return out


def draw_overlap_debug(image, overlaps):
    out = image.copy()
    for item in overlaps:
        mask = item.get('mask')
        if mask is None or not mask.any():
            continue
        out[mask.astype(bool)] = (255, 255, 255)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x = int(xs.mean())
        y = int(ys.mean())
        label = f"{item['a_id']}-{item['b_id']} iou:{item['iou']:.2f}"
        cv2.putText(out, label, (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_pose_debug(image, debug_items):
    out = ensure_bgr(image)
    for item in debug_items:
        if item.get('peduncle_mask') is not None:
            out = overlay_mask(out, item['peduncle_mask'], CLASS_COLORS['peduncle'], 0.28)
        if item.get('cluster_mask') is not None:
            out = overlay_mask(out, item['cluster_mask'], CLASS_COLORS['tomato_cluster'], 0.20)
        skeleton = item.get('skeleton')
        if skeleton is not None and skeleton.any():
            out[skeleton.astype(bool)] = (0, 255, 255)
        pick = item.get('pick')
        if pick is not None:
            u, v = int(round(pick[0])), int(round(pick[1]))
            cv2.circle(out, (u, v), 5, (0, 255, 0), -1)
            cv2.circle(out, (u, v), 9, (0, 0, 0), 1)
            label = f"id:{item.get('candidate_id', 0)} z:{item.get('depth', 0):.3f} s:{item.get('score', 0):.2f}"
            cv2.putText(out, label, (u + 8, max(15, v - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


def candidates_to_markers(candidates, frame_id, stamp):
    marker_array = MarkerArray()
    for idx, candidate in enumerate(candidates):
        pose = candidate.pose_camera
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'vision_pose_generate'
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 0.018
        marker.scale.y = 0.018
        marker.scale.z = 0.018
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker_array.markers.append(marker)

        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = stamp
        text.ns = 'vision_pose_generate_text'
        text.id = idx
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose = pose.pose
        text.pose.position.z += 0.035
        text.scale.z = 0.025
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f"{candidate.candidate_id}:{candidate.score:.2f}"
        marker_array.markers.append(text)
    return marker_array
