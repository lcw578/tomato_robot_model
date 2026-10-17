import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from rclpy.node import Node

from vision_tomato_interfaces.msg import PickPoseCandidate
from vision_tomato_interfaces.srv import GeneratePickPoses

from .geometry import (
    centroid_from_mask,
    deproject_pixel,
    direct_depth_at,
    fit_3d_tangent_from_pixels,
    largest_component,
    mask_msg_to_bool,
    min_mask_distance,
    normalize_depth_image,
    ordered_skeleton_from_endpoint,
    pose_orientation_from_tangent,
    pose_orientation_rpy_from_tangent,
    robust_depth_at,
    skeleton_endpoints,
    skeletonize,
    touches_border,
)
from .debug_viz import draw_pose_debug
from .maturity import estimate_fruit_maturity


class PoseEstimatorService(Node):
    def __init__(self):
        super().__init__('pose_estimator_service')
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 1.2)
        self.declare_parameter('depth_windows', [7, 11, 15, 21])
        self.declare_parameter('min_peduncle_area_px', 20)
        self.declare_parameter('min_skeleton_points', 8)
        self.declare_parameter('pick_offset_px', 10)
        self.declare_parameter('pick_point_mode', 'skeleton_midpoint')
        self.declare_parameter('tangent_fit_radius_px', 18)
        self.declare_parameter('min_tangent_fit_points', 4)
        self.declare_parameter('crop_border_margin_px', 5)
        self.declare_parameter('enable_maturity', True)
        self.declare_parameter('hand_eye_enabled', False)
        self.declare_parameter('hand_eye_matrix_file', '')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')

        self.bridge = CvBridge()
        self.srv = self.create_service(
            GeneratePickPoses,
            '/vision/tomato/generate_pick_poses',
            self.handle_generate,
        )
        self.get_logger().info('pose_estimator_service started at /vision/tomato/generate_pick_poses')

    def parse_instances(self, instances):
        parsed = []
        for inst in instances:
            try:
                mask = mask_msg_to_bool(inst.mask, self.bridge)
            except Exception as exc:
                self.get_logger().warning(f'failed to decode mask {inst.instance_id}: {exc}')
                continue
            parsed.append({
                'msg': inst,
                'id': int(inst.instance_id),
                'class_name': inst.class_name,
                'confidence': float(inst.confidence),
                'mask': mask,
                'bbox': list(inst.bbox_xyxy),
                'touches_border': bool(inst.touches_image_border),
                'centroid': centroid_from_mask(mask),
            })
        return parsed

    def associate(self, peduncles, clusters, fruits):
        associations = []
        for cluster in clusters:
            cluster_center = cluster['centroid']
            if cluster_center is None:
                continue
            best_peduncle = None
            best_dist = float('inf')
            for peduncle in peduncles:
                dist = min_mask_distance(peduncle['mask'], cluster['mask'])
                if dist < best_dist:
                    best_dist = dist
                    best_peduncle = peduncle
            if best_peduncle is None:
                continue

            cluster_fruits = []
            for fruit in fruits:
                c = fruit['centroid']
                if c is None:
                    continue
                x, y = int(round(c[0])), int(round(c[1]))
                inside = 0 <= y < cluster['mask'].shape[0] and 0 <= x < cluster['mask'].shape[1] and cluster['mask'][y, x]
                overlap = np.logical_and(fruit['mask'], cluster['mask']).sum()
                if inside or overlap > 0:
                    cluster_fruits.append(fruit)
            associations.append((best_peduncle, cluster, cluster_fruits, best_dist))
        return associations

    def choose_pick_point(self, peduncle_mask, cluster_mask):
        clean = largest_component(peduncle_mask)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        skeleton = skeletonize(clean)
        cluster_center = centroid_from_mask(cluster_mask)
        peduncle_center = centroid_from_mask(clean)
        min_skeleton_points = int(self.get_parameter('min_skeleton_points').value)
        pick_offset = int(self.get_parameter('pick_offset_px').value)
        pick_point_mode = self.get_parameter('pick_point_mode').value

        if skeleton.sum() >= min_skeleton_points and cluster_center is not None:
            endpoints = skeleton_endpoints(skeleton)
            if len(endpoints) == 0:
                skel_points = np.column_stack(np.where(skeleton))[:, ::-1].astype(np.float32)
                start = skel_points[np.argmin(np.linalg.norm(skel_points - cluster_center, axis=1))]
            else:
                start = endpoints[np.argmin(np.linalg.norm(endpoints - cluster_center, axis=1))]
            ordered = ordered_skeleton_from_endpoint(skeleton, start)
            if len(ordered) > 0:
                if pick_point_mode == 'skeleton_midpoint':
                    idx = len(ordered) // 2
                    reason = 'skeleton_midpoint'
                else:
                    idx = min(max(0, pick_offset), len(ordered) - 1)
                    reason = 'skeleton_offset'
                pick = ordered[idx]
                if len(ordered) > 3:
                    tail_idx = min(len(ordered) - 1, idx + 3)
                    tangent_2d = ordered[tail_idx] - ordered[max(0, idx - 3)]
                else:
                    tangent_2d = np.array([0.0, 1.0], dtype=np.float32)
                return pick, skeleton, tangent_2d, reason

        if peduncle_center is not None:
            return peduncle_center, skeleton, np.array([0.0, 1.0], dtype=np.float32), 'centroid_fallback'
        return None, skeleton, np.array([0.0, 1.0], dtype=np.float32), 'no_pick_point'

    def get_tangent_fit_pixels(self, skeleton, peduncle_mask, u, v):
        radius = int(self.get_parameter('tangent_fit_radius_px').value)
        min_points = int(self.get_parameter('min_tangent_fit_points').value)
        pixels = []
        if skeleton is not None and skeleton.any():
            ys, xs = np.where(skeleton)
            coords = np.column_stack((xs, ys)).astype(np.float32)
            dist = np.linalg.norm(coords - np.array([u, v], dtype=np.float32), axis=1)
            pixels = coords[dist <= radius]
            if len(pixels) >= min_points:
                return pixels

        ys, xs = np.where(peduncle_mask)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        coords = np.column_stack((xs, ys)).astype(np.float32)
        dist = np.linalg.norm(coords - np.array([u, v], dtype=np.float32), axis=1)
        pixels = coords[dist <= radius]
        if len(pixels) > 80:
            order = np.argsort(np.linalg.norm(pixels - np.array([u, v], dtype=np.float32), axis=1))
            pixels = pixels[order[:80]]
        return pixels

    def estimate_cluster_maturity(self, color, fruits):
        if not bool(self.get_parameter('enable_maturity').value) or not fruits:
            return 'unknown', 0.0
        ripe_like = 0.0
        labels = []
        for fruit in fruits:
            stats = estimate_fruit_maturity(color, fruit['mask'])
            labels.append(stats['maturity'])
            ripe_like += stats['ripe_score']
        ripe_ratio = float(ripe_like / max(1, len(fruits)))
        if labels.count('ripe') >= max(1, int(np.ceil(len(labels) * 0.5))):
            maturity = 'ripe'
        elif labels.count('unripe') >= max(1, int(np.ceil(len(labels) * 0.5))):
            maturity = 'unripe'
        else:
            maturity = 'semi_ripe'
        return maturity, ripe_ratio

    def score_candidate(self, peduncle, cluster, depth, skeleton, truncated, maturity, assoc_dist):
        score = 0.0
        score += 0.35 * peduncle['confidence']
        score += 0.25 * cluster['confidence']
        score += 0.20 if depth > 0 else -0.3
        score += min(0.15, float(skeleton.sum()) / 200.0) if skeleton is not None else 0.0
        score += 0.1 if maturity == 'ripe' else 0.03 if maturity == 'semi_ripe' else 0.0
        score -= min(0.2, assoc_dist / 300.0)
        if truncated:
            score -= 0.25
        return float(max(0.0, min(1.0, score)))

    def handle_generate(self, request, response):
        try:
            color = self.bridge.imgmsg_to_cv2(request.color_image, desired_encoding='bgr8')
            raw_depth = self.bridge.imgmsg_to_cv2(request.depth_image, desired_encoding='passthrough')
        except Exception as exc:
            response.success = False
            self.get_logger().error(f'failed to convert images: {exc}')
            return response

        depth = normalize_depth_image(raw_depth)
        instances = self.parse_instances(request.instances)
        peduncles = [i for i in instances if i['class_name'] == 'peduncle']
        clusters = [i for i in instances if i['class_name'] == 'tomato_cluster']
        fruits = [i for i in instances if i['class_name'] == 'tomato_fruit']

        min_area = int(self.get_parameter('min_peduncle_area_px').value)
        margin = int(self.get_parameter('crop_border_margin_px').value)
        min_depth = float(self.get_parameter('min_depth_m').value)
        max_depth = float(self.get_parameter('max_depth_m').value)
        windows = list(self.get_parameter('depth_windows').value)
        camera_frame = request.camera_info.header.frame_id or self.get_parameter('camera_frame').value
        candidate_id = 0
        best_candidate = None
        best_debug_item = None

        for peduncle, cluster, cluster_fruits, assoc_dist in self.associate(peduncles, clusters, fruits):
            if int(peduncle['mask'].sum()) < min_area:
                continue
            pick, skeleton, tangent_2d, pick_reason = self.choose_pick_point(peduncle['mask'], cluster['mask'])
            if pick is None:
                continue
            u, v = float(pick[0]), float(pick[1])
            z = direct_depth_at(depth, u, v, min_depth, max_depth)
            depth_source = 'direct_pixel_depth'
            if z <= 0:
                z = robust_depth_at(depth, None, u, v, windows[:2], min_depth, max_depth)
                depth_source = 'local_window_median'
            if z <= 0:
                z = robust_depth_at(depth, peduncle['mask'], u, v, windows, min_depth, max_depth)
                depth_source = 'peduncle_mask_median'
            if z <= 0:
                z = robust_depth_at(depth, cluster['mask'], u, v, windows, min_depth, max_depth)
                depth_source = 'cluster_mask_median'
            point = deproject_pixel(u, v, z, request.camera_info)
            if point is None:
                continue

            tangent_pixels = self.get_tangent_fit_pixels(skeleton, peduncle['mask'], u, v)
            tangent_ref = np.array([float(tangent_2d[0]), float(tangent_2d[1]), 0.0], dtype=np.float64)
            tangent_3d = fit_3d_tangent_from_pixels(
                tangent_pixels,
                depth,
                request.camera_info,
                min_depth,
                max_depth,
                reference_direction=tangent_ref,
            )
            tangent_source = '3d_pca_tangent'
            if tangent_3d is None:
                tangent_3d = tangent_ref
                tangent_source = '2d_tangent_fallback'
            qx, qy, qz, qw = pose_orientation_from_tangent(tangent_3d)
            roll, pitch, yaw = pose_orientation_rpy_from_tangent(tangent_3d)
            pose_camera = PoseStamped()
            pose_camera.header = request.camera_info.header
            pose_camera.header.frame_id = camera_frame
            pose_camera.pose.position.x = float(point[0])
            pose_camera.pose.position.y = float(point[1])
            pose_camera.pose.position.z = float(point[2])
            pose_camera.pose.orientation.x = qx
            pose_camera.pose.orientation.y = qy
            pose_camera.pose.orientation.z = qz
            pose_camera.pose.orientation.w = qw

            maturity, ripe_ratio = self.estimate_cluster_maturity(color, cluster_fruits)
            truncated = bool(
                peduncle['touches_border'] or cluster['touches_border'] or
                touches_border(peduncle['mask'], margin) or touches_border(cluster['mask'], margin)
            )
            score = self.score_candidate(peduncle, cluster, z, skeleton, truncated, maturity, assoc_dist)

            candidate = PickPoseCandidate()
            candidate.header = request.camera_info.header
            candidate.candidate_id = candidate_id
            candidate.peduncle_id = peduncle['id']
            candidate.cluster_id = cluster['id']
            candidate.fruit_ids = [f['id'] for f in cluster_fruits]
            candidate.image_u = u
            candidate.image_v = v
            candidate.depth = float(z)
            candidate.pose_xyzrpy_camera = [
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(roll),
                float(pitch),
                float(yaw),
            ]
            candidate.pose_xyzrpy_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            candidate.pose_camera = pose_camera
            candidate.pose_base = PoseStamped()
            candidate.has_pose_base = False
            candidate.score = score
            candidate.maturity = maturity
            candidate.ripe_ratio = float(ripe_ratio)
            candidate.truncated_by_crop = truncated
            candidate.reason = (
                f'{pick_reason}|{depth_source}|{tangent_source}|'
                f't=({tangent_3d[0]:.3f},{tangent_3d[1]:.3f},{tangent_3d[2]:.3f})'
            )
            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate
                best_debug_item = {
                    'candidate_id': candidate.candidate_id,
                    'depth': candidate.depth,
                    'score': candidate.score,
                    'peduncle_mask': peduncle['mask'],
                    'cluster_mask': cluster['mask'],
                    'skeleton': skeleton,
                    'pick': (u, v),
                }

            candidate_id += 1

        if best_candidate is None:
            response.success = False
            response.message = 'no valid pick pose found'
            self.get_logger().warning('no valid pick pose found')
            return response

        response.best = best_candidate
        response.debug_image = self.bridge.cv2_to_imgmsg(
            draw_pose_debug(color, [best_debug_item] if best_debug_item is not None else []),
            encoding='bgr8',
        )
        response.message = 'ok'
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
