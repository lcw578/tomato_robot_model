import cv2
import importlib
import numpy as np
import os
import rclpy
import sys
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node

from vision_tomato_interfaces.msg import InstanceMask
from vision_tomato_interfaces.srv import SegmentTomato

from .debug_viz import draw_instance_debug, draw_overlap_debug
from .geometry import bbox_from_mask, bool_mask_to_msg, touches_border


CLASS_NAMES = {
    0: 'peduncle',
    1: 'tomato_cluster',
    2: 'tomato_fruit',
}


class DetectionService(Node):
    def __init__(self):
        super().__init__('detection_service')
        self.declare_parameter('model_path', '')
        self.declare_parameter('ultralytics_source_dir', '/home/dong/codes_vision/ultralytics')
        self.declare_parameter('image_size', 640)
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('border_margin_px', 5)
        self.declare_parameter('overlap_iou_warn', 0.20)
        self.declare_parameter('same_class_duplicate_iou', 0.70)
        self.declare_parameter('filter_same_class_duplicates', True)

        self.bridge = CvBridge()
        self.model = None
        self.load_error = ''
        self.log_runtime_environment()
        self.load_model()
        self.srv = self.create_service(SegmentTomato, '/vision/tomato/segment', self.handle_segment)
        self.get_logger().info('detection_service started at /vision/tomato/segment')

    def log_runtime_environment(self):
        module_names = ['numpy', 'cv2', 'ultralytics', 'cv_bridge', 'rclpy']
        self.get_logger().info(f'python executable: {sys.executable}')
        self.get_logger().info(f'PYTHONNOUSERSITE={os.environ.get("PYTHONNOUSERSITE", "")}')
        for name in module_names:
            try:
                module = importlib.import_module(name)
                version = getattr(module, '__version__', 'n/a')
                path = getattr(module, '__file__', 'built-in')
                self.get_logger().info(f'module {name}: version={version} path={path}')
            except Exception as exc:
                self.get_logger().warning(f'module {name}: import failed: {exc}')
        for idx, path in enumerate(sys.path[:12]):
            self.get_logger().info(f'sys.path[{idx}]={path}')

    def analyze_overlaps(self, instances):
        overlaps = []
        duplicate_ids = set()
        warn_iou = float(self.get_parameter('overlap_iou_warn').value)
        duplicate_iou = float(self.get_parameter('same_class_duplicate_iou').value)

        for i in range(len(instances)):
            a = instances[i]
            mask_a = a['mask']
            area_a = int(mask_a.sum())
            if area_a == 0:
                continue
            for j in range(i + 1, len(instances)):
                b = instances[j]
                mask_b = b['mask']
                area_b = int(mask_b.sum())
                if area_b == 0:
                    continue
                inter_mask = np.logical_and(mask_a, mask_b)
                inter = int(inter_mask.sum())
                if inter == 0:
                    continue
                union = int(np.logical_or(mask_a, mask_b).sum())
                iou = float(inter / union) if union else 0.0
                contain_a = float(inter / area_a)
                contain_b = float(inter / area_b)
                same_class = a['class_name'] == b['class_name']
                if not same_class:
                    continue

                if iou >= duplicate_iou:
                    lower_score = a if a['confidence'] < b['confidence'] else b
                    duplicate_ids.add(lower_score['instance_id'])

                if iou >= warn_iou or contain_a >= 0.50 or contain_b >= 0.50:
                    overlaps.append({
                        'a_id': a['instance_id'],
                        'b_id': b['instance_id'],
                        'a_class': a['class_name'],
                        'b_class': b['class_name'],
                        'iou': iou,
                        'inter': inter,
                        'contain_a': contain_a,
                        'contain_b': contain_b,
                        'same_class': same_class,
                        'mask': inter_mask,
                    })
        return overlaps, duplicate_ids

    def load_model(self):
        model_path = self.get_parameter('model_path').value
        if not model_path:
            self.load_error = 'model_path is empty'
            self.get_logger().error(self.load_error)
            return
        model_path = self.resolve_model_path(model_path)
        try:
            from ultralytics import YOLO
        except Exception:
            source_dir = self.get_parameter('ultralytics_source_dir').value
            if source_dir and source_dir not in sys.path:
                sys.path.insert(0, source_dir)
            try:
                from ultralytics import YOLO
            except Exception as exc:
                self.load_error = f'failed to import ultralytics: {exc}'
                self.get_logger().error(self.load_error)
                return

        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f'loaded YOLO model: {model_path}')
        except Exception as exc:
            self.load_error = f'failed to load YOLO model: {exc}'
            self.get_logger().error(self.load_error)

    def resolve_model_path(self, model_path):
        path = str(model_path).strip()
        if path.startswith('package://'):
            rest = path[len('package://'):]
            package_name, _, relative_path = rest.partition('/')
            if package_name and relative_path:
                return os.path.join(get_package_share_directory(package_name), relative_path)
        if path.startswith('pkg://'):
            rest = path[len('pkg://'):]
            package_name, _, relative_path = rest.partition('/')
            if package_name and relative_path:
                return os.path.join(get_package_share_directory(package_name), relative_path)
        find_prefix = '$(find-pkg-share '
        if path.startswith(find_prefix):
            end = path.find(')')
            if end > len(find_prefix):
                package_name = path[len(find_prefix):end].strip()
                relative_path = path[end + 1:].lstrip('/')
                return os.path.join(get_package_share_directory(package_name), relative_path)
        return path

    def handle_segment(self, request, response):
        if self.model is None:
            response.success = False
            response.message = self.load_error or 'model is not loaded'
            return response

        try:
            color = self.bridge.imgmsg_to_cv2(request.color_image, desired_encoding='bgr8')
        except Exception as exc:
            response.success = False
            response.message = f'failed to convert color image: {exc}'
            return response

        try:
            # Force a real ndarray copy to avoid cv_bridge / OpenCV proxy objects
            # crossing Python environments and breaking downstream cv2 usage.
            color = np.array(color, copy=True)
            if color.ndim == 2:
                color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
            elif color.ndim == 3 and color.shape[2] == 4:
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            if color.dtype != np.uint8:
                color = color.astype(np.uint8)
            color = np.ascontiguousarray(color)
            # Recreate the image through the active cv2 binding. This avoids
            # ndarray / OpenCV ABI mismatches across mixed ROS/conda setups.
            ok, encoded = cv2.imencode('.png', color)
            if not ok:
                raise RuntimeError('cv2.imencode failed')
            color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if color is None:
                raise RuntimeError('cv2.imdecode returned None')
            _ = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)
            self.get_logger().info(
                f'segment input image type={type(color).__name__} shape={color.shape} dtype={color.dtype}'
            )
        except Exception as exc:
            response.success = False
            response.message = f'failed to normalize color image: {exc}'
            return response

        image_size = int(self.get_parameter('image_size').value)
        conf = float(self.get_parameter('conf_threshold').value)
        iou = float(self.get_parameter('iou_threshold').value)
        device = self.get_parameter('device').value
        border_margin = int(self.get_parameter('border_margin_px').value)

        try:
            results = self.model.predict(
                source=color,
                imgsz=image_size,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            response.success = False
            response.message = f'YOLO prediction failed: {exc}'
            return response

        instances_for_debug = []
        instance_id = 0
        result = results[0] if results else None
        if result is not None and result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
            scores = result.boxes.conf.detach().cpu().numpy()
            h, w = color.shape[:2]

            for idx, raw_mask in enumerate(masks):
                mask = raw_mask > 0.5
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                bbox = bbox_from_mask(mask)
                if bbox is None:
                    continue

                class_id = int(classes[idx])
                class_name = CLASS_NAMES.get(class_id, str(class_id))
                inst = InstanceMask()
                inst.header = request.color_image.header
                inst.instance_id = instance_id
                inst.class_id = class_id
                inst.class_name = class_name
                inst.confidence = float(scores[idx])
                inst.bbox_xyxy = [float(v) for v in boxes[idx]]
                inst.mask = bool_mask_to_msg(mask, self.bridge, request.color_image.header)
                inst.touches_image_border = touches_border(mask, border_margin)
                response.instances.append(inst)

                instances_for_debug.append({
                    'instance_id': instance_id,
                    'class_name': class_name,
                    'confidence': float(scores[idx]),
                    'bbox_xyxy': inst.bbox_xyxy,
                    'mask': mask,
                })
                instance_id += 1

        overlaps, duplicate_ids = self.analyze_overlaps(instances_for_debug)
        filter_duplicates = bool(self.get_parameter('filter_same_class_duplicates').value)
        filtered_duplicate_count = len(duplicate_ids) if filter_duplicates else 0
        if filter_duplicates and duplicate_ids:
            response.instances = [
                inst for inst in response.instances
                if int(inst.instance_id) not in duplicate_ids
            ]
            instances_for_debug = [
                inst for inst in instances_for_debug
                if int(inst['instance_id']) not in duplicate_ids
            ]
            overlaps, duplicate_ids_after_filter = self.analyze_overlaps(instances_for_debug)
        else:
            duplicate_ids_after_filter = duplicate_ids

        if overlaps:
            summary_parts = []
            for item in overlaps[:8]:
                summary_parts.append(
                    f"{item['a_id']}:{item['a_class']}<->{item['b_id']}:{item['b_class']} "
                    f"iou={item['iou']:.2f} contain=({item['contain_a']:.2f},{item['contain_b']:.2f})"
                )
            self.get_logger().warning('same-class mask overlaps: ' + ' | '.join(summary_parts))
        if duplicate_ids:
            action = 'filtered' if filter_duplicates else 'detected'
            self.get_logger().warning(f'same-class duplicate mask candidates {action}: {sorted(duplicate_ids)}')

        debug = draw_instance_debug(color, instances_for_debug)
        debug = draw_overlap_debug(debug, overlaps)
        response.debug_image = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        response.debug_image.header = request.color_image.header
        response.success = True
        response.message = (
            f'detected {len(response.instances)} instances, '
            f'same_class_overlaps={len(overlaps)}, '
            f'filtered_duplicates={filtered_duplicate_count}, '
            f'remaining_duplicate_candidates={len(duplicate_ids_after_filter)}'
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DetectionService()
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
