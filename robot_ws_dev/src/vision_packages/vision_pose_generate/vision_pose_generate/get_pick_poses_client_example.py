import argparse

import rclpy
from rclpy.node import Node

from vision_tomato_interfaces.srv import GetPickPoses


class GetPickPosesClientExample(Node):
    def __init__(self, service_name, timeout_sec):
        super().__init__('get_pick_poses_client_example')
        self.service_name = service_name
        self.timeout_sec = timeout_sec
        self.client = self.create_client(GetPickPoses, service_name)

    def call(self):
        self.get_logger().info(f'waiting for {self.service_name} ...')
        if not self.client.wait_for_service(timeout_sec=self.timeout_sec):
            self.get_logger().error(f'service not available: {self.service_name}')
            return 1

        request = GetPickPoses.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        if not future.done():
            self.get_logger().error('service call timeout')
            return 2

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'service call failed: {exc}')
            return 3

        self.get_logger().info(f'success={response.success}')
        if not response.success:
            self.get_logger().info(f'message={response.message}')
            return 4

        best_pose = response.best
        xyzrpy_camera = list(best_pose.pose_xyzrpy_camera)
        xyzrpy_base = list(best_pose.pose_xyzrpy_base)
        self.get_logger().info(
            'best camera xyzrpy[m,rad]='
            f'[{xyzrpy_camera[0]:.4f}, {xyzrpy_camera[1]:.4f}, {xyzrpy_camera[2]:.4f}, '
            f'{xyzrpy_camera[3]:.4f}, {xyzrpy_camera[4]:.4f}, {xyzrpy_camera[5]:.4f}]'
        )
        self.get_logger().info(
            'best base xyzrpy[m,rad]='
            f'[{xyzrpy_base[0]:.4f}, {xyzrpy_base[1]:.4f}, {xyzrpy_base[2]:.4f}, '
            f'{xyzrpy_base[3]:.4f}, {xyzrpy_base[4]:.4f}, {xyzrpy_base[5]:.4f}], '
            f'has_pose_base={best_pose.has_pose_base}'
        )
        self.get_logger().info(
            f'best ids: candidate={best_pose.candidate_id}, peduncle={best_pose.peduncle_id}, '
            f'cluster={best_pose.cluster_id}, fruits={list(best_pose.fruit_ids)}'
        )
        self.get_logger().info(
            f'best score={best_pose.score:.3f}, maturity={best_pose.maturity}, '
            f'ripe_ratio={best_pose.ripe_ratio:.3f}, reason={best_pose.reason}'
        )
        self.get_logger().info(f'message={response.message}')
        return 0


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Example client for /vision/tomato/get_pick_poses.')
    parser.add_argument(
        '--service',
        default='/vision/tomato/get_pick_poses',
        help='GetPickPoses service name.',
    )
    parser.add_argument(
        '--timeout-sec',
        type=float,
        default=15.0,
        help='Service wait and call timeout in seconds.',
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    rclpy.init()
    node = GetPickPosesClientExample(
        parsed.service,
        parsed.timeout_sec,
    )
    try:
        exit_code = node.call()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
