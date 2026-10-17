from glob import glob
from setuptools import find_packages, setup

package_name = 'vision_pose_generate'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/weights', glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dong',
    maintainer_email='dong@example.com',
    description='Tomato picking pose generation from RGB-D segmentation.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_preprocess_node = vision_pose_generate.image_preprocess_node:main',
            'detection_service = vision_pose_generate.detection_service:main',
            'pose_estimator_service = vision_pose_generate.pose_estimator_service:main',
            'pick_pose_service_node = vision_pose_generate.pick_pose_service_node:main',
            'get_pick_poses_client_example = vision_pose_generate.get_pick_poses_client_example:main',
        ],
    },
)
