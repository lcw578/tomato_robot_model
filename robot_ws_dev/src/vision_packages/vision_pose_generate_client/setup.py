from glob import glob
from setuptools import find_packages, setup

package_name = 'vision_pose_generate_client'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dong',
    maintainer_email='dong@example.com',
    description='Standalone client for the tomato pick pose service.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_pick_poses_client = vision_pose_generate_client.get_pick_poses_client:main',
            'pick_pose_visualizer = vision_pose_generate_client.pick_pose_visualizer:main',
        ],
    },
)
