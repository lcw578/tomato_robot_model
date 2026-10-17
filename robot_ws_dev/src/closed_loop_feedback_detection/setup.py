from setuptools import find_packages, setup

package_name = 'closed_loop_feedback_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name + '/weights', ['weights/best2.pt']),
        ('share/' + package_name + '/weights', ['weights/gripper_stem_xiongAn_20260726.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='onger777',
    maintainer_email='hhp-pc@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'image_captured_detection = closed_loop_feedback_detection.image_captured_detection:main',
        'image_captured_detection_sub = closed_loop_feedback_detection.image_captured_detection_sub:main',
        'fake_gripper_feedback = closed_loop_feedback_detection.fake_gripper_feedback:main',
        ],
    },
)
