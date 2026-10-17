from setuptools import find_packages, setup

package_name = 'hand_eye_calibration_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name + '/config', ['config/left/']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xc_home_wsl',
    maintainer_email='home_wsl@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_eye_calibration_node_python = hand_eye_calibration_python.hand_eye_calibration_node_python:main',
            'go_aruco_center = hand_eye_calibration_python.go_aruco_center:main',
            'click_to_move = hand_eye_calibration_python.click_to_move:main',
        ],
    },
)
