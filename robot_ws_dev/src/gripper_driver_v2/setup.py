from setuptools import find_packages, setup

package_name = 'gripper_driver_v2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'gripper_control_node = gripper_driver_v2.gripper_control_node:main', 
            'gripper_fake_node = gripper_driver_v2.gripper_fake:main',
        ],
    },
)
