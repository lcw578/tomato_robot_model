from setuptools import find_packages, setup

package_name = 'xc_car_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/navigation_file', ['navigation_file/navigation_file.csv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xc',
    maintainer_email='1312629819@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agv_control_node = xc_car_control.agv_control_node:main',
            'agv_navigation_node = xc_car_control.agv_navigation_node:main',
            'param_setter_node = xc_car_control.param_setter_node:main',
            # 'xc_robot_main_control_node = xc_robot_main_control.xc_robot_main_control_node:main'
            'agv_always_go_straght_node = xc_car_control.agv_always_go_straght:main',
            'agv_always_go_straght_node_only = xc_car_control.agv_always_go_straght_only:main',
            'vis_car_control_node = xc_car_control.vis_car_control:main'
        ],
    },
)
