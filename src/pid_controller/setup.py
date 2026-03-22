from setuptools import find_packages, setup

package_name = 'pid_controller'

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
    maintainer='Elaina',
    maintainer_email='1736843619@qq.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'circle_controller = pid_controller.test_sim:main',
            'stop_controller = pid_controller.test_stop:main',
            'pid_controller = pid_controller.pid_controller:main',
            'target_setter = pid_controller.target_setter:main',
        ],
    },
)
