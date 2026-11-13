from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robotic_tracking_system'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'meshes', 'panda', 'visual'), glob('meshes/panda/visual/*')),
        (os.path.join('share', package_name, 'meshes', 'panda', 'collision'), glob('meshes/panda/collision/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mahmoud Tafran',
    maintainer_email='mahmoudtafran40@gmail.com',
    description='Vision-guided perimeter tracking of parts moving on a conveyor.',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'home_arm_node = robotic_tracking_system.home_arm_node:main',
            'moving_shape_perimeter_tracker = robotic_tracking_system.moving_shape_perimeter_tracker:main',
            'gazebo_shape_stream_node = robotic_tracking_system.gazebo_shape_stream_node:main',
            'experiment_recorder = robotic_tracking_system.experiment_recorder:main',
            'tracking_supervisor = robotic_tracking_system.tracking_supervisor:main',
            'part_perception_node = robotic_tracking_system.part_perception_node:main',
            'perception_target_estimator = robotic_tracking_system.perception_target_estimator:main',
            'operator_dashboard = robotic_tracking_system.operator_dashboard:main',
        ],
    },
)
