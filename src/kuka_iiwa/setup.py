import os
from setuptools import find_packages, setup

package_name = 'kuka_iiwa'


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        install_path = os.path.join('share', package_name, path)
        files = [os.path.join(path, f) for f in filenames]
        if files:
            paths.append((install_path, files))
    return paths


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + package_files('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luka',
    maintainer_email='ostojic.luka36@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simulate_node = kuka_iiwa.simulate_basic:main'
        ],
    },
)
