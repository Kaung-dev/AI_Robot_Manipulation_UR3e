from setuptools import find_packages
from setuptools import setup

setup(
    name='ur_onrobot_moveit_config',
    version='0.0.1',
    packages=find_packages(
        include=('ur_onrobot_moveit_config', 'ur_onrobot_moveit_config.*')),
)
