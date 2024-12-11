from setuptools import setup


setup(
    name='xarm6world',
    author="Wenbo Li",
    author_email="1962672280@qq.com",
    description="xArm6 simulation environment for RL with a gym-style API.",
    url="https://github.com/OwenCaleb/xArm6-World.git",
    version='0.0.1',
    packages=['xarm6world'],
    include_package_data=True,
    install_requires=[
        'gym==0.21.0',
        'mujoco-py',
        'numpy>=1.18',
    ],
    license=open('LICENSE').read(),
    zip_safe=False,
)
