from setuptools import setup

package_name = 'bxi_example_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name + '/policy', ['policy/policy.jit']),
        # ('share/' + package_name + '/policy', ['policy/policy_slope.jit']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/example_launch.py']),
        ('share/' + package_name + '/launch', ['launch/example_launch_hw.py']),
        # ('share/' + package_name + '/launch', ['launch/example_launch_hw_slope.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liufq',
    maintainer_email='popsay@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bxi_example_py = bxi_example_py.bxi_example:main',
        ],
    },
)
