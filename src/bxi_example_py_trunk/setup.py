from setuptools import setup, find_packages
import os

package_name = 'bxi_example_py_trunk'

def get_data_files():
    data_files = []
    source_dir = 'policy'  # 源目录，相对于setup.py的位置
    target_dir = os.path.join('share', package_name, 'policy')  # 目标目录

    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 计算相对于源目录的相对路径，以保持子目录结构
            relative_path = os.path.relpath(root, source_dir)
            install_dir = os.path.join(target_dir, relative_path)
            data_files.append((install_dir, [file_path]))
    
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.inference'],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        # ('share/' + package_name + '/policy', ['policy/policy.jit']),
        # ('share/' + package_name + '/policy', ['policy/policy_slope.jit']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/example_launch.py']),
        ('share/' + package_name + '/launch', ['launch/example_launch_hw.py']),
        ('share/' + package_name + '/launch', ['launch/example_launch_xx.py']),
        ('share/' + package_name + '/launch', ['launch/example_launch_xx_terrain.py']),
        # ('share/' + package_name + '/launch', ['launch/example_launch_hw_slope.py']),
    ] + get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liufq',
    maintainer_email='popsay@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bxi_example_py_trunk = bxi_example_py_trunk.bxi_example:main',
            'xuxin_controller = bxi_example_py_trunk.xuxin_controller:main',
            'xuxin_controller_terrain = bxi_example_py_trunk.xuxin_controller_terrain:main',
        ],
    },
)
