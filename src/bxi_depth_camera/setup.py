from __future__ import annotations

import os

from setuptools import find_packages, setup

try:
    from setuptools._distutils.command.install_data import install_data
except ImportError:
    from distutils.command.install_data import install_data


PACKAGE_NAME = "bxi_depth_camera"


class PreserveSymlinkInstallData(install_data):
    def copy_file(self, src, dst, *args, **kwargs):
        if not os.path.islink(src):
            return super().copy_file(src, dst, *args, **kwargs)
        target = os.path.join(dst, os.path.basename(src)) if os.path.isdir(dst) else dst
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target):
            os.remove(target)
        os.symlink(os.readlink(src), target)
        return target, True


def tree_data(source: str, destination: str):
    result = []
    for root, directories, files in os.walk(source):
        directories[:] = [name for name in directories if name != "__pycache__"]
        selected = [
            os.path.join(root, name)
            for name in files
            if not name.endswith((".pyc", ".pyo"))
        ]
        if selected:
            relative = os.path.relpath(root, source)
            target = (
                destination if relative == "." else os.path.join(destination, relative)
            )
            result.append((target, selected))
    return result


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        ("share/" + PACKAGE_NAME, ["package.xml", "README.md"]),
    ]
    + tree_data("config", "share/" + PACKAGE_NAME + "/config")
    + tree_data("launch", "share/" + PACKAGE_NAME + "/launch")
    + tree_data("vendor", "share/" + PACKAGE_NAME + "/vendor"),
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="zwj",
    maintainer_email="popsay@163.com",
    description="Hot-pluggable multi-vendor ROS 2 depth camera publisher",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cameras = bxi_depth_camera.bootstrap:main",
            "cameras-inspect = bxi_depth_camera.inspect:main",
        ]
    },
    cmdclass={"install_data": PreserveSymlinkInstallData},
)
