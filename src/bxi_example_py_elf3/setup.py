from setuptools import find_packages, setup
import os
from pathlib import Path
import shutil

try:
    from setuptools._distutils.command.install_data import install_data
except ImportError:
    from distutils.command.install_data import install_data

try:
    from colcon_core.distutils.commands.symlink_data import symlink_data
except ImportError:
    symlink_data = None

package_name = "bxi_example_py_elf3"


class SyncInstalledModsMixin:
    """Remove built-in Mods that no longer exist in the source tree."""

    def run(self):
        self._remove_stale_mods()
        super().run()

    def _remove_stale_mods(self):
        source_root = Path(__file__).resolve().parent / "mods"
        installed_root = (
            Path(self.install_dir)
            / "share"
            / package_name
            / "mods"
        )

        # A symlinked root already follows the source tree and must never be
        # traversed for deletion: doing so could remove source files.
        if installed_root.is_symlink() or not installed_root.is_dir():
            return

        source_mods = {
            child.name
            for child in source_root.iterdir()
            if child.is_dir() and (child / "mod.yaml").is_file()
        } if source_root.is_dir() else set()

        for installed_mod in installed_root.iterdir():
            if installed_mod.name in source_mods:
                continue

            # Only direct children that contain a Mod manifest are eligible.
            # A broken manifest link is expected after deleting a Mod built
            # with --symlink-install.
            manifest = installed_mod / "mod.yaml"
            if installed_mod.is_symlink() and manifest.is_file():
                installed_mod.unlink()
            elif installed_mod.is_dir() and (
                manifest.is_file() or manifest.is_symlink()
            ):
                shutil.rmtree(installed_mod)
            else:
                continue

            self.announce(
                f"removing stale built-in Mod: {installed_mod}",
                level=2,
            )


class SyncModInstallData(SyncInstalledModsMixin, install_data):
    pass


if symlink_data is not None:
    class SyncModSymlinkData(SyncInstalledModsMixin, symlink_data):
        pass


command_classes = {"install_data": SyncModInstallData}
if symlink_data is not None:
    command_classes["symlink_data"] = SyncModSymlinkData


def get_data_files():
    data_files = []
    source_dir = "data"  # 源目录，相对于setup.py的位置
    target_dir = os.path.join("share", package_name, "data")  # 目标目录

    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [directory for directory in dirs if directory != "__pycache__"]
        for file in files:
            if file.endswith((".pyc", ".pyo")):
                continue
            file_path = os.path.join(root, file)
            # 计算相对于源目录的相对路径，以保持子目录结构
            relative_path = os.path.relpath(root, source_dir)
            install_dir = os.path.join(target_dir, relative_path)
            data_files.append((install_dir, [file_path]))

    return data_files


def get_launch_files():
    data_files = []
    source_dir = "launch"  # 源目录，相对于setup.py的位置
    target_dir = os.path.join("share", package_name, "launch")  # 目标目录

    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 计算相对于源目录的相对路径，以保持子目录结构
            relative_path = os.path.relpath(root, source_dir)
            install_dir = os.path.join(target_dir, relative_path)
            data_files.append((install_dir, [file_path]))

    return data_files


def get_config_files():
    data_files = []
    source_dir = "config"
    target_dir = os.path.join("share", package_name, "config")

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, source_dir)
            install_dir = (
                target_dir
                if relative_path == "."
                else os.path.join(target_dir, relative_path)
            )
            data_files.append((install_dir, [file_path]))

    return data_files


def get_mod_files():
    data_files = []
    source_dir = "mods"
    target_dir = os.path.join("share", package_name, "mods")

    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [directory for directory in dirs if directory != "__pycache__"]
        for file in files:
            if file.endswith((".pyc", ".pyo")):
                continue
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, source_dir)
            install_dir = (
                target_dir
                if relative_path == "."
                else os.path.join(target_dir, relative_path)
            )
            data_files.append((install_dir, [file_path]))

    return data_files


def get_schema_files():
    schema_dir = "schema"
    if not os.path.isdir(schema_dir):
        return []
    return [
        (os.path.join("share", package_name, "schema"), [
            os.path.join(schema_dir, file)
        ])
        for file in sorted(os.listdir(schema_dir))
        if file.endswith(".json")
    ]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("bin", ["../../tools/bxi-mod"]),
    ]
    + get_data_files()
    + get_launch_files()
    + get_config_files()
    + get_mod_files()
    + get_schema_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liufq",
    maintainer_email="popsay@163.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    cmdclass=command_classes,
    entry_points={
        "console_scripts": [
            "bxi_example_py_elf3_mjlab = bxi_example_py_elf3.bxi_example_mjlab:main",
            "bxi_example_py_elf3_demo = bxi_example_py_elf3.bxi_example_demo:main",
        ],
    },
)
