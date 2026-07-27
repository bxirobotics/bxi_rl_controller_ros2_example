from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from bxi_example_py_elf3._runtime import mod_loader
from bxi_example_py_elf3._runtime.mod_nodes import ModNodeManager, ModNodeSpec
from bxi_example_py_elf3._runtime.runtime_requirements import runtime_platform_tag


def _empty_requirements() -> dict[str, list[object]]:
    return {"python": [], "ros": [], "system": []}


class NativeModNodesTest(unittest.TestCase):
    def test_mod_executable_selects_current_platform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory) / "com.example.native"
            executable = mod_root / "bin" / runtime_platform_tag() / "detector_node"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            manifest = self._manifest(
                "com.example.native",
                runtime="executable",
                entrypoint="detector_node",
            )
            manifest["nodes"]["detector"].update(  # type: ignore[index,union-attr]
                {
                    "arguments": ["--device", "0"],
                    "namespace": "/vision",
                    "remappings": {"image": "/camera/image"},
                    "params": {"threshold": 0.5},
                }
            )
            self._write_manifest(mod_root, manifest)

            spec = self._load_only_spec(mod_root)

            self.assertEqual(spec.runtime, "executable")
            self.assertEqual(spec.execution, "process")
            self.assertEqual(spec.executable_path, executable.resolve())
            self.assertEqual(spec.arguments, ("--device", "0"))
            self.assertEqual(spec.namespace, "/vision")
            self.assertEqual(spec.remappings, {"image": "/camera/image"})
            self.assertIsNone(spec.factory)
            self.assertIsNone(spec.unavailable_error)

    def test_missing_platform_executable_marks_node_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory) / "com.example.missing"
            mod_root.mkdir()
            self._write_manifest(
                mod_root,
                self._manifest(
                    "com.example.missing",
                    runtime="executable",
                    entrypoint="detector_node",
                ),
            )

            spec = self._load_only_spec(mod_root)

            self.assertIsNone(spec.executable_path)
            self.assertEqual(
                spec.unavailable_error,
                "missing executable " f"'bin/{runtime_platform_tag()}/detector_node'",
            )

    def test_mod_executable_cannot_escape_platform_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory) / "com.example.unsafe"
            mod_root.mkdir()
            self._write_manifest(
                mod_root,
                self._manifest(
                    "com.example.unsafe",
                    runtime="executable",
                    entrypoint="../detector_node",
                ),
            )

            with self.assertRaisesRegex(ValueError, "safe relative executable"):
                self._load_only_spec(mod_root)

    def test_ros_runtime_resolves_ament_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mod_root = root / "com.example.ros"
            prefix = root / "install" / "detector_package"
            executable = prefix / "lib" / "detector_package" / "detector_node"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            mod_root.mkdir()
            self._write_manifest(
                mod_root,
                self._manifest(
                    "com.example.ros",
                    runtime="ros",
                    entrypoint="detector_package:detector_node",
                ),
            )

            with mock.patch.object(
                mod_loader,
                "get_package_prefix",
                return_value=str(prefix),
            ):
                spec = self._load_only_spec(mod_root)

            self.assertEqual(spec.runtime, "ros")
            self.assertEqual(spec.executable_path, executable.resolve())
            self.assertIsNone(spec.unavailable_error)

    def test_native_command_uses_ros_arguments_and_parameter_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "detector_node"
            executable.write_text("", encoding="utf-8")
            executable.chmod(0o755)
            captured: dict[str, object] = {}

            def process_factory(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return object()

            spec = ModNodeSpec(
                id="com.example.native/detector",
                mod_id="com.example.native",
                local_name="detector",
                node_name="com_example_native_detector",
                mod_root=root,
                manifest_path=root / "mod.yaml",
                entrypoint="detector_node",
                execution="process",
                lifecycle="mod",
                states=(),
                params={"threshold": 0.5, "enabled": True},
                manifest={"label": "Detector"},
                restart_max_attempts=3,
                restart_delay=1.0,
                factory=None,
                runtime="executable",
                arguments=("--device", "0"),
                remappings={"image": "/camera/image"},
                namespace="/vision",
                executable_path=executable,
            )
            manager = ModNodeManager([spec], process_factory=process_factory)
            try:
                manager._spawn_process(spec)
                command = captured["args"][0]  # type: ignore[index]
                parameter_path = Path(command[-1])
                parameters = yaml.safe_load(parameter_path.read_text("utf-8"))

                self.assertEqual(
                    command[:-2],
                    [
                        str(executable),
                        "--device",
                        "0",
                        "--ros-args",
                        "-r",
                        "__node:=com_example_native_detector",
                        "-r",
                        "__ns:=/vision",
                        "-r",
                        "image:=/camera/image",
                    ],
                )
                self.assertEqual(command[-2], "--params-file")
                self.assertEqual(parameter_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(
                    parameters,
                    {
                        "/vision/com_example_native_detector": {
                            "ros__parameters": {
                                "threshold": 0.5,
                                "enabled": True,
                            }
                        }
                    },
                )
                kwargs = captured["kwargs"]
                self.assertEqual(kwargs["cwd"], str(root))  # type: ignore[index]
                self.assertTrue(kwargs["start_new_session"])  # type: ignore[index]
                self.assertEqual(
                    kwargs["env"]["PYTHONDONTWRITEBYTECODE"],  # type: ignore[index]
                    "1",
                )
            finally:
                manager.close()
            self.assertFalse(parameter_path.exists())

    def test_native_process_is_started_and_stopped_by_manager(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "long_running_node"
            executable.write_text(
                "#!/usr/bin/python3\nimport time\ntime.sleep(30)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            spec = ModNodeSpec(
                id="com.example.native/worker",
                mod_id="com.example.native",
                local_name="worker",
                node_name="com_example_native_worker",
                mod_root=root,
                manifest_path=root / "mod.yaml",
                entrypoint="long_running_node",
                execution="process",
                lifecycle="mod",
                states=(),
                params={},
                manifest={"label": "Worker"},
                restart_max_attempts=0,
                restart_delay=0.0,
                factory=None,
                runtime="executable",
                executable_path=executable,
            )
            manager = ModNodeManager([spec])
            process = None
            try:
                manager.start()
                self.assertEqual(manager.snapshot()[0]["status"], "running")
                process = manager._running[spec.id].process
                self.assertIsNotNone(process)
                self.assertIsNone(process.poll())
            finally:
                manager.close()
            self.assertIsNotNone(process)
            self.assertIsNotNone(process.poll())

    @staticmethod
    def _manifest(
        mod_id: str,
        *,
        runtime: str,
        entrypoint: str,
    ) -> dict[str, object]:
        return {
            "schema": 1,
            "id": mod_id,
            "name": mod_id,
            "version": "1.0.0",
            "api": ">=1.2,<2",
            "enable": True,
            "entrypoint": None,
            "visibility": "protected",
            "requires": [],
            "conflicts": [],
            "python_exports": [],
            "runtime_requirements": _empty_requirements(),
            "nodes": {
                "detector": {
                    "runtime": runtime,
                    "entrypoint": entrypoint,
                    "execution": "process",
                    "lifecycle": "mod",
                    "params": {},
                    "arguments": [],
                    "remappings": {},
                    "namespace": "",
                    "manifest": {"label": "Detector"},
                    "runtime_requirements": _empty_requirements(),
                    "restart": {"max_attempts": 3, "delay": 1.0},
                }
            },
        }

    @staticmethod
    def _write_manifest(mod_root: Path, manifest: dict[str, object]) -> None:
        (mod_root / "mod.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _load_only_spec(mod_root: Path) -> ModNodeSpec:
        discovered = mod_loader._discover_mods((mod_root,))
        mod = next(iter(discovered.values()))
        package = mod_loader._create_dynamic_package(mod)
        try:
            specs = mod_loader._load_mod_node_specs(mod, package)
            return specs[0]
        finally:
            mod_loader._remove_module_prefixes((package.__name__,))


if __name__ == "__main__":
    unittest.main()
