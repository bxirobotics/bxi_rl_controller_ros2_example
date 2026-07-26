import os
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

from bxi_example_py_elf3._runtime import mod_loader
from bxi_example_py_elf3._runtime.mod_nodes import ModNodeManager, ModNodeSpec
from bxi_example_py_elf3._runtime.runtime_requirements import (
    RuntimeRequirements,
    check_runtime_requirements,
    read_runtime_requirements,
)


def _empty_requirements() -> dict[str, list[object]]:
    return {"python": [], "ros": [], "system": []}


class RuntimeRequirementsTest(unittest.TestCase):
    def test_runtime_requirements_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing fields"):
            read_runtime_requirements(
                {"python": [], "ros": []},
                "requirements",
            )

    def test_vendor_dependencies_are_checked_before_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory)
            vendor_python = mod_root / "vendor" / "python"
            vendor_library = mod_root / "vendor" / "lib"
            vendor_python.mkdir(parents=True)
            vendor_library.mkdir(parents=True)
            (vendor_python / "bxi_test_vendor.py").write_text("VALUE = 1\n")
            bundled_library = vendor_library / "libbxi_test_vendor.so"
            bundled_library.write_bytes(b"prebuilt-placeholder")

            requirements = read_runtime_requirements(
                {
                    "python": [{"import": "bxi_test_vendor"}],
                    "ros": [],
                    "system": [{"library": "bxi_test_vendor"}],
                },
                "requirements",
            )
            report = check_runtime_requirements(requirements, mod_root)

            self.assertTrue(report.available)
            self.assertTrue(report.vendor_python)
            self.assertEqual(
                report.vendor_libraries,
                (bundled_library.resolve(),),
            )

    def test_missing_dependencies_have_specific_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            requirements = read_runtime_requirements(
                {
                    "python": [{"import": "bxi_definitely_missing_python"}],
                    "ros": [{"package": "bxi_definitely_missing_ros"}],
                    "system": [{"library": "bxi_definitely_missing_system"}],
                },
                "requirements",
            )
            report = check_runtime_requirements(
                requirements,
                Path(temporary_directory),
            )

            self.assertEqual(
                report.errors,
                (
                    "missing Python module 'bxi_definitely_missing_python'",
                    "missing ROS package 'bxi_definitely_missing_ros'",
                    "missing system library 'bxi_definitely_missing_system'",
                ),
            )

    def test_unavailable_node_is_not_imported_or_started(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory) / "com.example.unavailable_node"
            mod_root.mkdir()
            manifest = {
                "schema": 1,
                "id": "com.example.unavailable_node",
                "name": "Unavailable node fixture",
                "version": "1.0.0",
                "api": 1,
                "enable": True,
                "entrypoint": None,
                "visibility": "protected",
                "requires": [],
                "conflicts": [],
                "python_exports": [],
                "runtime_requirements": _empty_requirements(),
                "nodes": {
                    "camera": {
                        "entrypoint": "camera:create_node",
                        "execution": "in_process",
                        "lifecycle": "mod",
                        "manifest": {"label": "Missing camera"},
                        "runtime_requirements": {
                            "python": [{"import": "bxi_missing_camera_driver"}],
                            "ros": [],
                            "system": [],
                        },
                    }
                },
            }
            (mod_root / "mod.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )
            (mod_root / "camera.py").write_text(
                "raise AssertionError('unavailable node module was imported')\n",
                encoding="utf-8",
            )

            discovered = mod_loader._discover_mods((mod_root,))
            mod = discovered["com.example.unavailable_node"]
            package = mod_loader._create_dynamic_package(mod)
            try:
                specs = mod_loader._load_mod_node_specs(
                    mod,
                    package,
                    vendor_session=mod_loader._VendorSession(),
                )
            finally:
                mod_loader._remove_module_prefixes((package.__name__,))

            self.assertEqual(len(specs), 1)
            self.assertIsNone(specs[0].factory)
            self.assertEqual(
                specs[0].unavailable_error,
                "missing Python module 'bxi_missing_camera_driver'",
            )

            manager = ModNodeManager(specs)
            manager.start()
            try:
                self.assertEqual(manager.snapshot()[0]["status"], "unavailable")
            finally:
                manager.close()

    def test_in_process_vendor_is_loaded_with_explicit_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory) / "com.example.vendor_node"
            vendor_python = mod_root / "vendor" / "python"
            vendor_python.mkdir(parents=True)
            (vendor_python / "bxi_test_inprocess_vendor.py").write_text(
                "VALUE = 7\n",
                encoding="utf-8",
            )
            manifest = self._manifest_header(
                "com.example.vendor_node",
                entrypoint=None,
            )
            manifest["nodes"] = {
                "camera": {
                    "entrypoint": "camera:create_node",
                    "execution": "in_process",
                    "lifecycle": "mod",
                    "manifest": {"label": "Vendor camera"},
                    "runtime_requirements": {
                        "python": [{"import": "bxi_test_inprocess_vendor"}],
                        "ros": [],
                        "system": [],
                    },
                }
            }
            (mod_root / "mod.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )
            (mod_root / "camera.py").write_text(
                "import bxi_test_inprocess_vendor\n"
                "def create_node(context):\n"
                "    return object()\n",
                encoding="utf-8",
            )

            discovered = mod_loader._discover_mods((mod_root,))
            mod = discovered["com.example.vendor_node"]
            package = mod_loader._create_dynamic_package(mod)
            vendor_session = mod_loader._VendorSession()
            try:
                specs = mod_loader._load_mod_node_specs(
                    mod,
                    package,
                    vendor_session=vendor_session,
                )
                self.assertIsNotNone(specs[0].factory)
                self.assertEqual(specs[0].unavailable_error, None)
                self.assertIn(
                    "process-global",
                    specs[0].warnings[0],
                )
            finally:
                mod_loader._remove_module_prefixes((package.__name__,))
                vendor_session.close()
                sys.modules.pop("bxi_test_inprocess_vendor", None)

    def test_process_node_receives_vendor_paths_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory)
            vendor_python = mod_root / "vendor" / "python"
            vendor_library = mod_root / "vendor" / "lib"
            vendor_python.mkdir(parents=True)
            vendor_library.mkdir(parents=True)
            captured: dict[str, object] = {}

            def process_factory(*args, **kwargs):
                captured["args"] = args
                captured["env"] = kwargs["env"]
                return object()

            spec = ModNodeSpec(
                id="com.example.vendor/process",
                mod_id="com.example.vendor",
                local_name="process",
                node_name="com_example_vendor_process",
                mod_root=mod_root,
                manifest_path=mod_root / "mod.yaml",
                entrypoint="node:create_node",
                execution="process",
                lifecycle="mod",
                states=(),
                params={},
                manifest={"label": "Vendor process"},
                restart_max_attempts=0,
                restart_delay=0.0,
                factory=None,
            )
            manager = ModNodeManager(
                [spec],
                process_factory=process_factory,
            )
            manager._spawn_process(spec)

            environment = captured["env"]
            self.assertEqual(
                environment["PYTHONPATH"].split(os.pathsep)[0],
                str(vendor_python),
            )
            self.assertEqual(
                environment["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                str(vendor_library),
            )

    def test_empty_runtime_requirements_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = check_runtime_requirements(
                RuntimeRequirements((), (), ()),
                Path(temporary_directory),
            )
        self.assertTrue(report.available)

    def test_mod_with_missing_runtime_requirement_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_base_mod(root)
            unavailable_root = root / "com.example.optional"
            unavailable_root.mkdir()
            unavailable_manifest = self._manifest_header(
                "com.example.optional",
                entrypoint="missing:create_mod",
            )
            unavailable_manifest["runtime_requirements"] = {
                "python": [{"import": "bxi_missing_optional_runtime"}],
                "ros": [],
                "system": [],
            }
            (unavailable_root / "mod.yaml").write_text(
                yaml.safe_dump(unavailable_manifest, sort_keys=False),
                encoding="utf-8",
            )

            runtime = mod_loader.load_mod_runtime(
                {"initial_state": "com.example.base/idle"},
                built_in_root=root,
            )
            try:
                self.assertEqual(
                    [mod.id for mod in runtime.mods],
                    ["com.example.base"],
                )
                self.assertEqual(len(runtime.unavailable_mods), 1)
                self.assertEqual(
                    runtime.unavailable_mods[0].error,
                    "missing Python module 'bxi_missing_optional_runtime'",
                )
            finally:
                runtime.close()

    def test_required_unavailable_mod_fails_with_dependency_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_base_mod(root)
            unavailable_root = root / "com.example.optional"
            unavailable_root.mkdir()
            unavailable_manifest = self._manifest_header(
                "com.example.optional",
                entrypoint="missing:create_mod",
            )
            unavailable_manifest["runtime_requirements"] = {
                "python": [{"import": "bxi_missing_optional_runtime"}],
                "ros": [],
                "system": [],
            }
            (unavailable_root / "mod.yaml").write_text(
                yaml.safe_dump(unavailable_manifest, sort_keys=False),
                encoding="utf-8",
            )

            dependent_root = root / "com.example.dependent"
            dependent_root.mkdir()
            dependent_manifest = self._manifest_header(
                "com.example.dependent",
                entrypoint="missing:create_mod",
            )
            dependent_manifest["requires"] = ["com.example.optional"]
            (dependent_root / "mod.yaml").write_text(
                yaml.safe_dump(dependent_manifest, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "com.example.dependent.*requires unavailable Mod.*"
                "com.example.optional",
            ):
                mod_loader.load_mod_runtime(
                    {"initial_state": "com.example.base/idle"},
                    built_in_root=root,
                )

    @staticmethod
    def _manifest_header(
        mod_id: str,
        *,
        entrypoint: str | None,
    ) -> dict[str, object]:
        return {
            "schema": 1,
            "id": mod_id,
            "name": mod_id,
            "version": "1.0.0",
            "api": 1,
            "enable": True,
            "entrypoint": entrypoint,
            "visibility": "protected",
            "requires": [],
            "conflicts": [],
            "python_exports": [],
            "runtime_requirements": _empty_requirements(),
        }

    def _write_base_mod(self, root: Path) -> None:
        mod_root = root / "com.example.base"
        mod_root.mkdir()
        manifest = self._manifest_header(
            "com.example.base",
            entrypoint="plugin:create_mod",
        )
        manifest["states"] = {
            "idle": {
                "manifest": {
                    "label": "Idle",
                    "priority": 1,
                    "group": "Test",
                    "icon": "test",
                }
            }
        }
        (mod_root / "mod.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        (mod_root / "plugin.py").write_text(
            "from bxi_example_py_elf3.mod_api import ModDefinition\n"
            "from bxi_example_py_elf3.mod_api.state import RobotControlState\n"
            "class Idle(RobotControlState):\n"
            "    def on_update(self, ctx, dt):\n"
            "        pass\n"
            "def build(context):\n"
            "    return Idle(context.name, context.state_id)\n"
            "def create_mod(context):\n"
            "    return ModDefinition(state_factories={'idle': build})\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
