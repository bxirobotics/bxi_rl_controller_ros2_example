import inspect
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
    runtime_platform_tag,
    runtime_python_tag,
)
from bxi_example_py_elf3.mod_api import MOD_API_VERSION, ResourceKey


def _empty_requirements() -> dict[str, list[object]]:
    return {"python": [], "ros": [], "system": []}


class _SeverityStickyLogger:
    """Model Humble's fixed-severity-per-Python-callsite behavior."""

    def __init__(self) -> None:
        self.callsite_levels: dict[tuple[str, int], str] = {}
        self.messages: list[tuple[str, str]] = []

    def _record(self, level: str, message: str) -> None:
        frame = inspect.currentframe()
        assert frame is not None
        method_frame = frame.f_back
        assert method_frame is not None
        caller_frame = method_frame.f_back
        assert caller_frame is not None
        callsite = (caller_frame.f_code.co_filename, caller_frame.f_lineno)
        previous = self.callsite_levels.setdefault(callsite, level)
        if previous != level:
            raise ValueError("Logger severity cannot be changed between calls.")
        self.messages.append((level, message))

    def info(self, message: str) -> None:
        self._record("info", message)

    def warning(self, message: str) -> None:
        self._record("warning", message)

    def error(self, message: str) -> None:
        self._record("error", message)


class RuntimeRequirementsTest(unittest.TestCase):
    def test_registration_controls_eager_and_lazy_resource_loading(self) -> None:
        for loading in ("eager", "lazy"):
            with self.subTest(loading=loading), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self._write_base_mod(root)
                mod_root = root / "com.example.resources"
                mod_root.mkdir()
                manifest = self._manifest_header(
                    "com.example.resources",
                    entrypoint="plugin:create_mod",
                )
                (mod_root / "mod.yaml").write_text(
                    yaml.safe_dump(manifest, sort_keys=False),
                    encoding="utf-8",
                )
                (mod_root / "plugin.py").write_text(
                    "from bxi_example_py_elf3.mod_api import (\n"
                    "    ModDefinition, ResourceKey\n"
                    ")\n"
                    "MODEL = ResourceKey('com.example.resources/model')\n"
                    "def load_model(context):\n"
                    "    (context.mod_root / 'loaded').write_text('yes')\n"
                    "    return object()\n"
                    "def create_mod(context):\n"
                    f"    context.register_resource(MODEL, load_model, "
                    f"loading={loading!r})\n"
                    "    return ModDefinition()\n",
                    encoding="utf-8",
                )

                runtime = mod_loader.load_mod_runtime(
                    {"initial_state": "com.example.base/idle"},
                    built_in_root=root,
                )
                try:
                    marker = mod_root / "loaded"
                    self.assertEqual(marker.exists(), loading == "eager")
                    if loading == "lazy":
                        runtime.resources.get(
                            ResourceKey[object]("com.example.resources/model")
                        )
                        self.assertTrue(marker.is_file())
                finally:
                    runtime.close()

    def test_mod_node_log_levels_use_distinct_rclpy_callsites(self) -> None:
        logger = _SeverityStickyLogger()
        manager = ModNodeManager([], logger=logger)

        manager._log("info", "started")
        manager._log("warning", "restart scheduled")
        manager._log("error", "restart limit reached")
        manager._log("warning", "another warning")

        self.assertEqual(
            logger.messages,
            [
                ("info", "started"),
                ("warning", "restart scheduled"),
                ("error", "restart limit reached"),
                ("warning", "another warning"),
            ],
        )

    def test_mod_api_version_is_public_and_manifest_range_is_checked(self) -> None:
        self.assertEqual(MOD_API_VERSION, "1.1.0")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mod_root = root / "com.example.incompatible"
            mod_root.mkdir()
            manifest = self._manifest_header(
                "com.example.incompatible",
                entrypoint=None,
            )
            manifest["api"] = ">=2,<3"
            (mod_root / "mod.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"requires Mod API.*>=2,<3.*framework provides.*1\.1\.0",
            ):
                mod_loader._discover_mods((root,))

    def test_requirement_version_constraint_is_validated_during_discovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mod_root = root / "com.example.invalid_requirement"
            mod_root.mkdir()
            manifest = self._manifest_header(
                "com.example.invalid_requirement",
                entrypoint=None,
            )
            manifest["requires"] = [
                {"id": "com.example.dependency", "version": "1.0.0"}
            ]
            (mod_root / "mod.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"requires\[0\]\.version is invalid",
            ):
                mod_loader._discover_mods((root,))

    def test_mod_requirement_version_range_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dependency_root = root / "com.example.dependency"
            dependency_root.mkdir()
            dependency = self._manifest_header(
                "com.example.dependency",
                entrypoint=None,
            )
            dependency["version"] = "1.5.0"
            (dependency_root / "mod.yaml").write_text(
                yaml.safe_dump(dependency, sort_keys=False),
                encoding="utf-8",
            )

            consumer_root = root / "com.example.consumer"
            consumer_root.mkdir()
            consumer = self._manifest_header(
                "com.example.consumer",
                entrypoint=None,
            )
            consumer["requires"] = [
                {"id": "com.example.dependency", "version": ">=2,<3"}
            ]
            (consumer_root / "mod.yaml").write_text(
                yaml.safe_dump(consumer, sort_keys=False),
                encoding="utf-8",
            )

            discovered = mod_loader._discover_mods((root,))
            with self.assertRaisesRegex(
                ValueError,
                r"requires 'com\.example\.dependency'.*>=2,<3.*found '1\.5\.0'",
            ):
                mod_loader._dependency_order(discovered)

    def test_runtime_requirements_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing fields"):
            read_runtime_requirements(
                {"python": [], "ros": []},
                "requirements",
            )

    def test_vendor_dependencies_are_checked_before_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory)
            vendor_python = mod_root / "vendor" / "python" / runtime_python_tag()
            vendor_library = mod_root / "vendor" / "lib" / runtime_platform_tag()
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
            self.assertEqual(report.vendor_python_paths, (vendor_python.resolve(),))
            self.assertEqual(
                report.vendor_libraries,
                (bundled_library.resolve(),),
            )

    def test_common_vendor_can_use_platform_python_and_library_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory)
            platform_python = mod_root / "vendor" / "python" / runtime_python_tag()
            common_python = mod_root / "vendor" / "python" / "common"
            platform_library = mod_root / "vendor" / "lib" / runtime_platform_tag()
            platform_python.mkdir(parents=True)
            common_python.mkdir(parents=True)
            platform_library.mkdir(parents=True)
            (platform_python / "bxi_platform_helper.py").write_text(
                "VALUE = 42\n",
                encoding="utf-8",
            )
            (common_python / "bxi_common_vendor.py").write_text(
                "import os\n"
                "from bxi_platform_helper import VALUE\n"
                f"assert os.environ['LD_LIBRARY_PATH'].split(os.pathsep)[0] "
                f"== {str(platform_library.resolve())!r}\n"
                "assert VALUE == 42\n",
                encoding="utf-8",
            )
            requirements = read_runtime_requirements(
                {
                    "python": [{"import": "bxi_common_vendor"}],
                    "ros": [],
                    "system": [],
                },
                "requirements",
            )

            report = check_runtime_requirements(requirements, mod_root)

            self.assertTrue(report.available)
            self.assertEqual(
                report.vendor_python_paths,
                (platform_python.resolve(), common_python.resolve()),
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

    def test_incompatible_vendor_python_falls_back_to_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory)
            incompatible = (
                mod_root / "vendor" / "python" / "foreign-platform-cpython-999" / "yaml"
            )
            incompatible.mkdir(parents=True)
            (incompatible / "__init__.py").write_text(
                "raise AssertionError('must not import incompatible vendor')\n",
                encoding="utf-8",
            )
            requirements = read_runtime_requirements(
                {"python": [{"import": "yaml"}], "ros": [], "system": []},
                "requirements",
            )

            report = check_runtime_requirements(requirements, mod_root)

            self.assertTrue(report.available)
            self.assertFalse(report.vendor_python)
            self.assertIn("incompatible targets", report.warnings[0])

    def test_matching_vendor_python_must_really_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory)
            package = (
                mod_root
                / "vendor"
                / "python"
                / runtime_python_tag()
                / "bxi_broken_vendor"
            )
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                "raise RuntimeError('broken native dependency')\n",
                encoding="utf-8",
            )
            requirements = read_runtime_requirements(
                {
                    "python": [{"import": "bxi_broken_vendor"}],
                    "ros": [],
                    "system": [],
                },
                "requirements",
            )

            report = check_runtime_requirements(requirements, mod_root)

            self.assertFalse(report.available)
            self.assertIn("is not importable", report.errors[0])
            self.assertIn("broken native dependency", report.errors[0])

    def test_unavailable_node_is_not_imported_or_started(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod_root = Path(temporary_directory) / "com.example.unavailable_node"
            mod_root.mkdir()
            manifest = {
                "schema": 1,
                "id": "com.example.unavailable_node",
                "name": "Unavailable node fixture",
                "version": "1.0.0",
                "api": ">=1,<2",
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
            vendor_python = mod_root / "vendor" / "python" / runtime_python_tag()
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
            vendor_python = mod_root / "vendor" / "python" / runtime_python_tag()
            vendor_library = mod_root / "vendor" / "lib" / runtime_platform_tag()
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
            "api": ">=1,<2",
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
