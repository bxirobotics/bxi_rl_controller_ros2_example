from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


if "ament_index_python.packages" not in sys.modules:
    ament_stub = ModuleType("ament_index_python")
    packages_stub = ModuleType("ament_index_python.packages")

    class PackageNotFoundError(Exception):
        pass

    packages_stub.PackageNotFoundError = PackageNotFoundError
    packages_stub.get_package_prefix = lambda _name: ""
    packages_stub.get_package_share_path = lambda _name: Path()
    ament_stub.packages = packages_stub
    sys.modules["ament_index_python"] = ament_stub
    sys.modules["ament_index_python.packages"] = packages_stub

from bxi_example_py_elf3.framework import StateMachineInspector


BASE_CONFIG = """
initial_state: com.example.demo/idle
mod_paths: []
graph:
  validate: true
  export:
    dot: should-not-be-written.dot
default_transition: instant
transition_profiles:
  instant:
    type: instant
"""


MOD_CONFIG = """
schema: 1
id: com.example.demo
name: Inspector Test
version: 1.0.0
api: ">=4,<5"
enable: true
entrypoint: plugin:create_mod
visibility: public
requires: []
conflicts: []
python_exports: []
runtime_requirements:
  python:
    - import: dependency_that_does_not_exist
  ros:
    - package: ros_package_that_does_not_exist
  system: []
events:
  activate:
    slot: btn_1
    value: 1
states:
  idle:
    inference_hz: 25
    manifest:
      label: Idle
      priority: 10
  active:
    manifest:
      label: Active
nodes:
  helper:
    runtime: command
    entrypoint: missing-program
    execution: process
    lifecycle: state
    states: [active]
    manifest:
      label: Helper
routes:
  - from: idle
    event: activate
    to: active
actions: []
"""


class StateMachineInspectorTest(unittest.TestCase):
    def _package(self, root: Path) -> Path:
        share = root / "share" / "bxi_example_py_elf3"
        (share / "config").mkdir(parents=True)
        mod = share / "mods" / "com.example.demo"
        mod.mkdir(parents=True)
        (share / "config" / "elf3_state_machine.yaml").write_text(
            BASE_CONFIG, encoding="utf-8"
        )
        (mod / "mod.yaml").write_text(MOD_CONFIG, encoding="utf-8")
        (mod / "plugin.py").write_text(
            "raise AssertionError('offline inspection imported Mod code')\n",
            encoding="utf-8",
        )
        return share

    def test_from_package_is_static_and_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            share = self._package(root)
            ament = ModuleType("ament_index_python")
            packages = ModuleType("ament_index_python.packages")
            packages.get_package_share_path = lambda _name: share
            ament.packages = packages
            with patch.dict(
                sys.modules,
                {
                    "ament_index_python": ament,
                    "ament_index_python.packages": packages,
                },
            ):
                inspector = StateMachineInspector.from_package()
            snapshot = inspector.snapshot(include_graph=True)

            self.assertTrue(snapshot["offline"])
            self.assertEqual(snapshot["package"], "bxi_example_py_elf3")
            self.assertEqual(snapshot["mods"][0]["status"], "enabled")
            self.assertEqual(snapshot["nodes"][0]["status"], "declared")
            graph = snapshot["graph"]
            self.assertEqual(len(graph["states"]), 2)
            self.assertEqual(graph["states"][0]["inference_hz"], 25.0)
            self.assertIsNone(graph["states"][0]["behavior"])
            self.assertEqual(graph["transitions"][0]["to"], "com.example.demo/active")
            json.dumps(snapshot)
            self.assertFalse((root / "should-not-be-written.dot").exists())

    def test_snapshot_without_graph_omits_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            share = self._package(Path(directory))
            inspector = StateMachineInspector(
                share / "config" / "elf3_state_machine.yaml",
                built_in_mod_root=share / "mods",
            )

            self.assertNotIn("graph", inspector.snapshot(include_graph=False))
            self.assertIn("states", inspector.graph())


if __name__ == "__main__":
    unittest.main()
