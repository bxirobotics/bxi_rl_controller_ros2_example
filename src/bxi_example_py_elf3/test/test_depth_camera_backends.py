from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from sensor_msgs.msg import Image

from bxi_example_py_elf3._runtime import mod_loader
from bxi_example_py_elf3.mod_api import NodeBuildContext


class DepthCameraBackendsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod_root = (
            Path(__file__).resolve().parents[1] / "mods" / "com.bxi.normal_depth"
        )
        discovered = mod_loader._discover_mods((cls.mod_root,))
        cls.mod = discovered["com.bxi.normal_depth"]
        cls.package = mod_loader._create_dynamic_package(cls.mod)
        cls.auto_module = mod_loader._load_mod_module(
            cls.mod, cls.package, "auto_depth_node"
        )
        cls.orbbec_module = mod_loader._load_mod_module(
            cls.mod, cls.package, "orbbec_depth_node"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        mod_loader._remove_module_prefixes((cls.package.__name__,))

    def test_usb_scan_and_preference_select_orbbec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sysfs = Path(temporary_directory)
            self._write_usb_device(sysfs / "1-1", "8086", "0b3a", "RealSense")
            self._write_usb_device(sysfs / "1-2", "2bc5", "0800", "Gemini 335")

            devices = self.auto_module.scan_usb_devices(sysfs)

        self.assertEqual(len(devices), 2)
        self.assertEqual(
            self.auto_module.choose_backend("auto", "orbbec", devices),
            "orbbec",
        )
        self.assertEqual(
            self.auto_module.choose_backend("auto", "realsense", devices),
            "realsense",
        )

    def test_other_orbbec_product_is_not_misidentified_as_gemini_335(self) -> None:
        devices = (self.auto_module.UsbDevice("2bc5", "0670", "Gemini 2"),)
        self.assertIsNone(self.auto_module.choose_backend("auto", "orbbec", devices))

    def test_factory_uses_selected_orbbec_backend(self) -> None:
        sentinel = object()
        captured: dict[str, object] = {}

        def build(node_name, mod_root, params):
            captured.update(
                node_name=node_name,
                mod_root=mod_root,
                params=params,
            )
            return sentinel

        context = NodeBuildContext(
            mod_id="com.bxi.normal_depth",
            node_id="com.bxi.normal_depth/depth_camera_publisher",
            node_name="depth_camera_publisher",
            mod_root=self.mod_root,
            params={
                "camera_backend": "auto",
                "camera_preference": "orbbec",
                "camera_detect_timeout_sec": 0.0,
                "camera_detect_interval_sec": 0.1,
            },
        )
        devices = (self.auto_module.UsbDevice("2bc5", "0800", "Gemini 335"),)
        with mock.patch.object(
            self.auto_module,
            "scan_usb_devices",
            return_value=devices,
        ), mock.patch.object(
            self.auto_module,
            "OrbbecDepthPublisher",
            side_effect=build,
        ):
            result = self.auto_module.create_node(context)

        self.assertIs(result, sentinel)
        self.assertEqual(captured["node_name"], "depth_camera_publisher")
        self.assertEqual(captured["mod_root"], self.mod_root)
        self.assertEqual(captured["params"]["depth_w"], 480)
        self.assertEqual(captured["params"]["depth_h"], 270)
        self.assertEqual(captured["params"]["depth_fps"], 60)
        self.assertNotIn("orbbec_depth_w", captured["params"])

    def test_sdk_depth_units_are_scaled_to_millimeters(self) -> None:
        source = np.array([[0, 1000], [2500, 65535]], dtype=np.uint16)

        depth = self.orbbec_module._depth_mm_from_sdk_data(
            source,
            2,
            2,
            1.0,
        )

        np.testing.assert_array_equal(depth, source)

    def test_fractional_sdk_depth_scale_is_converted_to_millimeters(self) -> None:
        source = np.array([[0, 12500], [65535, 10]], dtype=np.uint16)

        depth = self.orbbec_module._depth_mm_from_sdk_data(
            source,
            2,
            2,
            0.1,
        )

        np.testing.assert_array_equal(
            depth,
            np.array([[0, 1250], [6554, 1]], dtype=np.uint16),
        )

    def test_orbbec_camera_info_has_identity_rotation(self) -> None:
        image = Image()
        image.width = 848
        image.height = 480

        publisher = SimpleNamespace(_depth_distortion=[0.1, 0.2, 0.3, 0.4, 0.5])
        info = self.orbbec_module.OrbbecDepthPublisher._camera_info(
            publisher,
            image,
            400.0,
            401.0,
            423.5,
            239.5,
        )

        self.assertEqual(
            list(info.r),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(list(info.d), publisher._depth_distortion)

    def test_orbbec_backend_uses_sdk_parameters_without_ros_driver(self) -> None:
        defaults = self.orbbec_module.ORBBEC_DEFAULTS

        self.assertTrue(defaults["orbbec_enable_sdk_filters"])
        self.assertNotIn("orbbec_driver_package", defaults)
        self.assertNotIn("orbbec_input_depth_topic", defaults)

    def test_pyorbbecsdk_is_bundled_for_linux_x86_64_and_aarch64(self) -> None:
        python_root = self.mod_root / "vendor" / "python"
        extensions = (
            python_root
            / "linux-x86_64-cpython-310"
            / "pyorbbecsdk"
            / "pyorbbecsdk.cpython-310-x86_64-linux-gnu.so",
            python_root
            / "linux-aarch64-cpython-310"
            / "pyorbbecsdk"
            / "pyorbbecsdk.cpython-310-aarch64-linux-gnu.so",
        )

        for extension in extensions:
            self.assertTrue(extension.is_file(), extension)

    @staticmethod
    def _write_usb_device(
        root: Path,
        vendor: str,
        product: str,
        name: str,
    ) -> None:
        root.mkdir()
        (root / "idVendor").write_text(vendor + "\n", encoding="ascii")
        (root / "idProduct").write_text(product + "\n", encoding="ascii")
        (root / "product").write_text(name + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
