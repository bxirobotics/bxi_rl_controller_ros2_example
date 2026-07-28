from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from bxi_example_py_elf3.inference.history import HistoryBuffer
from bxi_example_py_elf3.inference.model import (
    ModelArtifact,
    ModelSpec,
    OnnxArtifact,
    OpenVinoArtifact,
    RknnArtifact,
)
from bxi_example_py_elf3.inference.runtime import (
    BackendRegistry,
    BackendUnavailableError,
    InferenceRuntime,
)
from bxi_example_py_elf3.inference.backends.base import (
    BackendAvailability,
    BackendFactory,
    InferenceBackend,
)
from bxi_example_py_elf3.inference.backends.openvino import (
    OpenVinoBackend,
    _safe_device,
)
from bxi_example_py_elf3.inference.backends.rknn import RknnBackendFactory
from bxi_example_py_elf3.inference.backends.rknn_builder import (
    RKNN_CONVERT_ENV,
    prepare_rknn_artifact,
)


class HistoryBufferTest(unittest.TestCase):
    def test_oldest_to_newest_order_and_stable_storage(self):
        history = HistoryBuffer(3, (1,))
        physical_id = id(history.storage)
        output = np.empty(3, dtype=np.float32)

        history.fill(np.array([1.0], dtype=np.float32))
        history.write_into(output)
        np.testing.assert_array_equal(output, [1.0, 1.0, 1.0])

        history.append(np.array([2.0], dtype=np.float32))
        history.write_into(output)
        np.testing.assert_array_equal(output, [1.0, 1.0, 2.0])

        history.append(np.array([3.0], dtype=np.float32))
        history.append(np.array([4.0], dtype=np.float32))
        history.write_into(output)
        np.testing.assert_array_equal(output, [2.0, 3.0, 4.0])
        self.assertEqual(physical_id, id(history.storage))

    def test_shape_and_dtype_are_explicit(self):
        history = HistoryBuffer(2, (3,))
        with self.assertRaises(ValueError):
            history.append(np.zeros(2, dtype=np.float32))
        with self.assertRaises(TypeError):
            history.append(np.zeros(3, dtype=np.float64))


class _TestArtifact(ModelArtifact):
    backend = "test"


class _MissingArtifact(ModelArtifact):
    backend = "missing"


class _TestBackend(InferenceBackend):
    backend_name = "test"

    def run(self, inputs):
        return {"actions": inputs["obs"]}


class _TestFactory(BackendFactory):
    backend_name = "test"

    def availability(self, artifact):
        return BackendAvailability(True, "test backend")

    def open(self, artifact, spec):
        return _TestBackend()


class _MissingFactory(BackendFactory):
    backend_name = "missing"

    def availability(self, artifact):
        return BackendAvailability(
            False,
            "optional runtime is not installed",
            "python3 -m pip install optional-runtime",
        )

    def open(self, artifact, spec):
        raise AssertionError("unavailable backend must not be opened")


class _FakePort:
    def __init__(self, name, shape):
        self._name = name
        self.shape = shape

    def get_any_name(self):
        return self._name


class _FakeTensor:
    def __init__(self, data, shared_memory=False):
        self.data = data


class _FakeRequest:
    def __init__(self):
        self.inputs = {}
        self.output = _FakeTensor(np.zeros((1, 3), dtype=np.float32))

    def set_tensor(self, name, tensor):
        self.inputs[name] = tensor

    def infer(self):
        self.output.data[:] = self.inputs["obs"].data * 2.0

    def get_tensor(self, name):
        return self.output


class _FakeCompiledModel:
    inputs = (_FakePort("obs", (1, 3)),)
    outputs = (_FakePort("actions", (1, 3)),)

    def __init__(self):
        self.request = _FakeRequest()

    def create_infer_request(self):
        return self.request


class _FakeOpenVinoCore:
    def read_model(self, path):
        return object()

    def compile_model(self, model, device_name, config):
        return _FakeCompiledModel()


class _FakeDeviceCore:
    available_devices = ("CPU", "GPU")

    def __init__(self, gpu_name):
        self.gpu_name = gpu_name

    def get_property(self, device, name):
        if name == "FULL_DEVICE_NAME" and device == "GPU":
            return self.gpu_name
        return device


class RuntimeTest(unittest.TestCase):
    def test_artifacts_are_backend_extensible(self):
        spec = ModelSpec(
            artifacts=(
                RknnArtifact("model.rknn", target="rk3588"),
                OpenVinoArtifact("model.onnx", device="CPU"),
                OnnxArtifact("model.onnx"),
            )
        )
        self.assertEqual(
            [item.backend for item in spec.artifacts],
            ["rknn", "openvino", "onnxruntime"],
        )

    def test_custom_backend_needs_no_model_spec_change(self):
        runtime = InferenceRuntime(registry=BackendRegistry((_TestFactory(),)))
        spec = ModelSpec(artifacts=(_TestArtifact("unused"),))
        backend = runtime.open_backend(spec)
        self.assertIsInstance(backend, _TestBackend)

    def test_auto_fallback_warns_with_install_command(self):
        runtime = InferenceRuntime(
            registry=BackendRegistry((_MissingFactory(), _TestFactory()))
        )
        spec = ModelSpec(
            artifacts=(_MissingArtifact("unused"), _TestArtifact("unused"))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = runtime.open_backend(spec)
        self.assertIsInstance(backend, _TestBackend)
        self.assertEqual(len(caught), 1)
        self.assertIn("pip install optional-runtime", str(caught[0].message))
        self.assertIn("selected test", str(caught[0].message))

    def test_explicit_unavailable_backend_reports_install_command(self):
        runtime = InferenceRuntime(registry=BackendRegistry((_MissingFactory(),)))
        spec = ModelSpec(artifacts=(_MissingArtifact("unused"),))
        with self.assertRaisesRegex(
            BackendUnavailableError,
            "pip install optional-runtime",
        ):
            runtime.open_backend(spec, backend="missing")


class OpenVinoBackendTest(unittest.TestCase):
    def test_non_intel_gpu_is_rejected_and_auto_uses_cpu(self):
        core = _FakeDeviceCore("NVIDIA GeForce RTX 3060 (dGPU)")
        with self.assertRaisesRegex(RuntimeError, "ONNX Runtime CUDA/TensorRT"):
            _safe_device(core, "GPU")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = _safe_device(core, "AUTO")
        self.assertEqual(selected, "CPU")
        self.assertIn("ignored unsupported non-Intel GPU", str(caught[0].message))

    def test_intel_gpu_remains_available(self):
        core = _FakeDeviceCore("Intel(R) Iris(R) Xe Graphics")
        self.assertEqual(_safe_device(core, "GPU"), "GPU")
        self.assertEqual(_safe_device(core, "AUTO"), "AUTO")

    def test_reuses_output_and_rebinds_replaced_input(self):
        artifact = OpenVinoArtifact(
            "unused.onnx",
            device="CPU",
            metadata=(("source", "test"),),
        )
        spec = ModelSpec(
            artifacts=(artifact,),
            input_names=("obs",),
            output_names=("actions",),
        )
        with patch(
            "bxi_example_py_elf3.inference.backends.openvino._openvino_api",
            return_value=(_FakeOpenVinoCore, _FakeTensor),
        ):
            backend = OpenVinoBackend(artifact, spec)

        value = np.ones((1, 3), dtype=np.float32)
        first = backend.run({"obs": value})["actions"]
        first_id = id(first)
        np.testing.assert_array_equal(first, np.full((1, 3), 2.0))

        value.fill(3.0)
        second = backend.run({"obs": value})["actions"]
        self.assertEqual(first_id, id(second))
        np.testing.assert_array_equal(second, np.full((1, 3), 6.0))

        replacement = np.full((1, 3), 4.0, dtype=np.float32)
        third = backend.run({"obs": replacement})["actions"]
        self.assertEqual(first_id, id(third))
        np.testing.assert_array_equal(third, np.full((1, 3), 8.0))
        backend.close()


class _FakeRknnToolkit:
    instances = []
    export_result = 0

    def __init__(self):
        self.calls = []
        self.released = False
        self.__class__.instances.append(self)

    def config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return 0

    def load_onnx(self, **kwargs):
        self.calls.append(("load_onnx", kwargs))
        return 0

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        return 0

    def export_rknn(self, path):
        self.calls.append(("export_rknn", path))
        if self.export_result == 0:
            Path(path).write_bytes(b"converted-rknn")
        return self.export_result

    def release(self):
        self.released = True


class RknnConversionTest(unittest.TestCase):
    def setUp(self):
        _FakeRknnToolkit.instances.clear()
        _FakeRknnToolkit.export_result = 0

    def _artifact(self, directory, **kwargs):
        source = Path(directory) / "model.onnx"
        source.write_bytes(b"onnx-v1")
        return RknnArtifact(
            Path(directory) / "model.rknn",
            source_onnx=source,
            target="rk3588",
            **kwargs,
        )

    def test_unset_environment_never_converts(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory)
            environment = os.environ.copy()
            environment.pop(RKNN_CONVERT_ENV, None)
            with patch.dict(os.environ, environment, clear=True), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api"
            ) as api:
                result = prepare_rknn_artifact(artifact)

            self.assertFalse(result.ready)
            self.assertFalse(artifact.resolved_path.exists())
            api.assert_not_called()

    def test_json_settings_convert_and_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.txt"
            calibration = Path(directory) / "sample.npy"
            calibration.write_bytes(b"sample")
            dataset.write_text(str(calibration), encoding="utf-8")
            artifact = self._artifact(directory)
            settings = json.dumps(
                {
                    "target": "rk3568",
                    "do_quantization": True,
                    "dataset": str(dataset),
                    "config": {"optimization_level": 3},
                }
            )
            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: settings}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api",
                return_value=_FakeRknnToolkit,
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = prepare_rknn_artifact(artifact)

            self.assertTrue(result.ready)
            self.assertTrue(result.converted)
            self.assertEqual(artifact.resolved_path.read_bytes(), b"converted-rknn")
            instance = _FakeRknnToolkit.instances[0]
            self.assertEqual(instance.calls[0][0], "config")
            self.assertEqual(instance.calls[0][1]["target_platform"], "rk3568")
            self.assertEqual(instance.calls[0][1]["optimization_level"], 3)
            self.assertEqual(
                instance.calls[2],
                (
                    "build",
                    {"do_quantization": True, "dataset": str(dataset.resolve())},
                ),
            )
            self.assertTrue(instance.released)
            manifest = Path(str(artifact.resolved_path) + ".build.json")
            self.assertEqual(json.loads(manifest.read_text())["schema"], 1)

    def test_current_fingerprint_does_not_convert_again(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory)
            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: "rk3588"}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api",
                return_value=_FakeRknnToolkit,
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first = prepare_rknn_artifact(artifact)
            self.assertTrue(first.converted)

            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: "rk3588"}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api"
            ) as api:
                second = prepare_rknn_artifact(artifact)
            self.assertTrue(second.ready)
            self.assertFalse(second.converted)
            api.assert_not_called()

    def test_changed_source_rebuilds_once(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory)
            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: "rk3588"}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api",
                return_value=_FakeRknnToolkit,
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prepare_rknn_artifact(artifact)
                Path(artifact.source_onnx).write_bytes(b"onnx-v2-with-new-size")
                result = prepare_rknn_artifact(artifact)

            self.assertTrue(result.converted)
            self.assertEqual(len(_FakeRknnToolkit.instances), 2)

    def test_environment_target_is_used_for_platform_check(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory)
            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: "rk3568"}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api",
                return_value=_FakeRknnToolkit,
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn.importlib.util.find_spec",
                return_value=object(),
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn._rockchip_compatible",
                return_value="rockchip,rk3568",
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                availability = RknnBackendFactory().availability(artifact)

            self.assertTrue(availability.available)

    def test_failed_conversion_preserves_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory)
            artifact.resolved_path.write_bytes(b"known-good-rknn")
            _FakeRknnToolkit.export_result = 9
            settings = json.dumps({"target": "rk3588", "force_rebuild": True})
            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: settings}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api",
                return_value=_FakeRknnToolkit,
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = prepare_rknn_artifact(artifact)

            self.assertFalse(result.ready)
            self.assertIn("export_rknn failed", result.reason)
            self.assertEqual(artifact.resolved_path.read_bytes(), b"known-good-rknn")
            temporary_models = [
                path
                for path in Path(directory).glob("*.rknn")
                if path != artifact.resolved_path
            ]
            self.assertEqual(temporary_models, [])
            self.assertTrue(_FakeRknnToolkit.instances[0].released)

    def test_quantization_requires_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory, do_quantization=True)
            with patch.dict(
                os.environ, {RKNN_CONVERT_ENV: "rk3588"}, clear=False
            ), patch(
                "bxi_example_py_elf3.inference.backends.rknn_builder._rknn_api"
            ) as api:
                result = prepare_rknn_artifact(artifact)

            self.assertFalse(result.ready)
            self.assertIn("requires a calibration dataset", result.reason)
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
