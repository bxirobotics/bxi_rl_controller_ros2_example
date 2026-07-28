from __future__ import annotations

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
from bxi_example_py_elf3.inference.backends.openvino import OpenVinoBackend


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


if __name__ == "__main__":
    unittest.main()
