from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib.util
from importlib.machinery import SourceFileLoader
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from bxi_example_py_elf3.utils.mod_system import (
    StateBuildContext,
    load_mod_runtime,
)
from bxi_example_py_elf3.utils.robot_state_builder import build_robot_states
from bxi_example_py_elf3.utils.state_library import (
    PolicyState,
    PoseState,
    ProceduralState,
)


_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "bxi-mod"
_TOOL_LOADER = SourceFileLoader("bxi_mod_tool", str(_TOOL_PATH))
_TOOL_SPEC = importlib.util.spec_from_loader("bxi_mod_tool", _TOOL_LOADER)
if _TOOL_SPEC is None or _TOOL_SPEC.loader is None:
    raise RuntimeError(f"cannot load test tool: {_TOOL_PATH}")
_TOOL_MODULE = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(_TOOL_MODULE)
mod_cli_main = _TOOL_MODULE.main


class _Context:
    def __init__(self):
        self.pos_last = np.array([0.1, 0.2], dtype=np.float32)
        self.joint_kp = np.array([20.0, 20.0], dtype=np.float32)
        self.joint_kd = np.array([1.0, 1.0], dtype=np.float32)
        self.last_target = None

    def set_motor_target(self, qpos, kp, kd):
        self.last_target = (qpos.copy(), kp.copy(), kd.copy())


class _Pose(PoseState):
    def target_position(self, ctx):
        return ctx.pos_last + 1.0


class _Procedure(ProceduralState):
    def compute_frame(self, ctx, elapsed):
        return self.frame(ctx, ctx.pos_last + elapsed)


class _Policy:
    def __init__(self):
        self.elapsed = 0.0


class _PolicyDriven(PolicyState[_Policy]):
    def create_policy(self, ctx):
        return _Policy()

    def reset_policy(self, ctx, policy):
        policy.elapsed = 0.0

    def policy_entry_position(self, ctx, policy):
        return ctx.pos_last

    def infer_position(self, ctx, policy, dt, *, advance):
        result = ctx.pos_last + policy.elapsed
        if advance:
            policy.elapsed += dt
        return result


class EasyStateTest(unittest.TestCase):
    def test_pose_state_supplies_transition_capabilities_and_output(self):
        context = _Context()
        state = _Pose("example/pose", 1)
        np.testing.assert_allclose(state.get_entry_frame(context).qpos, [1.1, 1.2])
        state.on_update(context, 0.02)
        np.testing.assert_allclose(context.last_target[0], [1.1, 1.2])

    def test_procedural_sampling_only_advances_when_requested(self):
        context = _Context()
        state = _Procedure("example/procedure", 2)
        state.sample_running_frame(context, 0.25, advance=False)
        self.assertEqual(state.elapsed, 0.0)
        state.sample_running_frame(context, 0.25, advance=True)
        self.assertEqual(state.elapsed, 0.25)
        frame = state.sample_running_frame(context, 0.25, advance=False)
        np.testing.assert_allclose(frame.qpos, [0.35, 0.45])

    def test_policy_state_lazily_creates_and_safely_samples_policy(self):
        context = _Context()
        state = _PolicyDriven("example/policy", 3)
        self.assertIsNone(state._policy)
        state.on_enter(context)
        self.assertIsNotNone(state._policy)
        state.sample_running_frame(context, 0.2, advance=False)
        self.assertEqual(state.policy.elapsed, 0.0)
        state.sample_running_frame(context, 0.2, advance=True)
        self.assertEqual(state.policy.elapsed, 0.2)


@dataclass(frozen=True)
class _Params:
    count: int
    gain: float = 0.5
    enabled: bool = True
    title: str = "hello"
    limit: int | None = None


class DataclassParamsTest(unittest.TestCase):
    def test_converts_numbers_and_uses_defaults(self):
        context = StateBuildContext("example/state", 1, {"count": 2, "gain": 3})
        params = context.dataclass_params(_Params)
        context.finish()
        self.assertEqual(params, _Params(count=2, gain=3.0))

    def test_reports_missing_wrong_and_unknown_params(self):
        with self.assertRaisesRegex(ValueError, "missing required param 'count'"):
            StateBuildContext("example/state", 1, {}).dataclass_params(_Params)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            StateBuildContext(
                "example/state", 1, {"count": True}
            ).dataclass_params(_Params)
        context = StateBuildContext(
            "example/state", 1, {"count": 1, "surprise": 2}
        )
        context.dataclass_params(_Params)
        with self.assertRaisesRegex(ValueError, "unknown params"):
            context.finish()


class ConventionModTest(unittest.TestCase):
    def test_two_file_mod_uses_defaults_shorthand_and_dataclass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mod = root / "com.example.wave"
            mod.mkdir()
            (mod / "mod.yaml").write_text(
                """id: com.example.wave
version: 1.0.0
states:
  wave:
    factory: state:WaveState
    label: Wave
    index: 7
    params: {amplitude: 0.25}
""",
                encoding="utf-8",
            )
            (mod / "state.py").write_text(
                """from dataclasses import dataclass
from bxi_example_py_elf3.utils.state_library import PoseState

@dataclass(frozen=True)
class WaveParams:
    amplitude: float = 0.1

class WaveState(PoseState[WaveParams]):
    Params = WaveParams
    def target_position(self, ctx):
        return ctx.pos_last + self.params.amplitude
""",
                encoding="utf-8",
            )
            runtime = load_mod_runtime(
                {"initial_state": "com.example.wave/wave"},
                built_in_root=root,
            )
            try:
                states = build_robot_states(runtime.config, runtime.state_factories)
                state = states["com.example.wave/wave"]
                self.assertEqual(state.params.amplitude, 0.25)
                self.assertEqual(state.manifest["label"], "Wave")
                self.assertEqual(state.manifest["index"], 7)
                composed = runtime.config["states"]["com.example.wave/wave"]
                self.assertNotIn("factory", composed)
                self.assertNotIn("label", composed)
            finally:
                runtime.close()

    def test_plugin_py_is_the_compatible_default_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mod = root / "com.example.plugin"
            mod.mkdir()
            (mod / "mod.yaml").write_text(
                """id: com.example.plugin
version: 1.0.0
states:
  hold:
    manifest: {label: Hold, index: 1}
""",
                encoding="utf-8",
            )
            (mod / "plugin.py").write_text(
                """from bxi_example_py_elf3.utils.state_library import PoseState
from bxi_example_py_elf3.utils.mod_system import ModDefinition
class Hold(PoseState):
    def target_position(self, ctx): return ctx.pos_last
def create_mod(context):
    return ModDefinition({'hold': lambda state: Hold(state.name, state.state_id)})
""",
                encoding="utf-8",
            )
            runtime = load_mod_runtime(
                {"initial_state": "com.example.plugin/hold"},
                built_in_root=root,
            )
            try:
                self.assertIn("com.example.plugin/hold", runtime.state_factories)
            finally:
                runtime.close()


class ToolingTest(unittest.TestCase):
    def test_repository_tool_bootstraps_source_tree(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [str(_TOOL_PATH), "--help"],
            cwd=_TOOL_PATH.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schema_is_json_and_cli_scaffolds_without_overwrite(self):
        schema = (
            Path(__file__).parents[1] / "schema" / "mod.schema.json"
        )
        document = json.loads(schema.read_text(encoding="utf-8"))
        self.assertEqual(
            document["$schema"],
            "https://json-schema.org/draft/2020-12/schema",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                result = mod_cli_main(
                    [
                        "new",
                        "com.example.generated",
                        "--root",
                        str(root),
                        "--state",
                        "wave",
                        "--template",
                        "procedural",
                        "--slot",
                        "test_generated_slot",
                    ]
                )
            self.assertEqual(result, 0)
            target = root / "com.example.generated"
            self.assertTrue((target / "mod.yaml").is_file())
            self.assertTrue((target / "state.py").is_file())
            common = [str(target)]
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(mod_cli_main(["validate", *common]), 0)
                self.assertEqual(mod_cli_main(["inspect", *common]), 0)
                self.assertEqual(
                    mod_cli_main(
                        ["new", "com.example.generated", "--root", str(root)]
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
