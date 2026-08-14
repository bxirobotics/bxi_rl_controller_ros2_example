from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


PICO_ROOT = (
    Path(__file__).resolve().parents[1]
    / "mods"
    / "com.bxi.sonic"
    / "pico"
)


def _load_python_runtime():
    name = "_sonic_python_runtime_test_module"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    sys.path.insert(0, str(PICO_ROOT))
    spec = importlib.util.spec_from_file_location(
        name,
        PICO_ROOT / "python_runtime.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load_python_runtime()


def test_resolve_python_preserves_virtualenv_launcher_symlink(tmp_path):
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))

    selected = runtime._resolve_python(str(launcher))

    assert selected == Path(os.path.abspath(launcher))
    assert selected != launcher.resolve()


def test_unique_paths_keeps_venv_and_base_interpreter_distinct(tmp_path):
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))

    selected = runtime._unique_paths((launcher, Path(sys.executable)))

    assert selected == (
        Path(os.path.abspath(launcher)),
        Path(os.path.abspath(sys.executable)),
    )


def test_explicit_python_selection_probes_venv_launcher(monkeypatch, tmp_path):
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))
    probed = []

    def fake_probe(interpreter, imports, *, vendor_directory=None):
        probed.append((interpreter, tuple(imports), vendor_directory))
        return True, ""

    monkeypatch.setenv("SONIC_PICO_PYTHON", str(launcher))
    monkeypatch.setattr(runtime, "_probe", fake_probe)

    selected = runtime.select_python("pico_manager", ("numpy",))

    expected = Path(os.path.abspath(launcher))
    assert selected.executable == expected
    assert probed == [(expected, ("numpy",), None)]
