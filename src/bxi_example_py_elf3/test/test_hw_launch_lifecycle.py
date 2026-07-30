import importlib.util
from pathlib import Path

import pytest


LAUNCH_FILE = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "example_demo_hw.launch.py"
)


def test_critical_control_node_exit_shuts_down_complete_hw_launch():
    source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert "target_action=hardware_node" in source
    assert "target_action=controller_node" in source
    assert 'Shutdown(reason="hardware_elf3 exited")' in source
    assert 'Shutdown(reason="bxi_example_py_elf3_demo exited")' in source
    assert 'sigterm_timeout="10"' in source
    assert 'sigkill_timeout="5"' in source
    assert "os.O_NOFOLLOW" in source
    assert "os.fchmod(fd, 0o666)" in source


def load_launch_module():
    spec = importlib.util.spec_from_file_location("elf3_hw_launch_test", LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_description_constructs_with_lifecycle_handlers(tmp_path, monkeypatch):
    module = load_launch_module()
    module.LOCK_FILE = str(tmp_path / "hardware.lock")
    monkeypatch.setattr(
        module,
        "get_package_share_path",
        lambda _: LAUNCH_FILE.parents[1],
    )
    try:
        description = module.generate_launch_description()
        executables = [
            getattr(entity, "node_executable", None)
            for entity in description.entities
        ]
        assert "hardware_elf3" in executables
        assert "bxi_example_py_elf3_demo" in executables
        assert "sonic_pico_runtime_supervisor" in executables
    finally:
        module._release_lock()


def test_hardware_lock_refuses_symlink_without_touching_target(tmp_path):
    module = load_launch_module()
    target = tmp_path / "must_not_change"
    target.write_text("sentinel", encoding="utf-8")
    lock = tmp_path / "hardware.lock"
    lock.symlink_to(target)
    module.LOCK_FILE = str(lock)

    with pytest.raises(SystemExit):
        module._acquire_lock()

    assert target.read_text(encoding="utf-8") == "sentinel"
