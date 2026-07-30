import ast
import os
from pathlib import Path
import subprocess
import textwrap

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "script"
REMOTE_CONTROLLER_CONFIG = (
    REPO_ROOT / "src/remote_controller/config/xbox_default.yaml"
)
HARDWARE_LAUNCH = (
    REPO_ROOT / "src/bxi_example_py_elf3/launch/example_demo_hw.launch.py"
)


def script(name: str) -> str:
    return (SCRIPT_DIR / name).read_text(encoding="utf-8")


def write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def assigned_call(source_path: Path, variable_name: str) -> ast.Call:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    matches = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
    ]
    assert len(matches) == 1, f"expected exactly one assignment to {variable_name}"
    return matches[0]


def fake_systemd_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    prefix = tmp_path / "opt" / "bxi_rl_controller_ros2_example"
    prefix.mkdir(parents=True)
    (prefix / "setup.bash").write_text("# test setup\n", encoding="utf-8")
    systemd_root = tmp_path / "systemd"
    dropin = systemd_root / "ros_elf_launch.service.d" / "sonic-runtime.conf"

    write_executable(
        fake_bin / "sudo",
        """
        #!/usr/bin/env bash
        exec "$@"
        """,
    )
    write_executable(
        fake_bin / "systemctl",
        """
        #!/usr/bin/env bash
        set -eu
        command="${1:-}"
        shift || true
        case "${command}" in
          cat)
            exit 0
            ;;
          is-active)
            exit 1
            ;;
          daemon-reload)
            exit 0
            ;;
          show)
            args=" $* "
            if [[ "${args}" == *" -p ExecStart "* ]]; then
              if [[ -f "${FAKE_DROPIN}" ]]; then
                printf 'argv[]=/bin/bash -lc source %s/setup.bash && exec ros2 launch remote_controller remote_controller.launch.py\n' "${CONTROLLER_PREFIX}"
              else
                printf '%s\n' "${FAKE_EXEC_START}"
              fi
            elif [[ "${args}" == *" -p Environment "* ]]; then
              if [[ -f "${FAKE_DROPIN}" && "${FAKE_CHANGE_ENV:-0}" == 1 ]]; then
                printf '%s\n' 'ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=bad'
              else
                printf '%s\n' 'ROS_DOMAIN_ID=22 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
              fi
            elif [[ "${args}" == *" -p LoadState "* ]]; then
              printf '%s\n' loaded
            elif [[ "${args}" == *" -p ActiveState "* ]]; then
              printf '%s\n' inactive
            elif [[ "${args}" == *" -p FragmentPath "* ]]; then
              printf '%s\n' /etc/systemd/system/ros_elf_launch.service
            elif [[ "${args}" == *" -p DropInPaths "* ]]; then
              [[ -f "${FAKE_DROPIN}" ]] && printf '%s\n' "${FAKE_DROPIN}" || true
            fi
            ;;
          *)
            exit 2
            ;;
        esac
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CONTROLLER_PREFIX": str(prefix),
            "SONIC_SYSTEMD_CONFIG_ROOT": str(systemd_root),
            "SONIC_SERVICE_BACKUP_DIR": str(tmp_path / "backups"),
            "FAKE_DROPIN": str(dropin),
            "FAKE_EXEC_START": (
                f"argv[]=/bin/bash -lc source {prefix}/setup.bash && "
                "exec ros2 launch remote_controller remote_controller.launch.py"
            ),
        }
    )
    return env, dropin, tmp_path / "backups"


def test_keyboard_gripper_shortcut_maps_to_btn_10_value_8():
    config = yaml.safe_load(REMOTE_CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    signal_name = "keyboard.sonic_teleop_gripper"
    event_name = "keyboard.sonic_teleop_gripper_event"

    keyboard_signals = config["sources"]["keyboard"]["signals"]
    assert keyboard_signals[signal_name] == {
        "from": "keyboard.key",
        "key": "g",
    }
    assert [
        name
        for name, signal in keyboard_signals.items()
        if signal.get("from") == "keyboard.key" and signal.get("key") == "g"
    ] == [signal_name]

    assert config["controls"][event_name] == {
        "type": "bool",
        "inputs": [{"source": signal_name}],
    }
    assert [
        rule
        for rule in config["outputs"]["level"]
        if rule.get("output") == "btn_10=8"
    ] == [
        {
            "output": "btn_10=8",
            "when": [event_name],
        }
    ]


def test_hardware_launch_supervises_both_sonic_modes_with_trigger_topics():
    # Importing generate_launch_description() would acquire the real hardware
    # single-instance lock.  Parse the launch source instead and inspect the
    # exact Node declaration without running any launch-time side effects.
    supervisor = assigned_call(HARDWARE_LAUNCH, "pico_supervisor_node")
    assert isinstance(supervisor.func, ast.Name)
    assert supervisor.func.id == "Node"
    keywords = {keyword.arg: keyword.value for keyword in supervisor.keywords}
    assert ast.literal_eval(keywords["executable"]) == "sonic_pico_runtime_supervisor"

    parameters = keywords["parameters"]
    assert isinstance(parameters, ast.List)
    literal_parameters = {}
    for parameter in parameters.elts:
        if not isinstance(parameter, ast.Dict):
            continue
        for key, value in zip(parameter.keys, parameter.values):
            if isinstance(key, ast.Constant) and key.value in {
                "target_state",
                "target_states",
            }:
                literal_parameters[key.value] = ast.literal_eval(value)

    assert literal_parameters["target_state"] == "sonic_teleop"
    assert literal_parameters["target_states"] == [
        "sonic_teleop",
        "sonic_teleop_gripper",
    ]
    assert ast.literal_eval(keywords["additional_env"]) == {
        "PICO_ENABLE_ROS_BUTTONS": "1",
    }


def test_one_click_installer_keeps_hardware_start_as_manual_acceptance():
    source = script("install_robot_sonic_bundle.sh")

    assert "deploy_robot_sonic_bundle.sh" in source
    assert "configure_robot_sonic_service.sh" in source
    assert "audit_robot_sonic_host.sh" in source
    assert "sha256sum -c MANIFEST.sha256" in source
    assert "flock -n" in source
    assert "systemctl stop \"${SERVICE}\"" in source
    assert "systemctl start" not in source
    assert not any(
        line.lstrip().startswith(("pkill ", "sudo pkill "))
        for line in source.splitlines()
    )
    assert "kill -9" not in source
    assert 'systemctl stop "${GATEWAY_SERVICE}"' not in source
    assert "systemctl stop bxi_rc_ros2.service" not in source
    assert "dpkg --configure -a" not in source
    assert "systemctl stop packagekit.service" not in source
    assert 'fuser -v "${HARDWARE_LOCK}"' in source
    assert "ensure_gateway_alive" in source
    assert "SERVICE_STOPPED" in source
    assert "not a recognized remote-controller service" in source
    assert "HARDWARE_LOCK_FD" in source
    assert "TERMINAL_ONLY == 0" in source
    assert "trap 'on_signal 129' HUP" in source
    assert "STANDARD_PATH_OVERRIDE_VARS" in source
    assert "refusing path override environment variable" in source
    assert 'final_service_state="$(systemctl show' in source
    assert '[[ "${final_service_state}" == "inactive" ]]' in source


def test_one_click_installer_rejects_standard_path_overrides(tmp_path):
    env = os.environ.copy()
    env["CONTROLLER_PREFIX"] = str(tmp_path / "unexpected-prefix")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "install_robot_sonic_bundle.sh"),
            "--yes",
            str(tmp_path / "not-a-bundle"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing path override environment variable CONTROLLER_PREFIX" in result.stderr
    assert "missing bundle manifest" not in result.stderr


def test_bundle_deployer_clears_lower_level_path_overrides():
    source = script("deploy_robot_sonic_bundle.sh")

    for variable in (
        "OPT_PREFIX",
        "BUILD_INSTALL",
        "BACKUP_DIR",
        "SONIC_OFFLINE_RUNTIME_DIR",
        "PICO_VENV",
        "CONTROLLER_PREFIX",
        "SONIC_XRT_SERVICE_DIR",
        "ROS_SETUP",
        "BXI_ROS_SETUP",
        "SONIC_PICO_PYTHON",
    ):
        assert f"-u {variable}" in source


def test_runtime_check_is_fail_closed_for_install_identity():
    source = script("check_robot_sonic_runtime.sh")

    assert 'EXPECTED_PREFIX="/opt/bxi/bxi_rl_controller_ros2_example"' in source
    assert source.count("expected exactly ${EXPECTED_PREFIX}") == 2
    assert 'fail "hardware launch does not expose sonic_pico_python' in source
    assert 'fail "installed hardware launch file not found' in source
    assert "hardware launch enables both SONIC modes" in source
    assert "PICO_ENABLE_ROS_BUTTONS" in source
    assert 'event.get("slot") != "btn_10"' in source
    assert '"sonic_teleop_gripper_event", 8, True' in source
    assert 'config.get("states", {}).get(state_name)' in source
    assert 'state.get("manifest")' in source
    assert 'state.get("behavior") != "SonicTeleopState"' in source
    assert 'get("hardware_gripper")' in source
    assert '[[ "${service_active}" == "inactive" ]]' in source


def test_runtime_check_stops_immediately_when_mktemp_fails(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "mktemp",
        """
        #!/usr/bin/env bash
        exit 73
        """,
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "check_robot_sonic_runtime.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "could not create a private runtime-check temporary directory" in result.stderr
    assert "== ROS environment ==" not in result.stdout


def test_service_dropin_preserves_robot_specific_environment():
    source = script("configure_robot_sonic_service.sh")

    assert "ExecStart=" in source
    assert "/opt/bxi/bxi_rl_controller_ros2_example" in source
    assert "source ${EXPECTED_SETUP}" in source
    assert "Environment=ROS_DOMAIN_ID" not in source
    assert "Environment=RMW_IMPLEMENTATION" not in source
    assert "AFTER_ENV" in source
    assert "BEFORE_ENV" in source
    assert "rollback_dropin" in source
    assert "trap 'handle_failure" in source
    assert "systemctl start" not in source


def test_terminal_hardware_runner_does_not_delete_single_instance_lock():
    source = script("run_robot_sonic_hw.sh")

    assert "rm -f /tmp/bxi_example_hw.lock" not in source


def test_pico_runners_use_event_driven_logs_and_current_sim2sim_source():
    source = script("run_sonic_pico_sources.sh")
    sim2sim_source = script("run_sonic_bxi_sim2sim.sh")
    gripper_source = script("run_sonic_bxi_sim2sim_gripper.sh")

    assert "BRIDGE_LOG_EVERY" not in source
    assert "--log-every" not in source
    assert 'src/bxi_example_py_elf3:${PYTHONPATH:-}' in sim2sim_source
    assert 'src/bxi_example_py_elf3:${PYTHONPATH:-}' in gripper_source
    assert "BXI_SIM_GRIPPER_ENABLE=1" in gripper_source
    assert "PICO_ENABLE_ROS_BUTTONS=1" in gripper_source
    assert "elf3_gripper.xml" in gripper_source
    assert "BXI_SIM_GRIPPER_ENABLE=0" in sim2sim_source
    assert "PICO_ENABLE_ROS_BUTTONS=0" in sim2sim_source
    assert "BXI_SONIC_GRIPPER_ENABLE" not in gripper_source
    assert "BXI_SONIC_GRIPPER_ENABLE" not in sim2sim_source
    assert "elf3_gripper.xml" not in sim2sim_source
    assert "PICO_MANAGER_ARGS+=(--vis_vr3pt --vis_smpl)" not in source


def test_bundle_source_export_excludes_repository_only_assets():
    source = script("prepare_robot_sonic_bundle.sh")

    assert 'git archive HEAD -- "${SOURCE_PATHS[@]}"' in source
    assert "src/bxi_example_py_elf3" in source
    assert "src/remote_controller" in source
    assert "THIRD_PARTY_NOTICES.md" in source
    assert "third_party" in source
    assert any(line.strip() == "docs" for line in source.splitlines())
    assert not any(
        line.strip() == "resources"
        for line in source.splitlines()
    )


def test_service_config_rejects_non_remote_controller_units(tmp_path):
    env, _, _ = fake_systemd_environment(tmp_path)

    protected = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "configure_robot_sonic_service.sh"),
            "--check",
            "--service",
            "ssh.service",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert protected.returncode == 2
    assert "refusing protected/non-control service" in protected.stderr

    env["FAKE_EXEC_START"] = "argv[]=/usr/sbin/unrelated-daemon"
    unrelated = subprocess.run(
        ["bash", str(SCRIPT_DIR / "configure_robot_sonic_service.sh"), "--check"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert unrelated.returncode == 2
    assert "not a recognized remote-controller service" in unrelated.stderr


def test_service_dropin_install_and_environment_rollback(tmp_path):
    env, dropin, backup_dir = fake_systemd_environment(tmp_path)

    installed = subprocess.run(
        ["bash", str(SCRIPT_DIR / "configure_robot_sonic_service.sh"), "--install"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    assert dropin.is_file()
    assert "Environment=" not in dropin.read_text(encoding="utf-8")
    assert list(backup_dir.glob("*.effective-state.txt"))

    dropin.unlink()
    env["FAKE_CHANGE_ENV"] = "1"
    rolled_back = subprocess.run(
        ["bash", str(SCRIPT_DIR / "configure_robot_sonic_service.sh"), "--install"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rolled_back.returncode != 0
    assert not dropin.exists()
    assert "effective Environment changed" in rolled_back.stderr
