"""Managed Python nodes contributed by Mods."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Protocol

import yaml

from bxi_example_py_elf3.framework.mod_api.node import ModNode, NodeBuildContext, NodeFactory
from bxi_example_py_elf3.framework.runtime.runtime_requirements import (
    vendor_library_paths,
    vendor_python_paths,
)


class ExecutorLike(Protocol):
    def add_node(self, node: object) -> object:
        ...

    def remove_node(self, node: object) -> object:
        ...


@dataclass(frozen=True)
class ModNodeSpec:
    id: str
    mod_id: str
    local_name: str
    node_name: str
    mod_root: Path
    manifest_path: Path
    entrypoint: str
    execution: str
    lifecycle: str
    states: tuple[str, ...]
    params: Mapping[str, object]
    manifest: Mapping[str, object]
    restart_max_attempts: int
    restart_delay: float
    factory: NodeFactory | None
    runtime: str = "python"
    arguments: tuple[str, ...] = ()
    remappings: Mapping[str, str] = field(default_factory=dict)
    namespace: str = ""
    executable_path: Path | None = None
    unavailable_error: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass
class _RunningNode:
    spec: ModNodeSpec
    instance: ModNode | None = None
    process: subprocess.Popen[bytes] | None = None
    attached: bool = False
    restart_attempts: int = 0
    next_restart_at: float = 0.0
    last_exit_code: int | None = None


@dataclass
class _StoppingProcess:
    node_id: str
    process: subprocess.Popen[bytes]
    kill_at: float


@dataclass
class _StoppingInstance:
    node_id: str
    instance: ModNode
    destroy_at: float


class ModNodeManager:
    """Starts, scopes, monitors and deterministically closes Mod nodes."""

    def __init__(
        self,
        specs: Sequence[ModNodeSpec],
        *,
        logger: object | None = None,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._specs = {spec.id: spec for spec in specs}
        self._logger = logger
        self._process_factory = process_factory
        self._executor: ExecutorLike | None = None
        self._running: dict[str, _RunningNode] = {}
        self._faults: dict[str, str] = {}
        self._fault_attempts: dict[str, int] = {}
        self._stopping_processes: dict[str, _StoppingProcess] = {}
        self._stopping_instances: dict[str, _StoppingInstance] = {}
        self._active_states: set[str] = set()
        self._prepared_states: set[str] = set()
        self._parameter_directory: Path | None = None
        self._parameter_files: dict[str, Path] = {}
        self._closed = False

    def start(self) -> None:
        for spec in self._specs.values():
            for warning in spec.warnings:
                self._log("warning", warning)
            if spec.unavailable_error is not None:
                self._log(
                    "warning",
                    f"Mod node '{spec.id}' is unavailable: "
                    f"{spec.unavailable_error}",
                )
        try:
            self._reconcile()
        except Exception:
            self.close()
            raise

    def activate_initial_state(self, state_name: str) -> None:
        self._active_states.add(state_name)
        try:
            self._reconcile()
            self._ensure_state_nodes_running(state_name)
        except Exception:
            self._active_states.discard(state_name)
            self._reconcile_safely()
            raise

    def prepare_state(self, state_name: str) -> None:
        if state_name in self._active_states or state_name in self._prepared_states:
            return
        self._prepared_states.add(state_name)
        try:
            self._reconcile()
            self._ensure_state_nodes_running(state_name)
        except Exception:
            self._prepared_states.discard(state_name)
            self._reconcile_safely()
            raise

    def cancel_prepared_state(self, state_name: str) -> None:
        self._prepared_states.discard(state_name)
        self._reconcile_safely()

    def finish_transition(self, from_state: str, to_state: str) -> None:
        self._active_states.discard(from_state)
        self._prepared_states.discard(to_state)
        self._active_states.add(to_state)
        self._reconcile_safely()

    def attach_executor(self, executor: ExecutorLike) -> None:
        if self._executor is executor:
            return
        if self._executor is not None:
            raise RuntimeError("Mod nodes are already attached to an executor")
        attached: list[_RunningNode] = []
        try:
            for handle in self._running.values():
                if handle.instance is None:
                    continue
                self._add_executor_node(executor, handle.instance)
                handle.attached = True
                attached.append(handle)
        except Exception:
            for handle in reversed(attached):
                try:
                    executor.remove_node(handle.instance)
                except Exception:
                    pass
                handle.attached = False
            raise
        self._executor = executor

    def detach_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is None:
            return
        for handle in reversed(tuple(self._running.values())):
            if not handle.attached or handle.instance is None:
                continue
            try:
                executor.remove_node(handle.instance)
            except Exception as exc:
                self._log(
                    "warning", f"failed to detach Mod node '{handle.spec.id}': {exc}"
                )
            handle.attached = False

    def poll(self) -> None:
        if self._closed:
            return
        desired = self._desired_node_ids()
        now = time.monotonic()
        self._poll_stopping_processes(now)
        self._poll_stopping_instances(now)
        for node_id, handle in tuple(self._running.items()):
            process = handle.process
            if process is None or node_id not in desired:
                continue
            exit_code = process.poll()
            if exit_code is None:
                continue
            handle.process = None
            handle.last_exit_code = exit_code
            if handle.restart_attempts >= handle.spec.restart_max_attempts:
                message = (
                    f"Mod node '{node_id}' exited with code {exit_code}; "
                    "restart limit reached"
                )
                self._faults[node_id] = message
                self._fault_attempts[node_id] = handle.restart_attempts
                self._running.pop(node_id, None)
                self._log("error", message)
                continue
            handle.restart_attempts += 1
            handle.next_restart_at = now + handle.spec.restart_delay
            self._log(
                "warning",
                f"Mod node '{node_id}' exited with code {exit_code}; "
                f"restart {handle.restart_attempts}/"
                f"{handle.spec.restart_max_attempts} scheduled",
            )

        for node_id, handle in tuple(self._running.items()):
            if (
                node_id not in desired
                or handle.spec.execution != "process"
                or handle.process is not None
                or now < handle.next_restart_at
            ):
                continue
            try:
                handle.process = self._spawn_process(handle.spec)
                self._log("info", f"restarted Mod node '{node_id}'")
            except Exception as exc:
                if handle.restart_attempts >= handle.spec.restart_max_attempts:
                    message = f"Mod node '{node_id}' restart failed: {exc}"
                    self._faults[node_id] = message
                    self._fault_attempts[node_id] = handle.restart_attempts
                    self._running.pop(node_id, None)
                    self._log("error", message)
                else:
                    handle.restart_attempts += 1
                    handle.next_restart_at = now + handle.spec.restart_delay
                    self._log("warning", f"Mod node '{node_id}' restart failed: {exc}")

    def snapshot(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        desired = self._desired_node_ids()
        for spec in self._specs.values():
            handle = self._running.get(spec.id)
            if spec.unavailable_error is not None:
                status = "unavailable"
            elif spec.id in self._faults:
                status = "faulted"
            elif (
                handle is not None
                and handle.process is None
                and spec.execution == "process"
            ):
                status = "restarting"
            elif handle is not None:
                status = "running"
            elif spec.id in desired:
                status = "starting"
            else:
                status = "stopped"
            result.append(
                {
                    "id": spec.id,
                    "runtime": spec.runtime,
                    "execution": spec.execution,
                    "lifecycle": spec.lifecycle,
                    "states": list(spec.states),
                    "status": status,
                    "restart_attempts": (
                        handle.restart_attempts
                        if handle
                        else self._fault_attempts.get(spec.id, 0)
                    ),
                    "error": spec.unavailable_error or self._faults.get(spec.id),
                    "warnings": list(spec.warnings),
                    **spec.manifest,
                }
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.detach_executor()
        for node_id in reversed(tuple(self._running)):
            self._stop_node(node_id, wait=True)
        for stopping in tuple(self._stopping_processes.values()):
            self._wait_for_process(stopping.process)
        self._stopping_processes.clear()
        for stopping in tuple(self._stopping_instances.values()):
            self._destroy_instance(stopping.node_id, stopping.instance)
        self._stopping_instances.clear()
        self._faults.clear()
        self._fault_attempts.clear()
        self._active_states.clear()
        self._prepared_states.clear()
        if self._parameter_directory is not None:
            shutil.rmtree(self._parameter_directory, ignore_errors=True)
            self._parameter_directory = None
            self._parameter_files.clear()

    def _desired_node_ids(self) -> set[str]:
        scoped_states = self._active_states | self._prepared_states
        return {
            spec.id
            for spec in self._specs.values()
            if spec.unavailable_error is None
            and (spec.lifecycle == "mod" or bool(set(spec.states) & scoped_states))
        }

    def _reconcile(self) -> None:
        if self._closed:
            raise RuntimeError("Mod node manager is closed")
        desired = self._desired_node_ids()
        for node_id in tuple(self._faults):
            if node_id not in desired:
                self._faults.pop(node_id, None)
                self._fault_attempts.pop(node_id, None)
        for node_id in reversed(tuple(self._running)):
            if node_id not in desired:
                self._stop_node(node_id)
        for node_id in desired:
            if node_id in self._running:
                continue
            if node_id in self._faults:
                continue
            self._start_node(self._specs[node_id])

    def _ensure_state_nodes_running(self, state_name: str) -> None:
        for spec in self._specs.values():
            if spec.lifecycle != "state" or state_name not in spec.states:
                continue
            if spec.unavailable_error is not None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' is unavailable: " f"{spec.unavailable_error}"
                )
            handle = self._running.get(spec.id)
            running = handle is not None and (
                handle.instance is not None or handle.process is not None
            )
            if not running:
                raise RuntimeError(
                    self._faults.get(spec.id, f"Mod node '{spec.id}' is not running")
                )

    def _reconcile_safely(self) -> None:
        try:
            self._reconcile()
        except Exception as exc:
            self._log("warning", f"Mod node lifecycle cleanup failed: {exc}")

    def _start_node(self, spec: ModNodeSpec) -> None:
        if spec.id in self._stopping_processes or spec.id in self._stopping_instances:
            raise RuntimeError(f"Mod node '{spec.id}' is still stopping")
        if spec.execution == "process":
            process = self._spawn_process(spec)
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' exited during startup with "
                    f"code {exit_code}"
                )
            self._running[spec.id] = _RunningNode(spec=spec, process=process)
            self._log("info", f"started process Mod node '{spec.id}'")
            return

        factory = spec.factory
        if factory is None:
            raise RuntimeError(f"Mod node '{spec.id}' has no in-process factory")
        context = NodeBuildContext(
            mod_id=spec.mod_id,
            node_id=spec.id,
            node_name=spec.node_name,
            mod_root=spec.mod_root,
            params=spec.params,
            arguments=spec.arguments,
            remappings=spec.remappings,
            namespace=spec.namespace,
        )
        instance: ModNode | None = None
        attached = False
        try:
            instance = factory(context)
            if not callable(getattr(instance, "destroy_node", None)):
                raise TypeError(
                    f"Mod node entrypoint '{spec.entrypoint}' must return "
                    "an rclpy Node"
                )
            if self._executor is not None:
                self._add_executor_node(self._executor, instance)
                attached = True
        except Exception:
            if attached and self._executor is not None and instance is not None:
                try:
                    self._executor.remove_node(instance)
                except Exception:
                    pass
            if instance is not None:
                try:
                    instance.destroy_node()
                except Exception:
                    pass
            raise
        self._running[spec.id] = _RunningNode(
            spec=spec,
            instance=instance,
            attached=attached,
        )
        self._log("info", f"started in-process Mod node '{spec.id}'")

    def _spawn_process(self, spec: ModNodeSpec) -> subprocess.Popen[bytes]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        inherited_paths: list[str] = []
        inherited_paths.extend(str(path) for path in vendor_python_paths(spec.mod_root))
        inherited_paths.extend(str(path) for path in sys.path if path)
        existing_python_path = environment.get("PYTHONPATH")
        if existing_python_path:
            inherited_paths.extend(existing_python_path.split(os.pathsep))
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(inherited_paths))
        library_paths: list[str] = []
        library_paths.extend(str(path) for path in vendor_library_paths(spec.mod_root))
        existing_library_path = environment.get("LD_LIBRARY_PATH")
        if existing_library_path:
            library_paths.extend(existing_library_path.split(os.pathsep))
        if library_paths:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                dict.fromkeys(library_paths)
            )
        if spec.runtime == "python":
            command = [
                sys.executable,
                "-m",
                "bxi_example_py_elf3.framework.runtime.mod_node_runner",
                "--manifest",
                str(spec.manifest_path),
                "--node",
                spec.local_name,
            ]
            cwd = None
        else:
            executable = spec.executable_path
            if executable is None:
                raise RuntimeError(f"Mod node '{spec.id}' has no resolved executable")
            command = [
                str(executable),
                *spec.arguments,
                "--ros-args",
                "-r",
                f"__node:={spec.node_name}",
            ]
            if spec.namespace:
                command.extend(("-r", f"__ns:={spec.namespace}"))
            for source, target in spec.remappings.items():
                command.extend(("-r", f"{source}:={target}"))
            if spec.params:
                command.extend(("--params-file", str(self._parameter_file(spec))))
            cwd = str(spec.mod_root)

        kwargs: dict[str, object] = {
            "env": environment,
            "start_new_session": True,
        }
        if cwd is not None:
            kwargs["cwd"] = cwd
        return self._process_factory(
            command,
            **kwargs,
        )

    def _parameter_file(self, spec: ModNodeSpec) -> Path:
        existing = self._parameter_files.get(spec.id)
        if existing is not None:
            return existing
        if self._parameter_directory is None:
            self._parameter_directory = Path(
                tempfile.mkdtemp(prefix="bxi-mod-node-params-")
            )
        path = self._parameter_directory / f"{spec.node_name}.yaml"
        namespace = spec.namespace.strip("/")
        node_fqn = "/" + "/".join(part for part in (namespace, spec.node_name) if part)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
                yaml.safe_dump(
                    {node_fqn: {"ros__parameters": dict(spec.params)}},
                    output_file,
                    sort_keys=False,
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._parameter_files[spec.id] = path
        return path

    @staticmethod
    def _add_executor_node(executor: ExecutorLike, node: object) -> None:
        added = executor.add_node(node)
        if added is False:
            raise RuntimeError("executor rejected Mod node")

    def _stop_node(self, node_id: str, *, wait: bool = False) -> None:
        handle = self._running.pop(node_id, None)
        if handle is None:
            return
        if handle.instance is not None:
            if handle.attached and self._executor is not None:
                try:
                    self._executor.remove_node(handle.instance)
                except Exception as exc:
                    self._log(
                        "warning", f"failed to remove Mod node '{node_id}': {exc}"
                    )
            if wait or self._executor is None:
                self._destroy_instance(node_id, handle.instance)
            else:
                self._stopping_instances[node_id] = _StoppingInstance(
                    node_id=node_id,
                    instance=handle.instance,
                    destroy_at=time.monotonic() + 0.1,
                )
        process = handle.process
        if process is not None and process.poll() is None:
            self._signal_process(process, signal.SIGTERM)
            if wait:
                self._wait_for_process(process)
            else:
                self._stopping_processes[node_id] = _StoppingProcess(
                    node_id=node_id,
                    process=process,
                    kill_at=time.monotonic() + 3.0,
                )
        self._log("info", f"stopped Mod node '{node_id}'")

    def _poll_stopping_processes(self, now: float) -> None:
        for node_id, stopping in tuple(self._stopping_processes.items()):
            if stopping.process.poll() is not None:
                self._stopping_processes.pop(node_id, None)
                continue
            if now < stopping.kill_at:
                continue
            self._signal_process(stopping.process, signal.SIGKILL)
            stopping.kill_at = float("inf")
            self._log("warning", f"killed unresponsive Mod node '{node_id}'")

    def _poll_stopping_instances(self, now: float) -> None:
        for node_id, stopping in tuple(self._stopping_instances.items()):
            if now < stopping.destroy_at:
                continue
            self._destroy_instance(node_id, stopping.instance)
            self._stopping_instances.pop(node_id, None)

    def _destroy_instance(self, node_id: str, instance: ModNode) -> None:
        try:
            instance.destroy_node()
        except Exception as exc:
            self._log("warning", f"failed to destroy Mod node '{node_id}': {exc}")

    @classmethod
    def _wait_for_process(cls, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            cls._signal_process(process, signal.SIGKILL)
            process.wait(timeout=3.0)

    @staticmethod
    def _signal_process(
        process: subprocess.Popen[bytes], value: signal.Signals
    ) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(process, subprocess.Popen) and isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, value)
                return
            except (ProcessLookupError, PermissionError):
                pass
        if value == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    def _log(self, level: str, message: str) -> None:
        logger = self._logger
        if logger is not None:
            # rclpy identifies a Python log call by its source location and
            # rejects later calls from that location with a different severity.
            # Keep every supported severity on its own call line.
            if level == "info":
                method = getattr(logger, "info", None)
                if callable(method):
                    method(message)
                    return
            elif level == "warning":
                method = getattr(logger, "warning", None)
                if callable(method):
                    method(message)
                    return
            elif level == "error":
                method = getattr(logger, "error", None)
                if callable(method):
                    method(message)
                    return
            else:
                method = getattr(logger, level, None)
                if callable(method):
                    method(message)
                    return
        print(f"{level}: {message}")


__all__ = ["ExecutorLike", "ModNodeManager", "ModNodeSpec"]
