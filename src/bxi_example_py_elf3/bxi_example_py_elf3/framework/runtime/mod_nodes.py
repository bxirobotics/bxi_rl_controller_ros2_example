"""Managed Python nodes contributed by Mods."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time
from typing import Protocol, cast

import yaml

from bxi_example_py_elf3.framework.mod_api.node import (
    ModNode,
    NodeBuildContext,
    NodeFactory,
)
from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    configure_current_thread,
    CpuAffinityPlan,
    CpuAffinityRole,
    CpuAffinitySpec,
    format_cpu_set,
)
from bxi_example_py_elf3.framework.runtime.logging import (
    ScopedLoggers,
    SubprocessLogRouter,
)
from bxi_example_py_elf3.framework.runtime.runtime_profiles import ResolvedRuntime


class ExecutorLike(Protocol):
    def add_node(self, node: object) -> object:
        ...

    def remove_node(self, node: object) -> object:
        ...


@dataclass(frozen=True)
class EnvironmentEdit:
    """One declarative child-process environment change."""

    value: str | None = None
    prepend: tuple[str, ...] = ()
    append: tuple[str, ...] = ()
    separator: str = os.pathsep
    existing_only: bool = False
    unset: bool = False


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
    interpreter: str | None = None
    environment: Mapping[str, EnvironmentEdit] = field(default_factory=dict)
    cwd: Path | None = None
    depends_on: tuple[str, ...] = ()
    shutdown_signal: signal.Signals = signal.SIGTERM
    shutdown_terminate_after: float | None = None
    shutdown_kill_after: float = 3.0
    unavailable_error: str | None = None
    warnings: tuple[str, ...] = ()
    restart_non_retryable_exit_codes: tuple[int, ...] = ()
    resolved_runtime: ResolvedRuntime = field(
        default_factory=lambda: ResolvedRuntime(name="host", mode="host")
    )
    cpu_affinity: CpuAffinitySpec = CpuAffinitySpec(
        role=CpuAffinityRole.SHARED
    )


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
    pgid: int | None
    terminate_at: float | None
    kill_at: float


@dataclass
class _StoppingInstance:
    node_id: str
    instance: ModNode
    destroy_at: float


@dataclass
class _WorkerRequest:
    operation: Callable[..., object]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    cpu_affinity: frozenset[int] | None
    completed: Event = field(default_factory=Event)
    result: object = None
    error: BaseException | None = None


class _ModRuntimeWorker:
    """Run foreign lifecycle code outside the real-time control thread."""

    def __init__(self, baseline_affinity: frozenset[int]) -> None:
        if not baseline_affinity:
            raise ValueError("Mod runtime baseline CPU affinity must not be empty")
        self._baseline_affinity = baseline_affinity
        self._requests: Queue[_WorkerRequest | None] = Queue()
        self._ready = Event()
        self._startup_error: BaseException | None = None
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name="bxi-mod-runtime",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self._closed = True
            raise RuntimeError(
                "cannot initialize Mod runtime worker with CPU affinity "
                f"{format_cpu_set(baseline_affinity)}: {self._startup_error}"
            ) from self._startup_error

    def call(
        self,
        operation: Callable[..., object],
        *args: object,
        cpu_affinity: frozenset[int] | None,
        **kwargs: object,
    ) -> object:
        if self._closed:
            raise RuntimeError("Mod runtime worker is closed")
        request = _WorkerRequest(
            operation,
            args,
            kwargs,
            cpu_affinity,
        )
        self._requests.put(request)
        request.completed.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def spawn(
        self,
        factory: Callable[..., subprocess.Popen[bytes]],
        command: Sequence[str],
        kwargs: dict[str, object],
        cpu_affinity: frozenset[int] | None,
    ) -> subprocess.Popen[bytes]:
        process = self.call(
            factory,
            command,
            cpu_affinity=cpu_affinity,
            **kwargs,
        )
        if process is None:
            raise RuntimeError("Mod runtime process creation returned no process")
        return cast(subprocess.Popen[bytes], process)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Mod runtime worker did not stop within timeout")

    def _run(self) -> None:
        try:
            self._configure_current_thread(self._baseline_affinity)
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._ready.set()
        if self._startup_error is not None:
            return

        while True:
            request = self._requests.get()
            if request is None:
                return
            target = request.cpu_affinity or self._baseline_affinity
            try:
                self._configure_current_thread(target)
                request.result = request.operation(*request.args, **request.kwargs)
            except BaseException as exc:
                request.error = exc
            finally:
                try:
                    self._configure_current_thread(self._baseline_affinity)
                except BaseException as exc:
                    if request.error is None:
                        request.error = RuntimeError(
                            "cannot restore Mod runtime worker scheduling on "
                            f"{format_cpu_set(self._baseline_affinity)}: {exc}"
                        )
                request.completed.set()

    @staticmethod
    def _configure_current_thread(cpus: frozenset[int]) -> None:
        configure_current_thread(cpus, realtime_priority=0)


class ModNodeManager:
    """Starts, scopes, monitors and deterministically closes Mod nodes."""

    def __init__(
        self,
        specs: Sequence[ModNodeSpec],
        *,
        loggers: ScopedLoggers,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        cpu_affinity_plan: CpuAffinityPlan | None = None,
    ) -> None:
        self._specs = {spec.id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("duplicate Mod node ids")
        self._ordered_ids = self._dependency_order(specs)
        self._loggers = loggers
        self._logger = loggers.framework("mod_nodes")
        self._node_loggers = {
            spec.id: loggers.node(spec.mod_id, spec.local_name) for spec in specs
        }
        self._process_factory = process_factory
        self._cpu_affinity_plan = cpu_affinity_plan or CpuAffinityPlan.discover()
        self._resolved_cpu_affinities = {
            spec.id: (
                self._cpu_affinity_plan.resolve(
                    spec.cpu_affinity,
                    context=f"Mod node '{spec.id}' scheduling.cpu_affinity",
                )
                or self._cpu_affinity_plan.roles[CpuAffinityRole.SHARED]
            )
            for spec in specs
        }
        has_process_nodes = any(spec.execution == "process" for spec in specs)
        self._runtime_worker = None
        self._log_router = None
        if has_process_nodes:
            self._log_router = SubprocessLogRouter(
                loggers.config.subprocess,
                cpu_affinity=self._cpu_affinity_plan.roles[CpuAffinityRole.SHARED],
            )
        if specs:
            try:
                self._runtime_worker = _ModRuntimeWorker(
                    self._cpu_affinity_plan.roles[CpuAffinityRole.SHARED]
                )
            except BaseException:
                if self._log_router is not None:
                    self._log_router.close()
                    self._log_router = None
                raise
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
        for node_id in self._ordered_ids:
            spec = self._specs[node_id]
            for warning in spec.warnings:
                self._log("warning", warning, node_id=node_id)
            if spec.unavailable_error is not None:
                self._log(
                    "warning",
                    f"Mod node '{spec.id}' is unavailable: "
                    f"{spec.unavailable_error}",
                    node_id=node_id,
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
                self._run_in_process(
                    handle.spec,
                    self._add_executor_node,
                    executor,
                    handle.instance,
                )
                handle.attached = True
                attached.append(handle)
        except Exception:
            for handle in reversed(attached):
                try:
                    self._run_in_process(
                        handle.spec,
                        executor.remove_node,
                        handle.instance,
                    )
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
                self._run_in_process(
                    handle.spec,
                    executor.remove_node,
                    handle.instance,
                )
            except Exception as exc:
                self._log(
                    "warning",
                    f"failed to detach Mod node '{handle.spec.id}': {exc}",
                    node_id=handle.spec.id,
                )
            handle.attached = False

    def poll(self) -> None:
        if self._closed:
            return
        desired = self._desired_node_ids()
        now = time.monotonic()
        self._poll_stopping_processes(now)
        self._poll_stopping_instances(now)
        for node_id in self._ordered_ids:
            handle = self._running.get(node_id)
            if handle is None:
                continue
            process = handle.process
            if process is None or node_id not in desired:
                continue
            exit_code = process.poll()
            if exit_code is None:
                continue
            stopping = self._make_stopping_process(node_id, process, handle.spec)
            if self._stopping_process_alive(stopping):
                self._signal_stopping_process(
                    stopping,
                    handle.spec.shutdown_signal,
                )
                self._stopping_processes[node_id] = stopping
            handle.process = None
            handle.last_exit_code = exit_code
            if exit_code in handle.spec.restart_non_retryable_exit_codes:
                message = (
                    f"Mod node '{node_id}' exited with non-retryable code "
                    f"{exit_code}; restart suppressed"
                )
                self._faults[node_id] = message
                self._fault_attempts[node_id] = handle.restart_attempts
                self._running.pop(node_id, None)
                self._log("error", message, node_id=node_id)
                self._fault_dependents(node_id, message)
                continue
            if handle.restart_attempts >= handle.spec.restart_max_attempts:
                message = (
                    f"Mod node '{node_id}' exited with code {exit_code}; "
                    "restart limit reached"
                )
                self._faults[node_id] = message
                self._fault_attempts[node_id] = handle.restart_attempts
                self._running.pop(node_id, None)
                self._log("error", message, node_id=node_id)
                self._fault_dependents(node_id, message)
                continue
            handle.restart_attempts += 1
            handle.next_restart_at = now + handle.spec.restart_delay
            self._log(
                "warning",
                f"Mod node '{node_id}' exited with code {exit_code}; "
                f"restart {handle.restart_attempts}/"
                f"{handle.spec.restart_max_attempts} scheduled",
                node_id=node_id,
            )

        for node_id in self._ordered_ids:
            handle = self._running.get(node_id)
            if handle is None:
                continue
            if (
                node_id not in desired
                or handle.spec.execution != "process"
                or handle.process is not None
                or node_id in self._stopping_processes
                or now < handle.next_restart_at
            ):
                continue
            try:
                self._ensure_dependencies_running(handle.spec)
                handle.process = self._spawn_process(handle.spec)
                self._log("info", f"restarted Mod node '{node_id}'", node_id=node_id)
            except Exception as exc:
                if handle.restart_attempts >= handle.spec.restart_max_attempts:
                    message = f"Mod node '{node_id}' restart failed: {exc}"
                    self._faults[node_id] = message
                    self._fault_attempts[node_id] = handle.restart_attempts
                    self._running.pop(node_id, None)
                    self._log("error", message, node_id=node_id)
                    self._fault_dependents(node_id, message)
                else:
                    handle.restart_attempts += 1
                    handle.next_restart_at = now + handle.spec.restart_delay
                    self._log(
                        "warning",
                        f"Mod node '{node_id}' restart failed: {exc}",
                        node_id=node_id,
                    )

    def snapshot(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        desired = self._desired_node_ids()
        for node_id in self._ordered_ids:
            spec = self._specs[node_id]
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
                    "depends_on": list(spec.depends_on),
                    "status": status,
                    "restart_attempts": (
                        handle.restart_attempts
                        if handle
                        else self._fault_attempts.get(spec.id, 0)
                    ),
                    "error": spec.unavailable_error or self._faults.get(spec.id),
                    "warnings": list(spec.warnings),
                    "runtime_profile": spec.resolved_runtime.name,
                    "runtime_mode": spec.resolved_runtime.mode,
                    "runtime_root": (
                        str(spec.resolved_runtime.root)
                        if spec.resolved_runtime.root is not None
                        else None
                    ),
                    "cpu_affinity": spec.cpu_affinity.label,
                    "resolved_cpu_affinity": (
                        format_cpu_set(self._resolved_cpu_affinities[spec.id])
                        if self._resolved_cpu_affinities[spec.id] is not None
                        else None
                    ),
                    **spec.manifest,
                }
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.detach_executor()
        for node_id in reversed(self._ordered_ids):
            self._stop_node(node_id, wait=True)
        for stopping in tuple(self._stopping_processes.values()):
            if not self._wait_for_stopping_process(stopping):
                self._log(
                    "error",
                    f"Mod node '{stopping.node_id}' still has live processes "
                    "after SIGKILL",
                    node_id=stopping.node_id,
                )
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
        if self._runtime_worker is not None:
            self._runtime_worker.close()
        if self._log_router is not None:
            self._log_router.close()

    def _desired_node_ids(self) -> set[str]:
        scoped_states = self._active_states | self._prepared_states
        desired = {
            spec.id
            for spec in self._specs.values()
            if spec.unavailable_error is None
            and (spec.lifecycle == "mod" or bool(set(spec.states) & scoped_states))
        }
        pending = list(desired)
        while pending:
            node_id = pending.pop()
            for dependency in self._specs[node_id].depends_on:
                if (
                    self._specs[dependency].unavailable_error is None
                    and dependency not in desired
                ):
                    desired.add(dependency)
                    pending.append(dependency)
        return desired

    def _reconcile(self) -> None:
        if self._closed:
            raise RuntimeError("Mod node manager is closed")
        desired = self._desired_node_ids()
        for node_id in tuple(self._faults):
            if node_id not in desired:
                self._faults.pop(node_id, None)
                self._fault_attempts.pop(node_id, None)
        for node_id in reversed(self._ordered_ids):
            if node_id not in desired:
                self._stop_node(node_id)
        for node_id in self._ordered_ids:
            if node_id not in desired:
                continue
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
            self._ensure_dependencies_running(spec)
            process = self._spawn_process(spec)
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' exited during startup with "
                    f"code {exit_code}"
                )
            self._running[spec.id] = _RunningNode(spec=spec, process=process)
            self._log(
                "info",
                f"started process Mod node '{spec.id}'",
                node_id=spec.id,
            )
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
            instance = cast(
                ModNode,
                self._run_in_process(spec, factory, context),
            )
            if not callable(getattr(instance, "destroy_node", None)):
                raise TypeError(
                    f"Mod node entrypoint '{spec.entrypoint}' must return "
                    "an rclpy Node"
                )
            if self._executor is not None:
                self._run_in_process(
                    spec,
                    self._add_executor_node,
                    self._executor,
                    instance,
                )
                attached = True
        except Exception:
            if attached and self._executor is not None and instance is not None:
                try:
                    self._run_in_process(
                        spec,
                        self._executor.remove_node,
                        instance,
                    )
                except Exception:
                    pass
            if instance is not None:
                try:
                    self._run_in_process(spec, instance.destroy_node)
                except Exception:
                    pass
            raise
        self._running[spec.id] = _RunningNode(
            spec=spec,
            instance=instance,
            attached=attached,
        )
        self._log(
            "info",
            f"started in-process Mod node '{spec.id}'",
            node_id=spec.id,
        )

    def _spawn_process(self, spec: ModNodeSpec) -> subprocess.Popen[bytes]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["BXI_MOD_ID"] = spec.mod_id
        environment["BXI_NODE_ID"] = spec.id
        environment["BXI_LOG_SCOPE"] = (
            f"mod.{spec.mod_id}.node.{spec.local_name}"
        )
        spec.resolved_runtime.apply_environment(environment)
        if spec.runtime == "python" or not spec.resolved_runtime.isolated:
            inherited_paths: list[str] = []
            if spec.resolved_runtime.isolated:
                inherited_paths.extend(self._framework_python_paths())
            else:
                inherited_paths.extend(str(path) for path in sys.path if path)
            existing_python_path = environment.get("PYTHONPATH")
            if existing_python_path:
                inherited_paths.extend(existing_python_path.split(os.pathsep))
            environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(inherited_paths))
        environment["BXI_MOD_ROOT"] = str(spec.mod_root)
        self._apply_environment(spec, environment)
        if spec.runtime == "python":
            python_executable = spec.resolved_runtime.python_executable or Path(
                sys.executable
            )
            command = [
                str(python_executable),
                "-m",
                "bxi_example_py_elf3.framework.runtime.mod_node_runner",
                "--manifest",
                str(spec.manifest_path),
                "--node",
                spec.local_name,
            ]
            cwd = None
        elif spec.runtime == "command":
            executable = spec.executable_path
            if executable is None:
                raise RuntimeError(f"Mod node '{spec.id}' has no resolved command")
            command = []
            if spec.interpreter is not None:
                if spec.interpreter == "bundled-python":
                    python_executable = spec.resolved_runtime.python_executable
                    if python_executable is None:
                        raise RuntimeError(
                            f"Mod node '{spec.id}' requested bundled-python but "
                            f"runtime profile '{spec.resolved_runtime.name}' does "
                            "not provide Python"
                        )
                    command.append(str(python_executable))
                else:
                    command.append(
                        self._resolve_interpreter(
                            self._expand_environment(spec.interpreter, environment),
                            spec.id,
                            environment,
                        )
                    )
            command.extend(
                (
                    str(executable),
                    *(
                        self._expand_environment(argument, environment)
                        for argument in spec.arguments
                    ),
                )
            )
            cwd = str(spec.cwd or spec.mod_root)
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
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
        }
        if cwd is not None:
            kwargs["cwd"] = cwd
        worker = self._runtime_worker
        if worker is None:
            raise RuntimeError("Mod runtime worker is not available")
        process = worker.spawn(
            self._process_factory,
            command,
            kwargs,
            self._resolved_cpu_affinities[spec.id],
        )
        if self._log_router is None:
            raise RuntimeError("Mod subprocess log router is not available")
        self._log_router.register(
            process,
            mod_id=spec.mod_id,
            node_name=spec.local_name,
        )
        return process

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

    def _run_in_process(
        self,
        spec: ModNodeSpec,
        operation: Callable[..., object],
        *args: object,
    ) -> object:
        worker = self._runtime_worker
        if worker is None:
            raise RuntimeError("Mod runtime worker is not available")
        return worker.call(
            operation,
            *args,
            cpu_affinity=self._resolved_cpu_affinities[spec.id],
        )

    def _stop_node(self, node_id: str, *, wait: bool = False) -> None:
        handle = self._running.pop(node_id, None)
        if handle is None:
            return
        if handle.instance is not None:
            if handle.attached and self._executor is not None:
                try:
                    self._run_in_process(
                        handle.spec,
                        self._executor.remove_node,
                        handle.instance,
                    )
                except Exception as exc:
                    self._log(
                        "warning",
                        f"failed to remove Mod node '{node_id}': {exc}",
                        node_id=node_id,
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
        if process is not None:
            stopping = self._make_stopping_process(node_id, process, handle.spec)
            if not self._stopping_process_alive(stopping):
                stopping = None
            else:
                self._signal_stopping_process(
                    stopping,
                    handle.spec.shutdown_signal,
                )
        else:
            stopping = None
        if stopping is not None:
            if wait:
                if not self._wait_for_stopping_process(stopping):
                    self._log(
                        "error",
                        f"Mod node '{node_id}' still has live processes after SIGKILL",
                        node_id=node_id,
                    )
            else:
                self._stopping_processes[node_id] = stopping
        self._log("info", f"stopped Mod node '{node_id}'", node_id=node_id)

    def _poll_stopping_processes(self, now: float) -> None:
        for node_id, stopping in tuple(self._stopping_processes.items()):
            if not self._stopping_process_alive(stopping):
                self._stopping_processes.pop(node_id, None)
                continue
            if stopping.terminate_at is not None and now >= stopping.terminate_at:
                self._signal_stopping_process(stopping, signal.SIGTERM)
                stopping.terminate_at = None
            if now < stopping.kill_at:
                continue
            self._signal_stopping_process(stopping, signal.SIGKILL)
            stopping.kill_at = float("inf")
            self._log(
                "warning",
                f"killed unresponsive Mod node '{node_id}'",
                node_id=node_id,
            )

    def _poll_stopping_instances(self, now: float) -> None:
        for node_id, stopping in tuple(self._stopping_instances.items()):
            if now < stopping.destroy_at:
                continue
            self._destroy_instance(node_id, stopping.instance)
            self._stopping_instances.pop(node_id, None)

    def _destroy_instance(self, node_id: str, instance: ModNode) -> None:
        try:
            self._run_in_process(
                self._specs[node_id],
                instance.destroy_node,
            )
        except Exception as exc:
            self._log(
                "warning",
                f"failed to destroy Mod node '{node_id}': {exc}",
                node_id=node_id,
            )

    @classmethod
    def _wait_for_stopping_process(cls, stopping: _StoppingProcess) -> bool:
        final_deadline: float | None = None
        while cls._stopping_process_alive(stopping):
            now = time.monotonic()
            if stopping.kill_at == float("inf"):
                if final_deadline is None:
                    cls._signal_stopping_process(stopping, signal.SIGKILL)
                    final_deadline = now + 3.0
                elif now >= final_deadline:
                    return False
                time.sleep(0.05)
                continue
            if stopping.terminate_at is not None and now >= stopping.terminate_at:
                cls._signal_stopping_process(stopping, signal.SIGTERM)
                stopping.terminate_at = None
                continue
            if now >= stopping.kill_at:
                cls._signal_stopping_process(stopping, signal.SIGKILL)
                stopping.kill_at = float("inf")
                final_deadline = now + 3.0
                continue
            deadlines = [stopping.kill_at]
            if stopping.terminate_at is not None:
                deadlines.append(stopping.terminate_at)
            time.sleep(min(0.05, max(0.0, min(deadlines) - now)))
        return True

    @staticmethod
    def _make_stopping_process(
        node_id: str,
        process: subprocess.Popen[bytes],
        spec: ModNodeSpec,
    ) -> _StoppingProcess:
        now = time.monotonic()
        pid = getattr(process, "pid", None)
        pgid = (
            pid
            if isinstance(process, subprocess.Popen)
            and isinstance(pid, int)
            and pid > 1
            and pid != os.getpgrp()
            else None
        )
        return _StoppingProcess(
            node_id=node_id,
            process=process,
            pgid=pgid,
            terminate_at=(
                now + spec.shutdown_terminate_after
                if spec.shutdown_terminate_after is not None
                else None
            ),
            kill_at=now + spec.shutdown_kill_after,
        )

    @staticmethod
    def _stopping_process_alive(stopping: _StoppingProcess) -> bool:
        leader_alive = stopping.process.poll() is None
        if stopping.pgid is None:
            return leader_alive
        try:
            os.killpg(stopping.pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    @classmethod
    def _signal_stopping_process(
        cls,
        stopping: _StoppingProcess,
        value: signal.Signals,
    ) -> None:
        if stopping.pgid is not None:
            try:
                os.killpg(stopping.pgid, value)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
            return
        cls._signal_process(stopping.process, value)

    @staticmethod
    def _signal_process(
        process: subprocess.Popen[bytes], value: signal.Signals
    ) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(process, subprocess.Popen) and isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, value)
                return
            except ProcessLookupError:
                return
            except PermissionError:
                return
        send_signal = getattr(process, "send_signal", None)
        if callable(send_signal):
            send_signal(value)
        elif value == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    def _ensure_dependencies_running(self, spec: ModNodeSpec) -> None:
        for dependency in spec.depends_on:
            dependency_spec = self._specs[dependency]
            if dependency_spec.unavailable_error is not None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' dependency '{dependency}' is unavailable: "
                    f"{dependency_spec.unavailable_error}"
                )
            handle = self._running.get(dependency)
            if handle is None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' dependency '{dependency}' is not running"
                )
            if dependency_spec.execution == "process" and handle.process is None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' dependency '{dependency}' is restarting"
                )
            if handle.process is not None and handle.process.poll() is not None:
                raise RuntimeError(
                    f"Mod node '{spec.id}' dependency '{dependency}' has exited"
                )

    def _fault_dependents(self, failed_node_id: str, reason: str) -> None:
        affected = {failed_node_id}
        for node_id in self._ordered_ids:
            if set(self._specs[node_id].depends_on) & affected:
                affected.add(node_id)
        for node_id in reversed(self._ordered_ids):
            if node_id == failed_node_id or node_id not in affected:
                continue
            self._stop_node(node_id)
            message = (
                f"Mod node '{node_id}' stopped because dependency "
                f"'{failed_node_id}' faulted: {reason}"
            )
            self._faults[node_id] = message
            self._fault_attempts[node_id] = 0
            self._log("error", message, node_id=node_id)

    @staticmethod
    def _dependency_order(specs: Sequence[ModNodeSpec]) -> tuple[str, ...]:
        by_id = {spec.id: spec for spec in specs}
        order: list[str] = []
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                cycle = visiting[visiting.index(node_id) :] + [node_id]
                raise ValueError("Mod node dependency cycle: " + " -> ".join(cycle))
            visiting.append(node_id)
            for dependency in by_id[node_id].depends_on:
                if dependency not in by_id:
                    raise ValueError(
                        f"Mod node '{node_id}' depends on unknown node "
                        f"'{dependency}'"
                    )
                visit(dependency)
            visiting.pop()
            visited.add(node_id)
            order.append(node_id)

        for spec in specs:
            visit(spec.id)
        return tuple(order)

    @classmethod
    def _apply_environment(
        cls,
        spec: ModNodeSpec,
        environment: dict[str, str],
    ) -> None:
        for name, edit in spec.environment.items():
            if edit.unset:
                environment.pop(name, None)
                continue
            if edit.value is not None:
                environment[name] = cls._expand_environment(edit.value, environment)
            current = environment.get(name, "")
            prepend = [
                cls._expand_environment(value, environment) for value in edit.prepend
            ]
            append = [
                cls._expand_environment(value, environment) for value in edit.append
            ]
            if edit.existing_only:
                prepend = [value for value in prepend if os.path.exists(value)]
                append = [value for value in append if os.path.exists(value)]
            parts = [*prepend]
            if current:
                parts.append(current)
            parts.extend(append)
            if edit.prepend or edit.append:
                environment[name] = edit.separator.join(
                    value for value in parts if value
                )

    @staticmethod
    def _expand_environment(value: str, environment: Mapping[str, str]) -> str:
        pattern = re.compile(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
            r"|\$([A-Za-z_][A-Za-z0-9_]*)"
        )

        def replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(3)
            default = match.group(2)
            current = environment.get(name, "")
            return current if current else (default or "")

        return pattern.sub(replace, value)

    @staticmethod
    def _resolve_interpreter(
        value: str,
        node_id: str,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        if not value:
            raise RuntimeError(f"Mod node '{node_id}' resolved an empty interpreter")
        if os.sep in value:
            candidate = Path(value).expanduser()
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise RuntimeError(
                    f"Mod node '{node_id}' interpreter is not executable: {value}"
                )
            return str(candidate)
        resolved = shutil.which(
            value,
            path=(environment or {}).get("PATH"),
        )
        if resolved is None:
            raise RuntimeError(
                f"Mod node '{node_id}' interpreter was not found: {value}"
            )
        return resolved

    @staticmethod
    def _framework_python_paths() -> tuple[str, ...]:
        """Return only paths needed to import the dedicated node runner.

        A portable isolated runtime must not regain the complete host
        ``sys.path`` merely because ``runtime: python`` uses the framework's
        process runner.
        """

        result: list[str] = []
        relative_runner = Path(
            "bxi_example_py_elf3/framework/runtime/mod_node_runner.py"
        )
        for raw_path in sys.path:
            if not raw_path:
                continue
            path = Path(raw_path).resolve()
            if (path / relative_runner).is_file():
                result.append(str(path))
        return tuple(dict.fromkeys(result))

    def _log(
        self,
        level: str,
        message: str,
        *,
        node_id: str | None = None,
    ) -> None:
        logger = self._logger if node_id is None else self._node_loggers[node_id]
        # rclpy identifies a Python log call by its source location and rejects
        # later calls from that location with a different severity.
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            raise ValueError(f"unsupported Mod node log level: {level}")


__all__ = ["EnvironmentEdit", "ExecutorLike", "ModNodeManager", "ModNodeSpec"]
