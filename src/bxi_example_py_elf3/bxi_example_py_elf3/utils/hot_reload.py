from __future__ import annotations

import importlib
from pathlib import Path

from bxi_example_py_elf3.utils.mod_system import ModRuntime
from bxi_example_py_elf3.utils.robot_state_builder import build_robot_states
from bxi_example_py_elf3.utils.state_machine import (
    RemoteEventAdapter,
    RobotStateMachine,
    load_state_machine_config,
)


class HotReloadMixin:
    """Atomically rebuilds all Mod contributions when a Mod changes on disk."""

    def log_mod_runtime(self, runtime: ModRuntime, *, action: str) -> None:
        logger = self.get_logger()
        remote_events = runtime.config.get("remote_events")
        remote_event_count = (
            len(remote_events) if isinstance(remote_events, dict) else 0
        )
        logger.info(
            f"{action} {len(runtime.mods)} Mods, "
            f"{len(runtime.disabled_mods)} disabled, "
            f"{len(runtime.state_factories)} states, "
            f"{remote_event_count} remote events; input conflicts validated"
        )
        for mod in runtime.mods:
            dependencies = (
                f"; requires={','.join(mod.requires)}" if mod.requires else ""
            )
            logger.info(f"Mod {mod.id}@{mod.version}: {mod.root}{dependencies}")
        for mod in runtime.disabled_mods:
            logger.info(f"Mod {mod.id}@{mod.version}: disabled; {mod.root}")

    def init_hot_reload(self) -> None:
        self.hot_reload_interval = 1.0
        self.hot_reload_elapsed = 0.0
        self.hot_reload_mtimes = self.current_hot_reload_mtimes()

    def current_hot_reload_mtimes(self) -> dict[str, int]:
        paths: set[Path] = {Path(self.state_machine_config_path)}
        built_in_root, extra_roots = self.mod_search_roots(
            self.base_state_machine_config
        )
        for root in (built_in_root, *extra_roots):
            if not root.exists():
                continue
            paths.add(root)
            paths.update(path for path in root.rglob("*") if path.is_file())
        return {
            str(path.resolve()): path.stat().st_mtime_ns
            for path in paths
            if path.exists()
        }

    def check_hot_reload(self, dt: float) -> None:
        if not self.hot_reload_enabled:
            return
        self.hot_reload_elapsed += dt
        if self.hot_reload_elapsed < self.hot_reload_interval:
            return
        self.hot_reload_elapsed = 0.0
        mtimes = self.current_hot_reload_mtimes()
        if mtimes == self.hot_reload_mtimes or self.state_machine.in_transition:
            return
        if self.reload_mod_runtime():
            self.hot_reload_mtimes = self.current_hot_reload_mtimes()

    def reload_mod_runtime(self) -> bool:
        old_runtime = self.mod_runtime
        old_state_machine = self.state_machine
        old_resources = self.resources
        current_state_name = old_state_machine.current_state_name
        runtime = None
        robot_states = None
        try:
            importlib.invalidate_caches()
            base_config = load_state_machine_config(self.state_machine_config_path)
            runtime = self.load_mod_runtime(base_config)
            runtime_config = dict(runtime.config)
            if current_state_name in (runtime.config.get("states") or {}):
                runtime_config["initial_state"] = current_state_name
            robot_states = build_robot_states(
                runtime_config,
                runtime.state_factories,
            )
            self.resources = runtime.resources
            self.bind_robot_states(robot_states)
            state_machine = RobotStateMachine(
                self,
                runtime_config,
                robot_states,
                enter_initial=False,
            )
            previous_values = getattr(self.remote_event_adapter, "_last_values", {})
            remote_event_adapter = RemoteEventAdapter(
                runtime_config.get("remote_events", {}),
                initial_values=previous_values,
            )
            self.state_machine = state_machine
            self.resources = runtime.resources
            state_machine.current.on_enter(self)
        except Exception as exc:
            if robot_states is not None:
                for state in robot_states.values():
                    try:
                        state.on_unbind(self)
                    except Exception as cleanup_exc:
                        self.get_logger().warning(
                            f"new Mod state cleanup failed: {cleanup_exc}"
                        )
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as cleanup_exc:
                    self.get_logger().warning(
                        f"new Mod runtime cleanup failed: {cleanup_exc}"
                    )
            self.state_machine = old_state_machine
            self.resources = old_resources
            self.get_logger().error(f"Mod hot reload failed: {exc}")
            return False

        assert runtime is not None
        assert robot_states is not None
        self.state_machine = old_state_machine
        self.resources = old_resources
        for state in self.robot_states.values():
            try:
                state.on_unbind(self)
            except Exception as exc:
                self.get_logger().warning(f"old Mod state cleanup failed: {exc}")
        try:
            old_runtime.close()
        except Exception as exc:
            self.get_logger().warning(f"old Mod runtime cleanup failed: {exc}")
        self.base_state_machine_config = base_config
        self.mod_runtime = runtime
        self.resources = runtime.resources
        self.state_machine_config = runtime_config
        self.speed_profiles = runtime_config.get("speed_profiles", {})
        self.robot_states = robot_states
        self.state_id_by_name = {
            name: state.state_id for name, state in robot_states.items()
        }
        self.state_name_by_id = {
            value: key for key, value in self.state_id_by_name.items()
        }
        self.state_machine = state_machine
        self.remote_event_adapter = remote_event_adapter
        self.state = state_machine.current_state_id
        self.pending_remote_events.clear()
        self.log_mod_runtime(runtime, action="hot reloaded")
        self.get_logger().info(
            "hot reloaded Mods at state " f"'{self.state_machine.current_state_name}'"
        )
        return True
