import importlib
import importlib.util
import os
import sys
import time
from threading import Lock, Thread

import bxi_example_py_elf3.robot_states as robot_states_module
from bxi_example_py_elf3.utils.robot_state_builder import build_robot_states
from bxi_example_py_elf3.utils.state_machine import (
    RemoteEventAdapter,
    RobotStateMachine,
    load_state_machine_config,
)


class HotReloadModelContext:
    def __init__(self, source):
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_assigned", {})

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __setattr__(self, name, value):
        self._assigned[name] = value

    @property
    def assigned(self):
        return self._assigned


class HotReloadMixin:
    def demo_module_path(self):
        return os.path.abspath(self.hot_reload_module_path)

    def init_hot_reload(self):
        self.hot_reload_interval = 1.0
        self.hot_reload_elapsed = 0.0
        self.hot_reload_mtimes = self.current_hot_reload_mtimes()
        self.hot_reload_model_lock = Lock()
        self.hot_reload_model_thread = None
        self.hot_reload_model_result = None

    def current_hot_reload_mtimes(self):
        mtimes = {}
        paths = [
            self.demo_module_path(),
            self.state_machine_config_path,
            getattr(robot_states_module, "__file__", ""),
        ]
        paths.extend(getattr(self, "model_file_paths", ()))
        for path in paths:
            if path and os.path.exists(path):
                path = os.path.abspath(path)
                mtimes[path] = os.path.getmtime(path)
        return mtimes

    def check_hot_reload(self, dt):
        if not self.hot_reload_enabled:
            return

        if self.apply_pending_model_reload():
            return

        if self.hot_reload_model_thread is not None:
            return

        self.hot_reload_elapsed += dt
        if self.hot_reload_elapsed < self.hot_reload_interval:
            return
        self.hot_reload_elapsed = 0.0

        mtimes = self.current_hot_reload_mtimes()
        if mtimes == self.hot_reload_mtimes:
            return
        if self.state_machine.in_transition:
            return

        changed_paths = {
            path
            for path in set(mtimes) | set(self.hot_reload_mtimes)
            if mtimes.get(path) != self.hot_reload_mtimes.get(path)
        }
        model_paths = {
            os.path.abspath(path)
            for path in getattr(self, "model_file_paths", ())
        }
        demo_paths = {self.demo_module_path()}
        demo_changed = any(path in demo_paths for path in changed_paths)
        model_changed = any(path in model_paths for path in changed_paths)
        state_machine_changed = any(
            path not in model_paths and path not in demo_paths
            for path in changed_paths
        )

        if state_machine_changed:
            reloaded = self.reload_state_machine()
            if reloaded and (model_changed or demo_changed):
                self.start_model_reload(reload_demo_code=demo_changed)
        else:
            self.start_model_reload(reload_demo_code=demo_changed)
            reloaded = False

        if reloaded:
            self.hot_reload_mtimes = self.current_hot_reload_mtimes()

    def load_demo_class_from_file(self):
        module_name = f"{__name__}_hot_reload_{os.getpid()}_{time.time_ns()}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            self.demo_module_path(),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module from {self.demo_module_path()}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)

        demo_class = getattr(module, self.__class__.__name__, None)
        if demo_class is None:
            raise RuntimeError(
                f"loaded module does not define {self.__class__.__name__}"
            )
        return demo_class

    def adopt_demo_class(self, demo_class):
        if self.__class__ is demo_class:
            return
        try:
            self.__class__ = demo_class
        except TypeError as exc:
            self.get_logger().warning(f"hot reload kept old node class: {exc}")

    def build_model_reload_attrs(self, demo_class=None):
        context = HotReloadModelContext(self)
        load_models = self.__class__.load_models
        if demo_class is not None:
            load_models = demo_class.load_models
        load_models(context)
        return dict(context.assigned)

    def start_model_reload(self, reload_demo_code=False):
        if (
            self.hot_reload_model_thread is not None
            and self.hot_reload_model_thread.is_alive()
        ):
            return

        with self.hot_reload_model_lock:
            self.hot_reload_model_result = None

        thread = Thread(
            target=self.run_model_reload,
            args=(reload_demo_code,),
            daemon=True,
        )
        self.hot_reload_model_thread = thread
        thread.start()

    def run_model_reload(self, reload_demo_code=False):
        try:
            demo_class = self.load_demo_class_from_file() if reload_demo_code else None
            model_attrs = self.build_model_reload_attrs(demo_class)
            result = {
                "ok": True,
                "reload_demo_code": reload_demo_code,
                "demo_class": demo_class,
                "model_attrs": model_attrs,
            }
        except Exception as exc:
            result = {
                "ok": False,
                "error": exc,
            }

        with self.hot_reload_model_lock:
            self.hot_reload_model_result = result

    def apply_pending_model_reload(self):
        thread = self.hot_reload_model_thread
        if thread is None:
            return False

        with self.hot_reload_model_lock:
            result = self.hot_reload_model_result

        if result is None:
            if thread.is_alive():
                return False
            thread.join(timeout=0.0)
            self.hot_reload_model_thread = None
            return False

        if self.state_machine.in_transition:
            return False

        thread.join(timeout=0.0)
        self.hot_reload_model_thread = None
        with self.hot_reload_model_lock:
            self.hot_reload_model_result = None

        if not result["ok"]:
            self.get_logger().error(f"hot reload models failed: {result['error']}")
            return False

        for attr_name, value in result["model_attrs"].items():
            setattr(self, attr_name, value)

        demo_class = result.get("demo_class")
        if demo_class is not None:
            self.adopt_demo_class(demo_class)

        for state in self.robot_states.values():
            if hasattr(state, "policy_configured"):
                state.policy_configured = False
        reload_message = "hot reloaded models"
        if result.get("reload_demo_code"):
            reload_message = "hot reloaded demo code and models"
        self.get_logger().info(reload_message)
        self.hot_reload_mtimes = self.current_hot_reload_mtimes()
        return True

    def reload_state_machine(self):
        current_state_name = self.state_machine.current_state_name
        try:
            importlib.invalidate_caches()
            importlib.reload(robot_states_module)
            state_machine_config = load_state_machine_config(
                self.state_machine_config_path
            )

            runtime_config = dict(state_machine_config)
            if current_state_name in (state_machine_config.get("states") or {}):
                runtime_config["initial_state"] = current_state_name

            robot_states = build_robot_states(runtime_config)
            self.bind_robot_states(robot_states)
            state_machine = RobotStateMachine(
                self,
                runtime_config,
                robot_states,
            )
            previous_remote_values = getattr(
                self.remote_event_adapter,
                "_last_values",
                {},
            )
            remote_event_adapter = RemoteEventAdapter(
                runtime_config.get("remote_events", {}),
                initial_values=previous_remote_values,
            )
        except Exception as exc:
            self.get_logger().error(f"hot reload failed: {exc}")
            return False

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
        self.state = self.state_machine.current_state_id
        self.pending_remote_events.clear()
        self.get_logger().info(
            f"hot reloaded state machine at state '{self.state_machine.current_state_name}'"
        )
        return True
