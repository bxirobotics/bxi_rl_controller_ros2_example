"""Control framework driven by a thin platform adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.joints import (
    JointCommandDefaults,
    JointCommandResolver,
    JointLayout,
    JointStateView,
)
from bxi_example_py_elf3.framework.inference import InferenceFrame, default_runtime
from bxi_example_py_elf3.framework.mod_api import (
    MotorFrame,
    RobotControlState,
    TransitionSpec,
)
from bxi_example_py_elf3.framework.mod_api.geometry import quaternion_to_euler_array
from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    CpuAffinityPlan,
    CpuAffinityRole,
)

from .mod_loader import ModRuntime, load_mod_runtime
from .logging import ScopedLoggers
from .mod_nodes import ExecutorLike, ModNodeManager
from .state_builder import build_robot_states
from .state_machine import RemoteEventAdapter, RobotStateMachine

if TYPE_CHECKING:
    from rclpy.node import Node
    from bxi_example_py_elf3.framework.platform.api import RobotObservation

class RobotControlFramework:
    """Own Mods, states, transitions and one control-cycle output frame."""

    def __init__(
        self,
        base_config: Mapping[str, object],
        *,
        built_in_mod_root: Path,
        extra_mod_roots: Sequence[Path] | None = None,
        command_defaults: JointCommandDefaults,
        ros_node: Node,
        loggers: ScopedLoggers,
        control_period: float = 0.02,
        cpu_affinity_plan: CpuAffinityPlan | None = None,
    ) -> None:
        self._ros_node = ros_node
        self._loggers = loggers
        self._logger = loggers.framework("controller")
        default_runtime().set_logger(loggers.framework("inference"))
        self._closed = True
        if control_period <= 0.0:
            raise ValueError("control_period must be greater than zero")
        self._default_control_period = float(control_period)
        self.dt = self._default_control_period
        self.loop_count = 0

        self.current_quat_xyzw = np.zeros(4, dtype=np.float64)
        self.current_quat_wxyz = np.zeros(4, dtype=np.float64)
        self.current_omega = np.zeros(3, dtype=np.float64)
        self.current_raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.current_cmd_vel = np.zeros(3, dtype=np.float32)
        self._command_defaults = command_defaults
        self._robot_layout: JointLayout | None = None
        self._robot_joints: JointStateView | None = None
        self._inference_frame: InferenceFrame | None = None
        self._command_resolver: JointCommandResolver | None = None
        self._resolved_motor_frame: MotorFrame | None = None
        self._last_motor_frame: MotorFrame | None = None
        self._direct_motor_layout: JointLayout | None = None
        self._motor_target: MotorFrame | None = None
        self._pending_state_requests: list[
            tuple[str, str, TransitionSpec, float, bool]
        ] = []

        runtime: ModRuntime | None = None
        node_manager: ModNodeManager | None = None
        states_bound = False
        try:
            if extra_mod_roots is None:
                raw_mod_paths = base_config.get("mod_paths", ())
                if not isinstance(raw_mod_paths, list) or not all(
                    isinstance(path, str) for path in raw_mod_paths
                ):
                    raise ValueError("mod_paths must be a list of directory strings")
                extra_mod_roots = tuple(Path(path) for path in raw_mod_paths)
            runtime = load_mod_runtime(
                base_config,
                built_in_root=built_in_mod_root,
                extra_roots=extra_mod_roots,
                resource_cpu_affinity=(
                    cpu_affinity_plan.roles[CpuAffinityRole.COMPUTE]
                    if cpu_affinity_plan is not None
                    else None
                ),
            )
            self.mod_runtime = runtime
            self.resources = runtime.resources
            self.config = runtime.config
            self.speed_profiles = self.config.get("speed_profiles", {})
            node_manager = ModNodeManager(
                runtime.node_specs,
                loggers=loggers,
                cpu_affinity_plan=cpu_affinity_plan,
            )
            self.node_manager = node_manager
            node_manager.start()
            for mod in (*runtime.mods, *runtime.unavailable_mods):
                mod_logger = loggers.mod(mod.id)
                for warning in mod.warnings:
                    mod_logger.warning(warning)

            states = build_robot_states(self.config, runtime.state_factories)
            self.robot_states = states
            self.state_id_by_name = {
                name: state.state_id for name, state in states.items()
            }
            self.state_name_by_id = {
                state_id: name for name, state_id in self.state_id_by_name.items()
            }
            self._bind_states(states)
            states_bound = True
            raw_initial = self.config.get("initial_state")
            initial_state = (
                str(raw_initial) if raw_initial is not None else next(iter(states))
            )
            if initial_state not in states:
                raise ValueError(f"unknown initial_state: {initial_state}")
            initial_resources = states[initial_state].required_resources
            unavailable_initial_resources = [
                resource.key.id
                for resource in initial_resources
                if resource.status != "ready"
            ]
            if unavailable_initial_resources:
                raise ValueError(
                    f"initial state '{initial_state}' requires resources that are "
                    "not policy='startup': "
                    f"{unavailable_initial_resources}"
                )
            node_manager.activate_initial_state(initial_state)
            self.state_machine = RobotStateMachine(
                self,
                self.config,
                states,
                node_lifecycle=node_manager,
                logger=loggers.framework("state_machine"),
                enter_initial=False,
            )
            self._initial_state_entered = False
            self.remote_event_adapter = RemoteEventAdapter(
                self.config.get("remote_events", {})
            )
        except BaseException:
            if states_bound:
                try:
                    self._unbind_states(self.robot_states)
                except Exception as cleanup_exc:
                    self._warn_cleanup_failure("state", cleanup_exc)
            if node_manager is not None:
                try:
                    node_manager.close()
                except Exception as cleanup_exc:
                    self._warn_cleanup_failure("Mod node", cleanup_exc)
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as cleanup_exc:
                    self._warn_cleanup_failure("Mod runtime", cleanup_exc)
            raise
        self._closed = False

    @property
    def ros_node(self) -> Node:
        return self._ros_node

    @property
    def current_state_id(self) -> int:
        return self.state_machine.current_state_id

    @property
    def current_state_name(self) -> str:
        return self.state_machine.current_state_name

    @property
    def desired_control_period(self) -> float:
        """Period requested by the current state or active transition."""

        default_hz = 1.0 / self._default_control_period
        requested_hz = self.state_machine.requested_inference_hz(default_hz)
        return 1.0 / requested_hz

    @property
    def robot_layout(self) -> JointLayout:
        if self._robot_layout is None:
            raise RuntimeError("robot joint layout is not bound to an observation yet")
        return self._robot_layout

    @property
    def robot_joints(self) -> JointStateView:
        if self._robot_joints is None:
            raise RuntimeError("robot joint state is not bound to an observation yet")
        return self._robot_joints

    @property
    def inference_frame(self) -> InferenceFrame:
        if self._inference_frame is None:
            raise RuntimeError("inference frame is not bound to an observation yet")
        return self._inference_frame

    @property
    def last_motor_frame(self) -> MotorFrame:
        if self._last_motor_frame is None:
            raise RuntimeError("motor frame is not bound to a robot layout yet")
        return self._last_motor_frame

    def update(
        self,
        observation: RobotObservation,
        events: Sequence[str],
        dt: float,
    ) -> MotorFrame | None:
        """Advance the framework once and return the final motor frame."""
        if self._closed:
            raise RuntimeError("RobotControlFramework is closed")

        self.dt = float(dt)
        self._set_observation(observation)
        self._apply_pending_state_requests()
        self.current_cmd_vel.fill(0.0)
        self._motor_target = None

        transition_active = self.state_machine.update(self.dt, events)
        if not transition_active:
            self.state_machine.update_current_state(self.dt)

        frame = self._motor_target
        if frame is not None:
            self.last_motor_frame.update(
                frame.qpos,
                frame.kp,
                frame.kd,
                vel=frame.vel,
                torque=frame.torque,
            )
        self.loop_count += 1
        return frame

    def maintenance_update(self) -> None:
        """Run non-control Mod supervision outside the control data path."""
        if self._closed:
            return
        self.node_manager.poll()

    def extract_remote_events(
        self,
        values: object,
        *,
        sync_only: bool = False,
    ) -> list[str]:
        return self.remote_event_adapter.extract_events(values, sync_only=sync_only)

    def request_state(
        self,
        state_name: str,
        *,
        trigger: str,
        transition: TransitionSpec = None,
        delay: float = 0.0,
        force: bool = False,
    ) -> bool:
        if not isinstance(force, bool):
            raise TypeError("state request force must be a bool")
        if self._robot_layout is None:
            self._pending_state_requests.append(
                (state_name, trigger, transition, float(delay), force)
            )
            return True
        return self.state_machine.request_transition(
            state_name,
            trigger=trigger,
            transition=transition,
            delay=delay,
            force=force,
        )

    def _apply_pending_state_requests(self) -> None:
        if not self._pending_state_requests:
            return
        pending = tuple(self._pending_state_requests)
        self._pending_state_requests.clear()
        for state_name, trigger, transition, delay, force in pending:
            self.state_machine.request_transition(
                state_name,
                trigger=trigger,
                transition=transition,
                delay=delay,
                force=force,
            )

    def set_motor_target(self, frame: MotorFrame) -> None:
        if frame.layout is self._direct_motor_layout:
            self._motor_target = frame
            return
        if (
            frame.layout is self.robot_layout
            or frame.layout.names == self.robot_layout.names
        ):
            self._direct_motor_layout = frame.layout
            self._motor_target = frame
            return
        output = self._resolved_motor_frame
        if output is None:
            raise RuntimeError("motor command resolver is not initialized")
        self.resolve_motor_frame(frame, output)
        self._motor_target = output

    def resolve_motor_frame(
        self,
        frame: MotorFrame,
        output: MotorFrame,
    ) -> MotorFrame:
        resolver = self._command_resolver
        if resolver is None:
            raise RuntimeError("motor command resolver is not initialized")
        resolver.resolve_into(frame, output)
        return output

    def snapshot(self, *, include_graph: bool = False) -> dict[str, object]:
        info = self.state_machine.snapshot(include_graph=include_graph)
        info.update(
            {
                "loop_count": self.loop_count,
                "inference_hz": 1.0 / self.desired_control_period,
                "cmd_vel": {
                    "x": float(self.current_cmd_vel[0]),
                    "y": float(self.current_cmd_vel[1]),
                    "yaw": float(self.current_cmd_vel[2]),
                },
                "mods": [
                    {
                        "id": mod.id,
                        "version": mod.version,
                        "status": mod.status,
                        "error": mod.error,
                        "warnings": list(mod.warnings),
                    }
                    for mod in (
                        *self.mod_runtime.mods,
                        *self.mod_runtime.unavailable_mods,
                        *self.mod_runtime.disabled_mods,
                    )
                ],
                "nodes": self.node_manager.snapshot(),
            }
        )
        return info

    def log_startup(self) -> None:
        events = self.config.get("remote_events")
        event_count = len(events) if isinstance(events, Mapping) else 0
        node_count = len(self.mod_runtime.node_specs)
        self._logger.info(
            f"loaded {len(self.mod_runtime.mods)} Mods, "
            f"{len(self.mod_runtime.unavailable_mods)} unavailable, "
            f"{len(self.mod_runtime.disabled_mods)} disabled, "
            f"{len(self.mod_runtime.state_factories)} states, "
            f"{node_count} nodes, "
            f"{event_count} remote events; input conflicts validated"
        )
        for mod in self.mod_runtime.mods:
            dependencies = (
                f"; requires={','.join(mod.requires)}" if mod.requires else ""
            )
            self._loggers.mod(mod.id).info(
                f"loaded v{mod.version}: {mod.root}{dependencies}"
            )
        for mod in self.mod_runtime.disabled_mods:
            self._loggers.mod(mod.id).info(
                f"disabled v{mod.version}: {mod.root}"
            )
        for mod in self.mod_runtime.unavailable_mods:
            self._loggers.mod(mod.id).warning(
                f"unavailable v{mod.version}: {mod.error}; {mod.root}"
            )

    def is_orientation_unsafe(self, quat_xyzw: object) -> bool:
        angles = quaternion_to_euler_array(quat_xyzw)
        angles[angles > math.pi] -= 2 * math.pi
        return bool(
            (np.abs(angles[0]) > (math.pi / 3.0))
            or (np.abs(angles[1]) > (math.pi / 3.0))
        )

    def preheat_model(
        self,
        model: object,
        command: object | None = None,
    ) -> None:
        if command is not None:
            command_array = np.asarray(command)
            if command_array.shape != self.current_cmd_vel.shape:
                raise ValueError(
                    f"preheat command shape is {command_array.shape}, expected "
                    f"{self.current_cmd_vel.shape}"
                )
            np.copyto(self.current_cmd_vel, command_array, casting="same_kind")
        model.reset(self.inference_frame)
        # Exactly one non-advancing run initializes lazy backend allocations
        # and history without moving the policy timeline.
        model.step(self.inference_frame, self.dt, advance=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        try:
            self._unbind_states(self.robot_states)
        except Exception as exc:
            first_error = exc
        try:
            self.node_manager.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                self._warn_cleanup_failure("Mod node", exc)
        try:
            self.mod_runtime.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                self._warn_cleanup_failure("Mod runtime", exc)
        if first_error is not None:
            raise first_error

    def attach_executor(self, executor: ExecutorLike) -> None:
        self.node_manager.attach_executor(executor)

    def detach_executor(self) -> None:
        self.node_manager.detach_executor()

    def _warn_cleanup_failure(self, component: str, exc: Exception) -> None:
        try:
            self._logger.warning(f"{component} cleanup also failed: {exc}")
        except Exception:
            pass

    def _set_observation(self, observation: RobotObservation) -> None:
        joints = observation.joints
        if self._robot_layout is None:
            self._bind_robot_layout(joints)
        elif (
            joints.layout is not self._robot_layout
            and joints.layout.names != self._robot_layout.names
        ):
            raise ValueError(
                "robot joint layout changed after startup: "
                f"initial={self._robot_layout.names}, current={joints.layout.names}"
            )
        self._robot_joints = joints
        self.inference_frame.joints = joints
        self.inference_frame.timestamp_ns = joints.timestamp_ns

        self._copy_vector(observation.quat_xyzw, self.current_quat_xyzw, "quat_xyzw")
        self._copy_vector(observation.quat_wxyz, self.current_quat_wxyz, "quat_wxyz")
        self._copy_vector(observation.omega, self.current_omega, "omega")
        self._copy_vector(
            observation.raw_cmd_vel,
            self.current_raw_cmd_vel,
            "raw_cmd_vel",
        )
        if not self._initial_state_entered:
            self.state_machine.current.on_enter(self)
            self._initial_state_entered = True

    def _bind_robot_layout(self, joints: JointStateView) -> None:
        layout = joints.layout
        self._robot_layout = layout
        self._robot_joints = joints
        self._command_resolver = JointCommandResolver(
            layout,
            self._command_defaults,
            warning_callback=self._logger.warning,
        )
        self._resolved_motor_frame = MotorFrame.empty(layout)
        self._last_motor_frame = MotorFrame.empty(layout)
        self._last_motor_frame.qpos[:] = joints.position
        self._last_motor_frame.kp.fill(0.0)
        self._last_motor_frame.kd.fill(0.0)
        self._last_motor_frame.vel.fill(0.0)
        self._last_motor_frame.torque.fill(0.0)
        self._inference_frame = InferenceFrame(
            joints=joints,
            quat_wxyz=self.current_quat_wxyz,
            angular_velocity=self.current_omega,
            command=self.current_cmd_vel,
            timestamp_ns=joints.timestamp_ns,
        )

    @staticmethod
    def _copy_vector(source: object, target: np.ndarray, name: str) -> None:
        array = np.asarray(source)
        if array.shape != target.shape:
            raise ValueError(
                f"robot observation {name} has shape {array.shape}, "
                f"expected {target.shape}"
            )
        np.copyto(target, array, casting="same_kind")

    def _bind_states(self, states: Mapping[str, RobotControlState]) -> None:
        bound: list[RobotControlState] = []
        try:
            for state in states.values():
                state._bind_logger(self._loggers.state(state.name))
                state.on_bind(self)
                bound.append(state)
        except BaseException:
            try:
                self._unbind_states({state.name: state for state in bound})
            except Exception as cleanup_exc:
                self._warn_cleanup_failure("partially bound state", cleanup_exc)
            raise

    def _unbind_states(self, states: Mapping[str, RobotControlState]) -> None:
        first_error: Exception | None = None
        for state in reversed(tuple(states.values())):
            try:
                state.on_unbind(self)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


__all__ = ["RobotControlFramework"]
