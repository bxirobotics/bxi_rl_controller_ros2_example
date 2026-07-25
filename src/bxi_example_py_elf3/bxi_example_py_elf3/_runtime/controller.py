"""Control framework driven by a thin platform adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.mod_api import MotorFrame, RobotControlState, TransitionSpec
from bxi_example_py_elf3.mod_api.geometry import quaternion_to_euler_array

from .mod_loader import ModRuntime, load_mod_runtime
from .state_builder import build_robot_states
from .state_machine import RemoteEventAdapter, RobotStateMachine

if TYPE_CHECKING:
    from rclpy.node import Node


@dataclass(frozen=True)
class RobotObservation:
    """One coherent robot observation supplied by the platform adapter."""

    q: object
    dq: object
    quat_xyzw: object
    quat_wxyz: object
    omega: object
    raw_cmd_vel: object


class RobotControlFramework:
    """Own Mods, states, transitions and one control-cycle output frame."""

    def __init__(
        self,
        base_config: Mapping[str, object],
        *,
        built_in_mod_root: Path,
        extra_mod_roots: Sequence[Path] | None = None,
        dof_num: int,
        ros_node: Node,
        inference_period: float = 0.02,
        inference_timeout_tolerance: float = 0.005,
    ) -> None:
        self._ros_node = ros_node
        self._closed = True
        self.dof_num = int(dof_num)
        self.dt = float(inference_period)
        self.loop_count = 0

        self.current_q = np.zeros(self.dof_num, dtype=np.float64)
        self.current_dq = np.zeros(self.dof_num, dtype=np.float64)
        self.current_quat_xyzw = np.zeros(4, dtype=np.float64)
        self.current_quat_wxyz = np.zeros(4, dtype=np.float64)
        self.current_omega = np.zeros(3, dtype=np.float64)
        self.current_raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.current_cmd_vel = np.zeros(3, dtype=np.float32)

        # Raw aliases retained by the current Mod API.
        self.qpos = self.current_q
        self.qvel = self.current_dq
        self.quat_xyzw = self.current_quat_xyzw
        self.quat_wxyz = self.current_quat_wxyz
        self.omega = self.current_omega

        self.pos_last = np.zeros(self.dof_num, dtype=np.float32)
        self.kp_last = np.zeros(self.dof_num, dtype=np.float32)
        self.kd_last = np.zeros(self.dof_num, dtype=np.float32)
        self.pos_last_state = np.zeros(self.dof_num, dtype=np.float32)
        self.kp_last_state = np.zeros(self.dof_num, dtype=np.float32)
        self.kd_last_state = np.zeros(self.dof_num, dtype=np.float32)
        self._motor_target: MotorFrame | None = None

        self.inference_period = float(inference_period)
        self.inference_timeout_tolerance = float(inference_timeout_tolerance)
        self.last_inference_frame_time: float | None = None
        self.inference_timeout_count = 0

        runtime: ModRuntime | None = None
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
            )
            self.mod_runtime = runtime
            self.resources = runtime.resources
            self.config = runtime.config
            self.speed_profiles = self.config.get("speed_profiles", {})

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
            self.state_machine = RobotStateMachine(self, self.config, states)
            self.remote_event_adapter = RemoteEventAdapter(
                self.config.get("remote_events", {})
            )
        except BaseException:
            if states_bound:
                try:
                    self._unbind_states(self.robot_states)
                except Exception as cleanup_exc:
                    self._warn_cleanup_failure("state", cleanup_exc)
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
        self.current_cmd_vel.fill(0.0)
        self._motor_target = None

        transition_active = self.state_machine.update(self.dt, events)
        if not transition_active:
            self.state_machine.update_current_state(self.dt)

        frame = self._motor_target
        if frame is not None:
            self.pos_last = frame.qpos
            self.kp_last = frame.kp
            self.kd_last = frame.kd
            self._check_inference_frame_timeout()
        self.loop_count += 1
        return frame

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
    ) -> None:
        self.state_machine.request_transition(
            state_name,
            trigger=trigger,
            transition=transition,
            delay=delay,
        )

    def set_motor_target(self, qpos: object, kp: object, kd: object) -> None:
        self._motor_target = MotorFrame.create(qpos, kp, kd)

    def snapshot(self, *, include_graph: bool = False) -> dict[str, object]:
        info = self.state_machine.snapshot(include_graph=include_graph)
        info.update(
            {
                "loop_count": self.loop_count,
                "cmd_vel": {
                    "x": float(self.current_cmd_vel[0]),
                    "y": float(self.current_cmd_vel[1]),
                    "yaw": float(self.current_cmd_vel[2]),
                },
                "inference_timeout_count": self.inference_timeout_count,
            }
        )
        return info

    def startup_messages(self) -> tuple[str, ...]:
        events = self.config.get("remote_events")
        event_count = len(events) if isinstance(events, Mapping) else 0
        messages = [
            f"loaded {len(self.mod_runtime.mods)} Mods, "
            f"{len(self.mod_runtime.disabled_mods)} disabled, "
            f"{len(self.mod_runtime.state_factories)} states, "
            f"{event_count} remote events; input conflicts validated"
        ]
        for mod in self.mod_runtime.mods:
            dependencies = (
                f"; requires={','.join(mod.requires)}" if mod.requires else ""
            )
            messages.append(f"Mod {mod.id}@{mod.version}: {mod.root}{dependencies}")
        for mod in self.mod_runtime.disabled_mods:
            messages.append(f"Mod {mod.id}@{mod.version}: disabled; {mod.root}")
        return tuple(messages)

    def reset_inference_timeout_monitor(self) -> None:
        self.last_inference_frame_time = None
        self.inference_timeout_count = 0

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
        with_cmd_vel: bool = False,
        cmd_vel: object = None,
    ) -> None:
        q = self.current_q.copy()
        dq = self.current_dq.copy()
        omega = self.current_omega.copy()
        quat_xyzw = self.current_quat_xyzw.copy()
        quat_wxyz = self.current_quat_wxyz.copy()
        command = (
            self.current_cmd_vel.copy()
            if cmd_vel is None
            else np.asarray(cmd_vel, dtype=np.float32)
        )
        history_len = int(getattr(model, "obs_history_len", 1))
        for _ in range(history_len * 2):
            infer_step = getattr(model, "infer_step", None)
            if callable(infer_step):
                infer_step(q, dq, quat_xyzw, omega, command)
                continue
            inference_step = getattr(model, "inference_step")
            if with_cmd_vel:
                inference_step(q, dq, quat_wxyz, omega, command)
            else:
                inference_step(q, dq, quat_wxyz, omega)

    def get_logger(self):
        return self._ros_node.get_logger()

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
            self.mod_runtime.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                self._warn_cleanup_failure("Mod runtime", exc)
        if first_error is not None:
            raise first_error

    def _warn_cleanup_failure(self, component: str, exc: Exception) -> None:
        try:
            self.get_logger().warning(f"{component} cleanup also failed: {exc}")
        except Exception:
            pass

    def _set_observation(self, observation: RobotObservation) -> None:
        self.current_q = np.asarray(observation.q, dtype=np.float64).copy()
        self.current_dq = np.asarray(observation.dq, dtype=np.float64).copy()
        self.current_quat_xyzw = np.asarray(
            observation.quat_xyzw, dtype=np.float64
        ).copy()
        self.current_quat_wxyz = np.asarray(
            observation.quat_wxyz, dtype=np.float64
        ).copy()
        self.current_omega = np.asarray(observation.omega, dtype=np.float64).copy()
        self.current_raw_cmd_vel = np.asarray(
            observation.raw_cmd_vel, dtype=np.float32
        ).copy()

        self.qpos = self.current_q
        self.qvel = self.current_dq
        self.quat_xyzw = self.current_quat_xyzw
        self.quat_wxyz = self.current_quat_wxyz
        self.omega = self.current_omega

    def _bind_states(self, states: Mapping[str, RobotControlState]) -> None:
        bound: list[RobotControlState] = []
        try:
            for state in states.values():
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

    def _check_inference_frame_timeout(self) -> None:
        now = time.perf_counter()
        last = self.last_inference_frame_time
        self.last_inference_frame_time = now
        if last is None or self.inference_period <= 0.0:
            return
        frame_delay = now - last
        timeout_threshold = self.inference_period + self.inference_timeout_tolerance
        if frame_delay <= timeout_threshold:
            return

        self.inference_timeout_count += 1
        state_id = self.current_state_id
        state_name = self.state_name_by_id.get(state_id, str(state_id))
        self.get_logger().warning(
            "inference timeout: "
            f"state={state_name}, delay={frame_delay * 1000.0:.2f}ms, "
            f"limit={self.inference_period * 1000.0:.2f}ms, "
            f"tolerance={self.inference_timeout_tolerance * 1000.0:.2f}ms, "
            f"count={self.inference_timeout_count}"
        )


__all__ = ["RobotControlFramework", "RobotObservation"]
