"""Internal state-machine execution engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Protocol, cast

import yaml

from bxi_example_py_elf3.framework.runtime.transition import compile_transition
from bxi_example_py_elf3.framework.mod_api.state import StateBehavior
from bxi_example_py_elf3.framework.mod_api.transition import (
    TransitionPlan,
    TransitionSession,
    TransitionSpec,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import LoggerLike, RobotControlContext


ConfigMap = dict[str, object]


def load_state_machine_config(path: str) -> ConfigMap:
    with open(path, "r", encoding="utf-8") as config_file:
        data: object = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"state machine config must be a YAML map: {path}")
    return cast(ConfigMap, data)


class RemoteEventAdapter:
    """Converts MotionCommands fields to named edge events."""

    def __init__(
        self,
        event_slots: Mapping[str, object],
        initial_values: Mapping[str, int] | None = None,
    ) -> None:
        self._event_slots = dict(event_slots)
        self._last_values = {name: 0 for name in self._event_slots}
        if initial_values:
            self._last_values.update(
                {
                    name: int(value)
                    for name, value in initial_values.items()
                    if name in self._last_values
                }
            )

    def extract_events(self, msg: object, sync_only: bool = False) -> list[str]:
        events: list[str] = []
        for event_name, slot_config in self._event_slots.items():
            if isinstance(slot_config, Mapping):
                slot_name = slot_config.get("slot")
                expected_value = slot_config.get("value")
            else:
                slot_name = slot_config
                expected_value = None
            if not isinstance(slot_name, str):
                raise ValueError(
                    f"remote event '{event_name}' must define a string slot"
                )
            value = int(getattr(msg, slot_name, 0))
            previous = self._last_values.get(event_name, 0)
            self._last_values[event_name] = value
            if sync_only:
                continue
            if expected_value is None and value != previous:
                events.append(event_name)
            elif (
                expected_value is not None
                and value == int(expected_value)
                and value != previous
            ):
                events.append(event_name)
        return events


@dataclass(frozen=True)
class ResolvedTransition:
    name: str
    plan: TransitionPlan


@dataclass(frozen=True)
class TransitionRule:
    to_state: str | None = None
    event: str | None = None
    after: float | None = None
    delay: float = 0.0
    action: str | None = None
    transition: ResolvedTransition | None = None
    force: bool = False


@dataclass
class PendingTransition:
    rule: TransitionRule
    trigger: str
    elapsed: float = 0.0


@dataclass
class PendingResourcePreparation:
    rule: TransitionRule
    trigger: str
    from_state: str


@dataclass
class ActiveTransition:
    from_state: StateBehavior[RobotControlContext]
    to_state: StateBehavior[RobotControlContext]
    transition: ResolvedTransition
    session: TransitionSession
    trigger: str
    force: bool


@dataclass(frozen=True)
class GraphDiagnostic:
    severity: str
    message: str


class StateNodeLifecycle(Protocol):
    def prepare_state(self, state_name: str) -> None:
        ...

    def cancel_prepared_state(self, state_name: str) -> None:
        ...

    def finish_transition(self, from_state: str, to_state: str) -> None:
        ...


class RobotStateMachine:
    def __init__(
        self,
        ctx: RobotControlContext,
        config: Mapping[str, object],
        states: Mapping[str, StateBehavior[RobotControlContext]],
        action_handlers: Mapping[str, Callable[[], None]] | None = None,
        node_lifecycle: StateNodeLifecycle | None = None,
        *,
        logger: LoggerLike,
        enter_initial: bool = True,
    ) -> None:
        self._ctx = ctx
        self._config = dict(config)
        self._states = dict(states)
        self._actions = dict(action_handlers or {})
        self._node_lifecycle = node_lifecycle
        self._logger = logger
        self._profile_configs = self._parse_profile_configs(
            self._mapping(config.get("transition_profiles"), "transition_profiles")
        )
        self._profiles = {
            name: self._compile_profile(name, raw)
            for name, raw in self._profile_configs.items()
        }
        self._default_transition = self._read_default_transition(config)
        self._rules = self._parse_state_rules(
            self._mapping(config.get("states"), "states")
        )

        raw_initial = config.get("initial_state")
        initial = (
            str(raw_initial) if raw_initial is not None else next(iter(self._states))
        )
        if initial not in self._states:
            raise ValueError(
                f"unknown initial_state in state machine config: {initial}"
            )

        self._run_graph_checks(initial)
        self._export_graph_from_config(initial)
        self.current = self._states[initial]
        self.state_elapsed = 0.0
        self._pending: PendingTransition | None = None
        self._preparing: PendingResourcePreparation | None = None
        self._active: ActiveTransition | None = None
        self._fired_after_rules: set[tuple[str, int]] = set()
        if enter_initial:
            self.current.on_enter(self._ctx)

    @property
    def current_state_id(self) -> int:
        return self.current.state_id

    @property
    def current_state_name(self) -> str:
        return self.current.name

    @property
    def in_transition(self) -> bool:
        return self._active is not None

    def _transition_request_in_flight(
        self,
        rule: TransitionRule,
        trigger: str,
    ) -> bool:
        """Return whether the same accepted request is already in progress."""

        target_name = rule.to_state
        if target_name is None:
            return False
        transition = rule.transition or self._default_transition

        def same_transition(candidate: ResolvedTransition | None) -> bool:
            resolved = candidate or self._default_transition
            return (
                resolved.name == transition.name
                and resolved.plan.snapshot() == transition.plan.snapshot()
            )

        active = self._active
        if (
            active is not None
            and active.to_state.name == target_name
            and active.trigger == trigger
            and active.force == rule.force
            and same_transition(active.transition)
        ):
            return True
        pending = self._pending
        if (
            pending is not None
            and pending.rule.to_state == target_name
            and pending.trigger == trigger
            and pending.rule.delay == rule.delay
            and pending.rule.force == rule.force
            and same_transition(pending.rule.transition)
        ):
            return True
        preparing = self._preparing
        return (
            preparing is not None
            and preparing.rule.to_state == target_name
            and preparing.from_state == self.current.name
            and preparing.trigger == trigger
            and preparing.rule.delay == rule.delay
            and preparing.rule.force == rule.force
            and same_transition(preparing.rule.transition)
        )

    def requested_inference_hz(self, default_hz: float) -> float:
        """Return the rate required by the current control path.

        A transition may sample both its source and target states.  Running it
        at the higher requested rate prevents either side's history or phase
        from being undersampled.  States without an explicit rate inherit the
        platform default.
        """

        def resolved(state: StateBehavior[RobotControlContext]) -> float:
            configured = state.inference_hz
            return default_hz if configured is None else configured

        active = self._active
        if active is None:
            return resolved(self.current)
        return max(resolved(active.from_state), resolved(active.to_state))

    def update(self, dt: float, events: Iterable[str]) -> bool:
        if self._active is not None:
            self._handle_events(events)
            if self._active is not None:
                self._update_active_transition(dt)
                return True
        else:
            self._handle_events(events)
            if self._active is not None:
                self._update_active_transition(dt)
                return True
        if self._pending is not None:
            self._pending.elapsed += dt
            if self._pending.elapsed >= self._pending.rule.delay:
                pending = self._pending
                self._pending = None
                self._begin_transition(pending.rule, pending.trigger)
        if self._active is not None:
            self._update_active_transition(dt)
            return True
        self.state_elapsed += dt
        self._handle_after_rules()
        if self._active is not None:
            self._update_active_transition(0.0)
            return True
        return False

    def update_current_state(self, dt: float) -> None:
        if self._active is None:
            self.current.on_update(self._ctx, dt)
            # Commit READY resources only after the source state has completed
            # this cycle. Safety or replacement requests issued by on_update()
            # therefore always supersede a stale asynchronous preparation.
            self._poll_resource_preparation()

    def request_transition(
        self,
        to_state: str,
        trigger: str = "code",
        transition: TransitionSpec = None,
        delay: float = 0.0,
        force: bool = False,
    ) -> bool:
        if delay < 0.0:
            raise ValueError("transition delay must be >= 0")
        if not isinstance(force, bool):
            raise TypeError("transition force must be a bool")
        resolved = self._resolve_transition(transition, f"request:{trigger}:{to_state}")
        rule = TransitionRule(
            to_state=to_state,
            delay=delay,
            transition=resolved,
            force=force,
        )
        if self._transition_request_in_flight(rule, trigger):
            return True
        if delay > 0.0:
            if not self._can_enter_transition(rule, trigger):
                return False
            self._cancel_active_transition()
            self._cancel_resource_preparation()
            self._pending = PendingTransition(rule, trigger)
            return True
        return self._begin_transition(rule, trigger)

    def snapshot(self, include_graph: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "mode": self._runtime_mode(),
            "current": {
                "name": self.current.name,
                "id": self.current.state_id,
                "elapsed": self.state_elapsed,
                "inference_hz": self.current.inference_hz,
            },
            "in_transition": self.in_transition,
            "transition": self._active_snapshot(),
            "pending": self._pending_snapshot(),
            "preparing": self._preparing_snapshot(),
        }
        if include_graph:
            result["graph"] = self._graph_snapshot()
        return result

    def _handle_events(self, events: Iterable[str]) -> None:
        event_set = set(events)
        for rule in self._rules.get(self.current.name, []):
            if rule.event not in event_set:
                continue
            if rule.action:
                self._run_action(rule.action)
            if not rule.to_state:
                return
            if self._transition_request_in_flight(rule, rule.event or ""):
                return
            if rule.delay > 0.0:
                if not self._can_enter_transition(rule, rule.event or ""):
                    return
                self._cancel_active_transition()
                self._cancel_resource_preparation()
                self._pending = PendingTransition(rule, rule.event or "")
            else:
                self._begin_transition(rule, rule.event or "")
            return

    def _handle_after_rules(self) -> None:
        for index, rule in enumerate(self._rules.get(self.current.name, [])):
            if rule.after is None:
                continue
            key = (self.current.name, index)
            if key in self._fired_after_rules or self.state_elapsed < rule.after:
                continue
            self._fired_after_rules.add(key)
            if rule.action:
                self._run_action(rule.action)
            if rule.to_state:
                self._begin_transition(rule, f"after:{rule.after}")
            return

    def _begin_transition(self, rule: TransitionRule, trigger: str) -> bool:
        if rule.to_state is None:
            if rule.action:
                self._run_action(rule.action)
            return True
        if self._transition_request_in_flight(rule, trigger):
            return True
        if not self._can_enter_transition(rule, trigger):
            return False
        if rule.to_state == self.current.name:
            if self._active is not None:
                self._cancel_active_transition()
            self._cancel_resource_preparation()
            return True
        transition = rule.transition or self._default_transition
        to_state = self._states[rule.to_state]
        resource_status = self._request_state_resources(to_state)
        if resource_status == "failed":
            self._report_resource_failure(to_state)
            return False
        if resource_status != "ready":
            self._cancel_active_transition()
            self._cancel_resource_preparation()
            self._pending = None
            self._preparing = PendingResourcePreparation(
                rule=rule,
                trigger=trigger,
                from_state=self.current.name,
            )
            self._logger.info(
                "state transition queued for resource preparation: "
                f"from={self.current.name}, to={to_state.name}, trigger={trigger}"
            )
            return True
        return self._commit_transition(rule, trigger, transition, to_state)

    def _commit_transition(
        self,
        rule: TransitionRule,
        trigger: str,
        transition: ResolvedTransition,
        to_state: StateBehavior[RobotControlContext],
    ) -> bool:
        self._cancel_active_transition()
        self._cancel_resource_preparation()
        if self._node_lifecycle is not None:
            try:
                self._node_lifecycle.prepare_state(to_state.name)
            except Exception as exc:
                self._report_node_start_failure(to_state.name, exc)
                return False
        self._logger.info(
            "state transition: "
            f"from={self.current.name}, to={to_state.name}, "
            f"plan={transition.name}, trigger={trigger}"
        )
        try:
            to_state.on_prepare(self._ctx, self.current)
        except Exception:
            if self._node_lifecycle is not None:
                self._node_lifecycle.cancel_prepared_state(to_state.name)
            raise
        try:
            session = transition.plan.create_session(self._ctx, self.current, to_state)
        except Exception:
            try:
                to_state.on_prepare_cancel(self._ctx, self.current)
            finally:
                if self._node_lifecycle is not None:
                    self._node_lifecycle.cancel_prepared_state(to_state.name)
            raise
        self._active = ActiveTransition(
            from_state=self.current,
            to_state=to_state,
            transition=transition,
            session=session,
            trigger=trigger,
            force=rule.force,
        )
        if session.duration <= 0.0:
            try:
                session.update(self._ctx, 0.0)
            except Exception:
                self._cancel_active_transition()
                raise
            self._finish_active_transition()
        return True

    def _poll_resource_preparation(self) -> None:
        preparing = self._preparing
        if preparing is None:
            return
        target_name = preparing.rule.to_state
        if target_name is None:
            self._preparing = None
            return
        to_state = self._states[target_name]
        resource_status = self._state_resource_status(to_state)
        if resource_status == "loading":
            return
        self._preparing = None
        if resource_status == "failed":
            self._report_resource_failure(to_state)
            return
        if self.current.name != preparing.from_state:
            self._logger.info(
                "discarded prepared state transition because source changed: "
                f"from={preparing.from_state}, current={self.current.name}, "
                f"to={to_state.name}"
            )
            return
        if not self._can_enter_transition(preparing.rule, preparing.trigger):
            return
        transition = preparing.rule.transition or self._default_transition
        self._commit_transition(
            preparing.rule,
            preparing.trigger,
            transition,
            to_state,
        )

    @staticmethod
    def _request_state_resources(
        state: StateBehavior[RobotControlContext],
    ) -> str:
        resources = getattr(state, "required_resources", ())
        for resource in resources:
            if resource.status == "failed":
                return "failed"
        for resource in resources:
            if resource.status == "unloaded":
                resource.request()
        return RobotStateMachine._state_resource_status(state)

    @staticmethod
    def _state_resource_status(
        state: StateBehavior[RobotControlContext],
    ) -> str:
        resources = getattr(state, "required_resources", ())
        statuses = tuple(resource.status for resource in resources)
        if "failed" in statuses:
            return "failed"
        if all(status == "ready" for status in statuses):
            return "ready"
        return "loading"

    def _can_enter_transition(self, rule: TransitionRule, trigger: str) -> bool:
        target_name = rule.to_state
        if target_name is None:
            return True
        if target_name not in self._states:
            raise ValueError(f"unknown transition target: {target_name}")
        if target_name == self.current.name:
            return True

        to_state = self._states[target_name]
        if not rule.force and not to_state.is_available(self._ctx):
            self._report_unavailable_state(to_state.name, trigger)
            return False
        transition = rule.transition or self._default_transition
        transition.plan.validate_states(self.current, to_state)
        return True

    def _update_active_transition(self, dt: float) -> None:
        active = self._active
        if active is None:
            return
        try:
            finished = active.session.update(self._ctx, dt)
        except Exception:
            self._cancel_active_transition()
            raise
        if finished:
            self._finish_active_transition()

    def _cancel_active_transition(self) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        try:
            active.to_state.on_prepare_cancel(self._ctx, active.from_state)
        finally:
            if self._node_lifecycle is not None:
                self._node_lifecycle.cancel_prepared_state(active.to_state.name)

    def _cancel_resource_preparation(self) -> None:
        self._preparing = None

    def _finish_active_transition(self) -> None:
        active = self._active
        if active is None:
            return
        active.from_state.on_exit(self._ctx)
        if self._node_lifecycle is not None:
            self._node_lifecycle.finish_transition(
                active.from_state.name,
                active.to_state.name,
            )
        self.current = active.to_state
        self.state_elapsed = 0.0
        self._pending = None
        self._active = None
        self._fired_after_rules.clear()
        self.current.on_enter(self._ctx)

    def _report_node_start_failure(self, state_name: str, exc: Exception) -> None:
        message = f"cannot enter state '{state_name}': Mod node startup failed: {exc}"
        self._logger.error(message)

    def _report_resource_failure(
        self,
        state: StateBehavior[RobotControlContext],
    ) -> None:
        resources = getattr(state, "required_resources", ())
        failures = [
            f"{resource.key.id}: {resource.error}"
            for resource in resources
            if resource.status == "failed"
        ]
        detail = "; ".join(failures) or "unknown resource error"
        self._logger.error(
            f"cannot enter state '{state.name}': resource preparation failed: "
            f"{detail}"
        )

    def _report_unavailable_state(self, state_name: str, trigger: str) -> None:
        message = (
            "state transition rejected because target is unavailable: "
            f"from={self.current.name}, target={state_name}, trigger={trigger}"
        )
        self._logger.warning(message)

    def _run_action(self, action_name: str) -> None:
        handler = self._actions.get(action_name)
        if handler is not None:
            handler()
        elif not self.current.on_action(self._ctx, action_name):
            raise ValueError(f"state machine action has no handler: {action_name}")

    def _parse_profile_configs(
        self, raw_profiles: Mapping[str, object]
    ) -> dict[str, Mapping[str, object]]:
        profiles: dict[str, Mapping[str, object]] = {}
        for name, raw in raw_profiles.items():
            profiles[name] = self._mapping(raw, f"transition_profiles.{name}")
        return profiles

    def _read_default_transition(
        self, config: Mapping[str, object]
    ) -> ResolvedTransition:
        profile_name = config.get("default_transition")
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError("state machine config must define default_transition")
        profile = self._profiles.get(profile_name)
        if profile is None:
            raise ValueError(
                "default_transition references unknown profile " f"'{profile_name}'"
            )
        return profile

    def _compile_profile(
        self, name: str, raw: Mapping[str, object]
    ) -> ResolvedTransition:
        return ResolvedTransition(name=name, plan=compile_transition(name, raw))

    def _resolve_transition(
        self, spec: TransitionSpec, context: str
    ) -> ResolvedTransition:
        if spec is None:
            return self._default_transition
        if isinstance(spec, str):
            profile = self._profiles.get(spec)
            if profile is None:
                raise ValueError(
                    f"transition '{context}' references unknown profile '{spec}'"
                )
            return profile
        raw = dict(self._mapping(spec, context))
        base_name = raw.pop("profile", None)
        if base_name is None:
            merged: dict[str, object] = {}
        else:
            if not isinstance(base_name, str) or base_name not in self._profile_configs:
                raise ValueError(
                    f"transition '{context}' references unknown profile '{base_name}'"
                )
            merged = dict(self._profile_configs[base_name])
        merged.update(raw)
        name = f"inline:{context}"
        return self._compile_profile(name, merged)

    def _parse_state_rules(
        self, states_config: Mapping[str, object]
    ) -> dict[str, list[TransitionRule]]:
        result: dict[str, list[TransitionRule]] = {}
        for state_name, raw_state in states_config.items():
            if state_name not in self._states:
                raise ValueError(f"unknown state in state machine config: {state_name}")
            state = self._mapping(raw_state, f"states.{state_name}")
            transitions = self._mapping(
                state.get("transitions"), f"states.{state_name}.transitions"
            )
            rules = self._parse_event_rules(
                state_name,
                self._mapping(
                    transitions.get("on_event"),
                    f"states.{state_name}.transitions.on_event",
                ),
            )
            raw_after = transitions.get("after", [])
            if not isinstance(raw_after, Sequence) or isinstance(
                raw_after, (str, bytes)
            ):
                raise ValueError(
                    f"states.{state_name}.transitions.after must be a list"
                )
            rules.extend(self._parse_after_rules(state_name, raw_after))
            result[state_name] = rules
        return result

    def _parse_event_rules(
        self, state_name: str, raw_rules: Mapping[str, object]
    ) -> list[TransitionRule]:
        rules: list[TransitionRule] = []
        for event, raw in raw_rules.items():
            if isinstance(raw, str):
                rules.append(
                    TransitionRule(
                        to_state=raw,
                        event=event,
                        transition=self._default_transition,
                    )
                )
                continue
            item = self._mapping(raw, f"{state_name}.{event}")
            rules.append(
                TransitionRule(
                    to_state=self._optional_string(item.get("to")),
                    event=event,
                    delay=self._number(
                        item.get("delay", 0.0),
                        f"{state_name}.{event}.delay",
                        minimum=0.0,
                    ),
                    action=self._optional_string(item.get("action")),
                    transition=self._resolve_transition(
                        item.get("transition"), f"{state_name}.{event}"
                    ),
                )
            )
        return rules

    def _parse_after_rules(
        self, state_name: str, raw_rules: Sequence[object]
    ) -> list[TransitionRule]:
        rules: list[TransitionRule] = []
        for index, raw in enumerate(raw_rules):
            item = self._mapping(raw, f"{state_name}.after[{index}]")
            seconds = item.get("seconds", item.get("after"))
            rules.append(
                TransitionRule(
                    to_state=self._optional_string(item.get("to")),
                    after=self._number(
                        seconds,
                        f"{state_name}.after[{index}].seconds",
                        minimum=0.0,
                    ),
                    action=self._optional_string(item.get("action")),
                    transition=self._resolve_transition(
                        item.get("transition"),
                        f"{state_name}.after[{index}]",
                    ),
                )
            )
        return rules

    def validate_graph(self, initial: str) -> list[GraphDiagnostic]:
        diagnostics: list[GraphDiagnostic] = []
        declared_events = set(
            self._mapping(self._config.get("remote_events"), "remote_events")
        )
        edges = self._graph_edges()
        for from_name, to_name, label, rule in edges:
            if not to_name and not rule.action:
                diagnostics.append(
                    GraphDiagnostic(
                        "error",
                        f"state '{from_name}' transition '{label}' must define "
                        "either 'to' or 'action'",
                    )
                )
            if to_name and to_name not in self._states:
                diagnostics.append(
                    GraphDiagnostic(
                        "error",
                        f"state '{from_name}' transition '{label}' targets "
                        f"unknown state '{to_name}'",
                    )
                )
                continue
            if to_name and rule.transition:
                try:
                    rule.transition.plan.validate_states(
                        self._states[from_name], self._states[to_name]
                    )
                except ValueError as exc:
                    diagnostics.append(
                        GraphDiagnostic(
                            "error",
                            f"invalid transition {from_name} -> {to_name} via "
                            f"'{rule.transition.name}': {exc}",
                        )
                    )
            if rule.event and rule.event not in declared_events:
                diagnostics.append(
                    GraphDiagnostic(
                        "warning",
                        f"state '{from_name}' listens to undeclared remote "
                        f"event '{rule.event}'",
                    )
                )
        reachable = self._reachable_states(initial, edges)
        for state_name in self._states:
            if state_name not in reachable:
                diagnostics.append(
                    GraphDiagnostic(
                        "warning",
                        f"state '{state_name}' is unreachable from '{initial}'",
                    )
                )
            if not self._rules.get(state_name):
                diagnostics.append(
                    GraphDiagnostic(
                        "warning",
                        f"state '{state_name}' has no configured outgoing transitions",
                    )
                )
        for cycle in self._after_transition_cycles():
            diagnostics.append(
                GraphDiagnostic(
                    "warning",
                    "automatic after-transition cycle detected: " + " -> ".join(cycle),
                )
            )
        diagnostics.append(
            GraphDiagnostic(
                "info",
                f"state graph loaded: {len(self._states)} states, "
                f"{len(edges)} transitions",
            )
        )
        return diagnostics

    def _run_graph_checks(self, initial: str) -> None:
        graph = self._mapping(self._config.get("graph"), "graph")
        if graph.get("validate", True) is False:
            return
        diagnostics = self.validate_graph(initial)
        for diagnostic in diagnostics:
            self._report_graph_diagnostic(diagnostic)
        errors = [item.message for item in diagnostics if item.severity == "error"]
        if errors:
            raise ValueError(
                "state machine graph validation failed: " + "; ".join(errors)
            )

    def _report_graph_diagnostic(self, diagnostic: GraphDiagnostic) -> None:
        if diagnostic.severity == "warning":
            self._logger.warning(diagnostic.message)
        elif diagnostic.severity == "error":
            self._logger.error(diagnostic.message)
        else:
            self._logger.info(diagnostic.message)

    def _graph_edges(self) -> list[tuple[str, str | None, str, TransitionRule]]:
        edges: list[tuple[str, str | None, str, TransitionRule]] = []
        for from_name, rules in self._rules.items():
            for rule in rules:
                label = rule.event or (
                    f"after {rule.after:g}s" if rule.after is not None else "transition"
                )
                if rule.action:
                    label += f" / {rule.action}"
                transition_name = (
                    rule.transition.name
                    if rule.transition
                    else self._default_transition.name
                )
                if transition_name != self._default_transition.name:
                    label += f" [{transition_name}]"
                edges.append((from_name, rule.to_state, label, rule))
        return edges

    def _reachable_states(
        self,
        initial: str,
        edges: Sequence[tuple[str, str | None, str, TransitionRule]],
    ) -> set[str]:
        adjacency: dict[str, list[str]] = {}
        for source, target, _label, _rule in edges:
            if target:
                adjacency.setdefault(source, []).append(target)
        reachable: set[str] = set()
        stack = [initial]
        while stack:
            state = stack.pop()
            if state in reachable:
                continue
            reachable.add(state)
            stack.extend(adjacency.get(state, []))
        return reachable

    def _after_transition_cycles(self) -> list[list[str]]:
        adjacency: dict[str, list[str]] = {}
        for from_state, rules in self._rules.items():
            for rule in rules:
                if rule.after is not None and rule.to_state:
                    adjacency.setdefault(from_state, []).append(rule.to_state)

        cycles: list[list[str]] = []
        stack: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(state: str) -> None:
            if state in visiting:
                cycle_start = stack.index(state)
                cycles.append(stack[cycle_start:] + [state])
                return
            if state in visited:
                return
            visiting.add(state)
            stack.append(state)
            for target in adjacency.get(state, []):
                visit(target)
            stack.pop()
            visiting.remove(state)
            visited.add(state)

        for state_name in adjacency:
            visit(state_name)
        return cycles

    def _graph_snapshot(self) -> dict[str, object]:
        return {
            "states": [
                {
                    "name": state.name,
                    "id": state.state_id,
                    "behavior": state.__class__.__name__,
                    "inference_hz": state.inference_hz,
                    **state.manifest,
                }
                for state in self._states.values()
            ],
            "transition_profiles": {
                name: transition.plan.snapshot()
                for name, transition in self._profiles.items()
            },
            "remote_events": dict(
                self._mapping(self._config.get("remote_events"), "remote_events")
            ),
            "actions": self._action_snapshots(),
            "transitions": [
                {
                    "from": source,
                    "to": target,
                    "label": label,
                    "event": rule.event,
                    "after": rule.after,
                    "delay": rule.delay,
                    "action": rule.action,
                    "transition": rule.transition.name
                    if rule.transition
                    else self._default_transition.name,
                    "transition_profile": (
                        rule.transition.plan.snapshot()
                        if rule.transition
                        else self._default_transition.plan.snapshot()
                    ),
                }
                for source, target, label, rule in self._graph_edges()
            ],
        }

    def _action_snapshots(self) -> list[dict[str, object]]:
        raw_actions = self._config.get("actions", ())
        if not isinstance(raw_actions, Sequence) or isinstance(
            raw_actions, (str, bytes)
        ):
            raise ValueError("actions must be a list")
        snapshots: list[dict[str, object]] = []
        for index, raw_action in enumerate(raw_actions):
            action = self._mapping(raw_action, f"actions[{index}]")
            manifest = self._mapping(
                action.get("manifest"), f"actions[{index}].manifest"
            )
            snapshots.append(
                {
                    "from": action.get("from"),
                    "event": action.get("event"),
                    "action": action.get("action"),
                    **manifest,
                }
            )
        return snapshots

    def _active_snapshot(self) -> dict[str, object] | None:
        active = self._active
        if active is None:
            return None
        return {
            "from": {"name": active.from_state.name, "id": active.from_state.state_id},
            "to": {"name": active.to_state.name, "id": active.to_state.state_id},
            "profile": active.transition.name,
            "type": active.transition.plan.type_name,
            "trigger": active.trigger,
            "force": active.force,
            "elapsed": active.session.elapsed,
            "duration": active.session.duration,
            "progress": active.session.progress,
        }

    def _pending_snapshot(self) -> dict[str, object] | None:
        pending = self._pending
        if pending is None:
            return None
        progress = (
            1.0
            if pending.rule.delay <= 0
            else min(pending.elapsed / pending.rule.delay, 1.0)
        )
        return {
            "to": pending.rule.to_state,
            "event": pending.rule.event,
            "trigger": pending.trigger,
            "elapsed": pending.elapsed,
            "delay": pending.rule.delay,
            "force": pending.rule.force,
            "progress": progress,
            "action": pending.rule.action,
            "transition": pending.rule.transition.name
            if pending.rule.transition
            else self._default_transition.name,
        }

    def _preparing_snapshot(self) -> dict[str, object] | None:
        preparing = self._preparing
        if preparing is None:
            return None
        target_name = preparing.rule.to_state
        target = self._states.get(target_name) if target_name is not None else None
        resources = getattr(target, "required_resources", ()) if target else ()
        return {
            "from": preparing.from_state,
            "to": target_name,
            "trigger": preparing.trigger,
            "force": preparing.rule.force,
            "resources": [
                {
                    "id": resource.key.id,
                    "status": resource.status,
                    "error": str(resource.error) if resource.error else None,
                }
                for resource in resources
            ],
            "transition": preparing.rule.transition.name
            if preparing.rule.transition
            else self._default_transition.name,
        }

    def _runtime_mode(self) -> str:
        if self._active is not None:
            return "transition"
        if self._preparing is not None:
            return "preparing"
        if self._pending is not None:
            return "pending"
        return "state"

    def _export_graph_from_config(self, initial: str) -> None:
        graph = self._mapping(self._config.get("graph"), "graph")
        export = self._mapping(graph.get("export"), "graph.export")
        dot_path = export.get("dot")
        mermaid_path = export.get("mermaid")
        if isinstance(dot_path, str) and dot_path:
            self._write_graph_file(dot_path, self._to_dot(initial))
        if isinstance(mermaid_path, str) and mermaid_path:
            self._write_graph_file(mermaid_path, self._to_mermaid(initial))

    def _to_dot(self, initial: str) -> str:
        lines = ["digraph robot_state_machine {", "  rankdir=LR;"]
        for name in self._states:
            shape = "doublecircle" if name == initial else "ellipse"
            lines.append(f'  "{name}" [shape={shape}];')
        for source, target, label, _rule in self._graph_edges():
            if target:
                lines.append(f'  "{source}" -> "{target}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _to_mermaid(self, initial: str) -> str:
        lines = ["stateDiagram-v2", f"  [*] --> {initial}"]
        for source, target, label, _rule in self._graph_edges():
            if target:
                lines.append(f"  {source} --> {target}: {label}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write_graph_file(path: str, content: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            output.write(content)

    @staticmethod
    def _mapping(value: object, context: str) -> Mapping[str, object]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{context} must be a map")
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{context} keys must be strings")
        return cast(Mapping[str, object], value)

    @staticmethod
    def _number(
        value: object,
        context: str,
        *,
        minimum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context} must be a number")
        result = float(value)
        if minimum is not None and result < minimum:
            raise ValueError(f"{context} must be >= {minimum}")
        return result

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"expected string, got {value!r}")
        return value
