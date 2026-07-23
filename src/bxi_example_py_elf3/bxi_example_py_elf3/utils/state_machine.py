from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import yaml

from bxi_example_py_elf3.utils.transition_core import (
    TransitionPlan,
    TransitionSession,
    TransitionSpec,
    compile_transition,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample


CtxT = TypeVar("CtxT")
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


class StateBehavior(Generic[CtxT]):
    """Small, transition-agnostic base class for user-defined states."""

    def __init__(self, name: str, state_id: int):
        self.name = name
        self.state_id = state_id
        self.manifest: dict[str, object] = {
            "label": "Unknown",
            "index": None,
            "group": "Base",
            "icon": "warning",
            "confirm": False,
            "confirm_message": "",
        }

    def on_bind(self, ctx: CtxT) -> None:
        pass

    def on_unbind(self, ctx: CtxT) -> None:
        """Release subscriptions, timers, or other state-owned handles."""
        pass

    def on_prepare(self, ctx: CtxT, from_state: StateBehavior[CtxT]) -> None:
        pass

    def on_prepare_cancel(
        self,
        ctx: CtxT,
        from_state: StateBehavior[CtxT],
    ) -> None:
        """Release resources when a prepared transition is cancelled."""
        pass

    def on_enter(self, ctx: CtxT) -> None:
        pass

    def on_update(self, ctx: CtxT, dt: float) -> None:
        pass

    def on_exit(self, ctx: CtxT) -> None:
        pass

    def on_action(self, ctx: CtxT, action_name: str) -> bool:
        return False


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


@dataclass
class PendingTransition:
    rule: TransitionRule
    trigger: str
    elapsed: float = 0.0


@dataclass
class ActiveTransition:
    from_state: StateBehavior[BxiExample]
    to_state: StateBehavior[BxiExample]
    transition: ResolvedTransition
    session: TransitionSession
    trigger: str


@dataclass(frozen=True)
class GraphDiagnostic:
    severity: str
    message: str


class RobotStateMachine:
    def __init__(
        self,
        ctx: BxiExample,
        config: Mapping[str, object],
        states: Mapping[str, StateBehavior[BxiExample]],
        action_handlers: Mapping[str, Callable[[], None]] | None = None,
        *,
        enter_initial: bool = True,
    ) -> None:
        self._ctx = ctx
        self._config = dict(config)
        self._states = dict(states)
        self._actions = dict(action_handlers or {})
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

    def update(self, dt: float, events: Iterable[str]) -> bool:
        if self._active is not None:
            self._handle_events(events)
            if self._active is not None:
                self._update_active_transition(dt)
            return True
        self._handle_events(events)
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

    def request_transition(
        self,
        to_state: str,
        trigger: str = "code",
        transition: TransitionSpec = None,
        delay: float = 0.0,
    ) -> None:
        if delay < 0.0:
            raise ValueError("transition delay must be >= 0")
        resolved = self._resolve_transition(transition, f"request:{trigger}:{to_state}")
        rule = TransitionRule(
            to_state=to_state,
            delay=delay,
            transition=resolved,
        )
        if delay > 0.0:
            self._cancel_active_transition()
            self._pending = PendingTransition(rule, trigger)
        else:
            self._begin_transition(rule, trigger)

    def snapshot(self, include_graph: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "mode": self._runtime_mode(),
            "current": {
                "name": self.current.name,
                "id": self.current.state_id,
                "elapsed": self.state_elapsed,
            },
            "in_transition": self.in_transition,
            "transition": self._active_snapshot(),
            "pending": self._pending_snapshot(),
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
            if rule.delay > 0.0:
                self._cancel_active_transition()
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

    def _begin_transition(self, rule: TransitionRule, trigger: str) -> None:
        if rule.to_state is None:
            if rule.action:
                self._run_action(rule.action)
            return
        if rule.to_state not in self._states:
            raise ValueError(f"unknown transition target: {rule.to_state}")
        if rule.to_state == self.current.name:
            if self._active is not None:
                self._cancel_active_transition()
            return
        self._cancel_active_transition()
        transition = rule.transition or self._default_transition
        to_state = self._states[rule.to_state]
        transition.plan.validate_states(self.current, to_state)
        print(
            f"switch {self.current.name} -> {to_state.name} "
            f"via {transition.name} ({trigger})"
        )
        to_state.on_prepare(self._ctx, self.current)
        try:
            session = transition.plan.create_session(self._ctx, self.current, to_state)
        except Exception:
            to_state.on_prepare_cancel(self._ctx, self.current)
            raise
        self._active = ActiveTransition(
            from_state=self.current,
            to_state=to_state,
            transition=transition,
            session=session,
            trigger=trigger,
        )
        if session.duration <= 0.0:
            try:
                session.update(self._ctx, 0.0)
            except Exception:
                self._cancel_active_transition()
                raise
            self._finish_active_transition()

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
        active.to_state.on_prepare_cancel(self._ctx, active.from_state)

    def _finish_active_transition(self) -> None:
        active = self._active
        if active is None:
            return
        active.from_state.on_exit(self._ctx)
        self.current = active.to_state
        self.state_elapsed = 0.0
        self._pending = None
        self._active = None
        self._fired_after_rules.clear()
        self.current.on_enter(self._ctx)

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
        logger_factory = getattr(self._ctx, "get_logger", None)
        if callable(logger_factory):
            logger = logger_factory()
            if diagnostic.severity == "warning":
                logger.warning(diagnostic.message)
            elif diagnostic.severity == "error":
                logger.error(diagnostic.message)
            else:
                logger.info(diagnostic.message)
        else:
            print(f"{diagnostic.severity}: {diagnostic.message}")

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
            "progress": progress,
            "action": pending.rule.action,
            "transition": pending.rule.transition.name
            if pending.rule.transition
            else self._default_transition.name,
        }

    def _runtime_mode(self) -> str:
        if self._active is not None:
            return "transition"
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
