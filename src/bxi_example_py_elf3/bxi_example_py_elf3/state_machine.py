from dataclasses import dataclass
import os
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import yaml


def load_state_machine_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"state machine config must be a YAML map: {path}")
    return data


class RemoteEventAdapter:
    """Converts legacy MotionCommands btn_N fields to named edge events."""

    def __init__(self, event_slots: Dict[str, str]):
        self._event_slots = dict(event_slots)
        self._last_values: Dict[str, int] = {}

    def extract_events(self, msg, sync_only: bool = False) -> List[str]:
        events: List[str] = []
        for event_name, slot_config in self._event_slots.items():
            if isinstance(slot_config, dict):
                slot_name = slot_config.get("slot")
                expected_value = slot_config.get("value")
            else:
                slot_name = slot_config
                expected_value = None

            value = int(getattr(msg, slot_name, 0))
            previous = self._last_values.get(event_name)
            self._last_values[event_name] = value
            if sync_only or previous is None:
                continue

            if expected_value is None and value != previous:
                events.append(event_name)
            elif expected_value is not None and value == int(expected_value) and value != previous:
                events.append(event_name)
        return events


class StateBehavior:
    """Base class for user-defined robot states."""

    def __init__(self, name: str, state_id: int):
        self.name = name
        self.state_id = state_id

    def on_enter(self, ctx) -> None:
        pass

    def on_update(self, ctx, dt: float) -> None:
        pass

    def on_exit(self, ctx) -> None:
        pass

    def on_prepare_enter(self, ctx, from_state: "StateBehavior", transition: "TransitionProfile") -> None:
        pass

    def on_exit_transition(
        self,
        ctx,
        to_state: "StateBehavior",
        progress: float,
        transition: "TransitionProfile",
    ) -> None:
        pass

    def on_enter_transition(
        self,
        ctx,
        from_state: "StateBehavior",
        progress: float,
        transition: "TransitionProfile",
    ) -> None:
        pass

    def on_action(self, ctx, action_name: str) -> bool:
        return False


@dataclass(frozen=True)
class TransitionProfile:
    name: str
    duration: float = 0.0
    exit_behavior: str = "hold_last_motor"
    enter_behavior: str = "none"


@dataclass(frozen=True)
class TransitionRule:
    to_state: Optional[str] = None
    event: Optional[str] = None
    after: Optional[float] = None
    delay: float = 0.0
    action: Optional[str] = None
    transition: str = "instant"


@dataclass
class PendingTransition:
    rule: TransitionRule
    trigger: str
    elapsed: float = 0.0


@dataclass
class ActiveTransition:
    from_state: StateBehavior
    to_state: StateBehavior
    profile: TransitionProfile
    trigger: str
    elapsed: float = 0.0


@dataclass(frozen=True)
class GraphDiagnostic:
    severity: str
    message: str


class RobotStateMachine:
    """Unity-style state machine with user-defined StateBehavior classes."""

    def __init__(
        self,
        ctx,
        config: Dict,
        states: Dict[str, StateBehavior],
        action_handlers: Optional[Dict[str, Callable[[], None]]] = None,
    ):
        self._ctx = ctx
        self._config = config
        self._states = dict(states)
        self._actions = action_handlers or {}
        self._profiles = self._parse_profiles(config.get("transition_profiles", {}))
        self._rules = self._parse_state_rules(config.get("states", {}))

        initial_state_name = config.get("initial_state")
        if initial_state_name is None:
            initial_state_name = next(iter(self._states))
        if initial_state_name not in self._states:
            raise ValueError(f"unknown initial_state in state machine config: {initial_state_name}")

        self._run_graph_checks(initial_state_name)
        self._export_graph_from_config(initial_state_name)

        self.current = self._states[initial_state_name]
        self.state_elapsed = 0.0
        self._pending: Optional[PendingTransition] = None
        self._active: Optional[ActiveTransition] = None
        self._fired_after_rules = set()
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
        transition: str = "instant",
        delay: float = 0.0,
    ) -> None:
        rule = TransitionRule(to_state=to_state, delay=delay, transition=transition)
        if delay > 0.0:
            self._pending = PendingTransition(rule=rule, trigger=trigger)
        else:
            self._begin_transition(rule, trigger)

    def _handle_events(self, events: Iterable[str]) -> None:
        event_set = set(events)
        if not event_set:
            return

        for rule in self._rules.get(self.current.name, []):
            if rule.event not in event_set:
                continue

            if rule.action and not rule.to_state:
                self._run_action(rule.action)
                continue

            if rule.delay > 0.0:
                self._pending = PendingTransition(rule=rule, trigger=rule.event or "")
                return

            self._begin_transition(rule, rule.event or "")
            return

    def _handle_after_rules(self) -> None:
        for index, rule in enumerate(self._rules.get(self.current.name, [])):
            if rule.after is None:
                continue
            key = (self.current.name, index)
            if key in self._fired_after_rules:
                continue
            if self.state_elapsed < rule.after:
                continue

            self._fired_after_rules.add(key)
            if rule.action and not rule.to_state:
                self._run_action(rule.action)
                continue
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
            return

        to_state = self._states[rule.to_state]
        profile = self._profiles.get(rule.transition, self._profiles["instant"])

        print(f"switch {self.current.name} -> {to_state.name} via {profile.name} ({trigger})")
        self.current.on_exit(self._ctx)
        to_state.on_prepare_enter(self._ctx, self.current, profile)

        self._active = ActiveTransition(
            from_state=self.current,
            to_state=to_state,
            profile=profile,
            trigger=trigger,
        )

        if profile.duration <= 0.0:
            self._finish_active_transition()

    def _update_active_transition(self, dt: float) -> None:
        if self._active is None:
            return

        self._active.elapsed += dt
        duration = max(self._active.profile.duration, 1e-6)
        progress = min(self._active.elapsed / duration, 1.0)

        self._active.from_state.on_exit_transition(
            self._ctx,
            self._active.to_state,
            progress,
            self._active.profile,
        )
        self._active.to_state.on_enter_transition(
            self._ctx,
            self._active.from_state,
            progress,
            self._active.profile,
        )

        if progress >= 1.0:
            self._finish_active_transition()

    def _finish_active_transition(self) -> None:
        if self._active is None:
            return

        self.current = self._active.to_state
        self.state_elapsed = 0.0
        self._pending = None
        self._active = None
        self._fired_after_rules.clear()
        self.current.on_enter(self._ctx)

    def _run_action(self, action_name: str) -> None:
        handler = self._actions.get(action_name)
        if handler is not None:
            handler()
            return

        if self.current.on_action(self._ctx, action_name):
            return

        raise ValueError(f"state machine action has no handler: {action_name}")

    def _parse_profiles(self, raw_profiles: Dict) -> Dict[str, TransitionProfile]:
        profiles = {
            "instant": TransitionProfile(name="instant", duration=0.0),
        }

        for name, raw_profile in raw_profiles.items():
            raw_profile = raw_profile or {}
            profiles[name] = TransitionProfile(
                name=name,
                duration=float(raw_profile.get("duration", 0.0) or 0.0),
                exit_behavior=raw_profile.get("exit_behavior", "hold_last_motor"),
                enter_behavior=raw_profile.get("enter_behavior", "none"),
            )

        return profiles

    def _parse_state_rules(self, states_config: Dict) -> Dict[str, List[TransitionRule]]:
        rules: Dict[str, List[TransitionRule]] = {}
        for state_name, state_config in states_config.items():
            if state_name not in self._states:
                raise ValueError(f"unknown state in state machine config: {state_name}")

            transitions = (state_config or {}).get("transitions", {})
            state_rules: List[TransitionRule] = []
            state_rules.extend(self._parse_event_rules(transitions.get("on_event", {})))
            state_rules.extend(self._parse_after_rules(transitions.get("after", [])))
            rules[state_name] = state_rules

        return rules

    def _parse_event_rules(self, raw_rules: Dict) -> List[TransitionRule]:
        rules: List[TransitionRule] = []
        for event_name, raw_rule in raw_rules.items():
            if isinstance(raw_rule, str):
                rules.append(TransitionRule(to_state=raw_rule, event=event_name))
                continue

            raw_rule = raw_rule or {}
            rules.append(
                TransitionRule(
                    to_state=raw_rule.get("to"),
                    event=event_name,
                    delay=float(raw_rule.get("delay", 0.0) or 0.0),
                    action=raw_rule.get("action"),
                    transition=raw_rule.get("transition", "instant"),
                )
            )
        return rules

    def _parse_after_rules(self, raw_rules: List[Dict]) -> List[TransitionRule]:
        rules: List[TransitionRule] = []
        for raw_rule in raw_rules:
            raw_rule = raw_rule or {}
            seconds = raw_rule.get("seconds", raw_rule.get("after"))
            rules.append(
                TransitionRule(
                    to_state=raw_rule.get("to"),
                    after=float(seconds),
                    action=raw_rule.get("action"),
                    transition=raw_rule.get("transition", "instant"),
                )
            )
        return rules

    def _graph_config(self) -> Dict:
        graph_config = self._config.get("graph", {}) or {}
        if graph_config is True:
            return {"validate": True}
        if not isinstance(graph_config, dict):
            return {}
        return graph_config

    def _run_graph_checks(self, initial_state_name: str) -> None:
        graph_config = self._graph_config()
        if graph_config.get("validate", True) is False:
            return

        diagnostics = self.validate_graph(initial_state_name)
        errors = [item for item in diagnostics if item.severity == "error"]
        for item in diagnostics:
            self._report_graph_diagnostic(item)
        if errors:
            messages = "; ".join(item.message for item in errors)
            raise ValueError(f"state machine graph validation failed: {messages}")

    def _report_graph_diagnostic(self, diagnostic: GraphDiagnostic) -> None:
        logger = getattr(self._ctx, "get_logger", None)
        if callable(logger):
            ros_logger = logger()
            if diagnostic.severity == "warning":
                if hasattr(ros_logger, "warning"):
                    ros_logger.warning(diagnostic.message)
                else:
                    ros_logger.warn(diagnostic.message)
            else:
                ros_logger.info(diagnostic.message)
            return
        print(f"{diagnostic.severity}: {diagnostic.message}")

    def validate_graph(self, initial_state_name: str) -> List[GraphDiagnostic]:
        diagnostics: List[GraphDiagnostic] = []
        declared_events = set((self._config.get("remote_events", {}) or {}).keys())

        edges = self._graph_edges()
        for from_state, to_state, label, rule in edges:
            if to_state and to_state not in self._states:
                diagnostics.append(GraphDiagnostic(
                    "error",
                    f"state '{from_state}' transition '{label}' targets unknown state '{to_state}'",
                ))
            if rule.transition not in self._profiles:
                diagnostics.append(GraphDiagnostic(
                    "error",
                    f"state '{from_state}' transition '{label}' references unknown profile '{rule.transition}'",
                ))
            if rule.event and rule.event not in declared_events:
                diagnostics.append(GraphDiagnostic(
                    "warning",
                    f"state '{from_state}' listens to undeclared remote event '{rule.event}'",
                ))

        reachable = self._reachable_states(initial_state_name, edges)
        for state_name in self._states:
            if state_name not in reachable:
                diagnostics.append(GraphDiagnostic("warning", f"state '{state_name}' is unreachable from '{initial_state_name}'"))
            if not self._rules.get(state_name):
                diagnostics.append(GraphDiagnostic("warning", f"state '{state_name}' has no configured outgoing transitions"))

        for cycle in self._after_transition_cycles():
            diagnostics.append(GraphDiagnostic(
                "warning",
                "automatic after-transition cycle detected: " + " -> ".join(cycle),
            ))

        diagnostics.append(GraphDiagnostic(
            "info",
            f"state graph loaded: {len(self._states)} states, {len(edges)} transitions",
        ))
        return diagnostics

    def _graph_edges(self) -> List[Tuple[str, Optional[str], str, TransitionRule]]:
        edges: List[Tuple[str, Optional[str], str, TransitionRule]] = []
        for from_state, rules in self._rules.items():
            for rule in rules:
                if rule.event:
                    label = rule.event
                elif rule.after is not None:
                    label = f"after {rule.after:g}s"
                elif rule.action:
                    label = f"action {rule.action}"
                else:
                    label = "transition"
                if rule.action:
                    label += f" / {rule.action}"
                if rule.transition != "instant":
                    label += f" [{rule.transition}]"
                edges.append((from_state, rule.to_state, label, rule))
        return edges

    def _reachable_states(
        self,
        initial_state_name: str,
        edges: List[Tuple[str, Optional[str], str, TransitionRule]],
    ) -> Set[str]:
        adjacency: Dict[str, List[str]] = {}
        for from_state, to_state, _label, _rule in edges:
            if to_state:
                adjacency.setdefault(from_state, []).append(to_state)

        reachable: Set[str] = set()
        stack = [initial_state_name]
        while stack:
            state = stack.pop()
            if state in reachable:
                continue
            reachable.add(state)
            stack.extend(adjacency.get(state, []))
        return reachable

    def _after_transition_cycles(self) -> List[List[str]]:
        adjacency: Dict[str, List[str]] = {}
        for from_state, rules in self._rules.items():
            for rule in rules:
                if rule.after is not None and rule.to_state:
                    adjacency.setdefault(from_state, []).append(rule.to_state)

        cycles: List[List[str]] = []
        stack: List[str] = []
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def dfs(state: str) -> None:
            if state in visiting:
                index = stack.index(state)
                cycles.append(stack[index:] + [state])
                return
            if state in visited:
                return
            visiting.add(state)
            stack.append(state)
            for target in adjacency.get(state, []):
                dfs(target)
            stack.pop()
            visiting.remove(state)
            visited.add(state)

        for state_name in adjacency:
            dfs(state_name)
        return cycles

    def _export_graph_from_config(self, initial_state_name: str) -> None:
        graph_config = self._graph_config()
        export_config = graph_config.get("export", {}) or {}
        if export_config is True:
            export_config = {
                "dot": "/tmp/elf3_state_machine.dot",
                "mermaid": "/tmp/elf3_state_machine.mmd",
            }
        if not isinstance(export_config, dict):
            return

        dot_path = export_config.get("dot")
        mermaid_path = export_config.get("mermaid")
        if dot_path:
            self.export_dot(dot_path, initial_state_name)
            self._report_graph_diagnostic(GraphDiagnostic("info", f"state graph dot exported: {dot_path}"))
        if mermaid_path:
            self.export_mermaid(mermaid_path, initial_state_name)
            self._report_graph_diagnostic(GraphDiagnostic("info", f"state graph mermaid exported: {mermaid_path}"))

    def export_dot(self, path: str, initial_state_name: Optional[str] = None) -> None:
        initial = initial_state_name or self.current_state_name
        lines = [
            "digraph RobotStateMachine {",
            "  rankdir=LR;",
            "  node [shape=box];",
            f"  __start [shape=point];",
            f"  __start -> \"{initial}\";",
        ]
        for state_name in self._states:
            lines.append(f"  \"{state_name}\";")
        for from_state, to_state, label, _rule in self._graph_edges():
            if to_state:
                lines.append(f"  \"{from_state}\" -> \"{to_state}\" [label=\"{label}\"];")
            else:
                lines.append(f"  \"{from_state}\" -> \"{from_state}\" [style=dashed, label=\"{label}\"];")
        lines.append("}")
        self._write_text(path, "\n".join(lines) + "\n")

    def export_mermaid(self, path: str, initial_state_name: Optional[str] = None) -> None:
        initial = initial_state_name or self.current_state_name
        lines = [
            "stateDiagram-v2",
            f"    [*] --> {initial}",
        ]
        for from_state, to_state, label, _rule in self._graph_edges():
            target = to_state or from_state
            lines.append(f"    {from_state} --> {target}: {label}")
        self._write_text(path, "\n".join(lines) + "\n")

    def _write_text(self, path: str, text: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as output_file:
            output_file.write(text)
