from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import cast

from bxi_example_py_elf3.utils.robot_state_base import RobotControlState


StateClass = type[RobotControlState]


def _walk_state_classes(base_class: StateClass) -> Iterator[StateClass]:
    for subclass in base_class.__subclasses__():
        yield subclass
        yield from _walk_state_classes(subclass)


def _state_behavior_classes() -> dict[str, StateClass]:
    # On hot reload, newer classes appear later and intentionally replace stale ones.
    return {cls.__name__: cls for cls in _walk_state_classes(RobotControlState)}


def _allocate_state_id(
    state_name: str,
    state_config: Mapping[str, object],
    used_ids: set[int],
    next_id: int,
) -> tuple[int, int]:
    configured_id = state_config.get("id")
    if configured_id is not None:
        if isinstance(configured_id, bool) or not isinstance(configured_id, int):
            raise ValueError(f"state '{state_name}'.id must be an integer")
        state_id = configured_id
        if state_id in used_ids:
            raise ValueError(f"duplicate state id {state_id} for state: {state_name}")
        used_ids.add(state_id)
        return state_id, max(next_id, state_id + 1)

    while next_id in used_ids:
        next_id += 1
    used_ids.add(next_id)
    return next_id, next_id + 1


def build_robot_states(
    config: Mapping[str, object],
) -> dict[str, RobotControlState]:
    states_config = _mapping(config.get("states"), "states")
    if not states_config:
        raise ValueError("state machine config must define states")

    behavior_classes = _state_behavior_classes()
    states: dict[str, RobotControlState] = {}
    used_ids: set[int] = set()
    next_id = 0

    for state_name, raw_state_config in states_config.items():
        state_config = _mapping(raw_state_config, f"states.{state_name}")
        behavior_name = state_config.get("behavior")
        if not isinstance(behavior_name, str) or not behavior_name:
            raise ValueError(
                f"state '{state_name}' must define string field 'behavior'"
            )

        behavior_class = behavior_classes.get(behavior_name)
        if behavior_class is None:
            raise ValueError(
                f"unknown state behavior '{behavior_name}' for state '{state_name}'"
            )

        state_id, next_id = _allocate_state_id(
            state_name, state_config, used_ids, next_id
        )
        params = _mapping(state_config.get("params"), f"states.{state_name}.params")
        constructor = cast(Callable[..., RobotControlState], behavior_class)
        state = constructor(state_name, state_id, **params)

        speed_profile = state_config.get("speed_profile")
        if speed_profile is not None and not isinstance(speed_profile, str):
            raise ValueError(f"states.{state_name}.speed_profile must be a string")
        state.speed_profile_name = speed_profile

        manifest = _mapping(
            state_config.get("manifest"), f"states.{state_name}.manifest"
        )
        state.manifest.update(manifest)
        states[state_name] = state

    return states


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a map")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return cast(Mapping[str, object], value)
