"""Internal construction of configured state instances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import cast
import zlib

from bxi_example_py_elf3.framework.mod_api import (
    RobotControlState,
    StateBuildContext,
    StateFactory,
)


@dataclass(frozen=True)
class ConfiguredState:
    """Pure configuration metadata shared by runtime and offline inspection."""

    name: str
    state_id: int
    params: Mapping[str, object]
    speed_profile: str | None
    inference_hz: float | None
    manifest: Mapping[str, object]


def _allocate_state_id(
    state_name: str,
    state_config: Mapping[str, object],
    used_ids: set[int],
) -> int:
    configured_id = state_config.get("id")
    if configured_id is not None:
        if isinstance(configured_id, bool) or not isinstance(configured_id, int):
            raise ValueError(f"state '{state_name}'.id must be an integer")
        if configured_id < 0 or configured_id > 0x7FFFFFFF:
            raise ValueError(f"state '{state_name}'.id must fit a non-negative int32")
        state_id = configured_id
        if state_id in used_ids:
            raise ValueError(f"duplicate state id {state_id} for state: {state_name}")
        used_ids.add(state_id)
        return state_id

    state_id = zlib.crc32(state_name.encode("utf-8")) & 0x7FFFFFFF
    if state_id in used_ids:
        raise ValueError(
            f"stable state id collision for '{state_name}'; set an explicit id"
        )
    used_ids.add(state_id)
    return state_id


def configured_states(config: Mapping[str, object]) -> tuple[ConfiguredState, ...]:
    """Compile state IDs and presentation metadata without importing a Mod."""

    states_config = _mapping(config.get("states"), "states")
    if not states_config:
        raise ValueError("state machine config must define states")

    result: list[ConfiguredState] = []
    used_ids: set[int] = set()

    for state_name, raw_state_config in states_config.items():
        state_config = _mapping(raw_state_config, f"states.{state_name}")
        state_id = _allocate_state_id(state_name, state_config, used_ids)
        params = _mapping(state_config.get("params"), f"states.{state_name}.params")
        speed_profile = state_config.get("speed_profile")
        if speed_profile is not None and not isinstance(speed_profile, str):
            raise ValueError(f"states.{state_name}.speed_profile must be a string")

        inference_hz = state_config.get("inference_hz")
        if inference_hz is not None:
            if isinstance(inference_hz, bool) or not isinstance(
                inference_hz, (int, float)
            ):
                raise ValueError(
                    f"states.{state_name}.inference_hz must be a number"
                )
            inference_hz = float(inference_hz)
            if not math.isfinite(inference_hz) or inference_hz <= 0.0:
                raise ValueError(
                    f"states.{state_name}.inference_hz must be finite and positive"
                )

        manifest = _mapping(
            state_config.get("manifest"), f"states.{state_name}.manifest"
        )
        result.append(
            ConfiguredState(
                name=state_name,
                state_id=state_id,
                params=params,
                speed_profile=cast(str | None, speed_profile),
                inference_hz=cast(float | None, inference_hz),
                manifest=manifest,
            )
        )

    return tuple(result)


def build_robot_states(
    config: Mapping[str, object],
    factories: Mapping[str, StateFactory],
) -> dict[str, RobotControlState]:
    states: dict[str, RobotControlState] = {}

    for configured in configured_states(config):
        factory = factories.get(configured.name)
        if factory is None:
            raise ValueError(
                f"no Mod factory registered for state '{configured.name}'"
            )

        build_context = StateBuildContext(
            configured.name,
            configured.state_id,
            configured.params,
        )
        state = factory(build_context)
        if not isinstance(state, RobotControlState):
            raise TypeError(
                f"state factory for '{configured.name}' must return RobotControlState"
            )
        build_context.finish()

        state.speed_profile_name = configured.speed_profile
        state.inference_hz = configured.inference_hz
        state.manifest.update(configured.manifest)
        states[configured.name] = state

    return states


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a map")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return cast(Mapping[str, object], value)
