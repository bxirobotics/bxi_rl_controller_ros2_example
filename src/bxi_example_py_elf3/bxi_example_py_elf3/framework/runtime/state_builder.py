"""Internal construction of configured state instances."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import cast
import zlib

from bxi_example_py_elf3.framework.mod_api import (
    RobotControlState,
    StateBuildContext,
    StateFactory,
)


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


def build_robot_states(
    config: Mapping[str, object],
    factories: Mapping[str, StateFactory],
) -> dict[str, RobotControlState]:
    states_config = _mapping(config.get("states"), "states")
    if not states_config:
        raise ValueError("state machine config must define states")

    states: dict[str, RobotControlState] = {}
    used_ids: set[int] = set()

    for state_name, raw_state_config in states_config.items():
        state_config = _mapping(raw_state_config, f"states.{state_name}")
        factory = factories.get(state_name)
        if factory is None:
            raise ValueError(f"no Mod factory registered for state '{state_name}'")

        state_id = _allocate_state_id(state_name, state_config, used_ids)
        params = _mapping(state_config.get("params"), f"states.{state_name}.params")
        build_context = StateBuildContext(state_name, state_id, params)
        state = factory(build_context)
        if not isinstance(state, RobotControlState):
            raise TypeError(
                f"state factory for '{state_name}' must return RobotControlState"
            )
        build_context.finish()

        speed_profile = state_config.get("speed_profile")
        if speed_profile is not None and not isinstance(speed_profile, str):
            raise ValueError(f"states.{state_name}.speed_profile must be a string")
        state.speed_profile_name = speed_profile

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
        state.inference_hz = inference_hz

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
